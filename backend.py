"""
backend.py — Data Ingestion Engine.

Handles:
  • PDF loading and chunking
  • Codebase / directory loading and chunking (with exclusions)
  • Embedding via local HuggingFace BGE model
  • Persistent storage in ChromaDB
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import tempfile
import pickle
import threading
import sqlite3
import json
import ast as _ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import BinaryIO, Callable, Any

from rank_bm25 import BM25Okapi
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import tree_sitter_languages
from tree_sitter import Node

from config import (
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CODE_CHUNK_SIZE,
    PDF_CHUNK_SIZE,
    CHROMA_DB_DIR,
    CODE_EXTENSIONS,
    EXCLUDED_FILE_PATTERNS,
    ZERO_CHUNK_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════════════════════
#  EMBEDDINGS  (cached at module level so we only load the model once)
# ═══════════════════════════════════════════════════════════════════════════

_embedding_model: HuggingFaceEmbeddings | None = None
_embed_lock = threading.Lock()

def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the singleton embedding model, downloading on first call (thread-safe)."""
    global _embedding_model
    with _embed_lock:
        if _embedding_model is None:
            _embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    return _embedding_model


# ═══════════════════════════════════════════════════════════════════════════
#  TEXT SPLITTERS
# ═══════════════════════════════════════════════════════════════════════════

# Extension → LangChain Language enum mapping for syntax-aware splitting.
_LANGUAGE_MAP: dict[str, Language] = {
    ".py":    Language.PYTHON,
    ".js":    Language.JS,
    ".ts":    Language.TS,
    ".jsx":   Language.JS,
    ".tsx":   Language.TS,
    ".java":  Language.JAVA,
    ".cpp":   Language.CPP,
    ".c":     Language.C,
    ".h":     Language.CPP,
    ".go":    Language.GO,
    ".rs":    Language.RUST,
    ".cs":    Language.CSHARP,
    ".html":  Language.HTML,
    ".md":    Language.MARKDOWN,
    # 💎 Hardware Design (Verilog / SystemVerilog / VHDL)
    ".v":     Language.CPP,  # Verilog syntax-aware splitting via C++ logic
    ".sv":    Language.CPP,  # SystemVerilog
    ".vhd":   Language.CPP,  # VHDL
}


def _get_splitter(
    extension: str = "",
    chunk_size_override: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Return a RecursiveCharacterTextSplitter, optionally language-aware.

    If the file extension maps to a known language, we use
    ``from_language()`` so chunk boundaries respect syntax (functions,
    classes, blocks) instead of cutting mid-statement.
    """
    size = chunk_size_override or CHUNK_SIZE
    lang = _LANGUAGE_MAP.get(extension.lower())
    if lang:
        return RecursiveCharacterTextSplitter.from_language(
            language=lang,
            chunk_size=size,
            chunk_overlap=CHUNK_OVERLAP,
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  AST CODE CHUNKER
# ═══════════════════════════════════════════════════════════════════════════

_PARSER_CACHE = {}
_PARSER_LOCK = threading.Lock()

class CodeASTChunker:
    """
    Syntax-aware chunking using Tree-Sitter.
    Extracts logical units (classes, functions, modules) as atomic chunks.
    """
    def __init__(self, chunk_size: int = CODE_CHUNK_SIZE):
        self.chunk_size = chunk_size

    def _enrich_chunk_header(self, func_name: str, code: str) -> str:
        """Prepend a plain-English descriptor so sparse utility chunks have retrieval signal."""
        enrichments = {
            "_est_": "Estimates",
            "_calc_": "Calculates",
            "_count_": "Counts",
            "_len": "measures length of",
            "_hash": "hashes",
            "_format_": "formats",
        }
        descriptor = next(
            (f"# {v} {func_name}\n" for k, v in enrichments.items() if k in func_name),
            f"# Function: {func_name}\n"
        )
        return descriptor + code

    def _extract_called_functions(self, code: str) -> list[str]:
        """Return all function names called in this code snippet."""
        try:
            tree = _ast.parse(code)
            calls = []
            for node in _ast.walk(tree):
                if isinstance(node, _ast.Call):
                    if isinstance(node.func, _ast.Name):
                        calls.append(node.func.id)
                    elif isinstance(node.func, _ast.Attribute):
                        calls.append(node.func.attr)
            return list(set(calls))
        except Exception:
            import re
            return list(set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)))

    def _extract_referenced_constants(self, code: str) -> list[str]:
        """Return ALL_CAPS names referenced (read or assigned) in this code snippet."""
        try:
            tree = _ast.parse(code)
            constants = set()
            for node in _ast.walk(tree):
                if isinstance(node, _ast.Name) and node.id.isupper() and len(node.id) > 3:
                    constants.add(node.id)
            return list(constants)
        except Exception:
            import re
            return list(set(re.findall(r"\b([A-Z][A-Z0-9_]{3,})\b", code)))

    def _get_parser(self, language: str):
        with _PARSER_LOCK:
            if language not in _PARSER_CACHE:
                # 🚀 Fix Cache Bloom: Limit to 10 parsers (LRU-style)
                if len(_PARSER_CACHE) >= 10:
                    # Remove the oldest entry
                    oldest_key = next(iter(_PARSER_CACHE))
                    del _PARSER_CACHE[oldest_key]
                    logger.debug(f"Evicted parser for {oldest_key} from cache.")
                
                try:
                    _PARSER_CACHE[language] = tree_sitter_languages.get_parser(language)
                except Exception as e:
                    logger.warning(f"Could not load tree-sitter parser for {language}: {e}")
                    _PARSER_CACHE[language] = None
            
            return _PARSER_CACHE[language]

    def chunk_file(self, content: str, filepath: str, extension: str) -> list[Document]:
        """Parse file and return chunks grounded with file/class context."""
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "cpp",
            ".cs": "c_sharp",
        }
        
        lang = lang_map.get(extension.lower())
        parser = self._get_parser(lang) if lang else None

        if not parser:
            # Fallback for Verilog/SystemVerilog or unsupported languages
            if extension.lower() in (".v", ".sv"):
                return self._regex_chunk_hdl(content, filepath, extension)
            return [] # Caller will handle LangChain fallback

        tree = parser.parse(bytes(content, "utf8"))
        root = tree.root_node
        
        chunks = []
        # Extract top-level nodes
        top_level_nodes = []
        for child in root.children:
            if child.type in ("class_definition", "function_definition", "decorated_definition", 
                              "method_definition", "export_statement", "lexical_declaration"):
                top_level_nodes.append(child)
            elif len(child.text) > 200: # Generic large blocks
                top_level_nodes.append(child)

        current_chunk_text = ""
        current_context = f"// File: {os.path.basename(filepath)}"
        
        for node in top_level_nodes:
            node_text = node.text.decode("utf8")
            
            # Extract class context if applicable
            class_name = None
            if node.type == "class_definition":
                for sub in node.children:
                    if sub.type == "identifier":
                        class_name = sub.text.decode("utf8")
                        break
            
            node_prefix = f"\n{current_context}"
            if class_name:
                node_prefix += f" | Class: {class_name}"
            
            # 🚀 Fix F & E: Semantic Enrichment and Call Graph Metadata
            func_name = None
            if node.type in ("function_definition", "method_definition", "decorated_definition"):
                for sub in node.children:
                    if sub.type == "identifier":
                        func_name = sub.text.decode("utf8")
                        break
                    if sub.type == "function_definition": # Handle decorated
                        for ssub in sub.children:
                            if ssub.type == "identifier":
                                func_name = ssub.text.decode("utf8")
                                break

            if func_name:
                node_text = self._enrich_chunk_header(func_name, node_text)

            calls = self._extract_called_functions(node_text)
            consts = self._extract_referenced_constants(node_text)

            # If a single node is too big, split it at internal boundaries (methods)
            if len(node_text) > self.chunk_size:
                # If we have a class, try to chunk into its methods
                if node.type == "class_definition":
                    methods = [c for c in node.children if c.type in ("function_definition", "method_definition")]
                    if methods:
                        for m in methods:
                            meth_text = m.text.decode("utf8")
                            m_name = None
                            for sub in m.children:
                                if sub.type == "identifier":
                                    m_name = sub.text.decode("utf8")
                                    break
                            if m_name:
                                meth_text = self._enrich_chunk_header(m_name, meth_text)

                            m_calls = " ".join(self._extract_called_functions(meth_text))
                            m_consts = " ".join(self._extract_referenced_constants(meth_text))
                            chunks.append(Document(
                                page_content=f"{node_prefix}\n{meth_text}",
                                metadata={"source": filepath, "ast_type": m.type, "calls_functions": m_calls, "references_constants": m_consts}
                            ))
                        continue

                # Recursive fallback for monoliths
                splitter = _get_splitter(extension)
                sub_chunks = splitter.split_text(node_text)
                for sc in sub_chunks:
                    sc_calls = " ".join(self._extract_called_functions(sc))
                    sc_consts = " ".join(self._extract_referenced_constants(sc))
                    chunks.append(Document(
                        page_content=f"{node_prefix} (part)\n{sc}",
                        metadata={"source": filepath, "ast_type": node.type, "calls_functions": sc_calls, "references_constants": sc_consts}
                    ))
            else:
                chunks.append(Document(
                    page_content=f"{node_prefix}\n{node_text}",
                    metadata={"source": filepath, "ast_type": node.type, "calls_functions": " ".join(calls), "references_constants": " ".join(consts)}
                ))

        return chunks

    def _regex_chunk_hdl(self, content: str, filepath: str, extension: str) -> list[Document]:
        """Surgical regex chunking for Verilog/SystemVerilog when AST is unavailable."""
        import re
        # Pattern to match modules, interfaces, and tasks/functions
        patterns = [
            r"(?s)module\s+\w+.*?(?=endmodule)endmodule",
            r"(?s)interface\s+\w+.*?(?=endinterface)endinterface",
            r"(?s)task\s+\w+.*?(?=endtask)endtask",
            r"(?s)function\s+\w+.*?(?=endfunction)endfunction",
        ]
        
        chunks = []
        fname = os.path.basename(filepath)
        
        found = False
        for p in patterns:
            for match in re.finditer(p, content):
                found = True
                block = match.group(0)
                # Extract name
                name_match = re.search(r"(module|interface|task|function)\s+(\w+)", block)
                name = name_match.group(2) if name_match else "unknown"
                
                prefix = f"// File: {fname} | Unit: {name}"
                
                if len(block) > self.chunk_size:
                    # Split logic for massive modules
                    splitter = _get_splitter(extension)
                    for sc in splitter.split_text(block):
                        chunks.append(Document(
                            page_content=f"{prefix} (part)\n{sc}",
                            metadata={"source": filepath, "hdl_type": "regex_block"}
                        ))
                else:
                    chunks.append(Document(
                        page_content=f"{prefix}\n{block}",
                        metadata={"source": filepath, "hdl_type": "regex_block"}
                    ))
        
        return chunks if found else []


# ═══════════════════════════════════════════════════════════════════════════
#  FILE-LEVEL EXCLUSION LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def _is_excluded(filepath: str) -> bool:
    """Return True if the file matches any exclusion pattern."""
    name = os.path.basename(filepath)
    full = filepath.replace("\\", "/")
    for pattern in EXCLUDED_FILE_PATTERNS:
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(full, pattern):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  PDF LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_and_chunk_pdf(file_path: str) -> list[Document]:
    """
    Load a PDF from *file_path* and split it into overlapping text chunks.

    This helper uses ``PyPDFLoader`` from LangChain to read the PDF and then
    applies the library‑wide text splitter (with the PDF‑specific chunk size).

    The function also implements *zero‑chunk* handling: if the combined text
    length of the PDF is below :data:`ZERO_CHUNK_THRESHOLD`, a single
    ``Document`` containing the whole content is returned and flagged with
    ``zero_chunk=True``.  Otherwise, the text is split into overlapping chunks
    and each chunk is enriched with ``chunk_index`` and a stable
    ``content_hash`` for deterministic caching.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        A list of :class:`langchain.docstore.document.Document` objects, each
        representing a chunk of the PDF (or a single merged document when
        zero‑chunking is applied).
    """
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()
    
    # ── Zero Chunking (Phase 3 Upgrade) ──────────────────────────────
    total_content = "\n".join([d.page_content for d in raw_docs])
    if len(total_content) < ZERO_CHUNK_THRESHOLD:
        # Create a single merged document
        merged_doc = Document(
            page_content=total_content,
            metadata={
                "source": file_path,
                "zero_chunk": True,
                "chunk_index": 0,
                "content_hash": hashlib.sha256(total_content.encode("utf-8")).hexdigest()
            }
        )
        return [merged_doc]

    splitter = _get_splitter(chunk_size_override=PDF_CHUNK_SIZE)
    chunks = splitter.split_documents(raw_docs)
    
    # Enrich metadata for cache-stable sorting
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["content_hash"] = _content_hash(chunk)
    return chunks


def load_and_chunk_pdf_upload(uploaded_file: BinaryIO, filename: str) -> list[Document]:
    """
    Accept a Streamlit UploadedFile, write it to a temp file,
    ingest it, then clean up.
    """
    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        return load_and_chunk_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
#  CODEBASE LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _collect_code_files(directory: str) -> list[str]:
    """
    Walk *directory* and return absolute paths of code files
    whose extension is in CODE_EXTENSIONS and that are NOT excluded.
    Also picks up extensionless files named 'Dockerfile'.
    """
    paths: list[str] = []
    for root, dirs, files in os.walk(directory):
        # Prune heavy directories early
        dirs[:] = [
            d for d in dirs
            if d not in {"node_modules", "venv", ".venv", "__pycache__",
                         ".git", "chroma_db", ".tox", ".mypy_cache"}
        ]
        for fname in files:
            fpath = os.path.join(root, fname)
            ext = Path(fname).suffix.lower()

            # Extensionless special files
            if fname in ("Dockerfile", "Makefile", "Jenkinsfile", ".dockerignore"):
                if not _is_excluded(fpath):
                    paths.append(fpath)
                continue

            if ext in CODE_EXTENSIONS and not _is_excluded(fpath):
                paths.append(fpath)
    return paths


def load_and_chunk_codebase(
    directory: str,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> list[Document]:
    """
    Recursively load a code directory, applying language-aware
    chunking per file and attaching source metadata.

    *on_progress(current, total, filename)* is called after each file.
    """
    all_chunks: list[Document] = []
    file_paths = _collect_code_files(directory)
    total = len(file_paths)

    for idx, fpath in enumerate(file_paths):
        if on_progress:
            on_progress(idx + 1, total, os.path.basename(fpath))
        ext = Path(fpath).suffix.lower()
        try:
            # 🚀 Reverting to stable autodetect now that 'chardet' is installed in the venv
            loader = TextLoader(fpath, autodetect_encoding=True)
            raw_docs = loader.load()
            if not raw_docs:
                # Fallback to UTF-8 if autodetect fails to find content
                loader = TextLoader(fpath, encoding="utf-8")
                raw_docs = loader.load()
            if not raw_docs:
                continue
            content = raw_docs[0].page_content
        except Exception:
            try:
                # Secondary Fallback: Force UTF-8 if autodetect crashes on certain characters
                loader = TextLoader(fpath, encoding="utf-8")
                raw_docs = loader.load()
                if not raw_docs:
                    continue
                content = raw_docs[0].page_content
            except Exception as exc:
                logger.warning("Skipping %s: %s", fpath, exc)
                continue

        # ── Zero Chunking (Phase 3 Upgrade) ──────────────────────────────
        if len(content) < ZERO_CHUNK_THRESHOLD:
            # Extract call-graph and constant-reference metadata even for
            # zero-chunks so FTS5 search_by_calls_batch / search_by_constants_batch
            # can find them.  Without this, config.py (the most common zero-chunk)
            # is invisible to Fix E and Fix G.
            _zc_chunker = CodeASTChunker()
            _zc_calls = " ".join(_zc_chunker._extract_called_functions(content))
            _zc_consts = " ".join(_zc_chunker._extract_referenced_constants(content))
            chunk = Document(
                page_content=content,
                metadata={
                    "source": fpath,
                    "source_type": "code",
                    "file_extension": ext,
                    "zero_chunk": True,
                    "chunk_index": 0,
                    "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "calls_functions": _zc_calls,
                    "references_constants": _zc_consts,
                }
            )
            all_chunks.append(chunk)
            continue

        # ── AST-Aware Chunking (New Upgrade) ──────────────────────────────
        ast_chunker = CodeASTChunker(chunk_size=CODE_CHUNK_SIZE)
        ast_chunks = ast_chunker.chunk_file(content, fpath, ext)
        
        if ast_chunks:
            # Enrich metadata and add to total
            for i, chunk in enumerate(ast_chunks):
                chunk.metadata["source_type"] = "code"
                chunk.metadata["file_extension"] = ext
                chunk.metadata["chunk_index"] = i
                chunk.metadata["content_hash"] = _content_hash(chunk)
            all_chunks.extend(ast_chunks)
            continue

        # Fallback to legacy splitter if AST returns nothing
        splitter = _get_splitter(ext, chunk_size_override=CODE_CHUNK_SIZE)
        chunks = splitter.split_documents(raw_docs)

        # Enrich metadata for citation and cache-stable sorting
        for i, chunk in enumerate(chunks):
            chunk.metadata["source_type"] = "code"
            chunk.metadata["file_extension"] = ext
            chunk.metadata["chunk_index"] = i
            chunk.metadata["content_hash"] = _content_hash(chunk)
        all_chunks.extend(chunks)

    return all_chunks


# ═══════════════════════════════════════════════════════════════════════════
#  CHROMADB INGESTION
# ═══════════════════════════════════════════════════════════════════════════

def _content_hash(doc: Document) -> str:
    """Return a SHA-256 hex digest of a document's page_content only."""
    content_str = doc.page_content
    return hashlib.sha256(content_str.encode("utf-8")).hexdigest()


def ingest_into_chroma(
    documents: list[Document],
    collection_name: str = "default",
) -> tuple[Chroma, int]:
    """
    Embed *documents* and upsert them into a persistent ChromaDB collection,
    skipping duplicates based on content hash.

    Returns (Chroma instance, number of new documents added).
    """
    if not documents:
        raise ValueError("No documents to ingest — check your file/folder path.")

    # Stamp content_hash only on documents that don't already have one.
    # load_and_chunk_codebase and load_and_chunk_pdf set this during chunking,
    # so recomputing it here is redundant for the normal ingestion path.
    for doc in documents:
        if "content_hash" not in doc.metadata:
            doc.metadata["content_hash"] = _content_hash(doc)

    embedding = get_embedding_model()

    # Try loading existing collection to deduplicate
    existing_db = load_existing_chroma(collection_name)
    if existing_db is not None:
        # Delete stale chunks from modified files before adding new ones.
        # Without this, re-ingesting a changed file leaves BOTH old and new
        # versions in the DB, feeding contradictory context to the LLM.
        #
        # Two-phase metadata scan (bounded RAM, no unfiltered full-dump):
        #   Phase A — targeted server-side filter for stale rows only:
        #             where={"source": {"$in": [...]}}. Gives us both
        #             stale_ids and stale_hashes in one call. Previously
        #             this did a full collection get() then python-side
        #             filtering — OOM risk on large corpora.
        #   Phase B — paged scan to build existing_hashes for dedup.
        #             We still must see every hash, but we page in chunks
        #             of PAGE so peak RAM is bounded regardless of
        #             collection size.
        incoming_sources = {d.metadata.get("source") for d in documents if d.metadata.get("source")}

        # Phase A: fetch only stale rows via server-side $in filter.
        stale_ids: list[str] = []
        stale_hashes: set[str] = set()
        if incoming_sources:
            src_list = list(incoming_sources)
            try:
                # Chroma supports {"$in": [...]} directly; a single-key
                # dict also works for a single source.
                if len(src_list) == 1:
                    where = {"source": src_list[0]}
                else:
                    where = {"source": {"$in": src_list}}
                stale_data = existing_db.get(where=where, include=["metadatas"])
                stale_ids = list(stale_data.get("ids", []))
                for meta in stale_data.get("metadatas", []) or []:
                    if meta and "content_hash" in meta:
                        stale_hashes.add(meta["content_hash"])
            except Exception as e:
                # Filter variants differ between Chroma versions; fall back
                # to a paged scan rather than crash.
                logger.warning("Chroma where-filter failed (%s); falling back to paged scan.", e)
                stale_ids = []
                stale_hashes = set()
                PAGE = 10000
                offset = 0
                while True:
                    batch = existing_db.get(limit=PAGE, offset=offset, include=["metadatas"])
                    ids = batch.get("ids", []) or []
                    metas = batch.get("metadatas", []) or []
                    if not ids:
                        break
                    for did, meta in zip(ids, metas):
                        if meta and meta.get("source") in incoming_sources:
                            stale_ids.append(did)
                            if "content_hash" in meta:
                                stale_hashes.add(meta["content_hash"])
                    if len(ids) < PAGE:
                        break
                    offset += PAGE

        if stale_ids:
            # Use the public delete() wrapper (avoids reaching into
            # ._collection, which is version-fragile).
            try:
                existing_db.delete(ids=stale_ids)
            except AttributeError:
                existing_db._collection.delete(ids=stale_ids)
            # Purge matching FTS5 entries so BM25 doesn't return stale content
            if stale_hashes:
                fts = SQLiteFTS5BM25(collection_name)
                fts.delete_by_hashes(stale_hashes)
            logger.info("🗑️ Ingestion: Deleted %d stale chunks from %d re-ingested files.", len(stale_ids), len(incoming_sources))

        # Phase B: paged scan to collect existing (non-stale) hashes for dedup.
        # Bounded peak RAM by PAGE rows regardless of collection size.
        existing_hashes: set[str] = set()
        stale_id_set = set(stale_ids)
        PAGE = 10000
        offset = 0
        while True:
            batch = existing_db.get(limit=PAGE, offset=offset, include=["metadatas"])
            ids = batch.get("ids", []) or []
            metas = batch.get("metadatas", []) or []
            if not ids:
                break
            for did, meta in zip(ids, metas):
                if not meta:
                    continue
                if did in stale_id_set:
                    continue  # about-to-delete; don't block new versions
                h = meta.get("content_hash")
                if h:
                    existing_hashes.add(h)
            if len(ids) < PAGE:
                break
            offset += PAGE

        new_docs = [d for d in documents if d.metadata["content_hash"] not in existing_hashes]
        if not new_docs:
            logger.info("⚡ Ingestion: All %d chunks are already in the database. 100%% De-duplicated.", len(documents))
            return existing_db, 0

        skipped = len(documents) - len(new_docs)
        if skipped > 0:
            logger.info("⚡ Ingestion: Adding %d new chunks. (Skipped %d duplicates)", len(new_docs), skipped)

        existing_db.add_documents(new_docs)
        # Keep BM25 in sync with ChromaDB — must update here too,
        # not just on the fresh-collection path below.
        _update_bm25_index(new_docs, collection_name)
        invalidate_collection_info_cache(collection_name)
        return existing_db, len(new_docs)

    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=CHROMA_DB_DIR,
        collection_name=collection_name,
    )

    # 🚀 Platinum Upgrade: Build Hybrid BM25 Index
    _update_bm25_index(documents, collection_name)
    invalidate_collection_info_cache(collection_name)

    return db, len(documents)


def _get_bm25_path(collection_name: str) -> str:
    """Return the filesystem path for the BM25 index pickle."""
    return os.path.join(CHROMA_DB_DIR, f"{collection_name}_bm25.pkl")


# 🚀 Platinum Optimization: In-process BM25 Registry
_bm25_cache: dict[str, dict] = {}
_bm25_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════
#  SQLITE BM25 ENGINE (New Scaling Infrastructure)
# ═══════════════════════════════════════════════════════════════════════════

_FTS5_GLOBAL_LOCK = threading.Lock()

# 🚀 SQLite Optimization: Use thread-local connections to prevent locking and speed up hybrid search
_sqlite_connections = threading.local()

class SQLiteFTS5BM25:
    """
    On-disk Full Text Search engine replacing RAM-heavy rank_bm25.
    Uses SQLite's FTS5 extension which is built into standard Python.
    Reuses a thread-local connection to avoid open/close overhead and handle concurrency.
    """
    def __init__(self, collection_name: str):
        self.db_path = os.path.join(CHROMA_DB_DIR, f"{collection_name}_fts5.db")
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a thread-local reusable connection."""
        if not hasattr(_sqlite_connections, "conn"):
            # Use high timeout and write-ahead logging (WAL) for better concurrency
            conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            _sqlite_connections.conn = conn
        return _sqlite_connections.conn

    def _init_db(self):
        conn = self._get_conn()
        # Enable FTS5 and create table with indexed symbols (calls + constants)
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(content, source_name, metadata_json UNINDEXED, calls, constants)")
        except sqlite3.OperationalError:
            conn.execute("DROP TABLE IF EXISTS docs_fts")
            conn.execute("CREATE VIRTUAL TABLE docs_fts USING fts5(content, source_name, metadata_json UNINDEXED, calls, constants)")
        conn.execute("CREATE TABLE IF NOT EXISTS hashes (content_hash TEXT PRIMARY KEY)")
        try:
            conn.execute("ALTER TABLE hashes ADD COLUMN rowid_ref INTEGER")
        except sqlite3.OperationalError:
            pass
        conn.commit()

    def add_documents(self, documents: list[Document]):
        """Incremental on-disk indexing."""
        conn = self._get_conn()
        try:
            for doc in documents:
                h = doc.metadata.get("content_hash", _content_hash(doc))
                # Skip if already indexed
                res = conn.execute("SELECT 1 FROM hashes WHERE content_hash = ?", (h,)).fetchone()
                if res: continue

                meta_json = json.dumps(doc.metadata)
                source_name = os.path.basename(doc.metadata.get("source", "")).lower()
                calls = doc.metadata.get("calls_functions", "")
                constants = doc.metadata.get("references_constants", "")
                cursor = conn.execute(
                    "INSERT INTO docs_fts(content, source_name, metadata_json, calls, constants) VALUES (?, ?, ?, ?, ?)",
                    (doc.page_content, source_name, meta_json, calls, constants)
                )
                rowid = cursor.lastrowid
                conn.execute("INSERT INTO hashes(content_hash, rowid_ref) VALUES (?, ?)", (h, rowid))
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # We no longer close the connection here to keep it persistent for the thread
        pass
        
    def close(self):
        """Explicitly close the connection (thread-local cleanup handled manually if needed)."""
        if hasattr(_sqlite_connections, "conn"):
            try:
                _sqlite_connections.conn.close()
                del _sqlite_connections.conn
            except Exception:
                pass
                self._conn = None

    def search(self, query: str, k: int = 10) -> list[Document]:
        """Fast keyword search via SQLite FTS5."""
        if not os.path.exists(self.db_path):
            return []

        # Clean query for FTS5 (strip special chars that break FTS5 grammar)
        # Preserve underscores and dots — critical for code identifiers
        clean_query = "".join(c if c.isalnum() or c.isspace() or c in "_.+#" else " " for c in query)
        clean_query = clean_query.strip()
        if not clean_query:
            return []

        with self._lock:
            conn = self._get_conn()
            try:
                # Use BM25 scoring via FTS5 'rank'
                # Extended query: match content OR source_name OR calls OR constants
                rows = conn.execute(
                    "SELECT content, metadata_json FROM docs_fts WHERE docs_fts MATCH ? ORDER BY rank LIMIT ?",
                    (clean_query, k)
                ).fetchall()
            except sqlite3.OperationalError:
                # Fallback for empty or invalid queries
                return []
        docs = []
        for content, meta_raw in rows:
            docs.append(Document(page_content=content, metadata=json.loads(meta_raw)))
        return docs

    def search_by_call(self, func_name: str, k: int = 10) -> list[Document]:
        """Directly query the 'calls' index for specific function call sites."""
        if not os.path.exists(self.db_path):
            return []
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT content, metadata_json FROM docs_fts WHERE calls MATCH ? LIMIT ?",
                    (func_name, k)
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [Document(page_content=c, metadata=json.loads(m)) for c, m in rows]

    def search_by_calls_batch(self, terms: list[str], k: int = 10) -> list[Document]:
        """Batch query: find chunks calling ANY of the given function names (FTS5 OR)."""
        if not terms or not os.path.exists(self.db_path):
            return []
        or_query = " OR ".join(f'"{t}"' for t in terms)
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT content, metadata_json FROM docs_fts WHERE calls MATCH ? LIMIT ?",
                    (or_query, k)
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [Document(page_content=c, metadata=json.loads(m)) for c, m in rows]

    def search_by_constant(self, const_name: str, k: int = 10) -> list[Document]:
        """Directly query the 'constants' index for chunks referencing a specific constant."""
        if not os.path.exists(self.db_path):
            return []
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT content, metadata_json FROM docs_fts WHERE constants MATCH ? LIMIT ?",
                    (const_name, k)
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [Document(page_content=c, metadata=json.loads(m)) for c, m in rows]

    def search_by_constants_batch(self, terms: list[str], k: int = 10) -> list[Document]:
        """Batch query: find chunks referencing ANY of the given constants (FTS5 OR)."""
        if not terms or not os.path.exists(self.db_path):
            return []
        or_query = " OR ".join(f'"{t}"' for t in terms)
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT content, metadata_json FROM docs_fts WHERE constants MATCH ? LIMIT ?",
                    (or_query, k)
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [Document(page_content=c, metadata=json.loads(m)) for c, m in rows]

    def delete_by_hashes(self, hashes: set[str]):
        """Remove stale entries from both FTS5 and the hash dedup table.

        FTS5 virtual tables don't support DELETE with arbitrary WHERE clauses
        on content columns.  We use the ``metadata_json`` (which stores the
        content_hash) to locate matching rowids, then delete by rowid.
        """
        if not hashes or not os.path.exists(self.db_path):
            return
        with self._lock:
            conn = self._get_conn()
            # Identify rowids using the optimized hashes mapping table
            hash_list = list(hashes)
            stale_rowids = []
            for i in range(0, len(hash_list), 900):
                batch = hash_list[i:i+900]
                placeholders = ",".join("?" * len(batch))
                # JOIN or simple lookup? Lookup is safer for virtual tables.
                cursor = conn.execute(f"SELECT rowid_ref FROM hashes WHERE content_hash IN ({placeholders})", batch)
                stale_rowids.extend([r[0] for r in cursor.fetchall() if r[0] is not None])
            if stale_rowids:
                # Batch DELETE via IN(...) placeholders — one statement per
                # table instead of N round-trips. Chunked to max 900 params.
                for i in range(0, len(stale_rowids), 900):
                    batch = stale_rowids[i:i+900]
                    placeholders = ",".join("?" * len(batch))
                    conn.execute(f"DELETE FROM docs_fts WHERE rowid IN ({placeholders})", batch)
                hash_list = list(hashes)
                for i in range(0, len(hash_list), 900):
                    batch = hash_list[i:i+900]
                    hph = ",".join("?" * len(batch))
                    conn.execute(f"DELETE FROM hashes WHERE content_hash IN ({hph})", batch)
                conn.commit()
                logger.info("FTS5: Deleted %d stale entries.", len(stale_rowids))


def _update_bm25_index(new_docs: list[Document], collection_name: str):
    """
    Transitioned to SQLite FTS5 for incremental, disk-based indexing.
    """
    fts = SQLiteFTS5BM25(collection_name)
    fts.add_documents(new_docs)


def load_existing_chroma(collection_name: str = "default") -> Chroma | None:
    """
    Load a previously-persisted ChromaDB collection.
    Returns None if the collection doesn't exist on disk or is empty.
    """
    if not os.path.isdir(CHROMA_DB_DIR):
        return None

    # 🚀 Platinum Safety: Verify the collection exists in the client's registry 
    # before attempting to instantiate the LangChain wrapper.
    try:
        available = list_collections()
        if collection_name not in available:
            logger.warning("Collection '%s' not found in registry.", collection_name)
            return None
    except Exception:
        # If client initialization fails, we can't load anything
        return None

    embedding = get_embedding_model()
    try:
        db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedding,
            collection_name=collection_name,
        )
        # Verify non-empty
        count = db._collection.count()
        if count == 0:
            return None
        return db
    except Exception as e:
        logger.error("Failed to load existing collection '%s': %s", collection_name, e)
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  CACHED CHROMADB CLIENT  (avoids re-opening SQLite on every sidebar render)
# ═══════════════════════════════════════════════════════════════════════════
_chroma_client = None
_chroma_client_lock = threading.Lock()

def _get_chroma_client():
    """Return a module-level PersistentClient, creating it once."""
    global _chroma_client
    if _chroma_client is None:
        with _chroma_client_lock:
            if _chroma_client is None:
                import chromadb
                _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return _chroma_client


def list_collections() -> list[str]:
    """Return the names of all user-facing ChromaDB collections on disk."""
    if not os.path.isdir(CHROMA_DB_DIR):
        return []
    _INTERNAL_COLLECTIONS = {"semantic_cache"}
    client = _get_chroma_client()
    return [c.name for c in client.list_collections() if c.name not in _INTERNAL_COLLECTIONS]


_collection_info_cache: dict[str, dict] = {}
_collection_info_lock = threading.Lock()

def get_collection_info(collection_name: str, use_cache: bool = True) -> dict:
    """Return chunk count and list of unique source files for a collection.

    Results are cached in-process so Streamlit sidebar reruns (which happen
    on every interaction) don't re-scan all metadata from ChromaDB.  The
    cache is invalidated on ingestion and collection deletion.
    """
    if not os.path.isdir(CHROMA_DB_DIR):
        return {"count": 0, "sources": []}

    if use_cache:
        with _collection_info_lock:
            cached = _collection_info_cache.get(collection_name)
            if cached is not None:
                return cached

    client = _get_chroma_client()
    try:
        coll = client.get_collection(collection_name)
    except Exception:
        return {"count": 0, "sources": []}
    count = coll.count()
    sources = set()
    
    # Fix OOM: Fetch metadata in pages to avoid massive RAM spikes
    limit = 1000
    offset = 0
    while offset < count:
        data = coll.get(include=["metadatas"], limit=limit, offset=offset)
        for meta in data.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        offset += limit
        
    result = {"count": count, "sources": sorted(sources)}
    with _collection_info_lock:
        _collection_info_cache[collection_name] = result
    return result


def invalidate_collection_info_cache(collection_name: str | None = None):
    """Clear cached collection info after ingestion or deletion."""
    with _collection_info_lock:
        if collection_name:
            _collection_info_cache.pop(collection_name, None)
        else:
            _collection_info_cache.clear()


def delete_collection(collection_name: str) -> bool:
    """Delete a ChromaDB collection and its associated BM25 index. Returns True if deleted."""
    if not os.path.isdir(CHROMA_DB_DIR):
        return False
    client = _get_chroma_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        return False
    # 🚀 Platinum Fix: Also delete the SQLite FTS5 index
    fts_path = os.path.join(CHROMA_DB_DIR, f"{collection_name}_fts5.db")
    if os.path.exists(fts_path):
        try:
            os.remove(fts_path)
        except Exception as e:
            logger.warning("Could not delete FTS5 index for '%s': %s", collection_name, e)

    with _bm25_lock:
        _bm25_cache.pop(collection_name, None)
    invalidate_collection_info_cache(collection_name)
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  ASYNC INGESTION (Phase 1a)
# ═══════════════════════════════════════════════════════════════════════════

class AsyncIngestionTask:
    """
    Run ingestion (loading, chunking, and embedding) in a background thread
    so the Streamlit UI stays responsive.
    """

    def __init__(self, target_path: str, collection_name: str = "default", is_pdf: bool = False):
        self.target_path = target_path
        self.collection_name = collection_name
        self.is_pdf = is_pdf
        self.progress: float = 0.0          # 0.0 to 1.0
        self.status: str = "pending"        # pending | running | done | error
        self.current_step: str = ""         # "Collecting files...", "Embedding chunks..."
        self.result: tuple | None = None    # (Chroma, added_count) on success
        self.error: str = ""
        self._thread: threading.Thread | None = None

    def start(self):
        """Launch the ingestion process in a background thread."""
        self.status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            # 1. Loading Phase
            self.current_step = f"Loading {'PDF' if self.is_pdf else 'codebase'}..."
            self.progress = 0.1
            
            if self.is_pdf:
                # Route through load_and_chunk_pdf so zero-chunking,
                # content_hash, and chunk_index metadata are all set
                # consistently — same as the sync ingestion path.
                chunks = load_and_chunk_pdf(self.target_path)
            else:
                # Codebase ingestion with file-by-file progress
                def _update_progress(curr, tot, name):
                    self.current_step = f"Collecting codebase: {name}"
                    # Loading phase covers 0.1 to 0.4 progress
                    self.progress = 0.1 + (curr / tot) * 0.3

                chunks = load_and_chunk_codebase(self.target_path, on_progress=_update_progress)

            if not chunks:
                self.error = "No relevant content found to ingest."
                self.status = "error"
                return

            # 2. Ingestion Phase
            self.current_step = f"Embedding {len(chunks)} chunks into ChromaDB..."
            self.progress = 0.5
            
            # Note: ChromaDB ingestion is synchronous but embedding happens here
            db, added = ingest_into_chroma(chunks, self.collection_name)
            
            self.progress = 1.0
            self.current_step = f"Successfully ingested {added} chunks!"
            self.result = (db, added)
            self.status = "done"
        except Exception as e:
            self.error = f"Ingestion failed: {str(e)}"
            self.status = "error"

    @property
    def is_done(self) -> bool:
        return self.status in ("done", "error")



def summarize_document_for_pin(file_path: str, max_chars: int = 3000) -> str:
    """
    Context Summarization (Phase 2b): Create a compact summary of
    a large file for pinning instead of the full content.

    Uses extractive summarization (no LLM call):
      - For code: extracts function/class signatures + docstrings
      - For text: extracts first N characters with paragraph boundaries
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return ""

    if not content:
        return ""

    ext = Path(file_path).suffix.lower()

    # Code files: extract signatures
    if ext in (".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"):
        return _extract_code_signatures(content, max_chars)

    # Text/PDF/markup: extract leading paragraphs
    paragraphs = content.split("\n\n")
    summary = ""
    for para in paragraphs:
        if len(summary) + len(para) > max_chars:
            break
        summary += para + "\n\n"
    return summary.strip() if summary else content[:max_chars]


def _extract_code_signatures(content: str, max_chars: int) -> str:
    """Extract function/class definitions and docstrings from code."""
    import re
    lines = content.split("\n")
    signatures = []
    total_len = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match common definition patterns
        if (stripped.startswith(("def ", "class ", "function ", "func ",
                                 "export ", "public ", "private ", "async def "))
                or re.match(r"^(const|let|var)\s+\w+\s*=\s*(async\s+)?\(", stripped)):
            # Include the signature line
            signatures.append(line)
            total_len += len(line)
            # Include docstring/comment on next line if present
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith(('"""', "'''", "//", "/*", "#", "*")):
                    signatures.append(lines[i + 1])
                    total_len += len(lines[i + 1])
            if total_len > max_chars:
                break

    if not signatures:
        return content[:max_chars]

    return "\n".join(signatures)
