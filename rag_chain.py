"""
rag_chain.py — Retrieval-Augmented Generation Query Pipeline.

Handles:
  • OpenRouter LLM configuration (free model by default)
  • ChromaDB retriever setup
  • LangChain retrieval chain construction
"""

from __future__ import annotations
import re
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import logging
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    DEFAULT_MODEL,
    CLOUDROUTER_MODELS,
    OLLAMA_BASE_URL,
    OLLAMA_MODELS,
    LLM_TEMPERATURE,
    RETRIEVER_K,
    MAX_TOKENS,
    ANTHROPIC_CACHE_BETA_HEADER,
    ENABLE_PROMPT_CACHING,
    ENABLE_AUTO_SPECIALIST,
    MAX_CACHE_CHECKPOINTS,
    SEMANTIC_CACHE_THRESHOLD,
    SENTINEL_MAX_TOKENS,
    SENTINEL_TOKEN_THRESHOLD,
    SENTINEL_INTERVAL,
    TRUST_NATIVE_CACHE,
    PROVIDER_CACHE_PROFILES,
    ENABLE_HYBRID_SEARCH,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    USE_RERANKER,
    RERANK_MODEL,
    RERANK_TOP_K,
    RERANK_CANDIDATES,
    PINNED_RELEVANCE_THRESHOLD,
    STICKY_PINNED_CONTEXT,
    SPECIALIST_MAPPING,
    GHOST_HISTORY_WINDOW,
    GHOST_HISTORY_MAX,
    AI_RESPONSE_MAX_CHARS,
    GHOST_AI_CHARS,
    MAX_HISTORY_TOKENS,
    MAX_ZERO_CHUNK_CHARS,
    AGENT_ROUTER_MODEL,
    OLLAMA_PREFIX,
    OLLAMA_CLOUD_API_KEY,
    OLLAMA_CLOUD_BASE_URL,
    OLLAMA_CLOUD_PREFIX,
)

logger = logging.getLogger(__name__)

# 🚀 Stability Scaling: Global Instance Cache
# Reusing instances prevents socket exhaustion (ephemeral port leaks) on Windows.
_llm_cache = {}
_llm_cache_lock = threading.Lock()

# 🚀 Platinum Scaling: Context Window Limits
MAX_CONTEXT_UNION = 15

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
)
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder
import hashlib
import pickle
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from backend import SQLiteFTS5BM25

# 🚀 Global Executor for non-blocking background tasks (Sentinel summaries)
# Shut down any stale executor left over from a previous Streamlit hot-reload
# before creating a fresh one.  Without this, each reload leaks 2 threads.
import atexit as _atexit

# Separate executors prevent priority inversion: a slow sentinel summary
# can't block a time-sensitive query rewrite (2s timeout).
_background_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sentinel")
_rewrite_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rewrite")
_atexit.register(_background_executor.shutdown, wait=False)
_atexit.register(_rewrite_executor.shutdown, wait=False)

def _background_summarize(history: list[BaseMessage]):
    """Background task to update sentinel state without stalling the main stream."""
    try:
        router = get_router()
        new_state = router.summarize_state_fast(history)
        return new_state
    except Exception as e:
        logger.error(f"❌ Background summary failed: {e}")
        return None

import functools

# ═══════════════════════════════════════════════════════════════════════════
#  EXACT-MATCH QUERY CACHE & PINNED CONTENT EMBEDDING CACHE
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_query(query: str) -> str:
    """Normalize query for cache-hit robustness.

    Preserves ``+`` and ``#`` so language names (C++, C#, F#) don't
    collide after stripping.  Keeps ``_`` and ``.`` for code identifiers.
    """
    import re
    return re.sub(r'[^\w\s+#.]', '', query).lower().strip()

@functools.lru_cache(maxsize=32)
def _get_pinned_embedding(pinned_prefix: str) -> list[float]:
    """Return a cached embedding for the pinned content prefix. Thread-safe."""
    from backend import get_embedding_model
    return get_embedding_model().embed_query(pinned_prefix)

def _get_max_tokens(specialty: str | None, query: str) -> int:
    """Return output token budget scaled to query complexity.

    CODE and REASONING tasks, or verbose queries (>200 chars), likely need
    the full budget.  Short factual/general questions rarely exceed 1024 tokens,
    so reserving 4096 for them wastes provider quota.
    """
    if specialty in ("CODE", "REASONING") or len(query) > 200:
        return MAX_TOKENS
    return 1024

# ═══════════════════════════════════════════════════════════════════════════
#  GHOST HISTORY ENCAPSULATION
# ═══════════════════════════════════════════════════════════════════════════

def _truncate_ai_in_history(history: list[BaseMessage]) -> list[BaseMessage]:
    """
    Cap AI response length in chat history to reduce token waste,
    while aggressively preserving code blocks so the LLM remembers
    the actual code it wrote.
    """
    import re
    truncated = []
    for msg in history:
        if isinstance(msg, AIMessage) and len(msg.content) > AI_RESPONSE_MAX_CHARS:
            code_blocks = re.findall(r"(```.*?```)", msg.content, flags=re.DOTALL)
            if code_blocks:
                gist = msg.content[:400]
                trimmed = f"{gist}\n... [prose truncated]\n\n" + "\n\n".join(code_blocks)
                if len(trimmed) > AI_RESPONSE_MAX_CHARS * 3:
                    trimmed = trimmed[:AI_RESPONSE_MAX_CHARS * 3] + "\n```\n... [code truncated]"
            else:
                trimmed = msg.content[:AI_RESPONSE_MAX_CHARS] + "\n... [truncated for context efficiency]"
            truncated.append(AIMessage(content=trimmed))
        else:
            truncated.append(msg)
    return truncated

def _est_tokens(msgs):
    """Estimate token count for a list of messages (safe for dense code).

    Uses chars//3 which is conservative (overestimates slightly for English,
    accurate for code).  Overestimating is the safe direction — it makes
    sentinel fire earlier and budget enforcement drop chunks sooner,
    preventing API-level truncation.
    """
    return sum(_content_len(m.content) for m in msgs) // 3

def compress_chat_history(history: list[BaseMessage], sentinel_state: str) -> list[BaseMessage]:
    """
    Intelligently trim the chat history based on Sentinel Summaries
    or Ghost History logic for token budget preservation.
    """
    if sentinel_state and sentinel_state != "No summary generated yet.":
        # Sentinel summary available — keep last 4 messages for immediate context
        # but ALWAYS protect the final user+AI pair so the LLM sees the
        # most recent exchange even if the budget is tight.
        keep = 4  # last 2 user+AI pairs
        truncated_history = history[-keep:]
        truncated_history = _truncate_ai_in_history(truncated_history)
    elif len(history) <= GHOST_HISTORY_MAX + 2:
        # Short enough that ghosting isn't needed yet (off-by-one fix:
        # +2 ensures we don't jump to ghost mode when ghost_section would
        # be empty, e.g. 11 messages with WINDOW=10 gives [2:1] = empty).
        truncated_history = _truncate_ai_in_history(history)
    else:
        anchor = history[:2]
        # Keep the last GHOST_HISTORY_WINDOW messages intact, ghost the middle
        window = history[-GHOST_HISTORY_WINDOW:]
        ghost_section = history[2:-GHOST_HISTORY_WINDOW]
        ghosts = []
        for msg in ghost_section:
            if isinstance(msg, AIMessage):
                trimmed = msg.content[:GHOST_AI_CHARS]
                if len(msg.content) > GHOST_AI_CHARS:
                    trimmed += "\n... [truncated]"
                ghosts.append(AIMessage(content=trimmed))
            else:
                ghosts.append(msg)
        truncated_history = _truncate_ai_in_history(anchor + ghosts + window)

    # Hard token budget: drop oldest ghost messages until under budget,
    # but ALWAYS protect the last 2 messages (most recent exchange).
    while _est_tokens(truncated_history) > MAX_HISTORY_TOKENS and len(truncated_history) > 4:
        # Remove the 3rd message (first ghost after anchor pair),
        # never touch the last 2 (protected tail).
        if len(truncated_history) > 4:
            truncated_history.pop(2)
        else:
            break

    return truncated_history

# 🚀 Elite Patterns: LLM Intent Routing
# We use a fast, local model (1B-3B parameters) for orchestration.


# ═══════════════════════════════════════════════════════════════════════════
#  LLM
# ═══════════════════════════════════════════════════════════════════════════

def is_cache_capable(model: str | None) -> bool:
    """
    Check if the model/provider supports Anthropic-style prompt caching blocks.
    
    In 2026, this includes Claude, Gemini 2+, Gemma 4 (Google), and GLM 5.
    OpenAI and DeepSeek use implicit prefix caching (no markers needed).
    """
    if not model or model.startswith(OLLAMA_PREFIX):
        return False
    
    m_lower = model.lower()
    # Broaden detection for SOTA models that favor explicit markers
    cache_brands = ["claude", "gemini", "gemma", "glm-5"]
    return any(brand in m_lower for brand in cache_brands)


def get_cache_profile(model: str | None) -> tuple[int, int]:
    """
    Cross-Provider Cache Router.
    Returns (max_checkpoints, min_tokens_for_cache) for the given model.
    Different providers have different cache economics:
      - Claude: 4 breakpoints, 1024 token minimum
      - Gemini: more breakpoints allowed, 1028 token minimum
      - DeepSeek/Qwen: similar to Claude
    Falls back to global defaults if model is unknown.
    """
    if not model:
        return (MAX_CACHE_CHECKPOINTS, 1024)
    m_lower = model.lower()
    for pattern, profile in PROVIDER_CACHE_PROFILES.items():
        if pattern in m_lower:
            return profile
    return (MAX_CACHE_CHECKPOINTS, 1024)


def format_message_content(text: str, model: str | None, use_cache: bool = False) -> str | list[dict]:
    """
    Return content as a plain string for non-cache models, 
    or a block-list for cache-capable ones.
    """
    # If caching is globally disabled or model can't handle it, 
    # always return a plain string.
    if not ENABLE_PROMPT_CACHING or not use_cache or not is_cache_capable(model):
        return text
    
    # Return Anthropic-style block format with cache markers
    return [
        {
            "type": "text", 
            "text": text, 
            "cache_control": {"type": "ephemeral"}
        }
    ]


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    streaming: bool = True,
):
    """
    Return a chat model instance.

    If *model* is ``OLLAMA_SENTINEL`` the function returns a local
    ``ChatOllama``; otherwise it returns a ``ChatOpenAI`` pointed at
    OpenRouter.
    """
    temp = temperature if temperature is not None else LLM_TEMPERATURE
    
    # 🚀 Cache Check (thread-safe)
    cache_key = (model, temp, streaming)
    with _llm_cache_lock:
        if cache_key in _llm_cache:
            return _llm_cache[cache_key]

    def _cache_and_return(llm_obj):
        with _llm_cache_lock:
            _llm_cache[cache_key] = llm_obj
        return llm_obj

    # ── Local Ollama path ──────────────────────────────────────────────
    if model and model.startswith(OLLAMA_PREFIX):
        ollama_model_name = model[len(OLLAMA_PREFIX):]
        return _cache_and_return(ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=ollama_model_name,
            temperature=temp,
            num_predict=MAX_TOKENS,
        ))

    # ── Ollama Cloud path ──────────────────────────────────────────────
    if model and model.startswith(OLLAMA_CLOUD_PREFIX):
        cloud_model_name = model[len(OLLAMA_CLOUD_PREFIX):]
        if not OLLAMA_CLOUD_API_KEY:
            raise ValueError(
                "OLLAMA_CLOUD_API_KEY is not set. "
                "Add it to your .env file."
            )
        return _cache_and_return(ChatOpenAI(
            base_url=OLLAMA_CLOUD_BASE_URL,
            api_key=OLLAMA_CLOUD_API_KEY,
            model=cloud_model_name,
            temperature=temp,
            streaming=streaming,
            max_tokens=MAX_TOKENS,
        ))

    # ── OpenRouter path ────────────────────────────────────────────────
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Create a .env file with: OPENROUTER_API_KEY=sk-or-v1-..."
        )
    # 🚀 Professional Polish: Conditional Header Safety
    # Only send Anthropic-specific headers when using a Claude model
    default_headers = {
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Private AI Knowledge Base",
    }
    
    current_model = model or DEFAULT_MODEL
    if "claude" in current_model.lower():
        default_headers["anthropic-beta"] = ANTHROPIC_CACHE_BETA_HEADER

    return _cache_and_return(ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=current_model,
        temperature=temp,
        streaming=streaming,
        max_tokens=MAX_TOKENS,
        default_headers=default_headers,
        # 🚀 Enable usage in stream for telemetry visibility
        model_kwargs={"stream_options": {"include_usage": True}}
    ))








def hybrid_search(
    db: Chroma,
    query: str,
    collection_name: str = "default",
    k: int = 10,
    exclude_file: str | None = None,
    filter_extensions: list[str] | None = None,
    query_embedding: list[float] | None = None,
) -> list[Document]:
    """
    Perform Hybrid Search (BM25 + Vector) with Reciprocal Rank Fusion (RRF).

    If *query_embedding* is provided, it is reused for the vector search
    via ``similarity_search_by_vector``, avoiding a redundant embedding
    inference that ChromaDB would otherwise perform internally.
    """
    if not ENABLE_HYBRID_SEARCH:
        # Fallback to standard vector search.
        # fetch_k is an MMR-only parameter and is not supported by
        # similarity_search / similarity_search_by_vector — omit it.
        chroma_filter = None
        if exclude_file:
            chroma_filter = {"source": {"$ne": exclude_file}}
        if query_embedding:
            return db.similarity_search_by_vector(query_embedding, k=k, filter=chroma_filter)
        return db.similarity_search(query, k=k, filter=chroma_filter)

    # Build a ChromaDB metadata filter from the caller's exclusion criteria so
    # the vector store never fetches docs that will be thrown away post-fetch.
    # BM25 has no filter API — _rank_docs() still handles that side.
    _conditions: list[dict] = []
    if exclude_file:
        _conditions.append({"source": {"$ne": exclude_file}})
    if filter_extensions:
        _conditions.append({"file_extension": {"$in": filter_extensions}})
    if len(_conditions) == 0:
        _chroma_filter = None
    elif len(_conditions) == 1:
        _chroma_filter = _conditions[0]
    else:
        _chroma_filter = {"$and": _conditions}

    # 1. 🔍 Vector Search (Semantic)
    # We fetch a larger candidate pool for RRF to merge.
    # Reuse pre-computed embedding when available to avoid double-embedding.
    if query_embedding:
        vector_docs = db.similarity_search_by_vector(query_embedding, k=k*3, filter=_chroma_filter)
    else:
        vector_docs = db.similarity_search(query, k=k*3, filter=_chroma_filter)
    
    # 2. 🔍 BM25 Keyword Search (Transitioned to SQLite FTS5)
    with SQLiteFTS5BM25(collection_name) as fts:
        # Note: SQLite search internalizes metadata filtering for better performance
        bm25_docs = fts.search(query, k=k*3)
    
    # Apply remaining excludes that are not yet in FTS SQL query
    if exclude_file or filter_extensions:
        bm25_docs = [
            d for d in bm25_docs
            if not (exclude_file and d.metadata.get("source") == exclude_file)
            and not (filter_extensions and d.metadata.get("file_extension") not in filter_extensions)
        ]

    # 3. 🧪 Reciprocal Rank Fusion (RRF)
    # RRF Score(d) = sum(1 / (k + rank))
    RRF_K = 60
    scores = {} # {doc_id: score}
    doc_map = {} # {doc_id: doc_object}
    
    def _rank_docs(docs, weight=1.0):
        for rank, doc in enumerate(docs):
            if exclude_file and doc.metadata.get("source") == exclude_file:
                continue
            if filter_extensions and doc.metadata.get("file_extension") not in filter_extensions:
                continue
                
            # 🚀 Fix: Use content excerpt to prevent collisions on zero-chunk docs with missing hashes
            doc_id = (
                doc.metadata.get("source"),
                doc.metadata.get("chunk_index", 0),
                doc.metadata.get("content_hash", doc.page_content[:64])
            )
            score = (1.0 / (RRF_K + rank + 1)) * weight
            scores[doc_id] = scores.get(doc_id, 0) + score
            doc_map[doc_id] = doc

    _rank_docs(vector_docs, weight=VECTOR_WEIGHT)
    _rank_docs(bm25_docs, weight=BM25_WEIGHT)
    
    # Sort by merged RRF score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    rrf_results = [doc_map[did] for did in sorted_ids[:k]]

    # Cross-encoder re-ranking is handled by LocalReRanker.rerank() in the
    # caller — applying it here as well would score the same docs twice with
    # the same model for zero quality gain.

    return rrf_results


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

# 🚀 CORE INSTRUCTIONS (Static/Cached)
CORE_INSTRUCTIONS = """\
You are an expert AI assistant specialising in code analysis and document comprehension.

INSTRUCTIONS:
1. Answer the user's question using ONLY the retrieved context, pinned source, and conversation state provided below.
2. If the context does not contain enough information, say so clearly — \
   do NOT fabricate an answer.
3. When discussing code, reference the source file and explain the logic.
4. Be concise, precise, and use markdown formatting where helpful.
5. If the user asks for code improvements, provide the improved version \
   with clear explanations.
6. The 'CONVERSATION STATE' section contains a summary of our past discussion. \
   You MUST use it to understand follow-up questions and you MUST report its contents if the user asks what it says.
7. CRITICAL OVERRIDE: If the user asks you to retrieve or read the 'CONVERSATION STATE', do NOT explain the python codebase or how variables like {sentinel_state} work. Look physically below at the text under the heading 'CONVERSATION STATE:' and copy it exactly word-for-word. Even if there are no bullet points and it says "No summary generated yet.", you must reply with exactly that text.
"""

# ═══════════════════════════════════════════════════════════════════════════
#  HISTORY CACHING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _prepare_history_with_cache(history: list[BaseMessage], model: str | None) -> list[BaseMessage]:
    """
    Prepare chat history with optional caching.

    For models with >4 breakpoints (Gemini), we add a cache marker to
    the second-to-last message (the most recent AI response) rather than
    the last message (user query that changes every turn).  This way the
    checkpoint is reusable across turns — only the new user message is
    uncached, not the entire history.
    For Claude (4 breakpoints), system blocks already consume the limit.
    """
    if not history:
        return history

    max_bp, _ = get_cache_profile(model)
    # If we have spare breakpoints (Gemini supports 8+), use one for history.
    # We use 4 for system blocks, so 5+ is the threshold.
    if max_bp > 4 and is_cache_capable(model) and ENABLE_PROMPT_CACHING:
        new_history = list(history)
        # Mark the second-to-last message (stable across turns) instead of
        # the last message (volatile user query) to avoid all-writes-no-reads.
        target_idx = -2 if len(new_history) >= 2 else -1
        target_msg = new_history[target_idx]
        if isinstance(target_msg.content, str):
            # CRITICAL: create a NEW message object instead of mutating in-place.
            # The history list shares objects with lc_history/full_history;
            # in-place mutation corrupts them (content: str → list) and garbles
            # the sentinel summary and token estimates downstream.
            cls = type(target_msg)  # HumanMessage or AIMessage
            new_msg = cls(content=[
                {
                    "type": "text",
                    "text": target_msg.content,
                    "cache_control": {"type": "ephemeral"}
                }
            ])
            new_history[target_idx] = new_msg
        return new_history

    return list(history)


# 🚀 Professional Polish: Linguistic Logic Gates (History vs Speed)
# ═══════════════════════════════════════════════════════════════════════════
#  PLATINUM STANDARD: RAG UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════


# 🚀 Local LLM Orchestrator (Agentic Router)
# ═══════════════════════════════════════════════════════════════════════════

class LocalReRanker:
    """
    Local Cross-Encoder "Critic" that re-scores retrieved chunks 
    to ensure surgical precision before the context is passed to the LLM.
    """
    def __init__(self):
        self.model = None
        self._init_model()

    def _init_model(self):
        if USE_RERANKER:
            self.model = self._get_cached_cross_encoder()

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _get_cached_cross_encoder():
        try:
            return CrossEncoder(RERANK_MODEL)
        except Exception as e:
            logger.error(f"❌ Re-ranker failed to load: {e}")
            return None

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """Re-score and filter documents using the Cross-Encoder."""
        if not self.model or not documents:
            return documents[:top_k]

        # Prepare pairs for cross-encoding (Query, Chunk)
        pairs = [[query, doc.page_content] for doc in documents]
        try:
            scores = self.model.predict(pairs)
            
            # Combine scores with docs and sort
            scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
            
            # 🚀 Phase 5: Store the top score for telemetry
            self.last_top_score = float(scored_docs[0][0]) if scored_docs else 0.0
            
            # Log the top score for telemetry
            if scored_docs:
                logger.info(f"🎯 Top Re-rank Relevance Score: {scored_docs[0][0]:.4f}")
            
            return [doc for score, doc in scored_docs[:top_k]]
        except Exception as e:
            logger.error(f"❌ Re-ranking execution failed: {e}")
            return documents[:top_k]
# 🚀 ARCH-1: Reranker Singleton
_reranker_instance = None
_reranker_lock = threading.Lock()
def get_reranker():
    global _reranker_instance
    if _reranker_instance is None and USE_RERANKER:
        with _reranker_lock:
            if _reranker_instance is None:  # double-check after lock
                try:
                    _reranker_instance = LocalReRanker()
                    logger.info("✅ Reranker initialized (LocalReRanker)")
                except Exception as e:
                    logger.error(f"❌ Reranker init failed: {e}")
    return _reranker_instance


# ═══════════════════════════════════════════════════════════════════════════
#  SEMANTIC CACHE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class SemanticCache:
    """
    Persistent Query/Response cache using ChromaDB.
    Bypasses RAG and LLM for repeat queries.
    """
    def __init__(self, collection_name: str = "semantic_cache"):
        from backend import get_embedding_model, CHROMA_DB_DIR
        self.db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=get_embedding_model(),
            collection_name=collection_name,
        )

    @staticmethod
    def _pinned_fingerprint(pinned_content: str | None) -> str:
        """Short stable hash of the pinned context.

        Using the content (not path) means "file A pinned then edited" and
        "file A pinned fresh" produce different fingerprints, so the cache
        won't serve pre-edit answers for post-edit content.
        Empty / None collapses to the sentinel "_none_" so unpinned turns
        share a cache bucket.
        """
        if not pinned_content:
            return "_none_"
        return hashlib.md5(pinned_content.encode("utf-8", errors="ignore")).hexdigest()[:16]

    @staticmethod
    def _model_fingerprint(model: str | None) -> str:
        """Normalize a model id so provider/size changes bust the cache.
        Different models phrase answers differently — don't cross-serve.
        """
        return (model or "_default_").strip().lower()

    def lookup(self, query: str, threshold: float = 0.95,
               collection_scope: str | None = None,
               pinned_content: str | None = None,
               model: str | None = None) -> str | None:
        """Find a cached answer if similarity exceeds threshold.

        Scope dimensions (all enforced):
          - collection_scope: which knowledge base the answer was grounded in
          - pinned_fp:        fingerprint of the pinned file content at time of answer
          - model_fp:         which model produced the answer

        Any mismatch is treated as a miss — safer to regenerate than to
        cross-serve an answer grounded in a different file / model.
        """
        results = self.db.similarity_search_with_relevance_scores(query, k=3)
        pinned_fp = self._pinned_fingerprint(pinned_content)
        model_fp = self._model_fingerprint(model)
        for doc, score in results:
            if score < threshold:
                break
            md = doc.metadata
            if collection_scope and md.get("source_collection") != collection_scope:
                continue
            # Only reject on a pinned-fp mismatch when BOTH sides declare one.
            # Legacy cache entries (pre-fix) have no pinned_fp and should not
            # be spuriously rejected — they'll age out as new entries overwrite.
            cached_pinned = md.get("pinned_fp")
            if cached_pinned and cached_pinned != pinned_fp:
                continue
            cached_model = md.get("model_fp")
            if cached_model and cached_model != model_fp:
                continue
            logger.info(f"⚡ Semantic Cache Hit (Score: {score:.4f})")
            return md.get("answer")
        return None

    def upsert(self, query: str, answer: str, collection_scope: str | None = None,
               pinned_content: str | None = None, model: str | None = None):
        """Save successful generation to cache, tagged with the full scope.

        Stores (collection, pinned_fp, model_fp) so the next lookup can
        reject the entry if any of those change.  Skips the expensive dedup
        lookup: SEMANTIC_CACHE_THRESHOLD (0.98) is tight enough that
        near-duplicate queries almost never produce different answers, and
        an extra cache entry for a slight paraphrase is cheaper than an
        embedding inference on every successful generation.
        """
        meta = {
            "answer": answer,
            "type": "cached_response",
            "pinned_fp": self._pinned_fingerprint(pinned_content),
            "model_fp": self._model_fingerprint(model),
        }
        if collection_scope:
            meta["source_collection"] = collection_scope
        self.db.add_texts(
            texts=[query],
            metadatas=[meta]
        )

_semantic_cache_instance = None
_semantic_cache_lock = threading.Lock()
def get_semantic_cache():
    global _semantic_cache_instance
    with _semantic_cache_lock:
        if _semantic_cache_instance is None:
            _semantic_cache_instance = SemanticCache()
        return _semantic_cache_instance

def reset_semantic_cache():
    """Drop the cached SemanticCache singleton so the next call rebuilds it."""
    global _semantic_cache_instance
    with _semantic_cache_lock:
        _semantic_cache_instance = None

# ═══════════════════════════════════════════════════════════════════════════

class VectorRouter:
    """
    Zero-latency decision engine using vector similarity to handle 
    classification and state management without LLM overhead.
    """
    def __init__(self):
        # We reuse the embedding model already loaded in backend.py
        pass

    def classify_intent(self, query: str, history: list[BaseMessage]) -> str:
        """
        Classify as NEW topic or FOLLOW-UP.
        Uses heuristic fast-paths first; falls through to local LLM only
        for ambiguous cases to avoid 200ms-2s latency on every query.
        """
        if not history:
            return "NEW"

        import re
        q = query.lower().strip()

        # Fast-path: pronouns/demonstratives strongly indicate follow-up
        if re.match(r"^(it|this|that|these|those|the same|above|previous|also|and |more )\b", q):
            return "FOLLOW-UP"
        # Fast-path: explicit new-topic signals
        if re.match(r"^(new topic|switch to|let's talk about|forget|start over)\b", q):
            return "NEW"

        try:
            llm = get_llm(model=AGENT_ROUTER_MODEL, temperature=0.0, streaming=False)
            
            history_text = "\n".join([
                f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:200]}" 
                for m in history[-2:]
            ])
            
            prompt = (
                f"Previous turns:\n{history_text}\n\n"
                f"Current query: '{query}'\n\n"
                "Output exactly 'FOLLOW-UP' if the user is referring to the current topic or files, "
                "or 'NEW' if they are asking about a different file or a fresh concept. "
                "Return ONLY the word."
            )
            response = llm.invoke(prompt)
            result = response.content.strip().upper()
            return "FOLLOW-UP" if "FOLLOW-UP" in result else "NEW"
        except Exception:
            # Ollama is down — heuristic fallback: if the current query
            # shares significant content words with the last user message,
            # it's likely a follow-up.  Defaulting to "NEW" here silently
            # destroys context union for every follow-up when Ollama is off.
            last_user = ""
            for m in reversed(history):
                if isinstance(m, HumanMessage):
                    last_user = m.content.lower()
                    break
            if last_user:
                _stop = {"the","is","a","an","in","of","to","for","and","or",
                         "how","does","what","it","this","that","can","do","i"}
                cur_words = set(q.split()) - _stop
                prev_words = set(last_user.split()) - _stop
                if cur_words and prev_words:
                    overlap = len(cur_words & prev_words) / max(len(cur_words), 1)
                    if overlap >= 0.3:
                        return "FOLLOW-UP"
            return "NEW"

    def detect_specialty(self, query: str) -> str:
        """
        Detect the best specialist for the query using robust regex word boundaries.
        Returns one of: ['CODE', 'REASONING', 'VISION', 'GENERAL']
        """
        import re
        q = query.lower()

        # 💻 Coding Specialist Triggers — checked BEFORE the short-question
        # fast-path so "What is the best way to implement a function in Python?"
        # correctly routes to CODE even though it starts with "what".
        code_triggers = [
            r"code", r"python", r"javascript", r"verilog", r"function", r"class",
            r"refactor", r"bug", r"debug", r"compile", r"script", r"hdl", r"rtl",
            r"implement", r"write a", r"api", r"library", r"sql", r"html",
            r"cpp", r"c\+\+", r"rust", r"golang"
        ]
        if any(re.search(rf"\b{t}\b", q) for t in code_triggers) or "```" in q:
            return "CODE"

        # Fast-path: short factual questions with no code signal → GENERAL
        if len(query) < 60 and re.match(r"^(what|where|who|when|which|is|does|can)\b", q):
            return "GENERAL"

        # 👁️ Vision Triggers
        vision_triggers = [r"image", r"plot", r"chart", r"diagram", r"vision", r"see this"]
        if any(re.search(rf"\b{t}\b", q) for t in vision_triggers):
            return "VISION"
            
        # 🧠 Reasoning / Math Triggers
        reasoning_triggers = [
            r"analyze", r"logic", r"math", r"derive", r"prove", r"step by step",
            r"complex", r"calculate", r"deepseek", r"reason", r"philosophy", 
            r"compare", r"architecture", r"design pattern", r"explain how"
        ]
        if any(re.search(rf"\b{t}\b", q) for t in reasoning_triggers):
            return "REASONING"
            
        # Default
        return "GENERAL"

    @staticmethod
    def _extractive_fallback(history: list[BaseMessage]) -> str:
        """
        Pure-Python extractive summary used when the local LLM (Ollama)
        is unavailable.  Keeps the first sentence of each recent human
        message to preserve topic continuity without any external calls.
        """
        bullets = []
        for m in history[-8:]:
            if not isinstance(m, HumanMessage):
                continue
            text = m.content.strip()
            # Take the first sentence (up to first period, question mark, or 120 chars)
            end = len(text)
            for ch in ".?!":
                idx = text.find(ch)
                if 0 < idx < end:
                    end = idx + 1
            snippet = text[:min(end, 120)].strip()
            if snippet:
                bullets.append(f"- {snippet}")
        return "\n".join(bullets[-3:]) if bullets else "No summary available."

    def summarize_state_fast(self, history: list[BaseMessage]) -> str:
        """
        Summarize conversation state.  Tries the local Ollama model first;
        falls back to a pure-Python extractive summary if Ollama is down
        so history compression is never silently skipped.
        """
        try:
            llm = get_llm(model=f"{OLLAMA_PREFIX}{AGENT_ROUTER_MODEL}", temperature=0.0, streaming=False)
            context = "\n".join([f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content[:500]}" for m in history[-6:]])
            prompt = (
                f"History:\n{context}\n\n"
                "Summarize the conversation state so far in exactly 3 dense bullet points. "
                "Focus on technical topics discussed. Reply ONLY with the bullet points."
            )
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            logger.warning("Ollama unavailable for sentinel — using extractive fallback")
            return self._extractive_fallback(history)


# 🚀 ARCH-3: Vector Router Singleton
_router_instance = None
_router_lock = threading.Lock()
def get_router():
    global _router_instance
    with _router_lock:
        if _router_instance is None:
            _router_instance = VectorRouter()
        return _router_instance


def _sort_docs_deterministically(
    docs: list[Document],
    stable_hashes: set[str] | None = None,
) -> list[Document]:
    """
    Prefix-Preserving Deterministic Sort.

    When *stable_hashes* is provided (the content hashes of docs that were
    already in the prompt on the previous turn), those docs sort FIRST
    (``_is_new=0``), preserving the exact byte prefix that the provider
    cache (Anthropic/Gemini) already stored.  New docs sort AFTER
    (``_is_new=1``) so they append to the end of the block without
    breaking the cached prefix.

    Within each group the order is fully deterministic:
    ``(source, chunk_index, content_hash, page_content)``.
    """
    def _sort_key(d):
        is_new = 0 if (
            stable_hashes
            and d.metadata.get("content_hash", "") in stable_hashes
        ) else (1 if stable_hashes else 0)
        return (
            is_new,
            str(d.metadata.get("source", "")),
            int(d.metadata.get("chunk_index", 0)),
            str(d.metadata.get("content_hash", "")),
            str(d.page_content),
        )
    return sorted(docs, key=_sort_key)


def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    if not vec1 or not vec2:
        return 0.0
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def _content_len(c) -> int:
    """Return the character length of a message content field.

    Content may be a plain string or a list of Anthropic cache-control
    dicts — handles both so token estimates stay accurate.
    """
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        return sum(len(b.get("text", "")) for b in c if isinstance(b, dict))
    return 0


def build_rag_chain(db: Chroma, model: str | None = None):
    """
    Build a retrieval chain with stable Full-Context Caching (Architecture A).
    """
    llm = get_llm(model=model)
    
    # 🚀 Professional Polish: Dynamic Retrieval Configuration
    # We build our retrievers inside the lambda to support the 
    # Pinned File exclusion filter.

    # Dual-Path Prompt Construction
    # Only Claude supports Anthropic-style cache_control blocks via OpenRouter.
    # All other models (Gemini/Qwen/DeepSeek/Ollama) get a clean string prompt.
    
    is_cc = is_cache_capable(model) and ENABLE_PROMPT_CACHING
    
    max_bp, _ = get_cache_profile(model)

    if is_cc:
        # Dynamic Cache Blocks — Claude only
        # Ordered from most stable to most volatile.  We only attach
        # cache_control markers to the first ``max_bp`` blocks; the
        # rest are plain text (no wasted cache writes).
        # Stable order: Instructions > Pinned > Sentinel > RAG context.
        static_system_text = CORE_INSTRUCTIONS
        # Order: most-stable → most-volatile.  The first max_bp blocks
        # get cache_control markers, so placing the volatile sentinel and
        # new-discoveries at the end avoids invalidating the prefix cache
        # every turn.
        block_specs = [
            static_system_text,
            "FULL SOURCE CONTEXT (PINNED):\n{full_source_context}",
            "STABLE RAG CONTEXT (DETERMINISTIC):\n{stable_context}",
            "CONVERSATION STATE:\n{sentinel_state}",
            "NEW RAG DISCOVERIES:\n{new_context}"
        ]
        system_blocks = []
        for idx, text in enumerate(block_specs):
            use_cache_marker = idx < max_bp  # only mark up to max_bp blocks
            formatted = format_message_content(text, model, use_cache=use_cache_marker)
            # format_message_content returns a list for cache-capable models
            if isinstance(formatted, list):
                system_blocks.append(formatted[0])
            else:
                # Plain string — wrap in the Anthropic text-block format
                # so the system_blocks list stays homogeneous.
                system_blocks.append({"type": "text", "text": formatted})

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_blocks),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
    else:
        # Mirror the cache-path ordering: most-stable → most-volatile.
        # Even non-cache providers (DeepSeek, Qwen) do implicit prefix
        # caching, so putting volatile sentinel AFTER stable RAG context
        # preserves more of the prefix across turns.
        system_text = (
            f"{CORE_INSTRUCTIONS}\n\n"
            "FULL SOURCE CONTEXT (PINNED):\n{full_source_context}\n\n"
            "STABLE RAG CONTEXT (DETERMINISTIC):\n{stable_context}\n\n"
            "CONVERSATION STATE:\n{sentinel_state}\n\n"
            "NEW RAG DISCOVERIES:\n{new_context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    # 🚀 Platinum Standard: Metadata-Aware LCEL Chain
    # We remove StrOutputParser to preserve the 'response_metadata' (for caching token counts)
    # inside the raw message chunks.
    # 🚀 Professional Polish: Instantiate Vector Router & Re-ranker
    router = get_router()
    reranker = get_reranker()

    question_answer_chain = prompt | llm
    _specialist_llm_cache: dict[str, object] = {}
    # Cooldown tracker: prevents sentinel from re-firing every turn once the
    # token threshold is crossed.  Stored as a mutable dict so the closure
    # can mutate it without a `nonlocal` declaration.
    _sentinel_cooldown: dict[str, int] = {"last_turn": 0}

    def _full_context_cache_chain(inputs: dict):
        """
        Unified chain with Agentic Routing, Hybrid Search,
        and Cross-Provider cache awareness.
        """
        user_input = inputs["input"]
        pinned_content = inputs.get("full_source_context", "")
        history = inputs.get("chat_history", [])
        coll_name = inputs.get("collection_name", "default")
        
        # 🚀 Fix: Get last query and its embedding from inputs
        last_query = inputs.get("last_query")
        last_query_emb = inputs.get("last_query_embedding")
        force_retrieval = inputs.get("force_retrieval", False)
        
        # 🚀 PHASE 3: Semantic Cache Lookup (Pre-Everything)
        # Scope by collection + pinned fingerprint + model so an answer
        # grounded in file A / model X is never served for file B / model Y.
        sem_cache = get_semantic_cache()
        if not force_retrieval:
            cached_ans = sem_cache.lookup(user_input, threshold=SEMANTIC_CACHE_THRESHOLD,
                                          collection_scope=coll_name,
                                          pinned_content=pinned_content,
                                          model=model)
            if cached_ans:
                yield {"answer": cached_ans, "intent": "CACHE_HIT"}
                return

        # 🚀 ASYNC SENTINEL TRIGGER
        background_future = None

        # 1. Initialize all prompt template variables to prevent KeyError
        inputs["full_source_context"] = inputs.get("full_source_context", "None pinned.")
        inputs["sentinel_state"] = inputs.get("sentinel_state", "No summarized state available.")
        inputs["stable_context"] = inputs.get("stable_context", "None previously established.")
        inputs["new_context"] = inputs.get("new_context", "No new discoveries.")
        inputs["chat_history"] = inputs.get("chat_history", [])

        # 2. Context Awareness (Latency-Free)
        # Use the true global turn count from app.py — compressed history
        # is capped at 4 messages (≤2 turns), which permanently deadlocks
        # the sentinel trigger if we count from it.
        turn_count = inputs.get("global_turn_count",
                                sum(1 for m in history if isinstance(m, HumanMessage)))
        # Full uncompressed history for sentinel summarization
        full_history = inputs.get("full_history", history)

        # Token-aware sentinel trigger: estimate from FULL history so the
        # budget reflects the real conversation size, not the compressed window.
        estimated_history_tokens = sum(_content_len(m.content) for m in full_history) // 3
        # Fire when history exceeds the token budget AND at least SENTINEL_INTERVAL
        # turns have passed since the last sentinel run.  Without the cooldown,
        # once the threshold is crossed it fires every single turn.
        should_summarize = (
            turn_count > 0
            and estimated_history_tokens >= SENTINEL_TOKEN_THRESHOLD
            and (turn_count - _sentinel_cooldown["last_turn"]) >= SENTINEL_INTERVAL
        )

        # Calculate semantic similarity once
        current_similarity = 0.0
        current_emb = None
        user_input_norm = _normalize_query(user_input)
        
        if not force_retrieval:
            # Layer 0: Exact-match cache (zero compute for identical queries)
            if last_query and last_query_emb and user_input_norm == _normalize_query(last_query):
                current_emb = last_query_emb
            else:
                from backend import get_embedding_model
                current_emb = get_embedding_model().embed_query(user_input)
            
            if last_query_emb:
                current_similarity = calculate_cosine_similarity(current_emb, last_query_emb)

        # Semantic Intent Detection (Latency-Free)

        # Semantic hit drives two behaviours:
        # 1. For Claude (cache-capable): retrieval still happens so the
        #    provider cache can fire on the deterministic prefix.
        # 2. For all other models: retrieval is skipped on a semantic hit
        #    since there's no provider-side cache benefit from re-fetching.
        is_semantic_hit = (
            current_similarity >= SEMANTIC_CACHE_THRESHOLD
        )
        
        # 3. Pinned context passthrough with Relevance Gate
        pinned_eligible = False
        if pinned_content and pinned_content != "None pinned.":
            if STICKY_PINNED_CONTEXT:
                pinned_eligible = True
            elif current_emb:
                # Use only the prefix to avoid massive embedding calls just for gating
                pinned_emb = _get_pinned_embedding(pinned_content[:2000])
                pinned_sim = calculate_cosine_similarity(current_emb, pinned_emb)
                if pinned_sim >= PINNED_RELEVANCE_THRESHOLD:
                    pinned_eligible = True
            else:
                pinned_eligible = True

        inputs["full_source_context"] = pinned_content if pinned_eligible else "None pinned."

        # 4. Define Previous Context Union
        previous_union = inputs.get("cached_docs") or []

        # 5. 🤖 Zero-Latency Vector Routing & Specialist Detection ─────────
        # Use LLM-based classification for high-precision follow-up detection
        intent = router.classify_intent(user_input, history) if history else "NEW"
        
        # Phase 4: Specialist Detection
        enable_auto = inputs.get("auto_specialist", ENABLE_AUTO_SPECIALIST)
        specialty = router.detect_specialty(user_input) if enable_auto else "GENERAL"
        
        # 🚀 Fix: Only switch models if we find a REAL specialty.
        # If it's just a GENERAL query, stay on the user's manual selection.
        specialist_model = (
            SPECIALIST_MAPPING.get(specialty)
            if (enable_auto and specialty != "GENERAL")
            else None
        )

        # Guard: never route to a cloud specialist when the user's chosen model
        # is local (Ollama). E.g. VISION maps to Gemma on OpenRouter by default —
        # that would silently bypass the user's local-only intent.
        if (specialist_model
                and not specialist_model.startswith(OLLAMA_PREFIX)
                and model
                and model.startswith(OLLAMA_PREFIX)):
            specialist_model = None

        pinned_file = inputs.get("exclude_file")
        ext_filter = inputs.get("filter_extensions")

        # Hybrid search (ChromaDB + BM25)
        # When the model supports provider-side prefix caching (Claude/Gemini/DeepSeek),
        # always retrieve so the deterministic sort can maximise cache hits.
        # For all other models the provider cache doesn't help, so skip
        # retrieval on semantic cache hits to save compute.
        k_fetch = RERANK_CANDIDATES if USE_RERANKER else RETRIEVER_K

        # 🚀 Fix: Include DeepSeek/Qwen as cache-capable for prefix stability, 
        # even if they don't use explicit Anthropic-style markers.
        provider_has_cache = is_cache_capable(model) or any(
            p in (model or "").lower() for p in ["deepseek", "qwen", "mistral"]
        )
        
        trust_native_cache = inputs.get("trust_native_cache", True)
        skip_retrieval = (
            not (trust_native_cache and provider_has_cache)
            and is_semantic_hit
            and bool(previous_union)
            and not force_retrieval
        )

        search_query = user_input
        if ENABLE_HYBRID_SEARCH and intent == "FOLLOW-UP":
            try:
                # 🚀 Fix Problem 1: Build formatted history string
                history_text = "\n".join([
                    f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
                    for m in history[-2:]
                ])
                # 🚀 Fix Problem 2: Call Ollama with a 2-second timeout to prevent blocking
                def _do_rewrite():
                    llm_rewrite = get_llm(model=f"{OLLAMA_PREFIX}{AGENT_ROUTER_MODEL}", temperature=0.0, streaming=False)
                    # 🚀 Fix: Direct timeout on invoke() to prevent thread leakage
                    llm_rewrite = llm_rewrite.with_config({"timeout": 2.0})
                    prompt_rewrite = (
                        f"History:\n{history_text}\n\n"
                        f"Rewrite this query to be standalone: '{user_input}'\n"
                        "Return ONLY the rewritten query, no explanation."
                    )
                    raw = llm_rewrite.invoke(prompt_rewrite).content.strip()
                    # Strip common LLM preamble that pollutes BM25 search
                    for prefix in ("Sure,", "Here's", "The rewritten query is:", "Rewritten query:"):
                        if raw.lower().startswith(prefix.lower()):
                            raw = raw[len(prefix):].strip().strip('"').strip("'")
                    # If rewriter returned something way longer than the input
                    # it's probably an explanation, not a query — fall back.
                    if len(raw) > len(user_input) * 3:
                        return user_input
                    return raw

                future = _rewrite_executor.submit(_do_rewrite)
                search_query = future.result(timeout=2.0)
            except Exception as e:
                logger.warning(f"Ollama rewrite failed or timed out: {e}")
                search_query = user_input

        # ── Fix C1: Anchor Term Injection ──────────────────────────────
        # Extract code identifiers from the user query.
        # all_identifiers: any word with underscore or ALL_CAPS (code-like tokens)
        # anchor_terms: strictly ALL_CAPS constants (ENABLE_PROMPT_CACHING, etc.)
        #   — used for propagation detection and targeted retrieval
        # This prevents "how does" from triggering propagation on every query.
        all_identifiers = list(set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b", user_input)))
        anchor_terms = [t for t in all_identifiers if t.isupper() and len(t) > 3]
        # Also include snake_case identifiers (contain underscore) for retrieval
        snake_case_ids = [t for t in all_identifiers if "_" in t and not t.isupper()]
        # Combined anchors for retrieval (ALL_CAPS + snake_case)
        retrieval_anchors = anchor_terms + snake_case_ids

        # Inject ALL_CAPS into search query so BM25 can find definition sites
        if anchor_terms:
            for term in anchor_terms:
                if term not in search_query:
                    search_query = f"{search_query} {term}"

        # ── Fix D: 3-Hop Propagation Detection ─────────────────────────
        # Only fires when the query contains ACTUAL code identifiers (ALL_CAPS
        # or snake_case), not just common English words that happen to be 4+ chars.
        # "changes" alone was too common ("what changes were made?") but
        # causality-tracing phrasings must still trigger propagation.
        # Using stem "chang" + context words avoids false positives while
        # catching "change when", "changes when", "changed by", etc.
        _PROPAGATION_KEYWORDS = (
            "propagat", "impact", "affect", "changes when", "change when",
            "changed by", "changed if",
            "trace", "alter", "flow", "step-by-step", "step by step",
            "how does", "ultimately",
        )
        is_propagation_query = bool(anchor_terms or snake_case_ids) and any(
            kw in user_input.lower() for kw in _PROPAGATION_KEYWORDS
        )

        # 🚀 Fix I: Aggregation query detection (L3)
        # Queries asking for completeness ("every place", "all functions") need
        # more results to survive reranking so scattered utility helpers aren't culled.
        _AGGREGATION_KEYWORDS = (
            "every ", "all places", "each place", "everywhere",
            "all functions", "list all", "every file", "all files",
            "all the places", "every place", "each function",
            "all methods", "every method", "scattered", "each file",
            "which files", "what files", "list down", "files in",
            "files you see", "list of files",
        )
        is_aggregation_query = any(
            kw in user_input.lower() for kw in _AGGREGATION_KEYWORDS
        )

        reranker_score = 0.0
        new_retrievals = []
        if not skip_retrieval and db:
            new_retrievals = hybrid_search(
                db, search_query,
                collection_name=coll_name,
                k=k_fetch,
                exclude_file=pinned_file,
                filter_extensions=ext_filter,
                query_embedding=current_emb if search_query == user_input else None,
            )

            # Fix D & E: extra retrieval passes for propagation queries.
            # All FTS5 and ChromaDB queries are batched (OR) to minimize
            # database round-trips instead of looping per anchor term.
            if is_propagation_query:
                with SQLiteFTS5BM25(coll_name) as fts:
                    top_anchors = retrieval_anchors[:3]

                    # 🚀 Fix E: Call-graph retrieval via FTS5 OR (L2)
                    try:
                        call_docs = fts.search_by_calls_batch(top_anchors, k=15)
                        if call_docs:
                            logger.info(f"📍 Fix E: Injected {len(call_docs)} chunks calling {top_anchors}")
                            new_retrievals.extend(call_docs)
                    except Exception as e:
                        logger.warning(f"Fix E FTS5 batch call retrieval failed: {e}")

                    # 🚀 Fix G: Constant-reference retrieval via FTS5 OR (L4)
                    try:
                        const_docs = fts.search_by_constants_batch(top_anchors, k=15)
                        if const_docs:
                            logger.info(f"📍 Fix G: Injected {len(const_docs)} chunks referencing constants {top_anchors}")
                            new_retrievals.extend(const_docs)
                    except Exception as e:
                        logger.warning(f"Fix G FTS5 batch constant retrieval failed: {e}")

                # 🚀 Fix J: Guaranteed anchor text retrieval via ChromaDB $or (L4)
                # BM25 misses config files (implicit AND + length penalty).
                # Vector search misses them (no semantic similarity).
                # Use ChromaDB where_document $or for exact substring matching —
                # guaranteed to find any chunk containing any anchor text.
                if db:
                    try:
                        if len(top_anchors) > 1:
                            where_doc = {"$or": [{"$contains": a} for a in top_anchors]}
                        else:
                            where_doc = {"$contains": top_anchors[0]}
                        text_docs = db.similarity_search(
                            user_input, k=10,
                            where_document=where_doc
                        )
                        if text_docs:
                            logger.info(f"📍 Fix J: Injected {len(text_docs)} chunks containing {top_anchors}")
                            new_retrievals.extend(text_docs)
                    except Exception as e:
                        logger.warning(f"Fix J text search failed: {e}")

                for anchor in retrieval_anchors[:2]:
                    for sub_q in (
                        f"function that reads {anchor}",
                        f"where {anchor} is used or evaluated",
                    ):
                        try:
                            extra = hybrid_search(
                                db, sub_q,
                                collection_name=coll_name,
                                k=6,
                                exclude_file=pinned_file,
                                filter_extensions=ext_filter,
                                query_embedding=None,
                            )
                            new_retrievals.extend(extra)
                        except Exception as e:
                            logger.warning(f"Sub-query retrieval failed for '{sub_q}': {e}")

                # Deduplicate the merged pool before reranking.
                seen_hashes_pre = set()
                deduped = []
                for d in new_retrievals:
                    h = d.metadata.get("content_hash") or d.page_content[:200]
                    if h not in seen_hashes_pre:
                        deduped.append(d)
                        seen_hashes_pre.add(h)
                new_retrievals = deduped
        elif skip_retrieval:
            # Semantic cache hit — reuse previous docs as the fresh set.
            new_retrievals = list(previous_union)

        # Save pre-rerank pool for Fix H post-reranker injection
        pre_rerank_pool = list(new_retrievals) if is_propagation_query else []

        # 3. 🎯 Local Re-ranking (Phase 3) ──────────────────────────────
        if USE_RERANKER and reranker and new_retrievals and not skip_retrieval:
            # Fix I: Widen reranker window for aggregation queries so scattered
            # utility helpers aren't culled from the top-k.
            effective_top_k = RERANK_TOP_K * 2 if is_aggregation_query else RERANK_TOP_K
            new_retrievals = reranker.rerank(
                search_query,
                new_retrievals,
                top_k=effective_top_k
            )
            # Capture the top relevance score for telemetry
            reranker_score = getattr(reranker, 'last_top_score', 0.0)

        # 🚀 Fix H: Post-reranker anchor injection (L4)
        # Force-include definition-site chunks the cross-encoder culled.
        # This ensures config constants and bridge functions survive reranking.
        if is_propagation_query and retrieval_anchors and pre_rerank_pool:
            reranked_hashes = {
                d.metadata.get("content_hash", d.page_content[:200])
                for d in new_retrievals
            }
            for anchor in retrieval_anchors[:3]:
                injected = 0
                for d in pre_rerank_pool:
                    if injected >= 2:
                        break
                    h = d.metadata.get("content_hash", d.page_content[:200])
                    if h in reranked_hashes:
                        continue
                    # Match definition sites (CONST = ...) or direct references
                    if f"{anchor}" in d.page_content:
                        ref_consts = d.metadata.get("references_constants", "")
                        is_zero = d.metadata.get("zero_chunk", False)
                        # Inject if: it's the definition file (zero-chunk with the constant)
                        # or it references the constant in its metadata
                        if is_zero or anchor in ref_consts:
                            new_retrievals.append(d)
                            reranked_hashes.add(h)
                            injected += 1
                            logger.info(f"📍 Fix H: Post-reranker injected chunk from {d.metadata.get('source', '?')} for '{anchor}'")

        # Intent-Aware Union Logic with Context Decay
        if intent == "FOLLOW-UP":
            # 🚀 Fix: Prevent "Knowledge Lock-in" by ensuring fresh retrievals 
            # always have priority. We calculate unique new docs first.
            seen_hashes = set()
            unique_new = []
            for d in new_retrievals:
                h = d.metadata.get("content_hash", d.page_content)
                if h not in seen_hashes:
                    unique_new.append(d)
                    seen_hashes.add(h)

            # Cap the new retrievals at MAX_CONTEXT_UNION
            unique_new = unique_new[:MAX_CONTEXT_UNION]
            
            # Rebuild seen_hashes based on the sliced unique_new to avoid dropping valid old docs
            seen_hashes = {d.metadata.get("content_hash", d.page_content) for d in unique_new}
            
            # Calculate how many slots are left for the older stable docs
            available_old_slots = MAX_CONTEXT_UNION - len(unique_new)

            # Eviction: keep old docs in the SAME ORDER they had in
            # previous_union so the established context block is
            # byte-stable across turns — critical for the provider
            # prefix cache (Anthropic / Gemini / DeepSeek).
            #
            # Earlier versions scored old docs by current-query keyword
            # overlap and re-sorted.  That changed membership AND order
            # whenever the user rephrased, destroying the cached prefix
            # on almost every turn.  Freshness priority is already
            # preserved by `unique_new` taking the first N slots; the
            # old docs just fill the tail in their original order.
            surviving_old = []
            for d in previous_union:
                if len(surviving_old) >= available_old_slots:
                    break
                h = d.metadata.get("content_hash", d.page_content)
                if h in seen_hashes:
                    continue  # already in unique_new
                surviving_old.append(d)
                seen_hashes.add(h)

            final_docs = surviving_old + unique_new
            protected_count = len(unique_new)
        else:
            final_docs = new_retrievals[:MAX_CONTEXT_UNION]
            protected_count = 0

        # Filter massive zero-chunks from retrieval results for all models.
        # Zero-chunks can be up to ZERO_CHUNK_THRESHOLD (100k chars / ~33k tokens) and destroy
        # signal-to-noise when surfaced via retrieval.  The pinned-file mechanism handles
        # deliberate full-file viewing; retrieved zero-chunks are almost never the right behaviour.
        final_docs = [
            d for d in final_docs
            if not (d.metadata.get("zero_chunk") and len(d.page_content) > MAX_ZERO_CHUNK_CHARS)
        ]

        # ── Context window budget enforcement ──────────────────────────
        # Estimate total prompt tokens and drop trailing RAG chunks until
        # we fit.  This prevents silent API failures on models with small
        # context windows (8K Ollama, 32K free-tier).
        # Specific patterns MUST appear before generic ones — the first
        # match wins, so "qwen2.5:3b" must precede "qwen", etc.
        _CONTEXT_BUDGETS = {
            "qwen2.5:3b": 6000, "llama3.2:1b": 4000,
            "ollama": 6000, "llama": 6000,
            "gemma": 28000, "gemini": 28000, "claude": 180000,
            "gpt-oss": 28000, "gpt": 120000,
            "deepseek": 60000, "qwen": 28000,
        }
        _budget = 28000  # default
        for _pattern, _limit in _CONTEXT_BUDGETS.items():
            if _pattern in (model or "").lower():
                _budget = _limit
                break
        # Estimate: system prompt + pinned + history + RAG + user query
        _sys_est = len(CORE_INSTRUCTIONS) // 3
        _pinned_est = _content_len(inputs.get("full_source_context", "")) // 3
        _hist_est = _est_tokens(history)
        _query_est = len(user_input) // 3
        _overhead = _sys_est + _pinned_est + _hist_est + _query_est + 500  # safety margin
        _rag_budget = _budget - _overhead
        # Drop chunks from the end (lowest relevance) until within budget.
        # Account for _format_docs overhead (~80 chars per chunk for
        # "SOURCE: ...\nCONTENT: " prefix).
        _FMT_OVERHEAD_PER_CHUNK = 80
        
        # 🚀 Fix Performance: Calculate total length once and decrement instead of re-summing in a loop (O(N) vs O(N^2))
        total_rag_chars = sum(len(d.page_content) + _FMT_OVERHEAD_PER_CHUNK for d in final_docs)
        
        while final_docs and (total_rag_chars // 3) > _rag_budget:
            # Pop from the tail end of surviving_old first, otherwise pop from unique_new
            if len(final_docs) > protected_count:
                idx = len(final_docs) - protected_count - 1
            else:
                idx = -1
                
            removed_doc = final_docs.pop(idx)
            total_rag_chars -= (len(removed_doc.page_content) + _FMT_OVERHEAD_PER_CHUNK)

        # 🚀 Split Context: Prefix cache hits on <established_context>, Relevance hits on <new_discoveries>
        stable_hashes = {
            d.metadata.get("content_hash", "")
            for d in previous_union
        } if previous_union and intent == "FOLLOW-UP" else None

        established_docs = []
        new_docs = []
        if stable_hashes:
            for d in final_docs:
                if d.metadata.get("content_hash", "") in stable_hashes:
                    established_docs.append(d)
                else:
                    new_docs.append(d)
        else:
            new_docs = final_docs

        # Only sort the established context deterministically to preserve identical byte-string
        established_docs = _sort_docs_deterministically(established_docs, stable_hashes=None)
        new_docs = _sort_docs_deterministically(new_docs, stable_hashes=None)

        def _format_docs(docs):
            return "\n\n".join([f"SOURCE: {d.metadata.get('source')}\nCONTENT: {d.page_content}" for d in docs]) if docs else ""

        stable_block = _format_docs(established_docs)
        new_block = _format_docs(new_docs)

        inputs["stable_context"] = f"<established_context>\n{stable_block}\n</established_context>" if stable_block else "None previously established."
        inputs["new_context"] = f"<new_discoveries>\n{new_block}\n</new_discoveries>" if new_block else "No new discoveries."
        inputs["context"] = established_docs + new_docs
        inputs["chat_history"] = _prepare_history_with_cache(history, model)
        
        # Dynamic Specialist Swap — cached LLM instances
        active_chain = question_answer_chain
        if enable_auto and specialist_model:
            current_m = getattr(
                active_chain.bound if hasattr(active_chain, "bound") else active_chain,
                "model_name", "",
            )
            if specialist_model != current_m:
                if specialist_model not in _specialist_llm_cache:
                    _specialist_llm_cache[specialist_model] = get_llm(
                        model=specialist_model, streaming=True
                    )
                active_chain = prompt | _specialist_llm_cache[specialist_model]

        # Dynamic output token budget — reduce for simple queries to free provider quota
        output_tokens = _get_max_tokens(specialty, user_input)
        if output_tokens != MAX_TOKENS:
            base_llm = (
                _specialist_llm_cache[specialist_model]
                if (enable_auto and specialist_model and specialist_model in _specialist_llm_cache)
                else llm
            )
            # ChatOllama uses 'num_predict' for output token budget; OpenAI/OpenRouter use 'max_tokens'.
            active_model_id = specialist_model or model or ""
            if active_model_id.startswith(OLLAMA_PREFIX):
                active_chain = prompt | base_llm.bind(num_predict=output_tokens)
            else:
                active_chain = prompt | base_llm.bind(max_tokens=output_tokens)
        
        # Ensure we always have an embedding to pass back for next turn.
        if current_emb is None:
            if is_semantic_hit and last_query_emb:
                current_emb = last_query_emb
            else:
                from backend import get_embedding_model
                current_emb = get_embedding_model().embed_query(user_input)

        # 🚀 ASYNC SENTINEL TRIGGER — uses full_history so the summary
        # covers the entire conversation, not just the compressed window.
        background_future = None
        if should_summarize and not inputs.get("sentinel_future_active"):
            _sentinel_cooldown["last_turn"] = turn_count
            # CRITICAL FIX: Pass a snapshot (shallow copy) to prevent thread race condition
            background_future = _background_executor.submit(_background_summarize, list(full_history))

        yield {
            "context": inputs["context"], 
            "intent": intent, 
            "query_embedding": current_emb,
            "specialist_active": specialist_model if enable_auto else None,
            "top_relevance_score": reranker_score,
            "sentinel_future": background_future # Pass future to UI for persistence
        }
        
        full_answer = ""
        for chunk in active_chain.stream(inputs):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_answer += content
            yield {"answer": content, "raw_chunk": chunk}
            
        # 🚀 PHASE 3: Update Semantic Cache with fresh generation
        # Tag with the full scope so future lookups can reject the entry
        # if the user re-pins, changes the file, or switches models.
        if not is_semantic_hit and len(full_answer) > 50:
            sem_cache.upsert(user_input, full_answer, collection_scope=coll_name,
                             pinned_content=pinned_content, model=model)

    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(_full_context_cache_chain)
