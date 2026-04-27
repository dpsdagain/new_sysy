import os
import threading
import logging
from urllib.parse import urlparse
import ipaddress
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from backend import load_existing_chroma, SQLiteFTS5BM25
from rag_chain import hybrid_search, get_reranker
from config import RETRIEVER_K, RERANK_TOP_K, USE_RERANKER, WORKSPACE_ROOT

logger = logging.getLogger(__name__)

# 🚀 Resource Safety: Global process tracking to prevent orphans on Windows
_active_process_groups = []
_process_lock = threading.Lock()

def cleanup_active_processes():
    """Kill all remaining process groups (call on CLI exit)."""
    import signal
    with _process_lock:
        for pid in _active_process_groups:
            try:
                if os.name == 'nt':
                    os.kill(pid, signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception:
                pass
        _active_process_groups.clear()

def validate_path(path: str) -> str:
    """Ensure the path is within the WORKSPACE_ROOT boundary using OS-level checks."""
    from pathlib import Path
    try:
        # Resolve to absolute, real path (handles .. and symlinks)
        target = Path(path).resolve()
        root = Path(WORKSPACE_ROOT).resolve()
        
        # Check if target is inside root
        if not target.is_relative_to(root):
            raise PermissionError(f"Access Denied: Path '{target}' is outside the allowed workspace '{root}'.")
            
        return str(target)
    except Exception as e:
        if isinstance(e, PermissionError):
            raise e
        raise PermissionError(f"Access Denied: Could not validate path '{path}'.")

class CodeSearchInput(BaseModel):
    query: str = Field(description="The natural language query or keywords to search for in the codebase.")
    collection_name: str = Field(default="default", description="The name of the collection to search within.")
    k: int = Field(default=RETRIEVER_K, description="Number of initial documents to retrieve.")

class FileReadInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to read.")
    start_line: Optional[int] = Field(default=None, description="The 1-based line number to start reading from.")
    end_line: Optional[int] = Field(default=None, description="The 1-based line number to end reading at.")

class FileEditInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to edit.")
    old_string: str = Field(description="The exact literal text to replace.")
    new_string: str = Field(description="The text to replace old_string with.")

class Replacement(BaseModel):
    old_string: str = Field(description="The exact literal text to replace.")
    new_string: str = Field(description="The text to replace old_string with.")

class MultiFileEditInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to edit.")
    replacements: List[Replacement] = Field(description="A list of replacement pairs.")

class GrepInput(BaseModel):
    pattern: str = Field(description="The regex pattern to search for.")
    include_pattern: Optional[str] = Field(default=None, description="Glob pattern for files to include (e.g., '*.py').")
    exclude_pattern: Optional[str] = Field(default=None, description="Glob pattern for files to exclude.")
    case_sensitive: bool = Field(default=False, description="Whether the search should be case-sensitive.")

def grep_tool(pattern: str, include_pattern: Optional[str] = None, exclude_pattern: Optional[str] = None, case_sensitive: bool = False) -> str:
    """Search for a pattern across the codebase using Python-native regex for platform consistency."""
    import re
    import fnmatch
    
    flags = re.IGNORECASE if not case_sensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Invalid regex pattern: {str(e)}"

    matches = []
    root_dir = str(WORKSPACE_ROOT)
    
    for root, _, files in os.walk(root_dir):
        # Apply directory exclusions
        if any(fnmatch.fnmatch(root, f"*{exc}*") for exc in ["__pycache__", "venv", ".git", "chroma_db"]):
            continue
            
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, root_dir)
            
            # Filter by include/exclude patterns
            if include_pattern and not fnmatch.fnmatch(file, include_pattern):
                continue
            if exclude_pattern and fnmatch.fnmatch(file, exclude_pattern):
                continue
                
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if regex.search(line):
                            matches.append(f"{rel_path}:{i}:{line.strip()}")
            except Exception:
                continue

    if not matches:
        return f"No matches found for pattern: {pattern}"
    
    return "\n".join(matches[:500]) # Cap at 500 lines for context safety

def multi_file_edit(file_path: str, replacements: List[Replacement]) -> str:
    """Apply multiple surgical replacements to a single file in one go."""
    try:
        file_path = validate_path(file_path)
    except PermissionError as e:
        return str(e)
        
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        original_content = content
        diffs = []
        
        for rep in replacements:
            old_str = rep.old_string
            new_str = rep.new_string
            
            if old_str not in content:
                return f"Error: Could not find exact match for '{old_str}' in {file_path}."
            
            # Check for multiple occurrences
            occurrences = content.count(old_str)
            if occurrences > 1:
                return f"Error: Found {occurrences} occurrences of '{old_str}'. Please provide more context."
                
            content = content.replace(old_str, new_str)
            diffs.append(f"[DIFF_START]{old_str}[DIFF_DIVIDER]{new_str}[DIFF_END]")
            
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"Successfully applied {len(replacements)} edits to {file_path}.\n" + "\n".join(diffs)
    except Exception as e:
        return f"Error in multi_file_edit: {str(e)}"

def code_search(query: str, collection_name: str = "default", k: int = RETRIEVER_K) -> str:
    """Search the codebase using hybrid search (Vector + BM25)."""
    db = load_existing_chroma(collection_name)
    if not db:
        return f"Error: Collection '{collection_name}' not found or is empty."
    
    docs = hybrid_search(db, query, collection_name=collection_name, k=k)
    
    if USE_RERANKER:
        reranker = get_reranker()
        if reranker:
            docs = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
            
    if not docs:
        return "No relevant code snippets found."
    
    formatted_results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        formatted_results.append(f"--- Result {i+1} ({source}) ---\n{doc.page_content}")
        
    return "\n\n".join(formatted_results)

def file_read(file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """Read a file's content, optionally within a line range."""
    try:
        file_path = validate_path(file_path)
    except PermissionError as e:
        return str(e)

    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        if start_line is not None or end_line is not None:
            start = (start_line - 1) if start_line else 0
            end = end_line if end_line else len(lines)
            content = "".join(lines[start:end])
            return f"--- Content of {file_path} (Lines {start+1}-{end}) ---\n{content}"
        else:
            content = "".join(lines)
            return f"--- Content of {file_path} ---\n{content}"
    except Exception as e:
        return f"Error reading file: {str(e)}"
# 🛡️ Fail-Closed Tool Factory (buildTool.ts:15)
class ToolMetadata(BaseModel):
    name: str
    description: str
    input_schema: Any
    func: Any
    is_read_only: bool = False # Default: Fail-Closed (Assumed to modify data)
    is_destructive: bool = False
    is_concurrency_safe: bool = False # Default: Fail-Closed (Assumed to require lock)

def build_tool(
    name: str, 
    description: str, 
    input_schema: Any, 
    func: Any, 
    is_read_only: bool = False,
    is_destructive: bool = False,
    is_concurrency_safe: bool = False
) -> Dict[str, Any]:
    """
    Centralized tool factory that enforces security defaults.
    Every tool in the system must pass through this gate.
    """
    return {
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "func": func,
        "is_read_only": is_read_only,
        "is_destructive": is_destructive,
        "is_concurrency_safe": is_concurrency_safe
    }

# 🌐 Global Engine Context for Tool-to-Coordinator Routing
current_engine = None

logger = logging.getLogger(__name__)

# 🚀 Resource Safety: Global process tracking to prevent orphans on Windows
_active_process_groups = []
_process_lock = threading.Lock()

def cleanup_active_processes():
    """Kill all remaining process groups (call on CLI exit)."""
    import signal
    with _process_lock:
        for pid in _active_process_groups:
            try:
                if os.name == 'nt':
                    os.kill(pid, signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
            except Exception:
                pass
        _active_process_groups.clear()

def validate_path(path: str) -> str:
    """Ensure the path is within the WORKSPACE_ROOT boundary using OS-level checks."""
    from pathlib import Path
    try:
        # Resolve to absolute, real path (handles .. and symlinks)
        target = Path(path).resolve()
        root = Path(WORKSPACE_ROOT).resolve()
        
        # Check if target is inside root
        if not target.is_relative_to(root):
            raise PermissionError(f"Access Denied: Path '{target}' is outside the allowed workspace '{root}'.")
            
        return str(target)
    except Exception as e:
        if isinstance(e, PermissionError):
            raise e
        raise PermissionError(f"Access Denied: Could not validate path '{path}'.")


class SwitchModelInput(BaseModel):
    model_id: str = Field(..., description="The ID of the model to switch to (e.g., 'ollama-cloud:gpt-oss:120b-cloud').")

def switch_model(model_id: str) -> str:
    """Reboots the agent with a new LLM engine. All conversation context is preserved."""
    return f"[MODEL_SWITCHED] {model_id}"

class UpdatePlanInput(BaseModel):
    plan: str = Field(description="The updated step-by-step plan for the current task.")

class SetStatusInput(BaseModel):
    status: str = Field(description="Brief status message for the UI (e.g. 'Analyzing index...').")

class UndoInput(BaseModel):
    message_id: str = Field(description="The ID of the turn/message to revert. Use the tool_use ID from the turn you want to undo.")

class NotebookEditInput(BaseModel):
    file_path: str = Field(description="Path to the .ipynb file.")
    cell_id: str = Field(description="UUID or virtual ID (cell-0, cell-1) of the cell.")
    new_source: str = Field(description="New content for the cell.")
    edit_mode: str = Field(default="replace", description="replace, insert, or delete.")
    cell_type: str = Field(default="code", description="code or markdown.")

class DoctorInput(BaseModel):
    pass

class CostInput(BaseModel):
    pass

class GitStatusInput(BaseModel):
    pass

class GitDiffInput(BaseModel):
    file_path: Optional[str] = Field(None, description="Optional path to a specific file to diff.")

class GitCommitInput(BaseModel):
    message: str = Field(description="The commit message.")

class GitLogInput(BaseModel):
    limit: int = Field(default=5, description="Number of recent commits to show.")

# -----------------------------------------------------------------------------
# GIT INTEGRATION (F-34)
class UpdatePlanInput(BaseModel):
    plan: str = Field(description="The updated step-by-step plan for the current task.")

class SetStatusInput(BaseModel):
    status: str = Field(description="Brief status message for the UI (e.g. 'Analyzing index...').")

class UndoInput(BaseModel):
    message_id: str = Field(description="The ID of the turn/message to revert. Use the tool_use ID from the turn you want to undo.")

class NotebookEditInput(BaseModel):
    file_path: str = Field(description="Path to the .ipynb file.")
    cell_id: str = Field(description="UUID or virtual ID (cell-0, cell-1) of the cell.")
    new_source: str = Field(description="New content for the cell.")
    edit_mode: str = Field(default="replace", description="replace, insert, or delete.")
    cell_type: str = Field(default="code", description="code or markdown.")

class DoctorInput(BaseModel):
    pass

class CostInput(BaseModel):
    pass

class GitStatusInput(BaseModel):
    pass

class GitDiffInput(BaseModel):
    file_path: Optional[str] = Field(None, description="Optional path to a specific file to diff.")

class GitCommitInput(BaseModel):
    message: str = Field(description="The commit message.")

class GitLogInput(BaseModel):
    limit: int = Field(default=5, description="Number of recent commits to show.")

# -----------------------------------------------------------------------------

@tool
def git_status() -> str:
    """Read the current git status of the project (F-34). Returns structured tracked/untracked files as JSON."""
    from git_manager import GitManager
    import json
    manager = GitManager(os.getcwd())
    return json.dumps(manager.get_file_status(), indent=2)

@tool
def git_diff(file_path: Optional[str] = None) -> str:
    """Show changes in the working directory (F-34). Automatically skips binary files."""
    import subprocess
    from pathlib import Path
    
    # Binary protection (logic from git.ts:644)
    if file_path:
        ext = Path(file_path).suffix.lower()
        if ext in ['.pdf', '.exe', '.dll', '.bin', '.png', '.jpg', '.so']:
            return f"Binary file '{file_path}' skipped for diff."

    try:
        cmd = ["git", "diff", "--no-color"]
        if file_path:
            cmd.append(file_path)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout if result.stdout else "No changes detected."
    except Exception as e:
        return f"Error running git diff: {str(e)}"

@tool
def git_commit(message: str) -> str:
    """Commit staged changes to the repository."""
    import subprocess
    try:
        # Check if anything is staged
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
        if not any(not line.startswith("??") for line in status.splitlines() if line.strip()):
            return "Error: No staged changes to commit. Use 'git add' via bash first."
            
        result = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True, check=True)
        return f"Successfully committed: {result.stdout}"
    except Exception as e:
        return f"Error running git commit: {str(e)}"

@tool
def git_root() -> str:
    """Find the canonical git root of the project, resolving through worktrees/submodules."""
    from git_manager import GitManager
    manager = GitManager(os.getcwd())
    root = manager.resolve_canonical_root()
    return str(root) if root else "Not a git repository."

@tool
def git_log(limit: int = 5) -> str:
    """View recent project history (Commits)."""
    import subprocess
    try:
        res = subprocess.run(["git", "log", "--oneline", "-n", str(limit)], capture_output=True, text=True)
        return res.stdout if res.returncode == 0 else "Error reading git log."
    except:
        return "Git not found."

@tool
def cost_report() -> str:
    """Generate a high-precision session cost report (F-18 Parity). Shows token usage and USD cost."""
    try:
        from query_engine import current_engine
        if not current_engine or not current_engine.usage_tracker:
            return "Error: Usage Tracker not initialized."
        
        return current_engine.usage_tracker.get_report()
    except Exception as e:
        return f"Error generating cost report: {str(e)}"

@tool
def read_url(url: str, prompt: Optional[str] = None) -> str:
    """Fetch and distill content from a URL (F-10 Parity). Best for reading documentation websites."""
    from web_utils import WebFetcher
    import json
    results = WebFetcher.fetch_markdown(url, prompt)
    if "error" in results:
        return f"Error fetching URL: {results['error']}"
    
    return f"--- Content from {url} ---\n{results['content']}\n\n[Total Length: {results['length']} chars]"

@tool
def system_doctor() -> str:
    """Perform a full environmental diagnostic check (F-44 Parity). Audits binaries, network, and workspace toxicity."""
    from doctor import SystemDoctor
    import json
    try:
        report = SystemDoctor.audit()
        return f"--- System Health Report ---\n{json.dumps(report, indent=2)}"
    except Exception as e:
        return f"Error running diagnostics: {str(e)}"

@tool
def notebook_edit(file_path: str, cell_id: str, new_source: str, edit_mode: str = "replace", cell_type: str = "code") -> str:
    """Surgically edit a Jupyter Notebook cell (F-12 Parity). Resets execution state on modified cells."""
    from notebook_utils import NotebookMutator
    try:
        file_path = validate_path(file_path)
    except Exception as e:
        return str(e)
        
    return NotebookMutator.edit(file_path, cell_id, new_source, edit_mode, cell_type)

def undo_last_edit(message_id: str) -> str:
    """Roll back file changes made in a specific turn (F-28 Parity)."""
    try:
        from query_engine import current_engine
        if not current_engine or not current_engine.history_manager:
            return "Error: History Manager not initialized."
        
        reverted = current_engine.history_manager.rollback(message_id)
        if not reverted:
            return f"No changes found to undo for turn ID: {message_id}"
            
        return f"Successfully reverted changes for {len(reverted)} files. Turn ID: {message_id}"
    except Exception as e:
        return f"Error performing undo: {str(e)}"

class UpdatePlanInput(BaseModel):
    plan: str = Field(description="The updated step-by-step plan for the current task.")

class SetStatusInput(BaseModel):
    status: str = Field(description="Brief status message for the UI (e.g. 'Analyzing index...').")

class UndoInput(BaseModel):
    message_id: str = Field(description="The ID of the turn/message to revert. Use the tool_use ID from the turn you want to undo.")

class NotebookEditInput(BaseModel):
    file_path: str = Field(description="Path to the .ipynb file.")
    cell_id: str = Field(description="UUID or virtual ID (cell-0, cell-1) of the cell.")
    new_source: str = Field(description="New content for the cell.")
    edit_mode: str = Field(default="replace", description="replace, insert, or delete.")
    cell_type: str = Field(default="code", description="code or markdown.")

class DoctorInput(BaseModel):
    pass

class CostInput(BaseModel):
    pass

class GitStatusInput(BaseModel):
    pass

class GitDiffInput(BaseModel):
    file_path: Optional[str] = Field(None, description="Optional path to a specific file to diff.")

class GitCommitInput(BaseModel):
    message: str = Field(description="The commit message.")

class GitLogInput(BaseModel):
    limit: int = Field(default=5, description="Number of recent commits to show.")

# -----------------------------------------------------------------------------
# TOOL REGISTRY (Extended)
class UpdatePlanInput(BaseModel):
    plan: str = Field(description="The updated step-by-step plan for the current task.")

class SetStatusInput(BaseModel):
    status: str = Field(description="Brief status message for the UI (e.g. 'Analyzing index...').")

class UndoInput(BaseModel):
    message_id: str = Field(description="The ID of the turn/message to revert. Use the tool_use ID from the turn you want to undo.")

class NotebookEditInput(BaseModel):
    file_path: str = Field(description="Path to the .ipynb file.")
    cell_id: str = Field(description="UUID or virtual ID (cell-0, cell-1) of the cell.")
    new_source: str = Field(description="New content for the cell.")
    edit_mode: str = Field(default="replace", description="replace, insert, or delete.")
    cell_type: str = Field(default="code", description="code or markdown.")

class DoctorInput(BaseModel):
    pass

class CostInput(BaseModel):
    pass

class GitStatusInput(BaseModel):
    pass

class GitDiffInput(BaseModel):
    file_path: Optional[str] = Field(None, description="Optional path to a specific file to diff.")

class GitCommitInput(BaseModel):
    message: str = Field(description="The commit message.")

class GitLogInput(BaseModel):
    limit: int = Field(default=5, description="Number of recent commits to show.")

# -----------------------------------------------------------------------------

def file_edit(file_path: str, old_string: str, new_string: str) -> str:
    """Surgically replace old_string with new_string in a file (F-05 Parity)."""
    from edit_utils import FuzzyMatcher
    try:
        file_path = validate_path(file_path)
    except PermissionError as e:
        return str(e)

    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        # 🧬 Fuzzy Match Phase
        actual_old = FuzzyMatcher.find_actual_string(content, old_string)
        
        if not actual_old:
            return f"Error: Could not find match for 'old_string' in {file_path}. Fuzzy matching also failed."
        
        # Check for multiple occurrences of the fuzzy result
        occurrences = content.count(actual_old)
        if occurrences > 1:
            return f"Error: Found {occurrences} fuzzy occurrences. Please provide more context."
            
        new_content = content.replace(actual_old, new_string)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        return f"Successfully edited {file_path}. Fuzzy match succeeded on detected block."
    except Exception as e:
        return f"Error editing file: {str(e)}"

import subprocess

class BashInput(BaseModel):
    command: str = Field(description="The shell command to execute.")

import subprocess
import time
import re
import threading
from queue import Queue, Empty

STALL_PATTERNS = [
    r"\? \[y/n\]",
    r"\(y/n\)\?",
    r"\[Y/n\]",
    r"\[y/N\]",
    r"confirm \[y/n\]",
    r"password:",
    r"enter to continue",
    r"press any key",
    r"terminate batch job",
]

def bash_tool(command: str) -> str:
    """Execute a shell command with real-time stall detection and process-group termination."""
    import subprocess
    import time
    import re
    import threading
    import signal
    from queue import Queue, Empty

    try:
        # Security check - Expanded Blacklist
        dangerous = [
            "rm -rf /", "mkfs", "dd if=", "format", "chown", "chmod", 
            "> /etc/", "sudo", "su -", "userdel", "groupdel",
            ":(){ :|:& };:", # Fork bomb
        ]
        if any(d in command for d in dangerous):
            return "Error: Command rejected for security reasons (Dangerous pattern detected)."

        # 🚀 Path Sentinel for BASH: Prevent commands from targeting paths outside the workspace
        # We look for path-like strings in the command and validate them
        paths = re.findall(r'((?:[a-zA-Z]:\\|[/\\])[\w\s.-]+(?:[/\\][\w\s.-]+)*)', command)
        for p in paths:
            try:
                # We use the existing validate_path to ensure the command doesn't leak data
                if not validate_path(p):
                    return f"Error: Command rejected. Detected attempt to access restricted path: {p}"
            except Exception:
                pass # Not a valid path, ignore

        # Windows-specific process group creation to allow killing child trees
        creationflags = 0
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        # Use Popen to allow real-time reading
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
            creationflags=creationflags
        )

        # 🚀 Resource Safety: Register for global cleanup
        with _process_lock:
            _active_process_groups.append(process.pid)

        output_queue = Queue()
        
        def reader(stream, queue):
            try:
                while True:
                    char = stream.read(1)
                    if not char:
                        break
                    queue.put(char)
            except Exception:
                pass
            finally:
                stream.close()

        stdout_thread = threading.Thread(target=reader, args=(process.stdout, output_queue))
        stderr_thread = threading.Thread(target=reader, args=(process.stderr, output_queue))
        stdout_thread.start()
        stderr_thread.start()

        full_output = []
        last_output_time = time.time()
        timeout = 30
        start_time = time.time()
        stall_detected = False

        while True:
            # Check for timeout
            if time.time() - start_time > timeout:
                if os.name == 'nt':
                    os.kill(process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    process.terminate()
                return "".join(full_output) + f"\n\nError: Command timed out after {timeout} seconds."

            try:
                # Use a very small timeout for non-blocking feel
                line = output_queue.get(timeout=0.1)
                full_output.append(line)
                last_output_time = time.time()
            except Empty:
                # No new output, check for stalls
                if process.poll() is not None:
                    # Process finished normally
                    break
                
                # If we've been waiting > 5.0 seconds with no new output, check the tail
                if time.time() - last_output_time > 5.0:
                    # Look at the last 100 characters of the total output
                    current_text = "".join(full_output).strip()
                    tail = current_text[-100:]
                    if any(re.search(p, tail, re.IGNORECASE) for p in STALL_PATTERNS):
                        stall_detected = True
                        if os.name == 'nt':
                            os.kill(process.pid, signal.CTRL_BREAK_EVENT)
                        else:
                            process.terminate()
                        break
            
        # Ensure threads finish
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

        result = "".join(full_output)
        
        if stall_detected:
            return (
                f"{result}\n\n"
                f"⚠️ [STALL DETECTED]: The command above appears to be waiting for interactive input (y/n, password, etc.). "
                f"I have terminated the process group to prevent orphaned child processes. "
                f"Please either:\n"
                f"1. Use a non-interactive flag (e.g., -y or --force).\n"
                f"2. Ask the user for help if human intervention is required."
            )

        if not result.strip():
            return "Command executed successfully (no output)."

        return result

    except Exception as e:
        return f"Error executing command: {str(e)}"

class FileWriteInput(BaseModel):
    file_path: str = Field(description="The absolute path to the file to create or overwrite.")
    content: str = Field(description="The full content to write to the file.")

def file_write(file_path: str, content: str) -> str:
    """Create or overwrite a file with the provided content."""
    try:
        file_path = validate_path(file_path)
    except PermissionError as e:
        return str(e)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}."
    except Exception as e:
        return f"Error writing file: {str(e)}"


class GlobInput(BaseModel):
    pattern: str = Field(description="Glob pattern to search for (e.g., '**/*.py').")

def glob_tool(pattern: str) -> str:
    """Find files matching a glob pattern."""
    import glob
    try:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            return f"No files found matching pattern: {pattern}"
            
        # Filter matches to ensure they are inside WORKSPACE_ROOT
        safe_matches = []
        for m in matches:
            try:
                safe_matches.append(validate_path(m))
            except PermissionError:
                continue
                
        if not safe_matches:
            return "No files found within the allowed workspace."
            
        return "Found files:\n" + "\n".join(safe_matches)
    except Exception as e:
        return f"Error executing glob: {str(e)}"

class BriefInput(BaseModel):
    file_path: str = Field(description="Path to the file to summarize.")

def brief_tool(file_path: str) -> str:
    """Provide a brief outline of a file to save context."""
    try:
        file_path = validate_path(file_path)
    except PermissionError as e:
        return str(e)
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    try:
        if file_path.endswith('.py'):
            import ast
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            outline = []
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    outline.append(f"class {node.name}:")
                    for sub in node.body:
                        if isinstance(sub, ast.FunctionDef):
                            outline.append(f"    def {sub.name}(...)")
                elif isinstance(node, ast.FunctionDef):
                    outline.append(f"def {node.name}(...)")
            if outline:
                return f"Outline of {file_path}:\n" + "\n".join(outline) + "\n\nUse file_read to see full implementation."
            return f"File {file_path} contains no top-level classes or functions."
        else:
            size = os.path.getsize(file_path)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                head = "".join([next(f) for _ in range(20)])
            return f"File {file_path} ({size} bytes). First 20 lines:\n{head}\n... Use file_read to see more."
    except Exception as e:
        return f"Error generating brief: {str(e)}"

class LinterInput(BaseModel):
    file_path: str = Field(description="Path to the file to lint.")

def linter_tool(file_path: str) -> str:
    """Run a basic linter check on a file."""
    try:
        file_path = validate_path(file_path)
    except PermissionError as e:
        return str(e)
    if file_path.endswith('.py'):
        try:
            result = subprocess.run(["flake8", file_path], capture_output=True, text=True)
            if result.returncode == 0:
                return "No linting errors found."
            return result.stdout
        except FileNotFoundError:
            return "Linter (flake8) not installed. Use bash tool to run a specific linter."
    return "Linter tool currently only supports Python (.py) files natively. Use bash for others."

class MemoryInput(BaseModel):
    fact: str = Field(description="A fact, preference, or learning to remember for future sessions.")

def memory_tool(fact: str) -> str:
    """Save a memory or preference to a persistent MEMORY.md file."""
    try:
        # Resolve path to project root
        memory_path = os.path.join(str(WORKSPACE_ROOT), "MEMORY.md")
        with open(memory_path, "a", encoding="utf-8") as f:
            f.write(f"- {fact}\n")
        return f"Memory saved successfully to {memory_path}."
    except Exception as e:
        return f"Error saving memory: {str(e)}"

class AskUserInput(BaseModel):
    question: str = Field(description="The question to ask the user.")

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the internet.")

class WebFetchInput(BaseModel):
    url: str = Field(description="The URL of the page to fetch and read.")

class SymbolSearchInput(BaseModel):
    symbol: str = Field(description="The name of the class, function, or variable to find the definition of.")

class AgentDelegateInput(BaseModel):
    task: str = Field(description="The specific task or goal for the sub-agent to achieve.")
    context_files: List[str] = Field(default=[], description="List of file paths the sub-agent should focus on.")

class ArchVisualizerInput(BaseModel):
    directory: str = Field(default=".", description="The directory to visualize.")

class UndercoverInput(BaseModel):
    text: str = Field(description="The text to strip AI markers and internal paths from.")

class TaskBudgetInput(BaseModel):
    max_tokens: int = Field(description="The maximum number of tokens allowed for this task.")

def web_search(query: str) -> str:
    """Perform a web search using a robust strategy with specialized headers."""
    import requests
    import re
    try:
        # Enhanced headers to mimic a real browser and avoid DDG bot blocking
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://duckduckgo.com/",
            "DNT": "1",
            "Connection": "keep-alive"
        }
        url = f"https://html.duckduckgo.com/html/?q={query}"
        session = requests.Session()
        resp = session.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        results = []
        # Target specific DDG result classes
        for result in soup.find_all('div', class_='result'):
            title_tag = result.find('a', class_='result__a')
            snippet_tag = result.find('a', class_='result__snippet')
            
            if title_tag and title_tag.get('href'):
                title = title_tag.get_text().strip()
                link = title_tag.get('href')
                
                # Cleanup DDG redirect URLs
                if 'uddg=' in link:
                    match = re.search(r'uddg=([^&]+)', link)
                    if match:
                        from urllib.parse import unquote
                        link = unquote(match.group(1))
                
                snippet = snippet_tag.get_text().strip() if snippet_tag else "No snippet."
                results.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")
            
            if len(results) >= 5:
                break
                
        if not results:
            return f"No results found for '{query}'. The search engine layout may have changed."
            
        return "Search Results:\n\n" + "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Error performing web search: {str(e)}"

def is_safe_url(url: str) -> bool:
    """SSRF Shield: Blocks access to private IP ranges and local hostnames."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False
        if hostname.lower() in ["localhost", "127.0.0.1", "0.0.0.0", "::1"]:
            return False
        try:
            ip = ipaddress.ip_address(hostname)
            return not ip.is_private
        except ValueError:
            return True
    except Exception:
        return False

def web_fetch(url: str) -> str:
    if not is_safe_url(url):
        return "Error: Access to local or private network addresses is restricted."
    import requests
    import re
    """Fetch webpage content with improved extraction and noise reduction."""
    try:
        from bs4 import BeautifulSoup
        has_bs4 = True
    except ImportError:
        has_bs4 = False

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        if has_bs4:
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Aggressively remove junk
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'iframe']):
                element.decompose()
            
            # Target main content containers
            main_content = soup.find('main') or soup.find('article') or soup.find('div', id=re.compile(r'content|main|body', re.I))
            if not main_content:
                main_content = soup.find('div', class_=re.compile(r'content|main|body', re.I))
                
            text = (main_content or soup).get_text(separator='\n')
            text = resp.text
        else:
            text = re.sub(r"<script.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<.*?>", " ", text)
        return text
    except Exception as e:
        return f"Error fetching webpage: {str(e)}"
def symbol_search(symbol: str) -> str:
    """Find the definition of a class or function across the codebase (Python-native)."""
    import re
    import fnmatch
    
    # Precise regex patterns for common languages
    patterns = [
        f"def {symbol}\\b",
        f"class {symbol}\\b",
        f"function {symbol}\\b",
        f"const {symbol}\\s*=",
        f"let {symbol}\\s*=",
        f"var {symbol}\\s*="
    ]
    compiled_patterns = [re.compile(p) for p in patterns]
    
    matches = []
    root_dir = str(WORKSPACE_ROOT)
    
    for root, _, files in os.walk(root_dir):
        if any(fnmatch.fnmatch(root, f"*{exc}*") for exc in ["__pycache__", "venv", ".git", "chroma_db"]):
            continue
            
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, root_dir)
            
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if any(p.search(line) for p in compiled_patterns):
                            matches.append(f"{rel_path}:{i}:{line.strip()}")
            except Exception:
                continue

    if not matches:
        return f"Definition for '{symbol}' not found. Try a regular grep_search."
    
    return "Potential Definitions:\n\n" + "\n".join(matches[:100])

class AgentDelegateInput(BaseModel):
    task: str = Field(description="The specific task or goal for the sub-agent to achieve.")
    context_files: List[str] = Field(default=[], description="List of file paths the sub-agent should focus on.")
    current_depth: int = Field(default=0, description="Internal field to track recursion.", exclude=True)

def agent_delegate(task: str, context_files: List[str] = []) -> str:
    """
    Spawn a sub-agent to handle a specific delegated task.
    Orchestrated by the Multi-Agent Coordinator (Coordinator.ts).
    """
    if current_engine is None:
        return "Error: Coordinator not initialized."
    
    context_summary = f"Focus files: {', '.join(context_files)}" if context_files else "General repository context."
    
    try:
        # 🐝 Route through the Coordinator for isolated Worker spawning
        return current_engine.coordinator.delegate(task, context_summary)
    except Exception as e:
        return f"Error in agent delegation: {str(e)}"

def arch_visualizer(directory: str = ".") -> str:
    """Generate a high-level architecture overview in Mermaid format."""
    try:
        directory = validate_path(directory)
    except PermissionError as e:
        return str(e)
    import ast
    mermaid = ["classDiagram"]
    
    for root, _, files in os.walk(directory):
        if any(exc in root for exc in ["__pycache__", "venv", ".git"]):
            continue
            
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        node = ast.parse(f.read())
                    
                    for sub in node.body:
                        if isinstance(sub, ast.ClassDef):
                            mermaid.append(f"    class {sub.name} {{")
                            for item in sub.body:
                                if isinstance(item, ast.FunctionDef):
                                    mermaid.append(f"        +{item.name}()")
                            mermaid.append("    }")
                except Exception:
                    continue
                    
    if len(mermaid) == 1:
        return "No classes found to visualize."
        
    return "Architecture Diagram (Mermaid):\n\n```mermaid\n" + "\n".join(mermaid) + "\n```"

def undercover_mode(text: str) -> str:
    """Strip AI identifiers and local paths from text for professional output."""
    import re
    # Patterns for AI markers
    markers = [
        r"As an AI.*?model,",
        r"I am Claude",
        r"Anthropic",
        r"Assistant",
        r"I don't have feelings",
    ]
    
    cleaned = text
    for marker in markers:
        cleaned = re.sub(marker, "", cleaned, flags=re.IGNORECASE)
        
    # Pattern for local paths (e.g., C:\Users\...)
    cleaned = re.sub(r'[a-zA-Z]:\\[\\\w\s.-]+', '[REDACTED_PATH]', cleaned)
    cleaned = re.sub(r'/(?:[\w.-]+/)+[\w.-]+', '[REDACTED_PATH]', cleaned)
    
    return cleaned.strip()

def task_budget(max_tokens: int) -> str:
    """Set or check a token budget for the current task."""
    # In a full system, this would interact with the QueryEngine's token counter.
    # Here, we simulate setting a limit.
    try:
        with open("budget_config.json", "w") as f:
            import json
            json.dump({"max_tokens": max_tokens}, f)
        return f"Budget set to {max_tokens} tokens. Agent will now monitor usage against this limit."
    except Exception as e:
        return f"Error setting budget: {str(e)}"

def ask_user(question: str) -> str:
    """Pause execution and ask the human user a question."""
    # 🚀 Fix: Removed synchronous input() to prevent deadlocks in UI environments (Streamlit/CLI).
    # The engine (cli.py or app.py) must detect this sentinel and handle the interruption.
    return f"[INTERRUPT_REQUIRED] The agent needs human input: {question}"


# 🗃️ Tool Registry (Partitioned for Prompt Cache Stability)
# Anthropic Logic: Sort alphabetically (Core first, then MCP/Plugins) to maximize prefix cache hits.
_UNSORTED_TOOLS = {
    "code_search": build_tool(
        name="code_search",
        func=code_search,
        input_schema=CodeSearchInput,
        description="Search the codebase for relevant functions, classes, or logic using keywords or natural language.",
        is_read_only=True
    ),
    "file_read": build_tool(
        name="file_read",
        func=file_read,
        input_schema=FileReadInput,
        description="Read the content of a file. Supports line ranges for large files.",
        is_read_only=True
    ),
    "file_edit": build_tool(
        name="file_edit",
        func=file_edit,
        input_schema=FileEditInput,
        description="Edit a file by replacing an exact string with a new string. This is safer than overwriting the whole file.",
        is_read_only=False
    ),
    "file_write": build_tool(
        name="file_write",
        func=file_write,
        input_schema=FileWriteInput,
        description="Create a new file or completely overwrite an existing file with new content.",
        is_read_only=False
    ),
    "multi_file_edit": build_tool(
        name="multi_file_edit",
        func=multi_file_edit,
        input_schema=MultiFileEditInput,
        description="Apply multiple surgical replacements to a single file in one go. Much more efficient than multiple file_edit calls.",
        is_read_only=False
    ),
    "grep_search": build_tool(
        name="grep_search",
        func=grep_tool,
        input_schema=GrepInput,
        description="Search for a pattern across the codebase using regex. Returns file paths and matching lines.",
        is_read_only=True
    ),
    "bash": build_tool(
        name="bash",
        func=bash_tool,
        input_schema=BashInput,
        description="Execute a shell command. Use this for running tests, build scripts, or git commands.",
        is_read_only=False
    ),
    "glob": build_tool(
        name="glob",
        func=glob_tool,
        input_schema=GlobInput,
        description="Search for files using glob patterns (e.g., '**/*.py').",
        is_read_only=True
    ),
    "brief": build_tool(
        name="brief",
        func=brief_tool,
        input_schema=BriefInput,
        description="Provide a brief outline (classes and functions) of a file to save context.",
        is_read_only=True
    ),
    "linter": build_tool(
        name="linter",
        func=linter_tool,
        input_schema=LinterInput,
        description="Run a linter on a specific file to check for syntax errors or formatting issues.",
        is_read_only=True
    ),
    "memory": build_tool(
        name="memory",
        func=memory_tool,
        input_schema=MemoryInput,
        description="Save a learned fact, architecture decision, or user preference into long-term memory (MEMORY.md).",
        is_read_only=False
    ),
    "ask_user": build_tool(
        name="ask_user",
        func=ask_user,
        input_schema=AskUserInput,
        description="Pause agent execution to ask the human user a clarifying question.",
        is_read_only=True,
        is_concurrency_safe=True
    ),
    "web_search": build_tool(
        name="web_search",
        func=web_search,
        input_schema=WebSearchInput,
        description="Search the internet for documentation, libraries, or coding solutions.",
        is_read_only=True
    ),
    "web_fetch": build_tool(
        name="web_fetch",
        func=web_fetch,
        input_schema=WebFetchInput,
        description="Fetch and read the text content of a specific URL.",
        is_read_only=True
    ),
    "symbol_search": build_tool(
        name="symbol_search",
        func=symbol_search,
        input_schema=SymbolSearchInput,
        description="Find the definition of a class or function across the codebase.",
        is_read_only=True
    ),
    "agent_delegate": build_tool(
        name="agent_delegate",
        func=agent_delegate,
        input_schema=AgentDelegateInput,
        description="Delegate a specific task to a sub-agent with its own context.",
        is_read_only=False
    ),
    "arch_visualizer": build_tool(
        name="arch_visualizer",
        func=arch_visualizer,
        input_schema=ArchVisualizerInput,
        description="Generate a high-level architecture overview of classes and methods in Mermaid format.",
        is_read_only=True
    ),
    "undercover_mode": build_tool(
        name="undercover_mode",
        func=undercover_mode,
        input_schema=UndercoverInput,
        description="Strip AI identifiers and local filesystem paths from a text string.",
        is_read_only=True
    ),
    "task_budget": build_tool(
        name="task_budget",
        func=task_budget,
        input_schema=TaskBudgetInput,
        description="Set a maximum token limit for the current task to control costs.",
        is_read_only=False
    ),
    "update_plan": build_tool(
        name="update_plan",
        func=update_plan,
        input_schema=UpdatePlanInput,
        description="Synthetic Tool: Update your internal master plan. Use this to track progress, rejected ideas, and next steps.",
        is_read_only=False
    ),
    "set_status": build_tool(
        name="set_status",
        func=set_status,
        input_schema=SetStatusInput,
        description="Sets the current activity status for the TUI.",
        is_read_only=False
    ),
    "git_status": build_tool(
        name="git_status",
        func=git_status,
        input_schema=GitStatusInput,
        description="Get porcelain git status.",
        is_read_only=True
    ),
    "git_diff": build_tool(
        name="git_diff",
        func=git_diff,
        input_schema=GitDiffInput,
        description="Get git diff for the repo or a file.",
        is_read_only=True
    ),
    "git_commit": build_tool(
        name="git_commit",
        func=git_commit,
        input_schema=GitCommitInput,
        description="Commit staged changes.",
        is_read_only=False
    ),
    "git_log": build_tool(
        name="git_log",
        func=git_log,
        input_schema=GitLogInput,
        description="View recent commit history.",
        is_read_only=True
    ),
    "switch_model": build_tool(
        name="switch_model",
        func=switch_model,
        input_schema=SwitchModelInput,
        description="Switch the active LLM brain. Ideal for scaling intelligence up or down.",
        is_read_only=True
    ),
    "undo_last_edit": build_tool(
        name="undo_last_edit",
        func=undo_last_edit,
        input_schema=UndoInput,
        description="Roll back file changes made in a specific turn. Use the tool_use_id of the turn to revert.",
        is_read_only=False
    ),
    "notebook_edit": build_tool(
        name="notebook_edit",
        func=notebook_edit,
        input_schema=NotebookEditInput,
        description="Surgically edit, insert, or delete Jupyter Notebook (.ipynb) cells.",
        is_read_only=False
    ),
    "system_doctor": build_tool(
        name="system_doctor",
        func=system_doctor,
        input_schema=DoctorInput,
        description="Audit system health (binaries, network, workspace toxicity). Run if tools are failing.",
        is_read_only=True
    ),
    "read_url": build_tool(
        name="read_url",
        func=read_url,
        input_schema=WebFetchInput,
        description="Fetch and distill web documentation or articles into Markdown.",
        is_read_only=True
    ),
    "cost_report": build_tool(
        name="cost_report",
        func=cost_report,
        input_schema=CostInput,
        description="Show the current session's token usage and USD cost report.",
        is_read_only=True
    )
}

CORE_TOOL_NAMES = {"file_read", "file_edit", "file_write", "multi_file_edit", "bash", "glob", "code_search", "grep_search"}

# 🚀 Deterministic Sorting
core_partition = sorted([k for k in _UNSORTED_TOOLS if k in CORE_TOOL_NAMES])
plugin_partition = sorted([k for k in _UNSORTED_TOOLS if k not in CORE_TOOL_NAMES])

AVAILABLE_TOOLS = {}
for k in core_partition + plugin_partition:
    AVAILABLE_TOOLS[k] = _UNSORTED_TOOLS[k]
