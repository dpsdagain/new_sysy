"""
config.py - Centralized configuration for the RAG Knowledge Base.
Updated for April 2026 OpenRouter Fleet.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_CLOUD_API_KEY = os.getenv("OLLAMA_CLOUD_API_KEY", "")
OLLAMA_CLOUD_BASE_URL = os.getenv("OLLAMA_CLOUD_BASE_URL", "https://api.ollama.com/v1")

GEMINI_MODEL = "google/gemini-2.0-flash-001"
QWEN_MODEL = "qwen/qwen3.6-plus:free"

# CLOUDROUTER MODELS - Verified 2026 Free Tier
CLOUDROUTER_MODELS = {
    "Preferred Fleet (2026 Free)": {
        "Gemma 4 (SOTA Free)": "google/gemma-4-31b-it:free",
        "Qwen 3 Coder (Free)": "qwen/qwen3-coder:free",
        "LFM 2.5 (Thinking Free)": "liquid/lfm-2.5-1.2b-thinking:free",
        "Llama 3.3 70B (Free)": "meta-llama/llama-3.3-70b-instruct:free",
        "Gemini 2.0 Flash": "google/gemini-2.0-flash-001",
        "Qwen 3.6 Plus (Free)": "qwen/qwen3.6-plus:free",
    },
    "Ollama Cloud Elite (2026)": {
        "MiniMax M2.5 (Agentic)": "ollama-cloud:minimax-m2.5:cloud",
        "Qwen 3.6 Coder (Logic)": "ollama-cloud:qwen3.6-coder:32b-cloud",
        "DeepSeek R1 (Finance)": "ollama-cloud:deepseek-r1:70b-cloud",
        "Llama 4 Scout (10M)": "ollama-cloud:llama4:scout-cloud",
        "DeepSeek V3.2 (Prose)": "ollama-cloud:deepseek-v3.2:405b-cloud",
        "Qwen 3.6 (Roleplay)": "ollama-cloud:qwen3.6:27b-cloud",
        "Gemma 4 (Daily)": "ollama-cloud:gemma4:31b-cloud",
    },
    "Google Elite (Free Tier)": {
        "Gemma 4 (31B)": "google/gemma-4-31b-it:free",
        "Gemma 4 (Light)": "google/gemma-4-26b-a4b-it:free",
        "Gemma 3 (27B)": "google/gemma-3-27b-it:free",
    },
    "Coding and Logic (Free)": {
        "Qwen 3 Coder": "qwen/qwen3-coder:free",
        "GPT-OSS (120B)": "openai/gpt-oss-120b:free",
    },
    "Deep Reasoning (Free)": {
        "LFM 2.5 Thinking": "liquid/lfm-2.5-1.2b-thinking:free",
        "Gemma 4 Reasoning": "google/gemma-4-31b-it:free",
    },
    "Ollama Cloud (New)": {
        "Gemma 4 Cloud (31B)": "ollama-cloud:gemma4:31b-cloud",
        "GPT-OSS Cloud (120B)": "ollama-cloud:gpt-oss:120b-cloud",
    }
}

# Default model updated to GPT-OSS Cloud (120B) as requested
DEFAULT_MODEL = "ollama-cloud:gpt-oss:120b-cloud"

ANTHROPIC_CACHE_BETA_HEADER = "prompt-caching-2024-07-31"
ENABLE_PROMPT_CACHING = True
CACHE_THRESHOLD_TOKENS = 1028
MAX_CACHE_CHECKPOINTS = 4

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODELS = ["llama3.1", "llama3.2:1b", "qwen2.5:3b"]
OLLAMA_PREFIX = "ollama:"
OLLAMA_CLOUD_PREFIX = "ollama-cloud:"
AGENT_ROUTER_MODEL = "ollama-cloud:llama4:scout-cloud"

LLM_TEMPERATURE = 0.0
MAX_TOKENS = 1024

# ═══════════════════════════════════════════════════════════════════════════
#  PATH CONFIGURATION (Universal Home)
# ═══════════════════════════════════════════════════════════════════════════

# Dynamically resolve APP_HOME based on this file's location
APP_HOME = Path(__file__).parent.resolve()
CHROMA_DB_DIR = str(APP_HOME / "chroma_db") 
SESSION_DIR = str(APP_HOME / "sessions")
TEMP_OUTPUT_DIR = str(APP_HOME / "temp_outputs")


# 🚀 The Workspace Boundary: Current directory where the agent was started
WORKSPACE_ROOT = Path(os.getcwd()).resolve()

# Ensure global directories exist
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
CODE_CHUNK_SIZE = 1000
PDF_CHUNK_SIZE = 1500
ZERO_CHUNK_THRESHOLD = 10000
MAX_ZERO_CHUNK_CHARS = 12000   # ~3000 tokens — cap for zero-chunks surfaced via retrieval

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
RETRIEVER_K = 12
RETRIEVER_FETCH_K = 30

MIN_PREV_QUERY_LENGTH = 15
MIN_CURRENT_QUERY_LENGTH = 10

SEMANTIC_CACHE_THRESHOLD = 0.98
PINNED_RELEVANCE_THRESHOLD = 0.40
STICKY_PINNED_CONTEXT = True
TRUST_NATIVE_CACHE = True

GHOST_HISTORY_WINDOW = 10
GHOST_HISTORY_MAX = 10
AI_RESPONSE_MAX_CHARS = 800
GHOST_AI_CHARS = 200
MAX_HISTORY_TOKENS = 2000

SENTINEL_INTERVAL = 3
SENTINEL_MAX_TOKENS = 500
SENTINEL_TOKEN_THRESHOLD = 1500

PROVIDER_CACHE_PROFILES = {
    "claude":    (4, 1024),
    "gemini":    (8, 1028),
    "gemma":     (8, 1028),
    "deepseek":  (4, 1024),
    "qwen":      (4, 1024),
    "nemotron":  (4, 1024),
    "glm":       (4, 1024),
    "gpt-5":     (4, 1024),
    "reka":      (4, 1024),
    "mistral":   (4, 1024),
    "gpt-oss":   (4, 1024),
}

USE_RERANKER = True
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 8
RERANK_CANDIDATES = 30

ENABLE_AUTO_SPECIALIST = True
SPECIALIST_MAPPING = {
    "CODE": "ollama-cloud:qwen3.6-coder:32b-cloud",
    "REASONING": "ollama-cloud:gpt-oss:120b-cloud",
    "VISION": "google/gemma-4-31b-it:free",
    "GENERAL": "ollama-cloud:gemma4:31b-cloud"
}

ENABLE_HYBRID_SEARCH = True
BM25_WEIGHT = 0.65
VECTOR_WEIGHT = 0.35

CODE_EXTENSIONS = [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", ".go", ".rs", ".cs", ".v", ".sv", ".vue", ".svelte", ".html", ".css", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh", ".bash", ".bat", ".ps1", ".sql"]
EXCLUDED_FILE_PATTERNS = ["*-lock.json", "*.lock", "*.csv", "*.log", "*.min.js", "*.min.css", "*.map", "*.svg", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.webp", "*node_modules*", "*venv*", "*__pycache__*", "*.pyc", "*chroma_db*", "*.env", "*archive*"]
