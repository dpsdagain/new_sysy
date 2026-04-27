import os
import logging
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from config import (
    OPENROUTER_BASE_URL, OPENROUTER_API_KEY,
    OLLAMA_CLOUD_BASE_URL, OLLAMA_CLOUD_API_KEY,
    OLLAMA_BASE_URL, OLLAMA_PREFIX, OLLAMA_CLOUD_PREFIX,
    DEFAULT_MODEL
)

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Anthropic-Grade Model Management (F-45).
    Implements semantic aliases, plan/act separation, and provider abstraction.
    """
    
    # Semantic Mapping: High-level aliases to specific implementation IDs
    # Inspired by model.ts:457
    SEMANTIC_ALIASES = {
        "best": "ollama-cloud:gpt-oss:120b-cloud",
        "fast": "google/gemini-2.0-flash-001", # High speed, low latency
        "coder": "ollama-cloud:qwen3.6-coder:32b-cloud", # Logical reasoning
        "reasoning": "liquid/lfm-2.5-1.2b-thinking:free", # Synthetic thinking
        "haiku": "google/gemma-4-26b-a4b-it:free", # Fast fallback
        "opus": "ollama-cloud:gpt-oss:120b-cloud" # SOTA fallback
    }

    @staticmethod
    def resolve_alias(model_id: str) -> str:
        """Resolves semantic aliases (best, fast, etc.) to canonical IDs."""
        id_lower = model_id.lower()
        # Handle [1m] suffix for extended context
        has_1m = "[1m]" in id_lower
        clean_id = id_lower.replace("[1m]", "")
        
        resolved = ModelFactory.SEMANTIC_ALIASES.get(clean_id, model_id)
        return f"{resolved}[1m]" if has_1m else resolved

    @staticmethod
    def create_model(model_id: str = None, temperature: float = 0.0) -> Any:
        # 🚀 1. Semantic Alias Resolution
        target_id = ModelFactory.resolve_alias(model_id or DEFAULT_MODEL)
        
        # 🚀 2. Context Length Injection (Simulated via token limits)
        max_tokens = 4096
        if "[1m]" in target_id:
            logger.info("Engaging Extended Context Mode")
            max_tokens = 16384 # Scaled for RAG performance
            target_id = target_id.replace("[1m]", "")

        logger.info(f"ModelFactory: Instantiating {target_id}")

        try:
            # 3. Provider Instantiation
            if target_id.startswith(OLLAMA_CLOUD_PREFIX):
                return ChatOpenAI(
                    base_url=OLLAMA_CLOUD_BASE_URL,
                    api_key=OLLAMA_CLOUD_API_KEY or "not-needed",
                    model=target_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    streaming=True
                )

            if target_id.startswith(OLLAMA_PREFIX):
                clean_name = target_id.replace(OLLAMA_PREFIX, "")
                return ChatOllama(
                    base_url=OLLAMA_BASE_URL,
                    model=clean_name,
                    temperature=temperature,
                    num_predict=max_tokens,
                    streaming=True
                )

            return ChatOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
                model=target_id,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=True
            )
            
        except Exception as e:
            logger.error(f"ModelFactory failover engaging: {e}")
            # Anthropic Pattern: Silent Peer Failover
            return ChatOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
                model="google/gemini-2.0-flash-001",
                temperature=0.0
            )

    @staticmethod
    def list_available_categories():
        """Helper for the UI to show categorized models from config."""
        from config import CLOUDROUTER_MODELS
        return CLOUDROUTER_MODELS
