import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# 📐 Anthropic-Parity Context Limits (utils/context.ts:149)
MODEL_LIMITS = {
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "default": 128_000
}

class ContextEstimator:
    """
    Anthropic-Grade Pre-Flight Estimator (F-32).
    Predicts context overflows before they reach the API.
    """
    
    @staticmethod
    def estimate_tokens(messages: List[Any]) -> int:
        """
        Performs a 'Rough-Cut' token estimation (chars / 3.8).
        Anthropic uses a similar local heuristic for fast decisions.
        """
        total_chars = 0
        for msg in messages:
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Handles multi-modal/tool blocks
                for block in content:
                    total_chars += len(str(block))
        
        # Heuristic: 1 token approx 4 chars in English
        return int(total_chars / 3.8)

    @staticmethod
    def check_flight_safety(model: str, messages: List[Any]) -> Dict[str, Any]:
        """
        Audits context against model limits (context.ts:118).
        """
        limit = MODEL_LIMITS.get(model, MODEL_LIMITS["default"])
        estimated = ContextEstimator.estimate_tokens(messages)
        usage_pct = (estimated / limit) * 100
        
        status = "SAFE"
        if usage_pct > 95:
            status = "CRITICAL" # Must compact
        elif usage_pct > 85:
            status = "WARNING" # Warn user
            
        return {
            "estimated_tokens": estimated,
            "limit": limit,
            "usage_pct": round(usage_pct, 2),
            "status": status
        }
