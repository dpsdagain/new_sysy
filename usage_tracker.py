import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# 💳 Current Model Price Map (Price per 1M tokens)
# Ported from Anthropic constants
MODEL_PRICES = {
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-3-5-haiku-20241022": {"input": 0.25, "output": 1.25, "cache_read": 0.03, "cache_write": 0.30},
    "default": {"input": 1.00, "output": 1.00, "cache_read": 1.00, "cache_write": 1.00}
}

class UsageTracker:
    """
    Anthropic-Grade Token & Cost Auditor (F-18).
    Maintains a high-precision session ledger.
    """
    
    def __init__(self):
        self.totals = {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "cost_usd": 0.0
        }
        self.model_usage = {}

    def track(self, model: str, usage: Dict[str, int]):
        """
        Updates the ledger with a new turn's usage (cost-tracker.ts:278).
        """
        prices = MODEL_PRICES.get(model, MODEL_PRICES["default"])
        
        in_t = usage.get("input_tokens", 0)
        out_t = usage.get("output_tokens", 0)
        c_read = usage.get("cache_read_input_tokens", 0)
        c_write = usage.get("cache_creation_input_tokens", 0)
        
        # 🧾 Calculate Cost in USD
        cost = (
            (in_t * prices["input"]) + 
            (out_t * prices["output"]) + 
            (c_read * prices["cache_read"]) + 
            (c_write * prices["cache_write"])
        ) / 1_000_000.0

        # Update Session Totals
        self.totals["input"] += in_t
        self.totals["output"] += out_t
        self.totals["cache_read"] += c_read
        self.totals["cache_write"] += c_write
        self.totals["cost_usd"] += cost

        # Update Model-Specific Tally
        if model not in self.model_usage:
            self.model_usage[model] = {"in": 0, "out": 0, "cost": 0.0}
        
        self.model_usage[model]["in"] += in_t
        self.model_usage[model]["out"] += out_t
        self.model_usage[model]["cost"] += cost

    def get_report(self) -> str:
        """
        Generates a professional cost report (cost-tracker.ts:228).
        """
        report = [
            "--- SESSION COST REPORT ---",
            f"Total Cost:            ${self.totals['cost_usd']:.4f}",
            f"Total Tokens:          {self.totals['input'] + self.totals['output']:,}",
            f"  - Input:             {self.totals['input']:,}",
            f"  - Output:            {self.totals['output']:,}",
            f"  - Cache Savings:     {self.totals['cache_read']:,} read, {self.totals['cache_write']:,} created",
            "\nBreakdown by Model:"
        ]
        
        for model, data in self.model_usage.items():
            report.append(f"  {model[:20]}...: ${data['cost']:.4f} ({data['in']:,} in, {data['out']:,} out)")
            
        return "\n".join(report)
