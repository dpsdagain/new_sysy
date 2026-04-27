import os
import logging

logger = logging.getLogger(__name__)

from config import TEMP_OUTPUT_DIR

class ResultManager:
    def __init__(self, max_chars: int = 5000):
        self.max_chars = max_chars
        self.temp_dir = TEMP_OUTPUT_DIR
        os.makedirs(self.temp_dir, exist_ok=True)

    def process_result(self, tool_name: str, result: str) -> str:
        """Truncate large results and save them to a file."""
        if len(result) <= self.max_chars:
            return result
        
        # Save full result to a file
        import hashlib
        result_hash = hashlib.md5(result.encode()).hexdigest()
        file_name = f"{tool_name}_{result_hash}.txt"
        file_path = os.path.join(self.temp_dir, file_name)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(result)
            
        summary = (
            f"Result from {tool_name} is too large ({len(result)} characters).\n"
            f"Showing first {self.max_chars//2} and last {self.max_chars//2} characters:\n\n"
            f"{result[:self.max_chars//2]}\n\n... [TRUNCATED] ...\n\n{result[-self.max_chars//2:]}\n\n"
            f"Full output saved to: {os.path.abspath(file_path)}\n"
            f"You can use 'file_read' on this path if you need the full content."
        )
        return summary
