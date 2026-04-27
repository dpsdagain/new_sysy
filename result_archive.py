import os
import uuid
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ResultArchive:
    """
    Lossless Tool Result Budgeter (Layer 1).
    Offloads massive tool outputs to disk to preserve context precision.
    (Mirrors utils/toolResultStorage.ts)
    """
    def __init__(self, storage_dir: str = ".antigravity/tool_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def offload_if_large(self, tool_name: str, content: str, threshold: int = 8000) -> str:
        """
        If content exceeds threshold, writes to disk and returns a pointer.
        Otherwise returns original content.
        """
        if len(content) <= threshold:
            return content

        file_id = f"res_{uuid.uuid4().hex[:8]}.log"
        file_path = self.storage_dir / file_id
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            logger.info(f"Layer 1: Offloaded {len(content)} chars from {tool_name} to {file_id}")
            
            # Return a 'Pointer Message' that informs the model how to recover the data
            return (
                f"--- [LAYER 1: LOSSLESS TRUNCATION] ---\n"
                f"OUTPUT FROM {tool_name} WAS TOO LARGE ({len(content)} chars).\n"
                f"FULL RESULT SAVED TO: {file_path.as_posix()}\n"
                f"PREVIEW OF START:\n{content[:2000]}\n"
                f"...\n"
                f"PREVIEW OF END:\n{content[-2000:]}\n"
                f"---------------------------------------\n"
                f"TIP: If you need the full output, use 'read_file' or 'run_command' on the path above."
            )
        except Exception as e:
            logger.error(f"Layer 1 Archive Failed: {e}")
            return content[:threshold] + "... [TRUNCATION FAILED: DATA LOST]"

    def cleanup(self):
        """Clears the session storage directory."""
        import shutil
        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir)
            self.storage_dir.mkdir()
