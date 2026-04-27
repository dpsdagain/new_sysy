import os
import shutil
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class HistoryManager:
    """
    Anthropic-Grade File History & Checkout System (F-28).
    Ported from utils/fileHistory.ts.
    
    Maintains a versioned backup of files modified during the session.
    """
    
    def __init__(self, session_dir: str):
        self.history_dir = os.path.join(session_dir, "file_history")
        os.makedirs(self.history_dir, exist_ok=True)
        # message_id -> { file_path -> backup_path }
        self.snapshots: Dict[str, Dict[str, str]] = {}

    def _get_backup_name(self, file_path: str, version: int) -> str:
        """Ported hash-based naming (fileHistory.ts:725)."""
        name_hash = hashlib.sha256(file_path.encode()).hexdigest()[:16]
        return f"{name_hash}_v{version}"

    def track_edit(self, file_path: str, message_id: str):
        """
        Creates a 'Pre-Edit' backup of a file (fileHistory.ts:86).
        Must be called BEFORE the tool modifies the file.
        """
        if not os.path.exists(file_path):
            # File is being created, backup is 'None' or non-existent
            return

        if message_id not in self.snapshots:
            self.snapshots[message_id] = {}

        if file_path in self.snapshots[message_id]:
            return # Already tracked for this turn

        version = len(self.snapshots) + 1
        backup_name = self._get_backup_name(file_path, version)
        backup_path = os.path.join(self.history_dir, backup_name)

        try:
            shutil.copy2(file_path, backup_path)
            self.snapshots[message_id][file_path] = backup_path
            logger.info(f"Checkpoint created for {file_path} (ID: {message_id})")
        except Exception as e:
            logger.error(f"Failed to create history checkpoint: {e}")

    def rollback(self, message_id: str) -> List[str]:
        """
        Rewinds the filesystem to the state before message_id (fileHistory.ts:347).
        """
        if message_id not in self.snapshots:
            return []

        reverted_files = []
        for file_path, backup_path in self.snapshots[message_id].items():
            try:
                shutil.copy2(backup_path, file_path)
                reverted_files.append(file_path)
                logger.info(f"Rollback: Restored {file_path}")
            except Exception as e:
                logger.error(f"Rollback failed for {file_path}: {e}")
        
        return reverted_files
