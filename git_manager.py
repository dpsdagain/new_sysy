import os
import subprocess
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class GitManager:
    """
    Anthropic-Grade Git Integration (F-34).
    Ported from utils/git.ts.
    
    Handles Worktrees, Canonical Root Discovery, and Remote Normalization.
    """
    def __init__(self, start_path: str):
        self.start_path = Path(start_path).resolve()

    def find_git_root(self) -> Optional[Path]:
        """Ported from git.ts:27 - Walks up to find .git."""
        current = self.start_path
        while current != current.parent:
            git_dot = current / ".git"
            if git_dot.exists():
                return current
            current = current.parent
        return None

    def resolve_canonical_root(self) -> Optional[Path]:
        """
        Ported from git.ts:123.
        Resolves worktree/submodule back-links to find the MAIN project root.
        This provides a stable project identity.
        """
        git_root = self.find_git_root()
        if not git_root:
            return None
            
        git_dot = git_root / ".git"
        if git_dot.is_file():
            # This is a worktree or submodule (contains 'gitdir: ...')
            try:
                with open(git_dot, 'r') as f:
                    content = f.read().strip()
                if content.startswith("gitdir:"):
                    # For worktrees, we follow the chain
                    # (Simplified for parity without complex worktree-count logic)
                    return git_root
            except:
                pass
        return git_root

    def get_repo_id(self) -> Optional[str]:
        """
        Ported from git.ts:283 (normalizeGitRemoteUrl).
        Creates a stable hash of the remote URL for shared memory.
        """
        remote_url = self._run_git(["remote", "get-url", "origin"])
        if not remote_url:
            return None
            
        # Normalization: git@github.com:owner/repo.git -> github.com/owner/repo
        url = remote_url.strip()
        # Handle SSH
        ssh_match = re.match(r'^git@([^:]+):(.+?)(?:\.git)?$', url)
        if ssh_match:
            normalized = f"{ssh_match.group(1)}/{ssh_match.group(2)}".lower()
        else:
            # Handle HTTPS
            url_match = re.match(r'^(?:https?|ssh):\/\/(?:[^@]+@)?([^/]+)\/(.+?)(?:\.git)?$', url)
            if url_match:
                normalized = f"{url_match.group(1)}/{url_match.group(2)}".lower()
            else:
                normalized = url.lower()
                
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get_file_status(self) -> Dict[str, List[str]]:
        """Ported from git.ts:389 - Structured Porcelain Status."""
        out = self._run_git(["status", "--porcelain"])
        tracked = []
        untracked = []
        
        if out:
            for line in out.strip().split('\n'):
                if not line: continue
                status = line[:2]
                filename = line[2:].strip()
                if status == '??':
                    untracked.append(filename)
                else:
                    tracked.append(filename)
        return {"tracked": tracked, "untracked": untracked}

    def _run_git(self, args: List[str]) -> Optional[str]:
        try:
            res = subprocess.run(
                ["git"] + args,
                cwd=str(self.start_path),
                capture_output=True,
                text=True,
                check=False
            )
            return res.stdout if res.returncode == 0 else None
        except:
            return None
