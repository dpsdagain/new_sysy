import os
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class PermissionDecision:
    def __init__(self, behavior: str, reason: str = ""):
        self.behavior = behavior # 'allow', 'deny', 'ask'
        self.reason = reason

class PermissionManager:
    """
    Anthropic-Grade Security Guardian (F-14).
    Ported from permissions.ts.
    
    Implements Three-Phase Evaluation:
    1. Static Blocklist (Regex)
    2. Fast-Path (Safe Tools)
    3. Guardian Check (AI Approval)
    """
    
    PROTECTED_PATTERNS = [
        r"\.ssh/", r"\.env", r"\.aws/", r"/etc/", r"/var/log/",
        r"\.netrc", r"id_rsa", r"id_ed25519"
    ]
    
    SAFE_TOOLS = ["git_status", "git_root", "list_dir", "view_file", "search_web", "symbol_search"]

    def __init__(self, mode: str = "ASK"):
        self.mode = mode # ASK, AUTO, DONT_ASK
        self.approved_tools = set() # Persistence within session

    def check_permission(self, tool_name: str, args: Dict[str, Any]) -> PermissionDecision:
        """Ported logic from hasPermissionsToUseTool in permissions.ts."""
        
        # 🛡️ Phase 1: High-Priority Blocklist (Regex)
        arg_str = str(args).lower()
        for pattern in self.PROTECTED_PATTERNS:
            if re.search(pattern, arg_str):
                logger.warning(f"Security Alert: Blocked access to sensitive pattern {pattern}")
                return PermissionDecision("deny", f"Access to sensitive path '{pattern}' is forbidden.")

        # 🚀 Phase 2: Fast-Path for Safe Tools
        if tool_name in self.SAFE_TOOLS:
            return PermissionDecision("allow", "Safe tool auto-approved.")

        # 🚀 Phase 3: Mode-Specific Logic
        if self.mode == "AUTO":
            return PermissionDecision("allow", "Auto-mode global approval.")
            
        if self.mode == "DONT_ASK":
            return PermissionDecision("deny", "Agent is in non-interactive mode. No permissions granted.")

        # Default to asking the human
        return PermissionDecision("ask", "Interactive approval required.")

    def update_mode(self, new_mode: str):
        self.mode = new_mode
