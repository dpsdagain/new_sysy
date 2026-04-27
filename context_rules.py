import os
import re
from pathlib import Path
from typing import List, Set, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ContextRules:
    """
    Anthropic-Grade Context Discovery Engine (F-16).
    Ported logic from utils/claudemd.ts.
    
    Implements hierarchical discovery of CLAUDE.md, .claude/rules/*.md,
    and HTML comment stripping to optimize context tokens.
    """
    
    TEXT_EXTENSIONS = {'.md', '.txt', '.text', '.py', '.js', '.sh'}
    MAX_MEMORY_CHARS = 40000

    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root)
        self.user_home = Path.home()

    def strip_html_comments(self, content: str) -> str:
        """Ported from claudemd.ts:292 - Removes comment blocks to save tokens."""
        return re.sub(r'<!--[\s\S]*?-->', '', content)

    def _discover_rule_files(self, start_dir: Path) -> List[Path]:
        """
        Recursive discovery of rules.
        Includes CLAUDE.md and everything in .claude/rules/
        """
        rule_files = []
        
        # 1. Check for CLAUDE.md in current dir
        claude_md = start_dir / "CLAUDE.md"
        if claude_md.exists():
            rule_files.append(claude_md)
            
        # 2. Check for .claude/rules/ directory
        rules_dir = start_dir / ".claude" / "rules"
        if rules_dir.is_dir():
            # Ported logic: Load all .md files in the rules dir
            for rule_file in rules_dir.glob("*.md"):
                rule_files.append(rule_file)
                
        return rule_files

    def get_instructions(self, current_dir: str = None) -> str:
        """
        Builds the consolidated instruction string following the priority hierarchy:
        Global/ETC -> User (~/.claude) -> Project Root -> Local Subdirectories
        """
        curr = Path(current_dir or os.getcwd())
        all_rules: List[str] = []
        processed_paths: Set[Path] = set()

        # 🚀 1. Load User Global Rules (~/.claude/CLAUDE.md)
        global_rule = self.user_home / ".claude" / "CLAUDE.md"
        if global_rule.exists():
            all_rules.append(f"### GLOBAL RULES (~/.claude)\n{self._read_and_strip(global_rule)}")
            processed_paths.add(global_rule)

        # 🚀 2. Hierarchical Discovery (Parent -> Child)
        # We traverse up to the workspace root, then reverse to maintain priority
        traversal = []
        temp_curr = curr
        while temp_curr.exists() and temp_curr != temp_curr.parent:
            traversal.append(temp_curr)
            if temp_curr == self.workspace_root:
                break
            temp_curr = temp_curr.parent
            
        # Reverse traversal so children's rules have higher priority (added later)
        for path in reversed(traversal):
            files = self._discover_rule_files(path)
            for f in files:
                if f not in processed_paths:
                    content = f"### RULES FROM {f.relative_to(self.workspace_root)}\n{self._read_and_strip(f)}"
                    all_rules.append(content)
                    processed_paths.add(f)

        if not all_rules:
            return ""

        consolidated = "\n\n".join(all_rules)
        # Truncate if extreme (Failsafe like MAX_MEMORY_CHARACTER_COUNT)
        if len(consolidated) > self.MAX_MEMORY_CHARS:
            logger.warning("Context instructions truncated (Too large)")
            consolidated = consolidated[:self.MAX_MEMORY_CHARS] + "\n... (truncated)"
            
        return consolidated

    def _read_and_strip(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return self.strip_html_comments(content).strip()
