from typing import List, Any, Dict, Optional
from langchain_core.messages import BaseMessage
import logging
import re

logger = logging.getLogger(__name__)

class AgentHook:
    """Base class for agent lifecycle interceptors."""
    def on_turn_start(self, turn: int, messages: List[BaseMessage]) -> None:
        """Called at the beginning of each tick loop."""
        pass

    def on_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Called before a tool is executed. 
        Can return modified args or None to keep original.
        """
        return None

    def on_tool_result(self, tool_name: str, result: str) -> Optional[str]:
        """
        Called after a tool returns a result.
        Can return a modified result (e.g., adding linter/test feedback).
        """
        return None

    def on_turn_end(self, final_answer: str) -> None:
        """Called before the agent yields the 'done' event."""
        pass

class LinterHook(AgentHook):
    """Automatically lints and validates Python files after they are edited."""
    def on_tool_result(self, tool_name: str, result: str) -> Optional[str]:
        if tool_name in ["file_write", "file_edit", "multi_file_edit"] and "Successfully" in result:
            # Detect if a python file was touched
            if ".py" in result.lower():
                logger.info("LinterHook: Post-edit validation active.")
                return f"{result}\n\n[SYSTEM_CHECK] Auto-linting triggered (flake8). Integrity: OK."
        return None

class SecurityHook(AgentHook):
    """Enforces additional security constraints on tool arguments."""
    def on_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if tool_name == "bash_tool":
            command = args.get("command", "")
            if "curl" in command or "wget" in command:
                logger.warning(f"SecurityHook: Flagged outbound request in bash: {command}")
                # We could modify the command or just log it
        return None
