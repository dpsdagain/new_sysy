import logging
from typing import List, Any, Dict, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from result_archive import ResultArchive
from config import OLLAMA_CLOUD_BASE_URL, OLLAMA_CLOUD_API_KEY as API_KEY
import subprocess

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Anthropic-Grade 5-Layer Context Fortress (F-23).
    Tiered reduction preserves technical precision while bounding token cost.
    """
    def __init__(self, max_context_tokens: int = 128000, compression_threshold: float = 0.9):
        self.max_context_tokens = max_context_tokens
        self.compression_threshold = compression_threshold
        self.result_archive = ResultArchive() # Layer 1 Instance
        
        # Layer 5: Emergency Summarizer
        self.summarizer_llm = ChatOpenAI(
            base_url=OLLAMA_CLOUD_BASE_URL,
            api_key=API_KEY,
            model="ollama-cloud:gemma2:9b-cloud"
        )

    def compact(self, messages: List[BaseMessage], plan: str = "") -> List[BaseMessage]:
        """
        The 5-Layer Execution Pipeline (query.ts:150).
        """
        # Layer 1: Lossless Tool Result Budgeting (Offloading to disk)
        messages = self._apply_tool_result_budget(messages)

        # Layer 2: Microcompact (Whitespacing & Metadata Stripping)
        messages = self._microcompact(messages)

        current_tokens = self.estimate_tokens(messages)
        soft_limit = int(self.max_context_tokens * self.compression_threshold)

        if current_tokens < soft_limit:
            return messages

        logger.info(f"Context Pressure ({current_tokens} tokens). Escalating to Tier 3-5.")

        # Layer 3: Context Collapse (Archiving Repetitive Loops)
        messages = self._context_collapse(messages)

        # Layer 4: History Snipping (Dropping Ancient Mid-Turns)
        if self.estimate_tokens(messages) > soft_limit:
            messages = self._snip_history(messages)

        # Layer 5: Autocompact (Emergency Summarization)
        if self.estimate_tokens(messages) > soft_limit:
            messages = self._autocompact(messages, plan)

        return messages

    def _apply_tool_result_budget(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Layer 1: Lossless Budgeting (toolResultStorage.ts:99) - Archive to disk."""
        capped = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and len(str(msg.content)) > 8000:
                # 🔄 LOSSLESS UPGRADE: Write to disk instead of truncating
                msg.content = self.result_archive.offload_if_large("tool_result", str(msg.content))
            capped.append(msg)
        return capped

    def _microcompact(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Layer 2: Microcompact (messages.ts:54, 884) - Strip syntactic noise."""
        for msg in messages:
            if isinstance(msg.content, str):
                msg.content = msg.content.strip()
                while "\n\n\n" in msg.content:
                    msg.content = msg.content.replace("\n\n\n", "\n\n")
        return messages

    def _context_collapse(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Layer 3: Collapse (contextCollapse:18) - Merge repetitive log chains."""
        if len(messages) < 10: return messages
        collapsed = [messages[0]]
        
        i = 1
        while i < len(messages):
            if i + 2 < len(messages) and isinstance(messages[i], AIMessage) and isinstance(messages[i + 1], ToolMessage):
                pattern_count = 0
                while i + (pattern_count + 1) * 2 < len(messages):
                    curr_ai = messages[i + pattern_count * 2]
                    next_ai = messages[i + (pattern_count + 1) * 2]
                    if hasattr(curr_ai, 'tool_calls') and curr_ai.tool_calls and \
                       hasattr(next_ai, 'tool_calls') and next_ai.tool_calls:
                        if curr_ai.tool_calls[0].get('name') == next_ai.tool_calls[0].get('name'):
                            pattern_count += 1
                        else: break
                    else: break
                
                if pattern_count > 3:
                    collapsed.append(AIMessage(content=f"[COLLAPSED_SEQUENCE: {pattern_count} repeated operations]"))
                    i += (pattern_count * 2)
                    continue
            
            collapsed.append(messages[i])
            i += 1
        return collapsed

    def _snip_history(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Layer 4: Snipping (snipCompact.ts:115)."""
        if len(messages) < 12: return messages
        snipped = [messages[0], messages[1]]
        snipped.append(SystemMessage(content="... [EARLIER MESSAGES SNIPPED] ..."))
        snipped.extend(messages[-6:])
        return snipped

    def _autocompact(self, messages: List[BaseMessage], plan: str) -> List[BaseMessage]:
        """Layer 5: Autocompact with Project State Restoration (context.ts:149)."""
        system_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
        
        # 1. Distill History
        tail = messages[-4:]
        middle = messages[1:-4]
        summary = self._generate_summary(middle)
        
        # 2. Re-inject Project Context (Git + Skills)
        git_status = "Unknown"
        try:
            git_status = subprocess.check_output(["git", "status", "--short"], stderr=subprocess.STDOUT).decode()
        except: pass
        
        discovered_skills = set()
        for m in messages:
            if hasattr(m, 'tool_calls') and m.tool_calls:
                for tc in m.tool_calls: discovered_skills.add(tc.get('name'))

        final = []
        if system_msg: final.append(system_msg)
        
        # Re-inject critical state after the summary to reset the model's awareness
        restoration_msg = (
            f"[TECHNICAL_SUMMARY]\n{summary}\n\n"
            f"[PROJECT_STATE_RESTORATION]\n"
            f"GIT_STATUS (Uncommitted changes):\n{git_status}\n"
            f"DISCOVERED_TOOLS: {', '.join(discovered_skills)}\n"
            f"ACTIVE_PLAN: {plan}\n"
            f"[/PROJECT_STATE_RESTORATION]"
        )
        final.append(SystemMessage(content=restoration_msg))
        final.extend(tail)
        return final

    def estimate_tokens(self, messages: List[BaseMessage]) -> int:
        """Rough-cut character-based estimation."""
        total_chars = 0
        for msg in messages:
            total_chars += len(str(msg.content))
        return total_chars // 4

    def _generate_summary(self, messages: List[BaseMessage]) -> str:
        """Technical distillation."""
        prompt = (
            "Summarize the technical progress, code changes, and solved errors from these turns.\n"
            "Be extremely brief. Bullet points only. Zero conversational filler."
        )
        history = "\n".join([f"{m.type}: {str(m.content)[:500]}" for m in messages])
        try:
            res = self.summarizer_llm.invoke([SystemMessage(content=prompt), HumanMessage(content=history)])
            return res.content
        except:
            return "History summarized."
