import logging
from typing import List, Any, Tuple
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from config import (
    OPENROUTER_API_KEY, 
    OPENROUTER_BASE_URL, 
    DEFAULT_MODEL, 
    MAX_TOKENS,
    OLLAMA_CLOUD_API_KEY,
    OLLAMA_CLOUD_BASE_URL
)

logger = logging.getLogger(__name__)

class ContextManager:
    def __init__(self, max_context_tokens: int = 40000, compression_threshold: float = 0.8):
        self.max_context_tokens = max_context_tokens
        self.compression_threshold = compression_threshold
        # Summarizer model: Using Gemma 4 Cloud (31B) for extreme speed and accuracy
        self.summarizer_llm = ChatOpenAI(
            base_url=OLLAMA_CLOUD_BASE_URL,
            api_key=OLLAMA_CLOUD_API_KEY,
            model="gemma4:31b-cloud", 
            temperature=0,
            max_tokens=1000
        )

    def estimate_tokens(self, messages: List[BaseMessage]) -> int:
        """Rough estimation of tokens (characters / 3)."""
        total_chars = 0
        for msg in messages:
            total_chars += len(str(msg.content))
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                total_chars += len(str(msg.tool_calls))
        return total_chars // 3

    def compact(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Apply compaction strategies if the context is too large."""
        current_tokens = self.estimate_tokens(messages)
        limit = int(self.max_context_tokens * self.compression_threshold)

        if current_tokens < limit:
            return messages

        logger.info(f"Context pressure detected ({current_tokens} tokens). Compacting...")
        
        # Strategy 1: Summarize stale turns
        # Keep System Prompt (index 0) and last 6 messages
        # Summarize everything in between
        if len(messages) > 10:
            system_msg = messages[0]
            stale_messages = messages[1:-6]
            recent_messages = messages[-6:]
            
            summary = self._summarize(stale_messages)
            summary_msg = SystemMessage(content=f"--- PREVIOUS CONVERSATION SUMMARY ---\n{summary}\n--- END SUMMARY ---")
            
            return [system_msg, summary_msg] + recent_messages
        
        # Strategy 2: Simple Snip (Fallback)
        return [messages[0]] + messages[-8:]

    def _summarize(self, messages: List[BaseMessage]) -> str:
        """Use LLM to condense a list of messages."""
        logger.info("Summarizing stale context...")
        prompt = "Summarize the following conversation between a User and an AI Assistant. Focus on the core goals, the files edited, and the current status. Be concise.\n\n"
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            if isinstance(msg, ToolMessage):
                role = f"Tool ({msg.name})"
            
            # CRITICAL FIX: Handle Anthropic-style list content (cache blocks)
            content = msg.content
            if isinstance(content, list):
                # Extract text from the first text block
                text = next((item["text"] for item in content if isinstance(item, dict) and "text" in item), str(content))
            else:
                text = str(content)
                
            prompt += f"{role}: {text[:500]}...\n" # Truncate large tool outputs for summary

        try:
            response = self.summarizer_llm.invoke([HumanMessage(content=prompt)])
            return str(response.content)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "[Error: Summarization failed, context was snipped instead.]"
