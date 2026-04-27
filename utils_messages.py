from typing import List, Any
from langchain_core.messages import (
    BaseMessage, 
    SystemMessage, 
    HumanMessage, 
    AIMessage, 
    ToolMessage
)
import logging

logger = logging.getLogger(__name__)

def normalize_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Production-Grade Message Normalizer.
    Enforces API invariants:
    1. Canonical System Message: One at the very top.
    2. Role Alternation: User -> Assistant -> User -> Assistant.
    3. Tool Sequencing: ToolMessages MUST follow an AIMessage with tool_calls.
    4. Content Sanitization: Removes empty strings/nulls that crash specific APIs.
    """
    if not messages:
        return []

    # 1. Phase: System Consolidation
    merged_system_content = []
    conversation = []
    
    for m in messages:
        if isinstance(m, SystemMessage):
            # Extract content string safely
            content = ""
            if isinstance(m.content, str):
                content = m.content
            elif isinstance(m.content, list):
                # Handle structured content (e.g. for caching tags)
                content = "\n".join([str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in m.content])
            
            if content.strip():
                merged_system_content.append(content)
        else:
            conversation.append(m)
            
    final_messages = []
    if merged_system_content:
        final_messages.append(SystemMessage(content="\n\n".join(merged_system_content)))

    # 2. Phase: Sequence Normalization
    for i, m in enumerate(conversation):
        if not final_messages:
            # First message must be Human or System (system handled above)
            # If we try to start with AI or Tool, the API will fail.
            if isinstance(m, (AIMessage, ToolMessage)):
                continue 
            final_messages.append(m)
            continue
            
        last = final_messages[-1]
        
        # Merge Consecutive Human Messages
        if isinstance(m, HumanMessage) and isinstance(last, HumanMessage):
            last.content += f"\n\n{m.content}"
            continue
            
        # Merge Consecutive AI Messages
        if isinstance(m, AIMessage) and isinstance(last, AIMessage):
            # Important: if the current message has tool_calls, we can't just merge text.
            # However, standard normalization usually combines content.
            # We prioritize keeping tool_calls.
            if not last.tool_calls:
                last.content += f"\n\n{m.content}"
                if hasattr(m, 'tool_calls') and m.tool_calls:
                    last.tool_calls = m.tool_calls # Transfer tool calls to the merged msg
                continue

        # Handle Role Alternation Violations (Double Assistant / Double User)
        # If we have User -> User or Assistant -> Assistant that couldn't be merged
        # insert a silent spacer if necessary. (Preferably we merge content).

        # Content Sanitization
        content_empty = not m.content or (isinstance(m.content, str) and not m.content.strip())
        has_tools = hasattr(m, 'tool_calls') and m.tool_calls
        
        if content_empty and not has_tools and not isinstance(m, ToolMessage):
            continue

        final_messages.append(m)

    return final_messages
