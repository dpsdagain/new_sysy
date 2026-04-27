import logging
import json
import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, 
    AIMessage, 
    SystemMessage, 
    ToolMessage,
    message_to_dict,
    messages_from_dict
)
from config import (
    OPENROUTER_API_KEY, 
    OPENROUTER_BASE_URL, 
    DEFAULT_MODEL, 
    LLM_TEMPERATURE, 
    MAX_TOKENS,
    OLLAMA_PREFIX,
    OLLAMA_BASE_URL,
    OLLAMA_CLOUD_PREFIX,
    OLLAMA_CLOUD_API_KEY,
    OLLAMA_CLOUD_BASE_URL,
    SESSION_DIR,
    ENABLE_PROMPT_CACHING,
    ANTHROPIC_CACHE_BETA_HEADER
)
from tools import AVAILABLE_TOOLS
from langchain_community.chat_models import ChatOllama
from permissions import PermissionManager
from result_manager import ResultManager
from context_manager import ContextManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an autonomous AI software engineer operating on a Windows (win32) system. You have access to a set of tools to research, read, and edit code, as well as execute shell commands.

Your workflow:
1. Research: Use 'code_search' to find relevant code snippets.
2. Analyze: Use 'file_read' to examine the full content of relevant files.
3. Act: Use 'file_write' to create new files or 'file_edit' to make surgical changes.
4. Verify: Use 'bash' to run tests/commands.

Shell Environment (Windows):
- Use 'dir' instead of 'ls' if 'ls' is not available.
- Use 'type' instead of 'cat' if 'cat' is not available.
- For creating files, PREFER the 'file_write' tool over bash redirects.

Constraints:
- Always check your changes by running tests if available.
- Be surgical with 'file_edit'. Only replace the minimal necessary string.
- If you get stuck, explain why and ask for clarification.
- Do not assume a file exists without searching for it first.

You are operating in a local environment. Be careful with destructive commands."""

class QueryEngine:
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = LLM_TEMPERATURE, permission_mode: str = "ASK", delegation_depth: int = 0):
        self.delegation_depth = delegation_depth
        temp = temperature
        
        # Build extra headers for prompt caching if enabled
        extra_headers = {}
        if ENABLE_PROMPT_CACHING:
            extra_headers["anthropic-beta"] = ANTHROPIC_CACHE_BETA_HEADER

        if model and model.startswith(OLLAMA_PREFIX):
            ollama_model_name = model[len(OLLAMA_PREFIX):]
            self.llm = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=ollama_model_name,
                temperature=temp,
                num_predict=MAX_TOKENS,
            )
        elif model and model.startswith(OLLAMA_CLOUD_PREFIX):
            cloud_model_name = model[len(OLLAMA_CLOUD_PREFIX):]
            self.llm = ChatOpenAI(
                base_url=OLLAMA_CLOUD_BASE_URL,
                api_key=OLLAMA_CLOUD_API_KEY,
                model=cloud_model_name,
                temperature=temp,
                max_tokens=MAX_TOKENS,
                extra_headers=extra_headers
            )
        else:
            self.llm = ChatOpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=OPENROUTER_API_KEY,
                model=model,
                temperature=temp,
                max_tokens=MAX_TOKENS,
                extra_headers=extra_headers
            )
        
        self.permission_manager = PermissionManager(mode=permission_mode)
        self.result_manager = ResultManager()
        self.context_manager = ContextManager()
        self.permission_callback = None # Set by CLI/API
        self.session_dir = SESSION_DIR
        os.makedirs(self.session_dir, exist_ok=True)

        # Convert AVAILABLE_TOOLS to LangChain tool format
        self.tools_metadata = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["input_schema"].model_json_schema(),
                },
            }
            for name, info in AVAILABLE_TOOLS.items()
        ]
        self.llm_with_tools = self.llm.bind_tools(self.tools_metadata)

    def save_session(self, session_id: str, messages: List[Any]):
        """Save the conversation history to a JSON file."""
        file_path = os.path.join(self.session_dir, f"{session_id}.json")
        serialized_messages = [message_to_dict(m) for m in messages]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serialized_messages, f, indent=2)
        logger.info(f"Session saved to {file_path}")

    def load_session(self, session_id: str) -> List[Any]:
        """Load conversation history from a JSON file."""
        file_path = os.path.join(self.session_dir, f"{session_id}.json")
        if not os.path.exists(file_path):
            return []
        with open(file_path, "r", encoding="utf-8") as f:
            serialized_messages = json.load(f)
        messages = messages_from_dict(serialized_messages)
        logger.info(f"Session {session_id} loaded with {len(messages)} messages.")
        return messages

    def list_sessions(self) -> List[str]:
        """List available session IDs."""
        if not os.path.exists(self.session_dir):
            return []
        return [f.replace(".json", "") for f in os.listdir(self.session_dir) if f.endswith(".json")]

    def execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool by name with provided arguments, checking permissions."""
        if name not in AVAILABLE_TOOLS:
            return f"Error: Tool '{name}' not found."
        
        # Check permissions with current depth to prevent recursion spirals
        if not self.permission_manager.check_permission(name, args, current_depth=self.delegation_depth):
            if self.permission_callback:
                allowed = self.permission_callback(name, args)
                if not allowed:
                    return f"Error: Permission denied by user for tool '{name}'."
            else:
                return f"Error: Permission denied for tool '{name}'. Set a permission_callback to handle 'ASK' mode."

        tool_info = AVAILABLE_TOOLS[name]
        try:
            # Validate args against pydantic schema
            validated_args = tool_info["input_schema"](**args)
            result = tool_info["func"](**validated_args.model_dump())
            
            # Process large results
            return self.result_manager.process_result(name, result)
        except Exception as e:
            # 🚀 Fix Reliability: Structured error feedback for the agent
            import traceback
            error_details = traceback.format_exc() if "DEBUG" in os.environ else str(e)
            return f"[TOOL_FAILURE] Tool '{name}' failed with error: {str(e)}. Please analyze the error and correct your arguments or approach."

    def process_query_stream(self, query: str, messages: Optional[List[Any]] = None, max_turns: int = 10):
        """Run the agentic loop and yield events (status, tool_calls, chunks)."""
        if messages is None or len(messages) == 0:
            sys_content = SYSTEM_PROMPT
            if ENABLE_PROMPT_CACHING:
                messages = [SystemMessage(
                    content=[{
                        "type": "text", 
                        "text": sys_content,
                        "cache_control": {"type": "ephemeral"}
                    }]
                )]
            else:
                messages = [SystemMessage(content=sys_content)]
        
        messages = self.context_manager.compact(messages)
        
        if ENABLE_PROMPT_CACHING and len(messages) > 1:
            last_msg = messages[-1]
            if hasattr(last_msg, 'content') and isinstance(last_msg.content, str):
                last_msg.content = [{
                    "type": "text",
                    "text": last_msg.content,
                    "cache_control": {"type": "ephemeral"}
                }]
        
        if query:
            messages.append(HumanMessage(content=query))
        
        turn = 0
        while turn < max_turns:
            turn += 1
            yield {"type": "status", "content": f"Thinking (Turn {turn})..."}
            
            # Use stream() for the final answer, or invoke() if tools are likely
            # For simplicity in the agentic loop, we'll use stream even for tools
            # to capture the thought process if the model supports it.
            
            full_response = None
            for chunk in self.llm_with_tools.stream(messages):
                if full_response is None:
                    full_response = chunk
                else:
                    full_response += chunk
                
                if chunk.content:
                    yield {"type": "chunk", "content": chunk.content}
            
            messages.append(full_response)
            
            if not full_response.tool_calls:
                yield {"type": "done", "messages": messages}
                return

            for tool_call in full_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                yield {"type": "status", "content": f"Executing {tool_name}..."}
                result = self.execute_tool(tool_name, tool_args)
                
                # 🚀 Fix: Handle Human-in-the-loop interruption
                if result.startswith("[INTERRUPT_REQUIRED]"):
                    yield {
                        "type": "interrupt", 
                        "content": result, 
                        "tool_name": tool_name, 
                        "tool_id": tool_id,
                        "messages": messages
                    }
                    return # Stop the current generator; CLI/UI will resume with the answer
                
                messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                yield {"type": "tool_result", "tool": tool_name, "result": result}
                
        yield {"type": "error", "content": "Maximum turns reached."}

    def process_query(self, query: str, messages: Optional[List[Any]] = None, max_turns: int = 10):
        """Run the agentic loop to process a user query."""
        if messages is None or len(messages) == 0:
            # Mark system prompt for caching
            sys_content = SYSTEM_PROMPT
            if ENABLE_PROMPT_CACHING:
                # Add cache control metadata (Anthropic style)
                messages = [SystemMessage(
                    content=[{
                        "type": "text", 
                        "text": sys_content,
                        "cache_control": {"type": "ephemeral"}
                    }]
                )]
            else:
                messages = [SystemMessage(content=sys_content)]
        
        # Apply context management (summarization/snipping) before the new turn
        messages = self.context_manager.compact(messages)
        
        # Mark the end of the history as a cache point if caching is enabled
        if ENABLE_PROMPT_CACHING and len(messages) > 1:
            last_msg = messages[-1]
            if hasattr(last_msg, 'content') and isinstance(last_msg.content, str):
                last_msg.content = [{
                    "type": "text",
                    "text": last_msg.content,
                    "cache_control": {"type": "ephemeral"}
                }]
        
        if query:
            messages.append(HumanMessage(content=query))
        
        turn = 0
        executed_tool_calls = [] # Track tool calls to detect repetitive failures
        
        while turn < max_turns:
            turn += 1
            logger.info(f"--- Turn {turn} ---")
            
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            # Check if LLM wants to call tools
            if not response.tool_calls:
                # Final answer reached
                return response.content, messages

            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                # Repetitive failure detection
                call_sig = (tool_name, json.dumps(tool_args, sort_keys=True))
                if executed_tool_calls.count(call_sig) >= 2:
                    error_msg = f"Error: The agent is repeating the same failing tool call: {tool_name}. Stopping to prevent infinite loop."
                    logger.error(error_msg)
                    return error_msg, messages
                
                executed_tool_calls.append(call_sig)
                
                logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
                result = self.execute_tool(tool_name, tool_args)
                
                messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                
        return "Error: Maximum turns reached without a final answer.", messages

# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Using the default model for the test
    # Set ASK mode for safety in production-ready code
    engine = QueryEngine(permission_mode="ASK")
    
    # Example: Searching for a function
    answer, history = engine.process_query("Search for the 'ingest_into_chroma' function and tell me what it does.")
    print(f"\nFinal Answer:\n{answer}")
