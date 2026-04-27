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
from context_manager import ContextManager
from context_rules import ContextRules
from model_factory import ModelFactory
from permission_manager import PermissionManager
from history_manager import HistoryManager
from dream_engine import DreamEngine
from estimator import ContextEstimator
from usage_tracker import UsageTracker
from rag_chain import CORE_INSTRUCTIONS as RAG_SYSTEM_PROMPT
from utils_messages import normalize_messages

PLANNER_PROMPT = """You are a strategic orchestrator for an AI agent. 
Based on the conversation history, decide the next logical step.

ACTIONS:
1. "rag": Use if you need to search codebase context, find documentation, or understand high-level architecture.
2. "tool": Use if you need to perform a specific action (read/write/edit files, run commands) or if you already know exactly which file to look at.
3. "final": Use ONLY if you have completed the user's request or have a definitive answer.

Respond ONLY with JSON:
{
    "action": "rag" | "tool" | "final",
    "input": "Search query for RAG, or instruction for tool use, or final response",
    "reason": "Brief justification"
}"""


logger = logging.getLogger(__name__)

# Global reference for tools that need engine context (like undo)
current_engine = None

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

class QueryState:
    """Ported from query.ts: State type. Tracks lifecycle of a single user turn."""
    def __init__(self, messages: List[Any], max_turns: int = 15):
        self.messages = messages
        self.turn_count = 1
        self.max_turns = max_turns
        self.recovery_count = 0
        self.has_attempted_reactive_compact = False
        self.executed_actions: List[str] = [] # Detect repeating loops
        self.is_terminal = False

class WorkerAgent:
    """
    Isolated Sub-Agent (Worker) with its own context window.
    Designed for surgical sub-tasks (Research, Testing, Linter fixes).
    """
    def __init__(self, delegation_depth: int, main_model: str):
        self.depth = delegation_depth
        self.model = main_model
        # Workers are always 'AUTO' permitted to prevent UI deadlocks during delegation
        self.engine = QueryEngine(
            session_id=f"worker_{os.getpid()}_{self.depth}",
            model_id=self.model,
            permission_mode="AUTO",
            delegation_depth=self.depth
        )

    def solve(self, task: str) -> str:
        """Executes the sub-task and returns the final report."""
        logger.info(f"Worker (Depth {self.depth}) starting task: {task[:50]}...")
        answer, _ = self.engine.process_query(task)
        return answer

class Coordinator:
    """
    Multi-Agent Manager (Coordinator).
    Orchestrates workers, parallelizes tasks, and prevents Strategic Drift.
    """
    def __init__(self, engine: Any):
        self.manager_engine = engine # The main QueryEngine session
        self.max_workers = 3
        self.active_workers: List[WorkerAgent] = []

    def delegate(self, task: str, context_summary: str = "") -> str:
        """
        Spawns an isolated Worker to solve a sub-problem.
        (Mirrors Coordinator.ts:delegation logic)
        """
        if self.manager_engine.delegation_depth >= 3:
            return "Error: Maximum delegation depth reached."
        
        worker = WorkerAgent(
            delegation_depth=self.manager_engine.delegation_depth + 1,
            main_model=self.manager_engine.model_id
        )
        
        # Hydrate the worker with relevant context from the manager
        full_task = f"ROLE: Assistant Worker. CONTEXT: {context_summary}\nTASK: {task}"
        result = worker.solve(full_task)
        return f"--- WORKER REPORT (Depth {worker.depth}) ---\n{result}\n--- END REPORT ---"

class QueryEngine:
    """The High-Fidelity Agentic Engine Loop (F-01)."""
    def __init__(self, session_id: str = "default", model_id: str = DEFAULT_MODEL, permission_mode: str = "ASK", delegation_depth: int = 0):
        self.session_id = session_id
        self.model_id = model_id
        self.permission_mode = permission_mode
        self.delegation_depth = delegation_depth
        
        # 🐝 Initialize the Multi-Agent Coordinator
        self.coordinator = Coordinator(self)
        
        # 🌐 Register engine globally for tool access (F-Coordinator)
        import tools
        tools.current_engine = self
        self.temperature = temperature
        self.root_dir = os.getcwd()
        
        global current_engine
        current_engine = self
        
        # Build extra headers for prompt caching if enabled
        self.extra_headers = {}
        if ENABLE_PROMPT_CACHING:
            self.extra_headers["anthropic-beta"] = ANTHROPIC_CACHE_BETA_HEADER

        self.llm = ModelFactory.create_model(self.model_id, temperature=self.temperature)
        
        # 🚀 Anthropic-Grade Plan/Act Separation
        self.planner_llm = ModelFactory.create_model("fast", temperature=0.0)
        
        self.permission_manager = PermissionManager(mode=permission_mode)
        self.context_manager = ContextManager()
        self.context_rules = ContextRules(self.root_dir)
        self.history_manager = HistoryManager(self.session_dir)
        self.usage_tracker = UsageTracker()
        self.permission_callback = None # Set by CLI/API
        self.session_dir = SESSION_DIR
        os.makedirs(self.session_dir, exist_ok=True)
        self.hooks: List[Any] = [] # List[AgentHook]
        self.current_plan: str = "No plan defined yet."
        self.current_status: str = "Initializing..."

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

    def save_session(self, session_id: str, messages: List[Any], append_only: bool = False):
        """Live Journaling: Persist history turns line-by-line for crash resiliency."""
        file_path = os.path.join(self.session_dir, f"{session_id}.jsonl")
        mode = "a" if append_only else "w"
        
        with open(file_path, mode, encoding="utf-8") as f:
            for m in messages:
                serialized = message_to_dict(m)
                f.write(json.dumps(serialized) + "\n")
        logger.debug(f"Journaled {len(messages)} turns to {file_path}")

    def load_session(self, session_id: str) -> List[Any]:
        """Load conversation history from JSONL (with legacy .json migration)."""
        file_path = os.path.join(self.session_dir, f"{session_id}.jsonl")
        old_json_path = os.path.join(self.session_dir, f"{session_id}.json")

        if os.path.exists(old_json_path) and not os.path.exists(file_path):
            try:
                with open(old_json_path, "r", encoding="utf-8") as f:
                    messages = messages_from_dict(json.load(f))
                    self.save_session(session_id, messages)
                    return messages
            except Exception as e:
                logger.error(f"Migration failed for {session_id}: {e}")

        if not os.path.exists(file_path):
            return []

        messages = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        messages.append(messages_from_dict([json.loads(line)])[0])
                    except Exception as e:
                        logger.error(f"Failed to parse journal line: {e}")
        return messages

    def list_sessions(self) -> List[str]:
        """List available session IDs."""
        if not os.path.exists(self.session_dir):
            return []
        # Support both .json and .jsonl in listing
        sessions = set()
        for f in os.listdir(self.session_dir):
            if f.endswith(".json"): sessions.add(f.replace(".json", ""))
            elif f.endswith(".jsonl"): sessions.add(f.replace(".jsonl", ""))
        return sorted(list(sessions))

    def switch_model(self, model_id: str) -> str:
        """Hot-swap the active LLM engine."""
        try:
            self.llm = ModelFactory.create_model(model_id, temperature=self.temperature)
            self.model_id = model_id
            logger.info(f"Engine: Successfully switched to {model_id}")
            return f"System: LLM Engine changed to {model_id}. All context preserved."
        except Exception as e:
            return f"Error: Failed to switch to {model_id}: {str(e)}"

    def execute_tool(self, name: str, args: Dict[str, Any], tool_id: str = "unknown") -> str:
        """Execute a tool by name with provided arguments, checking permissions."""
        if name not in AVAILABLE_TOOLS:
            return f"Error: Tool '{name}' not found."
        
        # Check permissions
        is_allowed = self.permission_callback(name, args) if self.permission_callback else True
        if not is_allowed:
            return f"Error: Permission denied to run tool '{name}'."

        tool_info = AVAILABLE_TOOLS[name]
        try:
            # Validate args against pydantic schema
            validated_args = tool_info["input_schema"](**args)
            result = tool_info["func"](**validated_args.model_dump())
            
            # 🚀 Handle Synthetic State Interception
            if isinstance(result, str):
                if result.startswith("[PLAN_UPDATED]"):
                    self.current_plan = result.replace("[PLAN_UPDATED] ", "")
                elif result.startswith("[STATUS_UPDATED]"):
                    self.current_status = result.replace("[STATUS_UPDATED] ", "")

            # 🚀 SYSTEM 2: Result Budget Gate (Archiving)
            offloaded_result = self.archive.maybe_offload(name, tool_id, result)
            
            # 🚀 SYSTEM 2: Working Memory Hydration
            if name == "file_read" and isinstance(result, str):
                self.context_manager.record_file_read(args.get("path", "unknown"), result)

            return offloaded_result
        except Exception as e:
            # 🚀 Fix Reliability: Structured error feedback for the agent
            import traceback
            error_details = traceback.format_exc() if "DEBUG" in os.environ else str(e)
            return f"[TOOL_FAILURE] Tool '{name}' failed with error: {str(e)}. Please analyze the error and correct your arguments or approach."

    def _call_planner(self, messages: List[Any], turn: int, action_history: List[str]) -> Dict[str, Any]:
        """Strategic Decision Layer with Strategy Evolution (Self-Awareness)."""
        try:
            # Construct a summary of recent history for the planner
            history_summary = ""
            for m in messages[-5:]:
                type_name = "User" if isinstance(m, HumanMessage) else "Assistant" if isinstance(m, AIMessage) else "Tool" if isinstance(m, ToolMessage) else "System"
                content = m.content if isinstance(m.content, str) else str(m.content)
                history_summary += f"{type_name}: {content[:150]}...\n"
            
            # Action history awareness
            action_hist_text = "\n".join(action_history[-5:]) if action_history else "No previous actions."
            
            prompt = f"""[AGENT_STATUS]
Current Turn: {turn}

Recent History:
{history_summary}

Recent Actions Taken:
{action_hist_text}

USER REQUEST: {messages[1].content if len(messages) > 1 else 'None'}
[/AGENT_STATUS]

Based on this status, determine the next action."""
            
            # 🚀 Use the specialized Planner Model for strategy overrides
            response = self.planner_llm.invoke([
                SystemMessage(content=PLANNER_PROMPT), 
                HumanMessage(content=prompt)
            ])
            
            # Cleanup JSON (handling cases where LLM adds Markdown blocks)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("{") and "}" in content:
                pass
            else:
                # Basic heuristic extraction if JSON is buried
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    content = content[start:end+1]
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"Planner failed: {e}. Defaulting to 'tool'.")
            return {"action": "tool", "input": "", "reason": "Planner error fallback"}

    def _apply_pre_query_grooming(self, messages: List[Any], plan: str) -> List[Any]:
        """
        Ported from query.ts:379-426.
        Surgically prunes context BEFORE every turn.
        """
        # 1. Update Context (Inject Rules + Plan)
        memory_instructions = self.context_rules.get_instructions()
        
        # 2. Compact if needed (System 2: Context Management)
        messages = self.context_manager.compact(messages, plan=plan)
        
        # 3. Apply Active System Directives
        system_instructions = f"{RAG_SYSTEM_PROMPT}\n\n[MASTER_PLAN]\n{plan}\n\n{memory_instructions}"
        if messages and isinstance(messages[0], SystemMessage):
            messages[0].content = system_instructions
        else:
            messages.insert(0, SystemMessage(content=system_instructions))
            
        return messages

    def process_query_stream(self, query: str, session_id: str = "default", messages: Optional[List[Any]] = None, max_turns: int = 15):
        """
        Anthropic-Grade Agentic Loop (F-01).
        Ported from queryLoop in query.ts.
        """
        if messages is None or len(messages) == 0:
            sys_content = f"{SYSTEM_PROMPT}\n\n[MASTER_PLAN_SCRATCHPAD]\n{self.current_plan}\n[/MASTER_PLAN_SCRATCHPAD]"
            messages = [SystemMessage(content=sys_content)]
        
        if query:
            user_msg = HumanMessage(content=query)
            messages.append(user_msg)
            self.save_session(session_id, [user_msg], append_only=True)

        state = QueryState(messages, max_turns)

        while not state.is_terminal:
            if state.turn_count > state.max_turns:
                yield {"type": "status", "content": "Reached max turn budget. Stopping."}
                state.is_terminal = True
                break

            # 🛠️ G-1: Grooming Phase (The Anthropic Secret)
            state.messages = self._apply_pre_query_grooming(state.messages, self.current_plan)
            
            # 🪦 T-1: Tombstone Recovery (query.ts:182)
            # We track the history length before the tick starts.
            # If the tick fails, we 'Burn' (Tombstone) any new messages to prevent poisoning.
            history_length_before_tick = len(state.messages)
            
            try:
                for event in self._execute_tick(state, session_id):
                    yield event
            except (Exception, KeyboardInterrupt) as e:
                # 🕯️ The Tombstone Phase: Burn the orphaned trail
                # 🛑 GHOST DISCARD: Force-terminate any active tool processes (bash, git, etc.) (Line 734)
                import tools
                tools.cleanup_active_processes()
                
                orphans = state.messages[history_length_before_tick:]
                if orphans:
                    logger.warning(f"Tombstone Action: Purging {len(orphans)} orphaned messages and terminating active tools.")
                    state.messages = state.messages[:history_length_before_tick]
                    
                    # Yield a tombstone event for the UI to clean up
                    yield {
                        "type": "tombstone", 
                        "content": f"Streaming Error: {str(e)[:100]}... Discarding partial history to prevent strategic drift.",
                        "purged_messages": len(orphans)
                    }
                
                if isinstance(e, KeyboardInterrupt):
                    raise e
                logger.error(f"Turn failure recovered via Tombstone: {e}")
                
            state.turn_count += 1

        # 🧠 F-50: Autonomous Post-Turn Reflection (AutoDream)
        self.dream_engine.reflect_and_learn(state.messages, self.model_id)

    def _execute_tick(self, state: QueryState, session_id: str):
        """One iteration of the agent's logic engine."""
        messages = normalize_messages(state.messages)
        
        # 📏 F-32: Pre-Flight Context Audit
        safety = ContextEstimator.check_flight_safety(self.model_id, messages)
        if safety["status"] == "CRITICAL":
            yield {"type": "status", "content": f"Context Critical ({safety['usage_pct']}%). Triggering Compaction..."}
            state.messages = self.context_manager.compact(state.messages)
            messages = normalize_messages(state.messages) # Refresh
        elif safety["status"] == "WARNING":
            yield {"type": "status", "content": f"Warning: Context is {safety['usage_pct']}% full."}

        # 1. Decision Layer (Planner)
        decision = self._call_planner(messages, state.turn_count, state.executed_actions)
        action = decision.get("action", "tool")
        action_input = decision.get("input", "")
        
        # Stall Check & Prevention
        action_sig = f"action:{action} input:{action_input[:40]}"
        if state.executed_actions.count(action_sig) >= 2:
            action = "tool" 
        state.executed_actions.append(action_sig)

        # 2. Action Execution
        if action == "rag":
            yield {"type": "status", "content": f"RAG: Exploring knowledge for '{action_input}'..."}
            try:
                rag_result = ""
                # Use the RAG chain directly
                for event in _full_context_cache_chain().stream({"question": action_input, "chat_history": messages}):
                    if isinstance(event, dict) and "answer" in event:
                        rag_result += event["answer"]
                
                # 🧠 F-50: Inject Project Memories
                memories = self.dream_engine.get_memories()
                
                rag_msg = SystemMessage(content=f"[RAG_CONTEXT]\n{rag_result}\n\n{memories}\n[/RAG_CONTEXT]")
                state.messages.append(rag_msg)
                self.save_session(session_id, [rag_msg], append_only=True)
            except Exception as e:
                logger.error(f"RAG failed: {e}")
                
        elif action == "tool":
            yield {"type": "status", "content": f"Thinking: {decision.get('reason', 'Processing...')}"}
            try:
                response = self.llm_with_tools.invoke(messages)
                state.messages.append(response)
                self.save_session(session_id, [response], append_only=True)
                
                if not response.tool_calls:
                    state.is_terminal = True
                else:
                    for tc in response.tool_calls:
                        # 🛡️ F-14: Permission Guardian Gate
                        tool_name = tc['name']
                        args = tc['args']
                        decision = self.permission_manager.check_permission(tool_name, args)
                        
                        if decision.behavior == "deny":
                            yield {"type": "status", "content": f"Blocked: {decision.reason}"}
                            result = f"Error: Permission denied. {decision.reason}"
                        elif decision.behavior == "ask":
                            # Interrupt and wait for user. Ported from interactiveHandler.ts
                            yield {"type": "permission_request", "tool": tool_name, "args": args}
                            return # Exit tick to wait for input
                        else:
                            # ⏪ F-28: Automatic Pre-Edit Checkpointing
                            if tool_name in ["file_write", "file_edit", "multi_file_edit"]:
                                self.history_manager.track_edit(args.get("file_path", ""), tc.get("id", "unknown"))
                            
                            yield {"type": "status", "content": f"Action: Running {tool_name}..."}
                            result = self.execute_tool(tool_name, args, tc.get("id", "unknown"))
                        
                        tool_msg = ToolMessage(content=str(result), tool_call_id=tc.get("id", "unknown"))
                        state.messages.append(tool_msg)
                        self.save_session(session_id, [tool_msg], append_only=True)
            except Exception as e:
                # Reactive Compact Recovery (F-40 / query.ts:168)
                if "context_length_exceeded" in str(e).lower() and not state.has_attempted_reactive_compact:
                    yield {"type": "status", "content": "Context Overloaded. Attempting Recovery..."}
                    state.messages = self.context_manager.compact(state.messages, plan=self.current_plan)
                    state.has_attempted_reactive_compact = True
                    state.turn_count -= 1 # Repeat this turn
                else:
                    logger.error(f"Execution failed: {e}")
                    state.is_terminal = True

            # 🚀 ENFORCED TOOL ACTION (ACTIVE NUDGE)
            if action == "tool":
                yield {"type": "status", "content": f"Action: {reason}"}
                
                # Active nudge to prevent model passivity (The "Lazy LLM" fix)
                nudge = SystemMessage(content="If external action is required (file read/write, bash), you MUST call a tool now. Do not provide a text-only response unless you are giving a final answer.")
                
                full_response = None
                for chunk in self.llm_with_tools.stream(messages + [nudge]):
                    if full_response is None:
                        full_response = chunk
                    else:
                        full_response += chunk
                    
                    if chunk.content:
                        yield {"type": "chunk", "content": chunk.content}
                
                messages.append(full_response)
                self.save_session(session_id, [full_response], append_only=True)
                
                # 💰 F-18: Capture and Audit Usage (Anthropic-Parity)
                if hasattr(full_response, "usage_metadata") and full_response.usage_metadata:
                    u = full_response.usage_metadata
                    self.usage_tracker.track(self.model_id, {
                        "input_tokens": u.get("input_tokens", 0),
                        "output_tokens": u.get("output_tokens", 0),
                        "cache_read_input_tokens": u.get("cache_read_input_tokens", 0),
                        "cache_creation_input_tokens": u.get("cache_creation_input_tokens", 0)
                    })
                
                if not full_response.tool_calls:
                    if turn > 1:
                        action = "final"
                    else:
                        return 

                for tool_call in full_response.tool_calls:
                    tool_name, tool_args, tool_id = tool_call["name"], tool_call["args"], tool_call["id"]
                    
                    # 🚀 HOOK: Pre-Tool Execution
                    for hook in self.hooks:
                        modified_args = hook.on_tool_call(tool_name, tool_args)
                        if modified_args is not None:
                            tool_args = modified_args

                    yield {"type": "status", "content": f"Executing {tool_name}..."}
                    result = self.execute_tool(tool_name, tool_args, tool_id=tool_id)
                    
                    # 🚀 HOOK: Post-Tool Result
                    for hook in self.hooks:
                        modified_result = hook.on_tool_result(tool_name, result)
                        if modified_result is not None:
                            result = modified_result

                    if result.startswith("[INTERRUPT_REQUIRED]"):
                        yield {
                            "type": "interrupt", 
                            "content": result, 
                            "tool_name": tool_name, 
                            "tool_id": tool_id,
                            "messages": messages
                        }
                        return
                    
                    tool_msg = ToolMessage(content=result, tool_call_id=tool_id)
                    messages.append(tool_msg)
                    self.save_session(session_id, [tool_msg], append_only=True)
                    yield {"type": "tool_result", "tool": tool_name, "result": result}
                
                continue # Re-plan after tool execution

            # 🚀 HIGH-FIDELITY FINAL ACTION
            if action == "final":
                final_prompt = """You are finishing the task. Based on all previous reasoning, tool results, and context:
1. Provide a clear and definitive final answer.
2. Be concise and professional.
3. Do not repeat unnecessary steps.
4. If code was changed, summarize the modifications."""
                
                response = self.llm.invoke(messages + [SystemMessage(content=final_prompt)])
                
                # 🚀 HOOK: Turn End
                for hook in self.hooks:
                    hook.on_turn_end(response.content)

                yield {"type": "chunk", "content": response.content}
                yield {"type": "done", "messages": messages}
                return

        yield {"type": "error", "content": f"Safety limit: Maximum turns ({max_turns}) reached."}


    def process_query(self, query: str, messages: Optional[List[Any]] = None, max_turns: int = 15):
        """Standard Agentic RAG Loop: Tick-based execution (sync version)."""
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
        
        from rag_chain import _full_context_cache_chain
        
        messages = self.context_manager.compact(messages)
        
        if query:
            messages.append(HumanMessage(content=query))
        
        turn = 0
        executed_actions = []

        while turn < max_turns:
            turn += 1
            logger.info(f"--- Strategic Planning (Turn {turn}) ---")
            
            decision = self._call_planner(messages, turn)
            action = decision.get("action", "tool")
            action_input = decision.get("input", "")

            # Safety: Stall Detection
            action_sig = f"{action}:{action_input[:50]}"
            if executed_actions.count(action_sig) >= 2:
                action = "tool"
            executed_actions.append(action_sig)

            # RAG Branch
            if action == "rag":
                logger.info(f"RAG Retrieval: {action_input}")
                rag_result = ""
                # For sync, we collect the stream
                for event in _full_context_cache_chain().stream({"question": action_input, "chat_history": messages}):
                    if isinstance(event, dict) and "answer" in event:
                        rag_result += event["answer"]
                
                messages.append(AIMessage(content=f"[INTERNAL_RETRIEVED_CONTEXT] {rag_result}"))
                continue 

            # TOOL Branch
            if action == "tool":
                response = self.llm_with_tools.invoke(messages)
                messages.append(response)
                
                if not response.tool_calls:
                    if turn > 1: action = "final"
                    else: continue

                for tool_call in response.tool_calls:
                    tool_name, tool_args, tool_id = tool_call["name"], tool_call["args"], tool_call["id"]
                    logger.info(f"Executing: {tool_name}")
                    result = self.execute_tool(tool_name, tool_args)
                    messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                
                continue

            # FINAL Branch
            if action == "final":
                final_answer = action_input or "Task completed."
                return final_answer, messages

        return "Error: Maximum turns reached.", messages

# Example usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = QueryEngine(permission_mode="ASK")
    
    # Example: Searching for a function
    # Note: ensure your rag_chain is populated before testing
    answer, history = engine.process_query("Search for the 'ingest_all.py' script and tell me its purpose.")
    print(f"\nFinal Answer:\n{answer}")

