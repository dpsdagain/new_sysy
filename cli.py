import sys
import time

# ⏱️ STARTUP PROFILER (Anthropic-Parity: entrypoint.ts:45)
_start_time = time.perf_counter()

# ⚡ FAST-PATH DISPATCHER 
if len(sys.argv) > 1:
    fast_cmd = sys.argv[1]
    if fast_cmd in ["--version", "-v"]:
        print("🤖 Antigravity Agent v2.4.1 (Anthropic-Parity Core)")
        sys.exit(0)
    elif fast_cmd == "--help":
        print("Usage: antigravity [COMMAND] [OPTIONS]\n\nCommands:\n  doctor   Audit system health\n  cost     View session costs\n  clear    Reset session\n\nOptions:\n  --model  Specify LLM model\n  --v      Show version")
        sys.exit(0)

# Check if Fast-Path is actually FAST (F-Profiler Gate)
_fast_path_latency = (time.perf_counter() - _start_time) * 1000
if _fast_path_latency > 50:
    # Print a "Developer Warning" in the background
    sys.stderr.write(f"[BOOT_WARNING] Fast-path took {_fast_path_latency:.2f}ms. Check for heavy top-level imports.\n")

import logging
import argparse
import time
import os
import re
import threading
from typing import List, Any, Optional, Dict

# Lazy-loading pointers
QueryEngine = None 
cleanup_active_processes = None
Console = None # Fast UI Path

def get_console():
    """Lazily initialize the rich console."""
    global Console
    if Console is None:
        from rich.console import Console as RichConsole
        from rich.theme import Theme
        theme = Theme({
            "info": "dim cyan", "warning": "magenta", "danger": "bold red",
            "user": "bold green", "agent": "bold blue", "status": "italic yellow",
            "tool": "bold cyan", "diff.add": "green", "diff.remove": "red",
            "model": "dim blue", "pill.think": "bold white on #4e4e4e",
            "pill.tool": "bold white on #005f5f",
        })
        Console = RichConsole(theme=theme)
    return Console

# Configure logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.ERROR)

# Constants for UI
DOT = "●"
INDENT = "  "
PIPE = "│ "
AGENT_ICON = "🤖"
USER_ICON = "👤"
TOOL_ICON = "🛠️"
THINK_ICON = "🧠"

class AgentCLI:
    def __init__(self, session_id: str, model_id: str, permission_mode: str = "ASK"):
        self.session_id = session_id
        self.model_id = model_id
        self.permission_mode = permission_mode
        self.engine = None
        self.is_ready = False
        self.history = []
        
        # UI components are loaded only when AgentCLI is instantiated
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.styles import Style as PtStyle
        from prompt_toolkit.shortcuts import CompleteStyle

        # Start Background Hydration immediately
        self._boot_thread = threading.Thread(target=self._boot_engine, daemon=True)
        self._boot_thread.start()

        # Instant UI Setup
        os.makedirs(".sessions", exist_ok=True)
        self.prompt_session = PromptSession(
            history=FileHistory(os.path.join(".sessions", f"{session_id}_history.txt")),
            complete_style=CompleteStyle.MULTI_COLUMN
        )
        self.completer = WordCompleter([
            "/model", "/session", "/help", "exit", "quit", "clear"
        ], ignore_case=True)
        
        self.pt_style = PtStyle.from_dict({
            'prompt': '#00ff00 bold',
        })

    def _boot_engine(self):
        """Heavy lifting happens here in the background."""
        global QueryEngine, cleanup_active_processes
        try:
            import atexit
            if QueryEngine is None:
                from query_engine import QueryEngine
            if cleanup_active_processes is None:
                from tools import cleanup_active_processes
                atexit.register(cleanup_active_processes)
            
            self.engine = QueryEngine(model=self.model_id, permission_mode=self.permission_mode)
            self.engine.permission_callback = self.permission_callback
            self.history = self.engine.load_session(self.session_id)
            self.is_ready = True
        except Exception as e:
            # We don't crash the UI thread, just log the error
            logging.error(f"Engine Boot Failed: {e}")

    def permission_callback(self, tool_name: str, tool_args: dict) -> bool:
        console = get_console()
        from rich.table import Table
        from rich.box import SIMPLE
        from rich.align import Align
        console.print("\n")
        console.print(f"{INDENT}[pill.tool] 🔑 PERMISSION [/] [bold white]Agent wants to use {tool_name}[/]")
        table = Table(border_style="warning", box=SIMPLE, expand=False, show_header=False)
        for k, v in tool_args.items():
            table.add_row(f"[tool]{k}[/]", str(v))
        console.print(Align.left(table, pad=True))
        answer = console.input(f"{INDENT}[bold yellow]Allow? (y/n) [y]: [/]").strip().lower()
        return answer in ["", "y", "yes"]

    def render_diff(self, file_path: str, old_str: str, new_str: str):
        console = get_console()
        from rich.syntax import Syntax
        from rich.panel import Panel
        from rich.box import ROUNDED
        from rich.align import Align
        from difflib import unified_diff
        diff = list(unified_diff(
            old_str.splitlines(keepends=True),
            new_str.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}"
        ))
        if diff:
            syntax = Syntax("".join(diff), "diff", theme="monokai")
            console.print(Align.left(Panel(syntax, title=f"Changes in {file_path}", border_style="tool", box=ROUNDED, width=min(console.width - 4, 100)), pad=True))

    def render_code_search(self, tool_result: str):
        console = get_console()
        from rich.syntax import Syntax
        from rich.panel import Panel
        from rich.box import MINIMAL
        from rich.align import Align
        
        parts = re.split(r"--- Result \d+ \((.*?)\) ---", tool_result)
        if len(parts) >= 2:
            for i in range(1, len(parts), 2):
                path = parts[i]
                content = parts[i+1].strip()
                ext = os.path.splitext(path)[1]
                lang = "python" if ext == ".py" else "javascript" if ext in [".js", ".ts", ".tsx"] else "text"
                syntax = Syntax(content, lang, theme="monokai", line_numbers=True)
                console.print(Align.left(Panel(syntax, title=f"🔍 {path}", border_style="info", box=MINIMAL, width=min(console.width - 4, 100)), pad=True))

    def print_help(self):
        console = get_console()
        from rich.table import Table
        from rich.box import ROUNDED
        from rich.align import Align
        
        table = Table(title="Interactive Commands", border_style="info", box=ROUNDED)
        table.add_column("Command", style="cyan")
        table.add_column("Description", style="white")
        table.add_row("/model <ID>", "Switch to a different LLM model")
        table.add_row("/session", "Show current session info")
        table.add_row("/help", "Show this help menu")
        table.add_row("clear", "Clear the terminal screen")
        table.add_row("exit | quit", "Save and exit the session")
        console.print(Align.left(table, pad=True))

    def run(self):
        console = get_console()
        from rich.panel import Panel
        from rich.box import ROUNDED
        from rich.text import Text as RichText
        from prompt_toolkit.shortcuts import CompleteStyle
        from prompt_toolkit.formatted_text import HTML
        
        console.print("\n")
        console.print(Panel(
            RichText.from_markup(f"Session: [bold cyan]{self.session_id}[/]\nModel: [bold blue]{self.model_id}[/]\nHydration: [bold yellow]Background Tick...[/]"),
            title=f"[bold agent]{AGENT_ICON} Autonomous Engineering Agent[/]",
            border_style="agent", box=ROUNDED, padding=(1, 2), expand=False
        ))
        console.print(f"{INDENT}[info]Type '/help' for commands. Greedy UI Boot active.[/info]\n")
        
        while True:
            try:
                query = self.prompt_session.prompt(
                    HTML(f'<b><ansigreen>{USER_ICON} User</ansigreen></b> > '),
                    completer=self.completer,
                    style=self.pt_style
                ).strip()
                
                if not query: continue
                if query.lower() in ["exit", "quit"]: break
                if query.lower() == "clear":
                    console.clear()
                    continue

                if query.startswith("/model "):
                    new_model = query.replace("/model ", "").strip()
                    with console.status(f"{INDENT}[status]Switching to {new_model}...[/status]"):
                        while not self.is_ready: time.sleep(0.1) # Wait if still booting
                        self.engine = QueryEngine(model=new_model, permission_mode=self.engine.permission_manager.mode)
                        self.engine.permission_callback = self.permission_callback
                        self.model_id = new_model
                    console.print(f"{INDENT}✅ Model switched to [bold blue]{self.model_id}[/]")
                    continue

                # --- ENSURE ENGINE IS READY ---
                if not self.is_ready:
                    with console.status(f"{INDENT}[status]Warming up query engine...[/status]"):
                        while not self.is_ready:
                            time.sleep(0.1)

                # --- AGENT TURN ---
                def run_agent_turn(agent_query, history_msgs):
                    console = get_console()
                    from rich.panel import Panel
                    from rich.box import ROUNDED
                    from rich.align import Align
                    from langchain_core.messages import ToolMessage
                    
                    full_answer = ""
                    active_status = None
                    has_printed_pipe = False
                    interrupted_event = None

                    for event in self.engine.process_query_stream(agent_query, session_id=self.session_id, messages=history_msgs):
                        if event["type"] == "status":
                            content = event['content']
                            icon = TOOL_ICON if "Executing" in content else THINK_ICON
                            status_msg = f"{INDENT}[pill.think] {icon} {content} [/]"
                            if active_status:
                                active_status.update(status_msg)
                            else:
                                active_status = console.status(status_msg)
                                active_status.start()
                        
                        elif event["type"] == "chunk":
                            if active_status:
                                active_status.stop()
                                active_status = None
                            
                            if not has_printed_pipe:
                                console.print(f"{PIPE}", end="")
                                has_printed_pipe = True
                                
                            content = event["content"]
                            full_answer += content
                            console.print(content.replace("\n", f"\n{PIPE}"), end="")

                        elif event["type"] == "tombstone":
                            if active_status: active_status.stop()
                            console.print(f"\n{INDENT}[warning] 🪦 {event['content']} [/]")
                        
                        elif event["type"] == "interrupt":
                            if active_status: active_status.stop()
                            interrupted_event = event
                            break 

                        elif event["type"] == "done":
                            if active_status: active_status.stop()
                            self.history = event["messages"]
                            console.print("\n")
                            
                        elif event["type"] == "error":
                            if active_status: active_status.stop()
                            console.print(f"\n{INDENT}[danger]Error: {event['content']}[/danger]")

                    if interrupted_event:
                        question = interrupted_event["content"].replace("[INTERRUPT_REQUIRED] The agent needs human input: ", "")
                        console.print("\n")
                        console.print(Align.left(Panel(
                            f"[bold yellow]🙋 Question:[/bold yellow] {question}",
                            border_style="warning", box=ROUNDED, width=min(console.width - 4, 100)
                        ), pad=True))
                        
                        answer = self.prompt_session.prompt(
                            HTML(f'<b><ansigellow>{USER_ICON} Answer</ansigellow></b> > '),
                            style=self.pt_style
                        ).strip()
                        
                        self.history.append(ToolMessage(content=f"User replied: {answer}", tool_call_id=interrupted_event["tool_id"]))
                        console.print(f"\n{DOT} [agent]Agent[/agent] [dim](Resuming...)[/dim]\n")
                        return run_agent_turn(None, self.history)
                    
                    return full_answer

                from rich.rule import Rule
                console.print(Rule(style="agent"))
                console.print(f"\n{DOT} [agent]Agent[/agent] [model]({self.model_id})[/model]\n")
                run_agent_turn(query, self.history)

                # --- POST-TURN RENDERS ---
                self.engine.save_session(self.session_id, self.history)
                console.print(Rule(style="dim"))
                
            except (KeyboardInterrupt, EOFError):
                console.print(f"\n{INDENT}[warning]Exiting and saving session...[/warning]")
                if self.engine: self.engine.save_session(self.session_id, self.history)
                break
            except Exception as e:
                console.print(f"\n{INDENT}[danger]Unexpected Error: {str(e)}[/danger]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    cli = AgentCLI(args.session or "default_session", args.model or "ollama-cloud:gpt-oss:120b-cloud")
    cli.run()
