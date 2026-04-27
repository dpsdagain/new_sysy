import sys
import logging
import argparse
import time
import os
import re
from typing import List, Any, Optional, Dict
from langchain_core.messages import AIMessage, HumanMessage
from query_engine import QueryEngine
from tools import cleanup_active_processes
import atexit

# 🚀 Resource Safety: Ensure background processes die with the CLI
atexit.register(cleanup_active_processes)

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.status import Status
from rich.table import Table
from rich.theme import Theme
from rich.syntax import Syntax
from rich.columns import Columns
from rich.align import Align
from rich.rule import Rule
from rich.text import Text as RichText
from rich.box import ROUNDED, MINIMAL, SIMPLE

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style as PtStyle
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import CompleteStyle

# Custom theme for the Anthropic/Claude look
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "user": "bold green",
    "agent": "bold blue",
    "status": "italic yellow",
    "tool": "bold cyan",
    "diff.add": "green",
    "diff.remove": "red",
    "model": "dim blue",
    "pill.think": "bold white on #4e4e4e",
    "pill.tool": "bold white on #005f5f",
})

console = Console(theme=custom_theme)

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
        self.engine = QueryEngine(model=model_id, permission_mode=permission_mode)
        self.engine.permission_callback = self.permission_callback
        self.history = self.engine.load_session(session_id)
        
        history_file = os.path.join(self.engine.session_dir, f"{session_id}_history.txt")
        self.prompt_session = PromptSession(
            history=FileHistory(history_file),
            complete_style=CompleteStyle.MULTI_COLUMN
        )
        self.completer = WordCompleter([
            "/model", "/session", "/help", "exit", "quit", "clear"
        ], ignore_case=True)
        
        self.pt_style = PtStyle.from_dict({
            'prompt': '#00ff00 bold',
        })

    def permission_callback(self, tool_name: str, tool_args: dict) -> bool:
        console.print("\n")
        console.print(f"{INDENT}[pill.tool] 🔑 PERMISSION [/] [bold white]Agent wants to use {tool_name}[/]")
        table = Table(border_style="warning", box=SIMPLE, expand=False, show_header=False)
        for k, v in tool_args.items():
            table.add_row(f"[tool]{k}[/]", str(v))
        console.print(Align.left(table, pad=True))
        answer = console.input(f"{INDENT}[bold yellow]Allow? (y/n) [y]: [/]").strip().lower()
        return answer in ["", "y", "yes"]

    def render_diff(self, file_path: str, old_str: str, new_str: str):
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
        console.print("\n")
        console.print(Panel(
            RichText.from_markup(f"Session: [bold cyan]{self.session_id}[/]\nModel: [bold blue]{self.model_id}[/]\nMessages: [bold]{len(self.history)}[/]"),
            title=f"[bold agent]{AGENT_ICON} Autonomous Engineering Agent[/]",
            border_style="agent", box=ROUNDED, padding=(1, 2), expand=False
        ))
        console.print(f"{INDENT}[info]Type '/help' for commands. History enabled.[/info]\n")
        
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
                        self.engine = QueryEngine(model=new_model, permission_mode=self.engine.permission_manager.mode)
                        self.engine.permission_callback = self.permission_callback
                        self.model_id = new_model
                    console.print(f"{INDENT}✅ Model switched to [bold blue]{self.model_id}[/]")
                    continue

                # --- AGENT TURN ---
                full_answer = ""
                console.print(Rule(style="agent"))
                console.print(f"\n{DOT} [agent]Agent[/agent] [model]({self.model_id})[/model]\n")
                
                # --- AGENT TURN ---
                def run_agent_turn(agent_query, history_msgs):
                    full_answer = ""
                    active_status = None
                    has_printed_pipe = False
                    interrupted_event = None

                    for event in self.engine.process_query_stream(agent_query, messages=history_msgs):
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
                            
                            if "\n" in content:
                                output = content.replace("\n", f"\n{PIPE}")
                                output = output.replace(f"{PIPE}{PIPE}", PIPE).replace(f"{PIPE}│", PIPE)
                                console.print(output, end="")
                            else:
                                console.print(content, end="")
                        
                        elif event["type"] == "interrupt":
                            if active_status: active_status.stop()
                            interrupted_event = event
                            break # Exit the stream to handle input

                        elif event["type"] == "done":
                            if active_status: active_status.stop()
                            self.history = event["messages"]
                            if not full_answer.strip() and not has_printed_pipe:
                                console.print(f"{PIPE}[dim]Agent provided a silent response.[/dim]")
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
                            HTML(f'<b><ansiyellow>{USER_ICON} Answer</ansiyellow></b> > '),
                            style=self.pt_style
                        ).strip()
                        
                        self.history.append(ToolMessage(content=f"User replied: {answer}", tool_call_id=interrupted_event["tool_id"]))
                        console.print(f"\n{DOT} [agent]Agent[/agent] [dim](Resuming...)[/dim]\n")
                        return run_agent_turn(None, self.history) # Recursive continuation
                    
                    return full_answer

                console.print(Rule(style="agent"))
                console.print(f"\n{DOT} [agent]Agent[/agent] [model]({self.model_id})[/model]\n")
                run_agent_turn(query, self.history)

                # --- POST-TURN SPECIAL RENDERS ---
                last_msg = self.history[-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for msg in reversed(self.history):
                        if isinstance(msg, HumanMessage): break
                        if hasattr(msg, "tool_call_id") and msg.content:
                            content = msg.content
                            if "[DIFF_START]" in content:
                                match = re.search(r"Successfully edited (.*?)\. \[DIFF_START\](.*?)\[DIFF_DIVIDER\](.*?)\[DIFF_END\]", content, re.DOTALL)
                                if match:
                                    self.render_diff(match.group(1), match.group(2), match.group(3))
                            elif "--- Result 1" in content:
                                self.render_code_search(content)

                self.engine.save_session(self.session_id, self.history)
                console.print(Rule(style="dim"))
                
            except (KeyboardInterrupt, EOFError):
                console.print(f"\n{INDENT}[warning]Exiting and saving session...[/warning]")
                self.engine.save_session(self.session_id, self.history)
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
