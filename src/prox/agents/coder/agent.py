"""Coder agent: writes/modifies files. MVP mentality."""
import os
import json
from typing import Any

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from prox.graph.state import AgentState
from prox.llm import get_model_for_role, Role, create_llm

CODER_INSTRUCTIONS = """Sei un agente Coder. Scrivi codice in Python e TypeScript/JavaScript.

REGOL:
- MVP mentality: versione minima funzionante, non perfetta.
- NON aggiungere commenti.
- Segui le convenzioni esistenti del progetto.
- Output: codice pronto da applicare.
- Rispondi SOLO con il risultato finale, senza spiegazioni.

TOOL A DISPOSIZIONE:
- write_file(path, content): scrive un file
- read_file(path): legge un file
- list_directory(path): elenca directory
"""


@tool
def read_file(file_path: str, offset: int = 0, limit: int = 200) -> str:
    """Legge un file dal filesystem."""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        return "".join(lines[offset:offset + limit])
    except Exception as e:
        return f"Errore: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Scrive contenuto in un file. Crea directory se non esistono."""
    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        return f"OK: {file_path}"
    except Exception as e:
        return f"Errore: {e}"


@tool
def list_directory(path: str = ".") -> str:
    """Elenca file e directory in un percorso."""
    try:
        entries = os.listdir(path)
        result = []
        for e in sorted(entries):
            full = os.path.join(path, e)
            prefix = "[DIR]" if os.path.isdir(full) else "[FILE]"
            result.append(f"{prefix} {e}")
        return "\n".join(result)
    except Exception as e:
        return f"Errore: {e}"


CODER_TOOLS = [read_file, write_file, list_directory]


def coder_node(state: AgentState) -> dict:
    model_name = get_model_for_role(Role.CODER)
    llm = create_llm(model_name, temperature=0.0, max_tokens=4096)
    llm_with_tools = llm.bind_tools(CODER_TOOLS)

    tasks = state.get("tasks", [])
    active_task = None
    for t in tasks:
        if t.get("agent") == "coder" and t.get("status") not in ("done", "blocked"):
            active_task = t
            break

    task_description = active_task.get("description", "") if active_task else state.get("user_query", "")

    project_dir = state.get("project_dir", ".")
    messages = [
        SystemMessage(content=CODER_INSTRUCTIONS),
        HumanMessage(content=f"Task: {task_description}\nProject directory: {project_dir}"),
    ]

    output = ""
    try:
        response = llm_with_tools.invoke(messages)
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                for t in CODER_TOOLS:
                    if t.name == tool_name:
                        result = t.invoke(tool_args)
                        output += f"[{tool_name}] {result}\n"
                        break
        else:
            output = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        output = f"Eseguito parzialmente. Errore: {e}"

    if active_task:
        active_task["status"] = "done"
        active_task["output"] = output

    return {
        "tasks": tasks,
        "messages": [{"role": "assistant", "content": output}],
    }
