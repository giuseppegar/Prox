import os
import json
from typing import Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from prox.graph.state import AgentState
from prox.llm import get_model_for_role, Role

CODER_INSTRUCTIONS = """Sei un agente Coder specializzato in Python e TypeScript/JavaScript.

REGOLO:
- MVP mentality: scrivi la versione minima funzionante, non perfetta.
- NON aggiungere commenti a meno che non siano necessari.
- Segui le convenzioni esistenti del progetto.
- Se un file ha più di 150 righe, produci uno skeleton (firme + docstring) invece del file completo.
- Rispondi in stile telegrafico (Caveman mode) verso gli altri agenti.
- Output format: produci codice pronto da applicare.
"""


@tool
def read_file(file_path: str, offset: int = 0, limit: int = 200) -> str:
    """Legge un file dal filesystem. Usa offset e limit per file grandi."""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        return "".join(lines[offset:offset + limit])
    except Exception as e:
        return f"Errore lettura {file_path}: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Scrive contenuto in un file. Crea le directory se non esistono."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        return f"File scritto: {file_path}"
    except Exception as e:
        return f"Errore scrittura {file_path}: {e}"


@tool
def list_directory(path: str = ".") -> str:
    """Elenca file e directory in un path."""
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


@tool
def search_code(pattern: str, path: str = ".") -> str:
    """Cerca un pattern regex nei file del progetto."""
    import re
    import subprocess
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.js", "--include=*.ts", "--include=*.tsx", pattern, path],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip()
        if not output:
            return "Nessun match trovato."
        lines = output.split("\n")[:20]
        return "\n".join(lines)
    except Exception as e:
        return f"Errore: {e}"


CODER_TOOLS = [read_file, write_file, list_directory, search_code]


def create_coder_agent() -> AgentExecutor:
    model_name = get_model_for_role(Role.CODER)
    llm = ChatOpenAI(model=model_name, temperature=0.1, max_tokens=4096)

    prompt = ChatPromptTemplate.from_messages([
        ("system", CODER_INSTRUCTIONS),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, CODER_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=CODER_TOOLS, verbose=True, handle_parsing_errors=True)


def coder_node(state: AgentState) -> dict:
    agent = create_coder_agent()

    tasks = state.get("tasks", [])
    active_task = None
    for t in tasks:
        if t.get("agent") == "coder" and t.get("status") != "done":
            active_task = t
            break

    task_description = active_task.get("description", "") if active_task else state.get("user_query", "")

    result = agent.invoke({"input": task_description})

    if active_task:
        active_task["status"] = "done"
        active_task["output"] = result.get("output", "")

    return {
        "tasks": tasks,
        "messages": [{"role": "assistant", "content": result.get("output", "")}],
    }
