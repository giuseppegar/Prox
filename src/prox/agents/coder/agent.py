"""Coder agent: writes/modifies files. MVP mentality."""
import os
from typing import Any

from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from prox.graph.state import AgentState
from prox.llm import get_model_for_role, Role, create_llm

CODER_INSTRUCTIONS = """Sei un agente Coder. Scrivi codice in Python e TypeScript/JavaScript.

REGOL FONDAMENTALI:
- Se il task e' CREARE un nuovo file: NON esplorare nulla. Chiama SUBITO write_file con il contenuto richiesto. Fine.
- Esplora (list_directory, read_file, search_code) SOLO se il task richiede di MODIFICARE codice esistente.
- MVP mentality: versione minima funzionante, non perfetta.
- NON aggiungere commenti al codice.
- Segui le convenzioni esistenti del progetto.
- Rispondi in stile telegrafico.

FLUSSO per CREAZIONE:
  1. write_file(path, content) → fatto. Stop.

FLUSSO per MODIFICA:
  1. read_file o list_directory per capire il contesto
  2. write_file con la modifica
"""


@tool
def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Legge un file dal filesystem."""
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
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        return f"File scritto: {file_path}"
    except Exception as e:
        return f"Errore scrittura {file_path}: {e}"


@tool
def list_directory(path: str = ".") -> str:
    """Elenca file e directory in un percorso."""
    try:
        entries = sorted(os.listdir(path))
        result = []
        for e in entries:
            full = os.path.join(path, e)
            prefix = "[DIR]" if os.path.isdir(full) else "[FILE]"
            result.append(f"{prefix} {e}")
        return "\n".join(result) if result else "(directory vuota)"
    except Exception as e:
        return f"Errore: {e}"


@tool
def search_code(pattern: str, path: str = ".") -> str:
    """Cerca un pattern nei file del progetto usando grep."""
    import subprocess
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.js", "--include=*.ts",
             "--include=*.tsx", "--include=*.md", "--include=*.json",
             pattern, path],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip()
        return output[:3000] if output else "Nessun match trovato."
    except Exception as e:
        return f"Errore ricerca: {e}"


CODER_TOOLS = [read_file, write_file, list_directory, search_code]


def create_coder_agent() -> AgentExecutor:
    model_name = get_model_for_role(Role.CODER)
    llm = create_llm(model_name, temperature=0.1, max_tokens=4096)

    prompt = ChatPromptTemplate.from_messages([
        ("system", CODER_INSTRUCTIONS),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, CODER_TOOLS, prompt)
    return AgentExecutor(
        agent=agent, tools=CODER_TOOLS, verbose=True,
        handle_parsing_errors=True, return_intermediate_steps=True,
    )


def coder_node(state: AgentState) -> dict:
    agent = create_coder_agent()

    tasks = state.get("tasks", [])
    active_task = None
    for t in tasks:
        if t.get("agent") == "coder" and t.get("status") not in ("done", "blocked"):
            active_task = t
            break

    task_description = active_task.get("description", "") if active_task else state.get("user_query", "")
    project_dir = state.get("project_dir", ".")

    full_input = f"Task: {task_description}\nProject directory: {project_dir}"

    try:
        result = agent.invoke({"input": full_input})
        output = result.get("output", "")
    except Exception as e:
        output = f"Eseguito parzialmente. Errore: {e}"

    if active_task:
        active_task["status"] = "done"
        active_task["output"] = output

    return {
        "tasks": tasks,
        "messages": [{"role": "assistant", "content": output}],
    }
