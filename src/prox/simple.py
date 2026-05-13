"""Simple agent mode: single agent with tools, no swarm orchestration.

Fast, lightweight, direct. Use for simple tasks. Use /swarm for complex tasks.
"""
import os
import subprocess
from typing import Optional

from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from prox.llm import create_llm


SIMPLE_INSTRUCTIONS = """Sei un agente di coding. Esegui il task dell'utente.

TOOL:
- write_file(path, content): crea o sovrascrivi un file
- read_file(path): leggi un file
- list_directory(path): elenca directory
- search_code(pattern, path): cerca codice con grep
- run_command(cmd): esegui comando shell

REGOL:
- Per CREARE un file: write_file subito, non esplorare.
- Per MODIFICARE: prima leggi il file, poi scrivi.
- MVP: versione minima funzionante.
- NON aggiungere commenti al codice.
- Rispondi in italiano, stile telegrafico.
"""


@tool
def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Legge un file dal filesystem."""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        return "".join(lines[offset:offset + limit])
    except Exception as e:
        return f"Errore: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Crea o sovrascrive un file. Crea directory se non esistono."""
    try:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)
        return f"OK: {file_path}"
    except Exception as e:
        return f"Errore: {e}"


@tool
def list_directory(path: str = ".") -> str:
    """Elenca file e directory."""
    try:
        entries = sorted(os.listdir(path))
        result = []
        for e in entries:
            full = os.path.join(path, e)
            result.append(f"[DIR] {e}" if os.path.isdir(full) else f"[FILE] {e}")
        return "\n".join(result) if result else "(vuota)"
    except Exception as e:
        return f"Errore: {e}"


@tool
def search_code(pattern: str, path: str = ".") -> str:
    """Cerca un pattern nel codice con grep."""
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.js", "--include=*.ts",
             "--include=*.tsx", "--include=*.md", "--include=*.json",
             pattern, path],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip()
        return output[:3000] if output else "Nessun match."
    except Exception as e:
        return f"Errore: {e}"


@tool
def run_command(command: str) -> str:
    """Esegue un comando shell."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60
        )
        out = result.stdout.strip() or result.stderr.strip()
        return out[:3000] if out else "(nessun output)"
    except Exception as e:
        return f"Errore: {e}"


SIMPLE_TOOLS = [read_file, write_file, list_directory, search_code, run_command]


def create_simple_agent(model: str = "deepseek/deepseek-v4-flash") -> AgentExecutor:
    """Crea un agente semplice con tool, senza swarm."""
    llm = create_llm(model, temperature=0.1, max_tokens=4096)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SIMPLE_INSTRUCTIONS),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, SIMPLE_TOOLS, prompt)
    return AgentExecutor(
        agent=agent, tools=SIMPLE_TOOLS, verbose=True,
        handle_parsing_errors=True, return_intermediate_steps=True,
    )


def run_simple(query: str, project_dir: Optional[str] = None, model: str = "deepseek/deepseek-v4-flash") -> str:
    """Esegue un task in modalità semplice e restituisce l'output."""
    agent = create_simple_agent(model)
    cwd = project_dir or os.getcwd()
    full_input = f"Project directory: {cwd}\nTask: {query}"
    result = agent.invoke({"input": full_input})
    return result.get("output", "")
