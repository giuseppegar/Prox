import subprocess
from typing import Any

from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from prox.graph.state import AgentState
from prox.llm import get_model_for_role, Role, create_llm

REVIEWER_INSTRUCTIONS = """Sei un agente Reviewer aggressivo. Il tuo compito e' trovare problemi, non complimenti.

REGOL:
- Analizza diff e codice per bug, problemi sicurezza, violazioni best practice.
- Pre-mortem: immagina che il codice fallisca in produzione. Perche'?
- NON riscrivere il codice, solo segnalare problemi.
- Rispondi in stile telegrafico (Caveman mode).
- Output format: lista di finding con criticita' (CRITICAL/HIGH/MEDIUM/LOW).
"""


@tool
def analyze_diff(file_path: str) -> str:
    """Analizza il diff di un file via git."""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD", "--", file_path],
            capture_output=True, text=True, timeout=10
        )
        diff = result.stdout.strip()
        if not diff:
            return f"Nessuna modifica in {file_path}"
        return diff
    except Exception as e:
        return f"Errore analisi diff: {e}"


@tool
def read_file_review(file_path: str) -> str:
    """Legge un file per la review."""
    try:
        with open(file_path, "r") as f:
            content = f.read()
        max_len = 5000
        if len(content) > max_len:
            content = content[:max_len] + f"\n... [truncated, {len(content) - max_len} more chars]"
        return content
    except Exception as e:
        return f"Errore lettura: {e}"


REVIEWER_TOOLS = [analyze_diff, read_file_review]


def create_reviewer_agent() -> AgentExecutor:
    model_name = get_model_for_role(Role.REVIEWER)
    llm = create_llm(model_name, temperature=0.0, max_tokens=4096)

    prompt = ChatPromptTemplate.from_messages([
        ("system", REVIEWER_INSTRUCTIONS),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, REVIEWER_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=REVIEWER_TOOLS, verbose=True,
                         handle_parsing_errors=True, max_iterations=3, max_execution_time=60)


def reviewer_node(state: AgentState) -> dict:
    agent = create_reviewer_agent()

    tasks = state.get("tasks", [])
    active_task = None
    for t in tasks:
        if t.get("agent") == "reviewer" and t.get("status") != "done":
            active_task = t
            break

    task_description = active_task.get("description", "") if active_task else state.get("user_query", "")

    try:
        result = agent.invoke({"input": task_description})
    except Exception as e:
        result = {"output": f"Analisi parziale. Errore: {e}"}

    if active_task:
        active_task["status"] = "done"
        active_task["output"] = result.get("output", "")

    return {
        "tasks": tasks,
        "messages": [{"role": "assistant", "content": result.get("output", "")}],
    }
