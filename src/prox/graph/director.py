import time
from typing import Any, Optional
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

from prox.graph.state import AgentState, Mode, Task
from prox.llm.models import ModelRegistry
from prox.vault.store import CredentialStore

MAX_ATTEMPTS = 3

DIRECTOR_INSTRUCTIONS = """Sei il Director Orchestrator. Ricevi il piano dai Mini Orchestrators e dispatchi task ai Worker.

TECNICHE OBBLIGATORIE:
- TIMEBOXING: max 3 tentativi per task. Dopo 3 fallimenti, il task va in BLOCKED.
- EAT THE FROG: il task piu' difficile/critico va eseguito per primo.
- CHUNKING: raggruppa task simili ed eseguili in blocco per ridurre context switch.
- MODE ENFORCEMENT: rispetta la modalita' (plan/auto/interactive).

MODALITA':
- PLAN: blocca qualsiasi write/execute. Solo analisi e proposte.
- AUTO: esegui tutto senza chiedere.
- INTERACTIVE: chiedi solo per operazioni fuori scope, distruttive, o credenziali.

OUTPUT: assegna il prossimo agente e task nel formato:
NEXT_AGENT: <nome>
TASK: <descrizione>
PRIORITY: <high/medium/low>
"""


def create_director_agent(model_registry: ModelRegistry) -> AgentExecutor:
    model_name = model_registry.get_model("director")
    llm = ChatOpenAI(model=model_name, temperature=0.1, max_tokens=4096)

    prompt = ChatPromptTemplate.from_messages([
        ("system", DIRECTOR_INSTRUCTIONS),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [], prompt)
    return AgentExecutor(agent=agent, tools=[], verbose=True, handle_parsing_errors=True)


def director_node(state: AgentState) -> dict:
    registry = ModelRegistry.from_config()
    agent = create_director_agent(registry)

    mode = Mode(state.get("mode", "interactive"))
    project_dir = state.get("project_dir", ".")
    tasks = state.get("tasks", [])
    parking_lot = state.get("parking_lot", [])
    director_plan = state.get("director_plan", {})

    if not tasks:
        return {"next_agent": "main_orchestrator", "tasks": tasks}

    blocked_count = sum(1 for t in tasks if t.get("status") == "blocked")
    done_count = sum(1 for t in tasks if t.get("status") == "done")

    if done_count + blocked_count >= len(tasks):
        return {
            "next_agent": "main_orchestrator",
            "tasks": tasks,
            "messages": [{"role": "assistant", "content": f"Tutti i task completati. {done_count} done, {blocked_count} blocked."}],
        }

    pending = [t for t in tasks if t.get("status") not in ("done", "blocked")]
    if not pending:
        return {"next_agent": "main_orchestrator", "tasks": tasks}

    pending.sort(key=lambda t: (
        0 if t.get("priority") == "high" else 1 if t.get("priority") == "medium" else 2,
        t.get("attempts", 0),
    ))

    current = pending[0]
    attempts = current.get("attempts", 0)

    if attempts >= MAX_ATTEMPTS:
        current["status"] = "blocked"
        parking_lot.append(f"BLOCKED: {current.get('description', '')} dopo {MAX_ATTEMPTS} tentativi")
        return {
            "tasks": tasks,
            "parking_lot": parking_lot,
            "next_agent": "main_orchestrator",
            "messages": [{"role": "assistant", "content": f"Task '{current.get('description', '')}' bloccato dopo {MAX_ATTEMPTS} tentativi."}],
        }

    current["attempts"] = attempts + 1
    current["status"] = "in_progress"
    next_agent = current.get("agent", "coder")

    scope_check = _check_scope(current, mode, project_dir)
    if scope_check:
        return scope_check

    return {
        "tasks": tasks,
        "next_agent": next_agent,
        "messages": [{"role": "assistant", "content": f"Director dispatcha a {next_agent}: {current.get('description', '')} (tentativo {attempts + 1}/{MAX_ATTEMPTS})"}],
    }


def _check_scope(task: dict, mode: Mode, project_dir: str) -> Optional[dict]:
    if mode == Mode.PLAN:
        if task.get("requires_write") or task.get("requires_execute"):
            task["status"] = "blocked"
            return {
                "tasks": task.get("_tasks", []),
                "needs_user_input": True,
                "user_question": f"Plan mode: '{task.get('description', '')}' richiede scrittura/esecuzione. Vuoi passare a interactive?",
                "messages": [{"role": "assistant", "content": f"PLAN MODE BLOCK: '{task.get('description')}' richiede scrittura/esecuzione."}],
            }
    return None
