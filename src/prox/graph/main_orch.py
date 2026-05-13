import os
import time
from typing import Any
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

from prox.graph.state import AgentState, Mode
from prox.llm import get_model_for_role, Role

MAIN_ORCH_INSTRUCTIONS = """Sei il Main Orchestrator di Prox. Sei l'unico punto di contatto con l'utente.

TECNICHE OBBLIGATORIE:
- WBS (Work Breakdown Structure): scomponi il task in deliverable, poi in sotto-task, finche' ogni pezzo e' completabile in una sessione.
- GTD INBOX: raccogli tutto, processa: elimina, delega, fai ora (<2 min), o schedula.
- SMART GOALS: ogni task deve essere Specifico, Misurabile, Raggiungibile, Rilevante, con Scadenza.
- MoSCoW: prioritizza Must have / Should have / Could have / Won't have.
- PARKING LOT: idee fuori perimetro vanno in lista separata, non inseguite.
- 2-MINUTE RULE: se un sotto-task richiede <2 minuti di lavoro, eseguilo subito (non delegare).

COMPORTAMENTO:
- Se il task e' complesso (3+ sotto-task), avvia Mini Orchestrators per dibattito.
- Se il task e' semplice (1-2 sotto-task), invia direttamente al Director.
- NON esegui mai codice direttamente. Il tuo compito e' pianificare e delegare.

OUTPUT FORMAT:
COMPLEXITY: simple/medium/complex
MINI_ORCH_NEEDED: yes/no (yes se complex)
TASKS: lista di task delegabili
PARKING_LOT: idee rimandabili
"""


def create_main_orch_agent() -> AgentExecutor:
    model_name = get_model_for_role(Role.MAIN_ORCHESTRATOR)
    llm = ChatOpenAI(model=model_name, temperature=0.2, max_tokens=4096)

    prompt = ChatPromptTemplate.from_messages([
        ("system", MAIN_ORCH_INSTRUCTIONS),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [], prompt)
    return AgentExecutor(agent=agent, tools=[], verbose=True, handle_parsing_errors=True)


def main_orchestrator_node(state: AgentState) -> dict:
    agent = create_main_orch_agent()

    mode = Mode(state.get("mode", "interactive"))
    project_dir = state.get("project_dir", os.getcwd())
    user_query = state.get("user_query", "")
    tasks = state.get("tasks", [])
    parking_lot = state.get("parking_lot", [])
    session_log = state.get("session_log", [])

    if state.get("needs_user_input", False):
        return {"next_agent": "end"}

    if tasks and all(t.get("status") == "done" for t in tasks):
        session_log.append(f"[{time.strftime('%H:%M')}] Sessione completata. {len(tasks)} task done.")
        return {
            "tasks": tasks,
            "parking_lot": parking_lot,
            "session_log": session_log,
            "next_agent": "end",
            "messages": [{"role": "assistant", "content": f"Sessione completata. {len(tasks)} task completati. Parking lot: {len(parking_lot)} idee rimandate."}],
        }

    if tasks and any(t.get("status") == "blocked" for t in tasks):
        blocked = [t for t in tasks if t.get("status") == "blocked"]
        blocked_desc = "\n".join(f"- {t.get('description','')}" for t in blocked)
        return {
            "tasks": tasks,
            "needs_user_input": True,
            "user_question": f"Alcuni task sono bloccati:\n{blocked_desc}\n\nCosa vuoi fare?",
            "next_agent": "end",
            "messages": [{"role": "assistant", "content": "Task bloccati. In attesa di input utente."}],
        }

    if not tasks:
        return _plan_task(user_query, project_dir, session_log, agent)

    pending = [t for t in tasks if t.get("status") not in ("done", "blocked")]
    if pending:
        pending.sort(key=lambda t: 0 if t.get("priority") == "high" else 1)
        current = pending[0]
        complexity = _estimate_complexity(tasks)

        if complexity == "complex":
            return {
                "tasks": tasks,
                "next_agent": "mini_orchestrator",
                "active_mini_orch_count": 3,
                "session_log": session_log,
                "messages": [{"role": "assistant", "content": f"Task complesso. Avvio {3} Mini Orchestrator per dibattito."}],
            }
        else:
            return {
                "tasks": tasks,
                "next_agent": "director",
                "session_log": session_log,
            }

    return {"next_agent": "end", "tasks": tasks}


def _plan_task(query: str, project_dir: str, session_log: list, agent: AgentExecutor) -> dict:
    context = f"User query: {query}\nProject directory: {project_dir}\n"
    context += "Analizza, applica WBS e SMART, e produci un piano."

    result = agent.invoke({"input": context})
    output = result.get("output", "")

    is_complex = "COMPLEXITY: complex" in output or "COMPLEXITY:COMPLEX" in output.upper()
    is_medium = "COMPLEXITY: medium" in output or "COMPLEXITY:MEDIUM" in output.upper()
    need_mini = "MINI_ORCH_NEEDED: yes" in output or "MINI_ORCH_NEEDED:YES" in output.upper()

    tasks = _extract_tasks(output, query)
    parking_lot = _extract_pl(output)

    session_log.append(f"[{time.strftime('%H:%M')}] MainOrch: WBS su '{query}' → {len(tasks)} task, {len(parking_lot)} in parking lot.")

    if need_mini or is_complex:
        return {
            "user_query": query,
            "tasks": tasks,
            "parking_lot": parking_lot,
            "session_log": session_log,
            "next_agent": "mini_orchestrator",
            "active_mini_orch_count": 3,
            "messages": [{"role": "assistant", "content": output}],
        }
    else:
        return {
            "user_query": query,
            "tasks": tasks,
            "parking_lot": parking_lot,
            "session_log": session_log,
            "next_agent": "director",
            "messages": [{"role": "assistant", "content": output}],
        }


def _extract_tasks(output: str, query: str) -> list[dict]:
    tasks = []
    in_tasks = False
    for line in output.split("\n"):
        if "TASKS:" in line:
            in_tasks = True
            continue
        if in_tasks and line.strip().startswith("-"):
            desc = line.strip()[1:].strip()
            tasks.append({
                "id": str(uuid4()),
                "description": desc,
                "agent": "coder",
                "status": "pending",
                "priority": "high" if not tasks else "medium",
                "output": "",
                "attempts": 0,
            })
        elif in_tasks and not line.strip().startswith("-"):
            in_tasks = False

    if not tasks:
        tasks.append({
            "id": str(uuid4()),
            "description": query,
            "agent": "coder",
            "status": "pending",
            "priority": "high",
            "output": "",
            "attempts": 0,
        })
    return tasks


def _extract_pl(output: str) -> list[str]:
    items = []
    in_pl = False
    for line in output.split("\n"):
        if "PARKING_LOT:" in line:
            in_pl = True
            continue
        if in_pl and line.strip().startswith("-"):
            items.append(line.strip()[1:].strip())
        elif in_pl and not line.strip().startswith("-"):
            in_pl = False
    return items


def _estimate_complexity(tasks: list) -> str:
    if len(tasks) > 3:
        return "complex"
    elif len(tasks) > 1:
        return "medium"
    return "simple"
