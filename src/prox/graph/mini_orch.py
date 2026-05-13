from typing import Any
from uuid import uuid4

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from prox.graph.state import AgentState, Mode
from prox.llm import get_model_for_role, Role, create_llm
from prox.synapse.freshness import FreshnessLayer

MINI_ORCH_INSTRUCTIONS = """Sei un Mini Orchestrator. Collabori con altri Mini Orchestrator per raffinare il piano.

IL TUO COMPITO:
- Ricevi il piano dal Main Orchestrator e lo discuti con gli altri Mini Orchestrator.
- Fai PRE-MORTEM: immagina che il task fallisca. Perche'?
- Proponi CHUNKING: raggruppa task simili per blocco.
- Sfida le priorita' con MoSCoW: e' davvero Must? O Could?
- Verifica FRESHNESS: le librerie citate sono aggiornate?
- Quando raggiungete il consenso, trasmettete il piano al Director.

OUTPUT FORMAT:
CONSENSUS: yes/no
PLAN: descrizione del piano convergente
WARNINGS: rischi identificati
CHUNKS: blocchi di task proposti
PARKING_LOT: task rimandabili
"""


def create_mini_orch_agent() -> AgentExecutor:
    model_name = get_model_for_role(Role.MINI_ORCHESTRATOR)
    llm = create_llm(model_name, temperature=0.3, max_tokens=4096)

    prompt = ChatPromptTemplate.from_messages([
        ("system", MINI_ORCH_INSTRUCTIONS),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [], prompt)
    return AgentExecutor(agent=agent, tools=[], verbose=True, handle_parsing_errors=True)


def mini_orchestrator_node(state: AgentState) -> dict:
    agent = create_mini_orch_agent()

    user_query = state.get("user_query", "")
    tasks = state.get("tasks", [])
    session_log = state.get("session_log", [])

    iteration = len([m for m in session_log if "MiniOrch" in str(m)]) + 1
    active_count = state.get("active_mini_orch_count", 3)

    debate_context = _build_debate_context(iteration, tasks, user_query, session_log)

    try:
        result = agent.invoke({"input": debate_context})
        output = result.get("output", "")
    except Exception as e:
        output = f"CONSENSUS: yes\nPLAN: esecuzione diretta\nWARNINGS: {e}\nCHUNKS:\n- {user_query}\nPARKING_LOT: none"

    session_log.append(f"[MiniOrch #{iteration}] {output[:500]}")

    consensus = "CONSENSUS: yes" in output or "CONSENSUS:YES" in output.upper()
    max_iterations = 3

    if consensus or iteration >= max_iterations:
        plan = _extract_plan(output, tasks, user_query)
        return {
            "consensus_reached": True,
            "director_plan": plan,
            "tasks": _plan_to_tasks(plan),
            "parking_lot": _extract_parking_lot(output),
            "session_log": session_log,
            "messages": [{"role": "assistant", "content": output}],
        }

    return {
        "consensus_reached": False,
        "active_mini_orch_count": active_count,
        "session_log": session_log,
        "messages": [{"role": "assistant", "content": output}],
    }


def _build_debate_context(iteration: int, tasks: list, query: str, log: list[str]) -> str:
    context = f"User query: {query}\n"
    if tasks:
        context += f"Tasks attuali: {tasks}\n"

    freshness = FreshnessLayer()
    check = freshness.pre_task_check(query)
    if check["stale"]:
        context += f"\nPACCHETTI DA AGGIORNARE (TTL scaduto): {', '.join(check['stale'])}\n"
        context += "Il Researcher deve verificare le ultime versioni di questi pacchetti.\n"
    if check["warnings"]:
        context += f"\nAVVISI FRESHNESS:\n"
        for w in check["warnings"]:
            context += f"  - {w}\n"

    context += f"Turno dibattito: {iteration}/3\n"
    context += "Analizza il piano e rispondi con il formato CONSENSUS/PLAN/WARNINGS/CHUNKS/PARKING_LOT.\n"
    context += "Prima di proporre librerie, verifica il Freshness report sopra.\n"
    if iteration > 1:
        context += "Dibattito precedente:\n" + "\n".join(log[-5:])
    return context


def _extract_plan(output: str, tasks: list, query: str) -> dict:
    return {
        "original_query": query,
        "mini_orch_output": output,
        "chunks": _extract_chunks(output),
    }


def _extract_chunks(output: str) -> list[str]:
    chunks = []
    in_chunks = False
    for line in output.split("\n"):
        if "CHUNKS:" in line:
            in_chunks = True
            continue
        if in_chunks and line.strip().startswith("-"):
            chunks.append(line.strip()[1:].strip())
        elif in_chunks and not line.strip().startswith("-"):
            in_chunks = False
    return chunks or ["Implement task"]


def _plan_to_tasks(plan: dict) -> list[dict]:
    chunks = plan.get("chunks", ["Implement task"])
    tasks = []
    agents = ["coder", "reviewer", "tester", "researcher", "memory"]
    for i, chunk in enumerate(chunks):
        tasks.append({
            "id": str(uuid4()),
            "description": chunk,
            "agent": agents[i % len(agents)],
            "status": "pending",
            "priority": "high" if i == 0 else "medium",
            "output": "",
            "attempts": 0,
        })
    return tasks


def _extract_parking_lot(output: str) -> list[str]:
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
