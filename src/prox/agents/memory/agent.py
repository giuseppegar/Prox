import time
from typing import Any

from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from prox.graph.state import AgentState
from prox.llm import get_model_for_role, Role, create_llm
from prox.synapse import NeuralStore, AttentionEngine, HebbianGraph, DecayScheduler, ConsolidationLoop

MEMORY_INSTRUCTIONS = """Sei l'agente Memory. Custode della memoria neurale Synapse.

OPERAZIONI:
- INGEST: ricevi contenuti e li salvi come Memory Trace in ChromaDB.
  Decidi cosa e' importante, scarta il rumore.
- QUERY: cerchi nella memoria globale usando Attention Engine.
  Supporti query project-scoped e cross-project.
- CONSOLIDATE: astrai pattern da trace recenti.
- RETROSPETTIVA: analizzi cosa ha funzionato e cosa no nella sessione.
- LINT: health check della memoria (decadimento, orfani, contraddizioni).

REGOL:
- Tutto passa per te: sei l'unico che legge/scrive Synapse.
- Rispondi in stile Caveman verso gli altri agenti.
- I valori di credenziali NON vanno MAI salvati nelle trace.
- Usa project_id per namespacing.
"""


def create_memory_agent() -> AgentExecutor:
    model_name = get_model_for_role(Role.MEMORY)
    llm = create_llm(model_name, temperature=0.1, max_tokens=4096)

    prompt = ChatPromptTemplate.from_messages([
        ("system", MEMORY_INSTRUCTIONS),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, [], prompt)
    return AgentExecutor(agent=agent, tools=[], verbose=True, handle_parsing_errors=True)


def memory_node(state: AgentState) -> dict:
    agent = create_memory_agent()

    store = NeuralStore()
    attention = AttentionEngine(store)
    hebbian = HebbianGraph()
    decay = DecayScheduler()
    consolidation = ConsolidationLoop(store, hebbian)

    tasks = state.get("tasks", [])
    active_task = None
    for t in tasks:
        if t.get("agent") == "memory" and t.get("status") != "done":
            active_task = t
            break

    task_description = active_task.get("description", "") if active_task else "retrospettiva di sessione"

    project_id = state.get("project_id", "global")
    session_log = state.get("session_log", [])

    ingest_context = "\n".join(session_log[-20:]) if session_log else ""
    full_input = f"{task_description}\n\nSession context:\n{ingest_context}\nProject: {project_id}"

    try:
        result = agent.invoke({"input": full_input})
    except Exception as e:
        result = {"output": f"Memoria aggiornata parzialmente. Errore: {e}"}

    if ingest_context:
        from prox.synapse.store import MemoryTrace
        trace = MemoryTrace(
            content=ingest_context,
            project_id=project_id,
            trace_type="session_log",
            metadata={"source": "memory_agent", "timestamp": time.time()},
        )
        store.add(trace)

    consolidation.consolidate(project_id=project_id)
    decay.prune_candidates(store.list_all(project_id=project_id))

    if active_task:
        active_task["status"] = "done"
        active_task["output"] = result.get("output", "")

    return {
        "tasks": tasks,
        "messages": [{"role": "assistant", "content": result.get("output", "")}],
    }
