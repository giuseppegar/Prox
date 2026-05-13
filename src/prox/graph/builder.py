import os
from typing import Any, Callable, Optional

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState


def create_prox_graph(
    main_orch_node: Callable,
    mini_orch_node: Callable,
    director_node: Callable,
    worker_router: Callable,
    coder_node: Callable,
    reviewer_node: Callable,
    researcher_node: Callable,
    tester_node: Callable,
    memory_node: Callable,
    checkpointer: Optional[Any] = None,
) -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("main_orchestrator", main_orch_node)
    workflow.add_node("mini_orchestrator", mini_orch_node)
    workflow.add_node("director", director_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("tester", tester_node)
    workflow.add_node("memory", memory_node)

    workflow.set_entry_point("main_orchestrator")

    workflow.add_conditional_edges(
        "main_orchestrator",
        _route_main_orch,
        {
            "mini_orchestrator": "mini_orchestrator",
            "director": "director",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "mini_orchestrator",
        _route_mini_orch,
        {
            "mini_orchestrator": "mini_orchestrator",
            "director": "director",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "director",
        _route_director,
        {
            "coder": "coder",
            "reviewer": "reviewer",
            "researcher": "researcher",
            "tester": "tester",
            "memory": "memory",
            "main_orchestrator": "main_orchestrator",
            "end": END,
        },
    )

    for worker in ["coder", "reviewer", "researcher", "tester", "memory"]:
        workflow.add_edge(worker, "director")

    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


def _route_main_orch(state: AgentState) -> str:
    needs_input = state.get("needs_user_input", False)
    if needs_input:
        return "end"

    next_agent = state.get("next_agent", "")
    if next_agent == "mini_orchestrator":
        return "mini_orchestrator"
    elif next_agent == "director":
        return "director"

    tasks = state.get("tasks", [])
    if tasks and all(t.get("status") == "done" for t in tasks):
        return "end"

    return "end"


def _route_mini_orch(state: AgentState) -> str:
    if state.get("consensus_reached", False):
        return "director"

    active = state.get("active_mini_orch_count", 0)
    if active > 0:
        return "mini_orchestrator"

    return "end"


def _route_director(state: AgentState) -> str:
    next_agent = state.get("next_agent", "")
    if next_agent in ("coder", "reviewer", "researcher", "tester", "memory"):
        return next_agent

    tasks = state.get("tasks", [])
    if all(t.get("status") == "done" for t in tasks):
        return "main_orchestrator"

    return "end"
