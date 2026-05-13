from enum import Enum
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class Mode(Enum):
    PLAN = "plan"
    AUTO = "auto"
    INTERACTIVE = "interactive"


class AgentStatus(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    WORKING = "working"
    WAITING = "waiting"
    DONE = "done"
    BLOCKED = "blocked"


class Task(TypedDict, total=False):
    id: str
    description: str
    agent: str
    status: str
    priority: str
    output: str
    attempts: int


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str
    project_dir: str
    project_id: str
    tasks: list[dict]
    parking_lot: list[str]
    session_log: list[str]
    context_tokens: int
    max_tokens: int
    warn_tokens: int
    auto_save_tokens: int
    active_mini_orch_count: int
    consensus_reached: bool
    director_plan: dict
    worker_results: dict
    next_agent: str
    user_query: str
    needs_user_input: bool
    user_question: str
    needs_passphrase: bool
