import subprocess
from typing import Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from prox.graph.state import AgentState
from prox.llm.models import ModelRegistry

TESTER_INSTRUCTIONS = """Sei un agente Tester. Scrivi ed esegui test automatici.

REGOL:
- Genera test per il codice prodotto dal Coder.
- Usa pytest per Python, jest/vitest per TypeScript.
- Esegui i test e riporta risultati (pass/fail/error).
- Se i test falliscono, riporta esattamente cosa e' fallito.
- Output in stile Caveman: solo risultati essenziali.
"""


@tool
def run_python_tests(test_path: str = ".") -> str:
    """Esegue test Python con pytest."""
    try:
        result = subprocess.run(
            ["pytest", test_path, "-v", "--tb=short"],
            capture_output=True, text=True, timeout=120,
        )
        output = result.stdout + result.stderr
        return output[:4000]
    except subprocess.TimeoutExpired:
        return "Test timeout dopo 120 secondi."
    except Exception as e:
        return f"Errore esecuzione test: {e}"


@tool
def run_command(command: str) -> str:
    """Esegue un comando shell e restituisce l'output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60
        )
        return result.stdout[:3000] or result.stderr[:3000] or "(nessun output)"
    except subprocess.TimeoutExpired:
        return "Comando timeout dopo 60 secondi."
    except Exception as e:
        return f"Errore: {e}"


TESTER_TOOLS = [run_python_tests, run_command]


def create_tester_agent(model_registry: ModelRegistry) -> AgentExecutor:
    model_name = model_registry.get_model("worker")
    llm = ChatOpenAI(model=model_name, temperature=0.0, max_tokens=4096)

    prompt = ChatPromptTemplate.from_messages([
        ("system", TESTER_INSTRUCTIONS),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, TESTER_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TESTER_TOOLS, verbose=True, handle_parsing_errors=True)


def tester_node(state: AgentState) -> dict:
    registry = ModelRegistry.from_config()
    agent = create_tester_agent(registry)

    tasks = state.get("tasks", [])
    active_task = None
    for t in tasks:
        if t.get("agent") == "tester" and t.get("status") != "done":
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
