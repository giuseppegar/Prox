from typing import Any

from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
import httpx

from prox.graph.state import AgentState
from prox.llm import get_model_for_role, Role, create_llm

RESEARCHER_INSTRUCTIONS = """Sei un agente Researcher. Unico con accesso a internet e ricerca.

REGOL:
- Cerca documentazione aggiornata, changelog, best practice.
- Verifica versioni librerie e breaking changes.
- 5-minute rule: inizia solo 5 minuti di ricerca, poi riporta.
- Output in stile Caveman: solo fatti essenziali.
- Segnala CVE, deprecation, alternative migliori.
"""


@tool
def web_search(query: str) -> str:
    """Cerca sul web documentazione e informazioni aggiornate."""
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        with httpx.Client(timeout=15.0) as client:
            response = client.get(search_url, headers={"User-Agent": "Prox/0.1"})
            if response.status_code != 200:
                return f"Search error: HTTP {response.status_code}"
            text = response.text[:4000]
            return f"Risultati ricerca per '{query}'\n{text}"
    except Exception as e:
        return f"Errore ricerca web: {e}"


@tool
def fetch_docs(url: str) -> str:
    """Scarica documentazione da un URL."""
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            response = client.get(url, headers={"User-Agent": "Prox/0.1"})
            if response.status_code != 200:
                return f"HTTP {response.status_code}"
            text = response.text[:5000]
            return text
    except Exception as e:
        return f"Errore fetch: {e}"


@tool
def check_package_version(package_name: str) -> str:
    """Controlla l'ultima versione di un pacchetto PyPI."""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"https://pypi.org/pypi/{package_name}/json")
            if response.status_code != 200:
                return f"Pacchetto '{package_name}' non trovato su PyPI."
            data = response.json()
            version = data.get("info", {}).get("version", "sconosciuta")
            summary = data.get("info", {}).get("summary", "")
            return f"📦 {package_name}@{version}: {summary}"
    except Exception as e:
        return f"Errore: {e}"


RESEARCHER_TOOLS = [web_search, fetch_docs, check_package_version]


def create_researcher_agent() -> AgentExecutor:
    model_name = get_model_for_role(Role.RESEARCHER)
    llm = create_llm(model_name, temperature=0.1, max_tokens=4096)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RESEARCHER_INSTRUCTIONS),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, RESEARCHER_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=RESEARCHER_TOOLS, verbose=True,
                         handle_parsing_errors=True, max_iterations=3, max_execution_time=60)


def researcher_node(state: AgentState) -> dict:
    agent = create_researcher_agent()

    tasks = state.get("tasks", [])
    active_task = None
    for t in tasks:
        if t.get("agent") == "researcher" and t.get("status") != "done":
            active_task = t
            break

    task_description = active_task.get("description", "") if active_task else state.get("user_query", "")

    try:
        result = agent.invoke({"input": task_description})
    except Exception as e:
        result = {"output": f"Ricerca parziale. Errore: {e}"}

    if active_task:
        active_task["status"] = "done"
        active_task["output"] = result.get("output", "")

    return {
        "tasks": tasks,
        "messages": [{"role": "assistant", "content": result.get("output", "")}],
    }
