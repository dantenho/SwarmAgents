"""Supervisor-and-workers swarm workflow built on LangGraph."""

from __future__ import annotations

from typing import Any, TypedDict, cast

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from swarmagents.providers import get_chat_model

DEFAULT_SUPERVISOR_PROMPT = (
    "You are the supervisor of a research swarm. Plan the work, delegate "
    "each step to exactly one of the available workers, and synthesize "
    "their results into a final answer. Use tools only through delegation."
)

WORKER_PROMPTS = {
    "researcher": (
        "You are a web research specialist. Given a research question, "
        "decide what facts are needed and report them clearly."
    ),
    "analyst": (
        "You are a quantitative analyst. Take the supplied findings and "
        "produce structured insights, trade-offs, and recommendations."
    ),
    "writer": (
        "You are a technical writer. Turn the analysis into a clear, "
        "well-structured final report for the end user."
    ),
}

_WORKER_DESCRIPTIONS = {
    "researcher": "Gathers facts and background for a question.",
    "analyst": "Interprets findings and produces structured insights.",
    "writer": "Drafts the polished final report.",
}


@tool
def delegate(worker: str, instruction: str) -> dict[str, str]:
    """Delegate a unit of work to a named swarm worker.

    Args:
        worker: One of the registered worker names.
        instruction: The concrete task for that worker.
    """
    return {"worker": worker, "instruction": instruction}


@tool
def finish(summary: str) -> dict[str, str]:
    """Finish the workflow by writing the final summary for the user."""
    return {"summary": summary}


class SwarmState(TypedDict, total=False):
    """State shared across the swarm workflow."""

    question: str
    messages: list[Any]
    worker: str
    instruction: str
    result: str
    scratchpad: list[dict[str, str]]
    answer: str


def build_worker(model: BaseChatModel, role: str) -> Any:
    """Create a worker agent node for a role."""

    @tool
    def submit_result(result: str) -> dict[str, str]:
        """Submit this worker's result back to the supervisor."""
        return {"result": result}

    return create_agent(
        model,
        tools=[submit_result],
        system_prompt=(
            f"{WORKER_PROMPTS[role]}\n"
            "When done, call 'submit_result' with your complete findings."
        ),
        name=role,
    )


def build_supervisor(
    model: BaseChatModel,
    workers: dict[str, Any],
) -> Any:
    """Create the supervisor agent that routes work to workers."""
    delegate_options = ", ".join(
        f"'{name}' ({_WORKER_DESCRIPTIONS[name]})" for name in workers
    )
    return create_agent(
        model,
        tools=[delegate, finish],
        system_prompt=(
            f"{DEFAULT_SUPERVISOR_PROMPT}\n"
            f"Available workers: {delegate_options}.\n"
            "Delegate with 'delegate', then call 'finish' with the final "
            "answer once the workers' results are sufficient."
        ),
        name="supervisor",
    )


def _route(state: SwarmState) -> str:
    if state.get("worker"):
        return state["worker"]
    return "supervisor"


def build_swarm(
    supervisor_model: BaseChatModel,
    worker_models: dict[str, BaseChatModel] | None = None,
) -> Any:
    """Assemble the supervisor-worker swarm as a compiled LangGraph."""
    roles = tuple(WORKER_PROMPTS)
    models = worker_models or {role: supervisor_model for role in roles}

    supervisor = build_supervisor(supervisor_model, {r: models[r] for r in roles})
    agents = {role: build_worker(models[role], role) for role in roles}

    graph = StateGraph(cast(Any, SwarmState))
    graph.add_node("supervisor", supervisor)
    for role, agent in agents.items():
        graph.add_node(role, agent)
        graph.add_edge(role, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route,
        {role: role for role in roles} | {"supervisor": END},
    )
    graph.add_edge(START, "supervisor")
    return graph.compile()


def run_swarm(
    question: str,
    provider: str = "openai",
    model: str | None = None,
    worker_provider: str | None = None,
    worker_model: str | None = None,
    **model_kwargs: Any,
) -> str:
    """Run the swarm workflow on a question using named LLM providers.

    Args:
        question: The user question for the swarm.
        provider: Provider for the supervisor model.
        model: Supervisor model name (defaults to the provider default).
        worker_provider: Provider for the workers (defaults to provider).
        worker_model: Worker model name (defaults to provider default).
        **model_kwargs: Extra kwargs forwarded to the chat models.
    """
    supervisor_model = get_chat_model(provider, model, **model_kwargs)
    worker_provider_name = worker_provider or provider
    worker_models: dict[str, BaseChatModel] | None = None
    if worker_provider_name != provider or worker_model is not None:
        shared_worker_model = get_chat_model(
            worker_provider_name, worker_model, **model_kwargs
        )
        worker_models = {role: shared_worker_model for role in WORKER_PROMPTS}

    swarm = build_swarm(supervisor_model, worker_models)
    final_state = swarm.invoke({"messages": [("user", question)]})
    return str(final_state.get("answer") or final_state["messages"][-1].content)
