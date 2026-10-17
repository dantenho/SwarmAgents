"""Handoff-based LangGraph Swarm: worker agents transfer control between each other.

Uses the official `langgraph-swarm` multi-agent pattern where each agent can
invoke a handoff tool to route the conversation to another agent.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph_swarm import create_handoff_tool, create_swarm

from swarmagents.providers import get_chat_model

AGENT_ROLES: dict[str, str] = {
    "researcher": (
        "You are a web research specialist. Answer factual questions and "
        "gather background. Delegate writing or data analysis to other agents."
    ),
    "analyst": (
        "You are a quantitative analyst. Interpret findings, compute "
        "insights, and structure recommendations."
    ),
    "writer": (
        "You are a technical writer. Compose the final user-facing report "
        "from the provided findings. Do not delegate."
    ),
}


def build_swarm_agent(
    role: str,
    model: BaseChatModel,
    handoff_tools: list[BaseTool],
    extra_tools: list[BaseTool] | None = None,
) -> Any:
    """Create one swarm agent with handoff tools to the other agents."""
    others = ", ".join(sorted(set(AGENT_ROLES) - {role}))
    return create_agent(
        model,
        tools=list(extra_tools or []) + handoff_tools,
        system_prompt=(
            f"{AGENT_ROLES[role]}\n"
            f"When the request belongs to {others}, use the transfer tool. "
            "Once you have fully handled your part, reply directly to the user."
        ),
        name=role,
    )


def create_langgraph_swarm(
    provider: str = "openai",
    model: str | None = None,
    roles: tuple[str, ...] = tuple(AGENT_ROLES),
    **model_kwargs: Any,
) -> Any:
    """Build a handoff swarm where control passes agent-to-agent.

    Args:
        provider: LLM API provider name from the 20-provider registry.
        model: Model name override (defaults to the provider's default).
        roles: Which agent roles to include in the swarm.
        **model_kwargs: Extra kwargs forwarded to the chat model.
    """
    chat_model = get_chat_model(provider, model, **model_kwargs)

    handoffs: dict[str, list[BaseTool]] = {}
    for role in roles:
        handoffs[role] = [
            create_handoff_tool(
                agent_name=other,
                description=AGENT_ROLES[other],
            )
            for other in roles
            if other != role
        ]

    agents = [
        build_swarm_agent(role, chat_model, handoffs[role])
        for role in roles
    ]
    return create_swarm(
        agents,
        default_active_agent=roles[0],
    )


def run_langgraph_swarm(
    message: str,
    provider: str = "openai",
    model: str | None = None,
    roles: tuple[str, ...] = tuple(AGENT_ROLES),
    **model_kwargs: Any,
) -> str:
    """Run the handoff swarm on a single user message.

    Args:
        message: The user's request for the swarm to handle.
        provider: LLM API provider name from the 20-provider registry.
        model: Model name override.
        roles: Which agent roles to include.
        **model_kwargs: Extra kwargs forwarded to the chat model.
    """
    swarm = create_langgraph_swarm(provider, model, roles, **model_kwargs)
    result = swarm.invoke(
        {"messages": [{"role": "user", "content": message}]},
    )
    return str(result["messages"][-1].content)
