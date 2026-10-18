"""Specialized swarm agents: reviewer, debugger, coder, memory, research,
guide, judge, and caching roles with handoff wiring.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph_swarm import create_handoff_tool, create_swarm

SPECIALIST_ROLES: dict[str, str] = {
    "codereviewer": (
        "You are a senior code reviewer. Examine diffs for correctness, "
        "security, performance, and style. Report findings with severity "
        "and file:line references. Delegate fixes to the coder, deep "
        "context to deepresearch, and quality arbitration to the judge."
    ),
    "debugger": (
        "You are a debugging specialist. Reproduce failures from stack "
        "traces and symptoms, isolate root causes, and propose minimal "
        "fixes. Hand implementation to the coder once the cause is known."
    ),
    "coder": (
        "You are an expert software engineer. Write clean, tested, "
        "idiomatic code implementing the given requirements. After "
        "writing, hand the diff to the codereviewer for verification."
    ),
    "memory": (
        "You are the swarm's memory manager. Recall prior decisions, "
        "store new facts and embeddings via the vector store, and answer "
        "questions about what the swarm already knows. Do not write code."
    ),
    "deepresearch": (
        "You are a deep-research analyst. Decompose open questions, "
        "gather and synthesize source material, and produce cited "
        "findings. Hand writing tasks to the guidecreator."
    ),
    "guidecreator": (
        "You are a technical writer of guides and documentation. Turn "
        "research findings and code changes into clear, structured "
        "documentation with examples."
    ),
    "judge": (
        "You are an impartial quality judge. Compare candidate answers "
        "or code against the requirements, score them on explicit "
        "criteria, and pick the best with justification."
    ),
    "caching": (
        "You are a caching and performance specialist. Decide what to "
        "cache, choose eviction policies, and analyze hit rates and "
        "latency budgets. You advise; the coder implements."
    ),
}

DEFAULT_SPECIALISTS: tuple[str, ...] = tuple(SPECIALIST_ROLES)


def build_specialist(
    role: str,
    model: BaseChatModel,
    handoffs: list[BaseTool],
    tools: list[BaseTool] | None = None,
) -> Any:
    """Create one specialist agent with handoff tools to the others.

    Args:
        role: One of the SPECIALIST_ROLES keys.
        model: Chat model driving the agent.
        handoffs: Handoff tools to the other specialists.
        tools: Optional domain tools for this agent.

    Returns:
        A compiled agent node for the swarm.
    """
    others = ", ".join(sorted(set(SPECIALIST_ROLES) - {role}))
    return create_agent(
        model,
        tools=list(tools or []) + handoffs,
        system_prompt=(
            f"{SPECIALIST_ROLES[role]}\n"
            f"Available specialists: {others}. Transfer control when a "
            "request belongs to another specialty; answer directly when "
            "it is yours."
        ),
        name=role,
    )


def create_specialist_swarm(
    model: BaseChatModel | Any,
    roles: tuple[str, ...] = DEFAULT_SPECIALISTS,
    tools_by_role: dict[str, list[BaseTool]] | None = None,
) -> Any:
    """Build a handoff swarm of the specialist agents.

    Args:
        model: Chat model for every specialist.
        roles: Which specialist roles to include.
        tools_by_role: Optional per-role domain tools.

    Returns:
        A langgraph-swarm StateGraph of the specialists.
    """
    tools_by_role = tools_by_role or {}
    agents = []
    for role in roles:
        handoffs = [
            create_handoff_tool(
                agent_name=other,
                description=SPECIALIST_ROLES[other],
            )
            for other in roles
            if other != role
        ]
        agents.append(
            build_specialist(
                role,
                model,
                handoffs,
                tools_by_role.get(role),
            )
        )
    return create_swarm(agents, default_active_agent=roles[0])


__all__ = (
    "DEFAULT_SPECIALISTS",
    "SPECIALIST_ROLES",
    "build_specialist",
    "create_specialist_swarm",
)
