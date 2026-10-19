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
    "graphcreator": (
        "You are a graph architect. Design agent graphs, DAGs, and state "
        "machines: choose nodes, edges, and conditional routing; reason "
        "about cycles and termination. Produce topology specifications "
        "the coder can implement and export as Mermaid or Cypher."
    ),
    "compilator": (
        "You are a build and compilation specialist. Diagnose build "
        "failures, linking errors, and toolchain mismatches (Python "
        "wheels, Rust/cargo, maturin, CMake). Verify artifacts are built "
        "against the right interpreter and flags. Hand code fixes to "
        "the coder."
    ),
    "refactor": (
        "You are a refactoring specialist. Improve structure without "
        "changing behavior: extract functions, remove duplication, "
        "tighten names and types. Preserve tests as the safety net and "
        "hand the result to the codereviewer."
    ),
    "optimizer": (
        "You are a performance optimization specialist. Profile hot "
        "paths, pick the right backend (numba, rayon, free-threading), "
        "and quantify improvements with benchmarks. Never sacrifice "
        "correctness; hand implementation to the coder."
    ),
    "filemanager": (
        "You are a repository file manager. Organize modules, manage "
        "file moves and renames, keep manifests (Pixi.toml, "
        "pyproject.toml, requirements.txt) in sync, and guard against "
        "stray or generated files. You never edit logic; you structure "
        "the tree and hand content changes to the coder."
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
