"""Parallel fan-out/fan-in over specialist agents.

Runs several agents concurrently on the same task and merges their
outputs. Thread-based, so it parallelizes for real on free-threaded
builds (PYTHON_GIL=0) and degrades to sequential on GIL builds.

Patterns:
- fan_out: same prompt to N agents in parallel, collect all answers.
- judge_over_candidates: run N generator agents in parallel, then feed
  all candidates to the judge for arbitration.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from swarmagents.agents import SPECIALIST_ROLES, build_specialist
from swarmagents.kernels import gil_enabled
from swarmagents.providers import get_chat_model

_MAX_WORKERS = min(32, (os.cpu_count() or 4) * 4)


def _compile(graph: Any) -> Any:
    if hasattr(graph, "compile") and not hasattr(graph, "invoke"):
        return graph.compile()
    return graph


def fan_out(
    roles: tuple[str, ...],
    prompt: str,
    model: Any,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Run several specialist agents concurrently on one prompt.

    Args:
        roles: Specialist roles to invoke (from SPECIALIST_ROLES).
        prompt: The task prompt given to every agent.
        model: Chat model (or provider name) driving the agents.
        max_workers: Thread pool size; defaults to CPU-based cap.

    Returns:
        Mapping of role name to its answer (or error string).
    """
    if any(role not in SPECIALIST_ROLES for role in roles):
        unknown = [r for r in roles if r not in SPECIALIST_ROLES]
        raise ValueError(f"unknown roles: {unknown}")
    chat_model = get_chat_model(model) if isinstance(model, str) else model
    workers = max_workers or min(len(roles), _MAX_WORKERS)

    def run_one(role: str) -> tuple[str, Any]:
        agent = build_specialist(
            role,
            chat_model,
            handoffs=[],
            tools=[],
        )
        compiled = _compile(agent)
        try:
            result = compiled.invoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            return role, str(result["messages"][-1].content)
        except Exception as err:  # noqa: BLE001
            return role, f"<error: {err}>"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pairs = list(pool.map(run_one, roles))
    return dict(pairs)


def judge_over_candidates(
    generator_roles: tuple[str, ...],
    prompt: str,
    model: Any,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """Generate candidates in parallel, then arbitrate with the judge.

    Args:
        generator_roles: Roles producing candidate answers in parallel.
        prompt: The task prompt for the generators.
        model: Chat model (or provider name).
        max_workers: Thread pool size for the generation phase.

    Returns:
        Dict with candidates (role -> answer) and judge verdict.
    """
    candidates = fan_out(generator_roles, prompt, model, max_workers)
    judge = build_specialist(
        "judge",
        model if not isinstance(model, str) else get_chat_model(model),
        handoffs=[],
        tools=[],
    )
    compiled = _compile(judge)
    joined = "\n\n".join(
        f"## Candidate from {role}\n{answer}" for role, answer in candidates.items()
    )
    verdict_prompt = (
        f"Original request:\n{prompt}\n\n"
        f"Candidates:\n{joined}\n\n"
        "Score each candidate on correctness, completeness, and clarity, "
        "then declare the best with justification."
    )
    result = compiled.invoke(
        {"messages": [{"role": "user", "content": verdict_prompt}]}
    )
    return {
        "candidates": candidates,
        "verdict": str(result["messages"][-1].content),
    }


def parallel_backend() -> str:
    """Report whether thread fan-out parallelizes on this interpreter."""
    if not gil_enabled():
        return f"threads-parallel (GIL off, up to {_MAX_WORKERS} workers)"
    return "threads-serialized (GIL on; use free-threaded build)"


__all__ = (
    "fan_out",
    "judge_over_candidates",
    "parallel_backend",
)
