"""Token usage tracking and estimation for swarm runs.

Learned from 2026 token-optimization research:
- Gartner: agentic workflows use 5-30x more tokens per task than chat
- Prompt caching: 41-80% cost reduction (arXiv:2601.06007) — we keep a
  STABLE prompt prefix (agent identity/skills) so providers can cache it,
  and put dynamic content (task description) at the END.
- Biggest win is sending less context: we estimate tokens per prompt and
  expose metrics so the orchestrator can budget and prune.
"""

from __future__ import annotations

from pydantic import BaseModel


class PromptSpec(BaseModel):
    """Cache-friendly prompt: stable prefix first, dynamic tail last."""

    agent_name: str
    agent_skills: list[str]
    task_description: str

    def render(self) -> str:
        # Stable, cacheable prefix (identical across tasks for one agent):
        prefix = f"You are {self.agent_name} with skills {self.agent_skills}. "
        # Dynamic tail (changes per task, placed last to protect the cache):
        return f"{prefix}Complete the task: {self.task_description}"


class TokenUsage(BaseModel):
    """Per-run and per-agent token accounting."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cache_hits: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.calls += 1

    def register_cache_hit(self) -> None:
        self.cache_hits += 1


def estimate_tokens(text: str) -> int:
    """Cheap heuristic: ~4 chars/token for English; good enough for budgets."""
    return max(1, len(text) // 4)


class TokenMeter:
    """Aggregates usage per agent; exposes budget checks."""

    def __init__(self, budget: int | None = None) -> None:
        self.budget = budget
        self.per_agent: dict[str, TokenUsage] = {}

    def usage_for(self, agent_name: str) -> TokenUsage:
        if agent_name not in self.per_agent:
            self.per_agent[agent_name] = TokenUsage()
        return self.per_agent[agent_name]

    def record(
        self, agent_name: str, prompt: str, completion: str, cached_prefix: bool = False
    ) -> int:
        """Record one call; returns estimated total tokens (0 if over budget)."""
        u = self.usage_for(agent_name)
        pt = estimate_tokens(prompt)
        # Cached stable prefix is discounted (providers bill ~10-25% for it)
        if cached_prefix:
            pt = pt // 4
            u.register_cache_hit()
        ct = estimate_tokens(completion)
        if self.budget is not None and self.total + pt + ct > self.budget:
            return 0  # budget exceeded: caller should skip this call
        u.add(pt, ct)
        return pt + ct

    @property
    def total(self) -> int:
        return sum(u.total_tokens for u in self.per_agent.values())

    def summary(self) -> dict[str, int]:
        return {
            "total_tokens": self.total,
            "calls": sum(u.calls for u in self.per_agent.values()),
            "cache_hits": sum(u.cache_hits for u in self.per_agent.values()),
        }
