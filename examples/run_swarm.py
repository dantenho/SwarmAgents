"""Example: run the LangGraph handoff swarm.

Set an API key for your chosen provider, e.g.:
    export OPENAI_API_KEY=sk-...
"""

from swarmagents import PROVIDER_NAMES, run_langgraph_swarm

if __name__ == "__main__":
    print("Available providers:", ", ".join(PROVIDER_NAMES))
    answer = run_langgraph_swarm(
        "Research the trade-offs of SQLite vs Postgres for an embedded "
        "analytics workload, analyze the findings, and write a short report.",
        provider="openai",
    )
    print(answer)
