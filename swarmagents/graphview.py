"""Write the swarm agent graph to Neo4j for visualization.

Exports the LangSwarm topology (agents and handoff edges) as Neo4j nodes
and relationships so it can be explored in Neo4j Browser or Bloom. Uses
langchain-neo4j when a server is reachable; otherwise degrades to
cypher-script and Mermaid export without a server.
"""

from __future__ import annotations

import os
from typing import Any

from swarmagents.swarm import AGENT_ROLES

NODE_CYPHER = """
MERGE (a:Agent {name: $name})
SET a.description = $description
"""

EDGE_CYPHER = """
MATCH (src:Agent {name: $source})
MATCH (dst:Agent {name: $target})
MERGE (src)-[h:HANDS_OFF_TO]->(dst)
"""


def swarm_topology() -> dict[str, Any]:
    """Return agent nodes and handoff edges for the default swarm.

    Returns:
        A dict-shaped topology with agents and directed handoff edges.
    """
    agents = [
        {"name": role, "description": prompt.split(".")[0] + "."}
        for role, prompt in AGENT_ROLES.items()
    ]
    edges = []
    roles = list(AGENT_ROLES)
    for src in roles:
        for dst in roles:
            if src != dst:
                edges.append({"source": src, "target": dst})
    return {"agents": agents, "edges": edges}


def to_cypher(topology: dict[str, Any]) -> str:
    """Render the topology as an idempotent Cypher script.

    Args:
        topology: Output of swarm_topology().

    Returns:
        A Cypher script creating the Agent nodes and HANDS_OFF_TO edges.
    """
    lines = ["// LangSwarm topology — idempotent (MERGE)"]
    for agent in topology["agents"]:
        name = agent["name"].replace('"', '\\"')
        desc = agent["description"].replace('"', '\\"')
        lines.append(
            f'MERGE (a:Agent {{name: "{name}"}}) SET a.description = "{desc}";'
        )
    for edge in topology["edges"]:
        src = edge["source"].replace('"', '\\"')
        dst = edge["target"].replace('"', '\\"')
        lines.append(
            f'MATCH (s:Agent {{name: "{src}"}}), (t:Agent {{name: "{dst}"}}) '
            f"MERGE (s)-[:HANDS_OFF_TO]->(t);"
        )
    return "\n".join(lines)


def to_mermaid(topology: dict[str, Any]) -> str:
    """Render the topology as a Mermaid flowchart.

    Args:
        topology: Output of swarm_topology().

    Returns:
        Mermaid graph definition text.
    """
    lines = ["graph LR"]
    for agent in topology["agents"]:
        lines.append(f'    {agent["name"]}["{agent["name"]}"]')
    for edge in topology["edges"]:
        lines.append(f"    {edge['source']} -->|hands off| {edge['target']}")
    return "\n".join(lines)


def push_to_neo4j(topology: dict[str, Any]) -> bool:
    """Write the topology to a Neo4j server when credentials are set.

    Args:
        topology: Output of swarm_topology().

    Returns:
        True when the write succeeded; False when no server is available.
    """
    url = os.getenv("NEO4J_URI", "")
    user = os.getenv("NEO4J_USERNAME", "")
    password = os.getenv("NEO4J_PASSWORD", "")
    if not (url and user and password):
        return False
    try:
        from langchain_neo4j import Neo4jGraph

        graph = Neo4jGraph(url=url, username=user, password=password)
        for agent in topology["agents"]:
            graph.query(NODE_CYPHER, params=agent)
        for edge in topology["edges"]:
            graph.query(EDGE_CYPHER, params=edge)
        return True
    except Exception:
        return False


def export(path: str | None = None) -> dict[str, Any]:
    """Export the swarm graph for viewing: Neo4j, Cypher, and Mermaid.

    Args:
        path: Optional directory to write swarm_graph.cypher and
            swarm_graph.mmd into.

    Returns:
        A summary dict with neo4j_pushed, cypher, and mermaid fields.
    """
    topology = swarm_topology()
    pushed = push_to_neo4j(topology)
    cypher = to_cypher(topology)
    mermaid = to_mermaid(topology)
    if path:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "swarm_graph.cypher"), "w") as fh:
            fh.write(cypher + "\n")
        with open(os.path.join(path, "swarm_graph.mmd"), "w") as fh:
            fh.write(mermaid + "\n")
    return {"neo4j_pushed": pushed, "cypher": cypher, "mermaid": mermaid}


__all__ = ("export", "push_to_neo4j", "swarm_topology", "to_cypher", "to_mermaid")
