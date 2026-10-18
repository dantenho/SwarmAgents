"""Render the LangSwarm architecture as a PNG canvas with matplotlib."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

NODES = {
    "user": (0.5, 0.92),
    "supervisor": (0.5, 0.70),
    "researcher": (0.16, 0.42),
    "analyst": (0.5, 0.42),
    "writer": (0.84, 0.42),
    "pgvector\n(supabase)": (0.16, 0.14),
    "faiss\n(avx512 int8)": (0.5, 0.14),
    "redis\n(cache)": (0.84, 0.14),
}

EDGES = [
    ("user", "supervisor"),
    ("supervisor", "researcher"),
    ("supervisor", "analyst"),
    ("supervisor", "writer"),
    ("researcher", "analyst"),
    ("analyst", "writer"),
    ("researcher", "pgvector\n(supabase)"),
    ("researcher", "faiss\n(avx512 int8)"),
    ("analyst", "redis\n(cache)"),
    ("writer", "user"),
]


def render(path: str = "langswarm/architecture.png") -> str:
    """Draw the swarm architecture canvas and save it as a PNG."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#0d1117")

    for src, dst in EDGES:
        x1, y1 = NODES[src]
        x2, y2 = NODES[dst]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.4),
        )

    palette = {
        "user": "#f0883e",
        "supervisor": "#a371f7",
        "researcher": "#58a6ff",
        "analyst": "#58a6ff",
        "writer": "#58a6ff",
    }
    for name, (x, y) in NODES.items():
        kind = next((k for k in palette if k in name.lower()), None)
        color = palette.get(
            next((k for k in palette if k in name.lower()[: len(k)]), ""),
            "#238636",
        )
        if any(k in name.lower() for k in palette):
            color = next(
                v for k, v in palette.items() if k in name.lower()
            )
        box = dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="none")
        ax.text(x, y, name, ha="center", va="center", fontsize=11,
               color="white", fontweight="bold", bbox=box, linespacing=1.4)
        del kind

    ax.set_title(
        "LangSwarm — SwarmAgents Architecture",
        fontsize=16,
        color="white",
        pad=20,
        fontweight="bold",
    )
    fig.savefig(path, dpi=72, facecolor="#0d1117", bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(render())
