# SwarmAgents

LangChain/LangGraph agentic swarm with a 20-provider LLM registry, int8
MiniLM-L6-v2 embeddings, Redis caching, and a polyglot toolchain.

## Features

- **20 LLM API providers** in one registry (`swarmagents/providers.py`):
  OpenAI, Anthropic, Google GenAI, Bedrock, Mistral, Groq, Fireworks,
  Together, DeepSeek, xAI, Cohere, Perplexity, HuggingFace, Ollama, vLLM,
  OpenRouter, Novita, NVIDIA, Cerebras, Azure OpenAI.

- **LangGraph Swarm** (`swarmagents/swarm.py`): handoff-based multi-agent
  workflow (researcher → analyst → writer) using `langgraph-swarm`, plus a
  supervisor-worker variant in `swarmagents/graph.py`.

- **Embeddings** (`swarmagents/embeddings.py`): all-MiniLM-L6-v2, 384-dim,
  dynamically quantized to int8, with Redis caching (TTL-based, degrades
  gracefully without Redis). Heavy imports are lazy per PEP 810.

- **Observability**: OpenTelemetry + Prometheus exporter, LangSmith, logfire,
  loguru, rich.

- **Vector stores**: pgvector / Supabase, faiss-cpu, DuckDB, Polars, PyArrow.

## Quick start

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

export OPENAI_API_KEY=sk-...
.venv/bin/python examples/run_swarm.py
```

Embeddings without any API key:

```python
from swarmagents import embed, embed_batch, EMBEDDING_DIM

vec = embed("hello world")           # 384-dim int8 list, Redis-cached
vecs = embed_batch(["a", "b"])
```

Swarm with any provider:

```python
from swarmagents import run_langgraph_swarm, PROVIDER_NAMES

print(PROVIDER_NAMES)
answer = run_langgraph_swarm("Compare SQLite and DuckDB", provider="mistralai")
```

## Environment

- Python >=3.15, Node.js >=26.11.1, Rust >=1.99, Ruff 1.99
- LLVM/clang 23.1.3, Node linters: biome 2.5.15, oxlint 1.88.0
- pyo3 0.30.1 + maturin for Rust↔Python bindings
- @xenova/transformers (transformers.js) for JS-side models
- See `Pixi.toml` for the full pinned dependency set

## Development

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m ruff check .
```

Tests are offline — no API keys or network required.

See `AGENTS.md` for repo conventions and known caveats.

## License

See `LICENSE`.
