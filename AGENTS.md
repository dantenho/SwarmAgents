# AGENTS.md

Guidance for AI agents and contributors working on SwarmAgents.

## Project

LangChain/LangGraph agentic swarm with a 20-provider LLM registry, int8
MiniLM-L6-v2 embeddings, Redis caching, and a Supabase pgvector/faiss vector
layer (in progress).

## Toolchain pins

- Python: >=3.15 (sandbox verification runs on 3.12; language features used
  stay 3.12-compatible)
- Node.js: >=26.11.1
- Rust: >=1.99 (pyo3 0.30.1 via crates.io, maturin for bindings)
- Ruff: 1.99 (`ruff check .` must pass)
- clang/clangxx/llvm: 23.1.3 (Pixi pin; sandbox uses system gcc)
- ty, biome 2.5.15, oxlint 1.88.0 available as additional linters

## Package manager

Use `uv pip` (uv 0.11.x) into the project venv at `.venv/`:

    uv pip install --python .venv/bin/python <pkgs>

Dependencies are recorded in three places — keep all three in sync:
`Pixi.toml`, `pyproject.toml` `[project.dependencies]`, `requirements.txt`.

## Repo layout

- `swarmagents/providers.py` — 20-provider LLM registry (name, default model,
  API-key env var, integration or OpenAI-compatible base URL). Add providers
  by appending a `ProviderSpec` to `REGISTRY`; `test_registry_has_20_providers`
  asserts exactly 20, so update it if the count changes intentionally.
- `swarmagents/swarm.py` — handoff swarm via `langgraph-swarm`
  (`create_swarm` + `create_handoff_tool`), roles: researcher/analyst/writer.
- `swarmagents/graph.py` — supervisor-worker variant via `StateGraph`.
- `swarmagents/embeddings.py` — MiniLM-L6-v2 int8 (384-dim) with Redis
  caching. Heavy imports are lazy (PEP 810 style); do not add module-level
  imports of torch/transformers here. `ruff` will flag undefined `torch` if
  you forget a local lazy import.
- `tests/test_swarm.py` — offline tests only (no API keys, no network).
- `examples/run_swarm.py` — end-to-end example.

## Conventions

- No comments in code; docstrings for public functions only.
- Ruff: line-length 88, target py315, rules E/F/W/I, E203 ignored.
- Error type for user-facing failures: `ProviderError` from `swarmagents.providers`.

## Verification

    .venv/bin/python -m pytest tests/ -q
    .venv/bin/python -m ruff check .

Both must pass before committing. Never commit `.venv/`, keys, or lockfiles
for unsupported managers.

## Known caveats (do not re-litigate)

- PyPI `qwen` is an abandoned broken wrapper; Qwen 0.6B models are used via
  `transformers`.
- faiss-cpu AVX-512/LTO/PGO source build requires system BLAS + swig; the
  standard wheel is used until that build is completed.
- `swig` is pip-installable (`uv pip install swig`, provides `swig` binary in
  `.venv/bin`).
- Redis AVX-512/ThinLTO/Polly/mimalloc are from-source build flags; pinned as
  intent in Pixi, not built here.
- "lifeguard" (Python) is a data-quality validation lib, not a service.
