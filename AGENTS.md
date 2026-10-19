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

## Benchmarks (recorded 2026-10-18, Xeon AVX-512, best of 5)

`benchmarks/bench_kernels.py` measures int8 cosine similarity on 384-dim
MiniLM vectors across pure Python, Numba `@njit`, and the pyo3 Rust extension.

- Single pair: python 39.9us, numba 0.36us (110x), rust 7.0us (5.7x)
- 10k-pair batch: python 362ms, numba-parallel 188us (1922x), rust 59ms (6.1x)
- Rule: keep hot numeric loops in Numba (`@njit(cache=True, parallel=True)`
  with `prange`); pure Python only for orchestration. pyo3 per-call list
  marshalling dominates Rust timings — pass NumPy arrays if that path ever
  becomes hot. Re-run with `.venv/bin/python benchmarks/bench_kernels.py`.

## Python 3.15 status (recorded 2026-10-18)

- CPython 3.15.0b1 (GIL and free-threaded builds) installed via uv.
- Free-threading verified working: 8 OS threads scale 3.84x on pure-Python
  CPU work with `PYTHON_GIL=0` (PEP 703). `sys._is_gil_enabled()` to check.
- numpy 2.5.4 wheels are NOT yet 3.15-ABI compatible: `numpy.random` and
  `@`/matmul segfault on 3.15.0b1 (import and basic ops are fine). Wait for
  wheels built against 3.15 stable; do not pin numpy in a 3.15 venv yet.
- Numba 0.68/llvmlite 0.50 segfault on import under 3.15.0b1 (both GIL and
  free-threaded). Numba supports free-threading on 3.14t (verified working,
  ~1635x on the int8 batch kernel). Keep numba imports lazy
  (`importlib.util.find_spec`) so 3.15 paths degrade gracefully.
- The pyo3 extension builds from source per-interpreter via maturin, so it
  is the 3.15-ready path for native speed today.

## numpy on 3.15 (updated)

Tested 2.1.3, 2.2.6, 2.3.4, 2.5.4 on 3.15.0b1: ALL segfault in
numpy.random/matmul. No older wheel fixes this — the 3.15 beta ABI predates
every published wheel. numpy stays on 3.12/3.14 until 3.15-stable wheels
exist. The kernels dispatcher avoids numpy entirely outside the numba path.

## Type checking policy (ruff + ty, 2026-10-18)

- Gates before every commit: `ruff check .`, `ruff format .`,
  `ty check`, and `pytest tests/ -q`.
- ty config lives in pyproject.toml `[tool.ty.src]`; overrides ignore
  `unresolved-import` (pyo3 module has no stubs) and `invalid-assignment`
  (benchmark monkeypatches) for `swarmagents/kernels.py` and `benchmarks/`.
- ty does NOT honor `# type: ignore[...]` comments — use pyproject
  overrides, not inline ignores.
- Remaining accepted diagnostic: 1 deprecation warning for
  `torch.ao.quantization.quantize_dynamic` (only quantization API available).
- LangGraph `StateGraph(TypedDict)` needs `cast(Any, schema)` to satisfy
  ty's structural TypedDictLike bound (runtime is fine either way).

## Multiagentes (resumo operacional)

- Dois padrões: handoff (swarm.py/agents.py) e supervisor (graph.py).
- 8 especialistas em `swarmagents/agents.py`; entrada padrão codereviewer.
- Sempre `.compile()` o StateGraph antes de invoke; fakes de teste devem
  retornar AIMessage real; prompts de papel precisam declarar quando
  transferir e quando responder direto. Detalhes completos em
  docs/KNOWLEDGE.md seção 10.
