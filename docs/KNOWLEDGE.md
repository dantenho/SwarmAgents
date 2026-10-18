# SwarmAgents Knowledge Base

Single source of truth for the LangSwarm stack: architecture, verified
benchmarks, toolchain constraints, and operational knowledge. Built and
verified on 2026-10-18.

## 1. What this is

A LangChain/LangGraph agentic swarm system with:

- **20 LLM API providers** in one registry (`swarmagents/providers.py`):
  OpenAI, Anthropic, Google GenAI, Bedrock, Mistral, Groq, Fireworks,
  Together, DeepSeek, xAI, Cohere, Perplexity, HuggingFace, Ollama, vLLM,
  OpenRouter, Novita, NVIDIA, Cerebras, Azure OpenAI. Each entry has a
  default model, API-key env var, and integration or OpenAI-compatible
  base URL. `get_chat_model(provider)` resolves any of them.
- **Two swarm patterns**:
  - Handoff swarm (`swarmagents/swarm.py`, langgraph-swarm): agents pass
    control via handoff tools; roles researcher/analyst/writer.
  - Supervisor swarm (`swarmagents/graph.py`): StateGraph supervisor
    delegates and synthesizes.
- **Int8 embeddings** (`swarmagents/embeddings.py`): all-MiniLM-L6-v2,
  384-dim, dynamically quantized to int8, Redis-cached (SHA-256 content
  keys, TTL, graceful degradation without Redis).
- **Adaptive kernels** (`swarmagents/kernels.py`): dispatches int8 cosine
  and top-k to the best backend at call time — numba `@njit(parallel)`,
  free-threaded ThreadPool (GIL off), pyo3 Rust, or pure Python.
- **Vector stores** (`swarmagents/vectordb.py`): `SupabaseVectorStore`
  (pgvector `vector(384)` column + match RPC) and `LocalVectorStore`
  (in-memory via kernels). `create_vector_store()` auto-selects.
- **Tracing** (`swarmagents/tracing.py`): LangSmith `@traceable` wrapper +
  LangChainTracer callbacks; inert without `LANGSMITH_TRACING/API_KEY`.
- **Graph view** (`swarmagents/graphview.py`): swarm topology as Neo4j
  Cypher (MERGE `(:Agent)-[:HANDS_OFF_TO]->()`), Mermaid, or live push via
  langchain-neo4j.
- **Rust extension** (`rust_ext/`): pyo3 0.30, `cosine_similarity_int8` +
  rayon-parallel `cosine_batch_int8`, ThinLTO release.

## 2. Verified benchmarks (Xeon AVX-512, best-of-5 / avg-of-20)

| Workload | Python | Numba | Rust (pyo3) |
|---|---|---|---|
| Single cosine (384-dim int8) | 39.9 µs | 0.36 µs (110x) | 7.0 µs (5.7x) |
| 10k-pair batch | 362 ms | 188 µs (1,922x) | 59–99 ms (6x) |
| Rerank top-10 / 2k corpus (avg) | 64–81 ms | — | 6.1x per-call |
| Free-thread rerank (3.15t, 8 thr) | 64.1 → 43.5 ms (1.47x) | — | — |
| Free-thread CPU scaling (3.15t) | 3.84x on 8 threads | — | — |

Observability overhead (loguru + Prometheus + OTel): <1% per call
(+256/−85/+117 µs on a 32 ms rerank — noise level).

LangChain stack metrics: registry resolve 0.002 ms; cache-hit embed
0.007 ms; swarm build (fake model) 14 ms; batch_cosine 1k vectors 18.2 ms
(~110k comparisons/sec).

## 3. Interpreter matrix (hard-won knowledge)

| Runtime | Works | Notes |
|---|---|---|
| CPython 3.12 (main `.venv`) | full stack | dev environment; all tests/benchmarks |
| CPython 3.14t free-threaded | numba + freethread | numba works on 3.14t (~1635x batch); `PYTHON_GIL=0` |
| CPython 3.15.0b1 (GIL) | python-only | **numpy segfaults** (random/matmul) in ALL versions tested (2.1.3–2.5.4): no wheel predates the 3.15 beta ABI. Numba segfaults on import. |
| CPython 3.15.0b1 free-threaded | freethread | GIL off verified: 3.84x scaling; kernels dispatcher runs numpy-free |

**Rules from this**: keep numba imports lazy (`importlib.util.find_spec`)
so 3.15 paths degrade; the kernels dispatcher avoids numpy outside the
numba path; pyo3/maturin builds per-interpreter from source, making Rust
the 3.15-native speed path today.

## 4. Toolchain

- **uv pip** into project venvs (`.venv` = 3.12 dev; `.venv-315t`,
  `.venv-315`, `.venv-ft` for interpreter experiments)
- **Pixi.toml**: Python ≥3.15, Node ≥26.11.1, Rust ≥1.99, ruff 1.99,
  clang/clangxx/llvm 23.1.3, meson ≥1.12, maturin ≥1.15, ty ≥1, 44 deps;
  tasks `lint`/`test`/`bench`/`build-ext`; environments `default`
  (node+rust) and `llvm` (clang)
- **Rust**: pyo3 0.30.1, rayon, ThinLTO; build with
  `cd rust_ext && maturin develop --release`. Note: `cargo test` cannot
  link (pyo3 needs libpython); test the extension from Python instead.
- **Quality gates (all must pass before commit)**:
  `ruff check .`, `ruff format .`, `ty check` (1 accepted deprecation
  warning: `quantize_dynamic`), `pytest tests/ -q` (43 tests incl.
  Hypothesis property tests)
- ty config in pyproject `[tool.ty.src]` + overrides; **ty does not honor
  `# type: ignore`** — use pyproject overrides
- LangGraph `StateGraph(TypedDict)` needs `cast(Any, schema)` for ty

## 5. Known caveats

- PyPI `qwen` is an abandoned broken wrapper — Qwen 0.6B models come via
  `transformers`
- `langswarm` PyPI package pins old langchain semantics; after any fresh
  install verify `langchain==1.4.6` survives (`create_agent` API)
- faiss-cpu AVX-512/LTO source build blocked on system BLAS + swig (swig
  is pip-installable; BLAS bundled in the wheel is usable via
  `-DBLAS_LIBRARIES=`)
- Redis AVX-512/LTO: source build works (1,896 zmm instructions verified)
  but lives in /tmp, not the repo
- pgvector compile needs PostgreSQL server headers (not in sandbox);
  Neo4j server needs Java (not in sandbox) — client integrations only
- Gmail connector: denied (user refused auth); Google Drive used instead

## 6. Environment variables

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` (etc., per provider) | LLM provider auth |
| `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` | LangSmith tracing on |
| `LANGSMITH_PROJECT` | LangSmith project (default swarmagents) |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | live graph push |
| `SUPABASE_URL` / `SUPABASE_KEY` | pgvector store selection |
| `SWARM_REDIS_URL` | embedding cache (default localhost:6379/0) |
| `PYTHON_GIL=0` | disable GIL on free-threaded builds |
| `PROMETHEUS_PORT` | expose /metrics from the observability bench |

## 7. Repo layout

```
swarmagents/          # package: providers, swarm, graph, embeddings,
                      # kernels, vectordb, tracing, graphview
rust_ext/             # pyo3 extension + Cargo.toml
benchmarks/           # bench_kernels, bench_freethread, bench_langchain,
                      # bench_rerank, bench_observability
tests/                # test_swarm.py, test_properties.py (Hypothesis)
langswarm/            # canvas renderer + exported graph artifacts
examples/             # run_swarm.py
Pixi.toml             # pinned polyglot environment
AGENTS.md             # agent guidance + verified findings
```

## 8. Links

- Repo PR (code): https://github.com/dantenho/SwarmAgents/pull/1
- Branch: `vibe/swarm-workflow-2a5989`
- Google Drive folder: https://drive.google.com/drive/folders/13m7fR-8qRKHYwnwdNkNWTHln29qECaY0
