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

## 10. Multiagentes: padrões e conhecimento operacional (2026-10-18)

### Padrões de orquestração implementados neste repo

| Padrão | Módulo | Quando usar |
|---|---|---|
| Handoff swarm (langgraph-swarm) | `swarmagents/swarm.py`, `swarmagents/agents.py` | Controle passa agente→agente via `create_handoff_tool`; sem ponto único; cada agente decide transferir. Ideal para especialidades |
| Supervisor-worker (StateGraph) | `swarmagents/graph.py` | Supervisor planeja, delega e sintetiza; melhor quando se quer ponto de coordenação e ciclo plan→delegate→collect |

### Papéis especializados (13) e seus fluxos

Novos papéis (2ª leva): **graphcreator** (arquitetura de grafos/DAGs de
agentes, exportáveis como Mermaid/Cypher), **compilator** (builds,
toolchain, linking — Python wheels, cargo/maturin, CMake), **refactor**
(reestruturação sem mudança de comportamento, testes como rede de
segurança), **optimizer** (profiling, escolha de backend
numba/rayon/freethread, benchmarks quantificados), **filemanager**
(organização do repositório e sincronização dos manifests — nunca edita
lógica).

- `coder → codereviewer`: loop de verificação obrigatório pós-escrita
- `debugger → coder`: debugger isola causa-raiz, coder implementa a correção
- `deepresearch → guidecreator`: pesquisa vira documentação
- `judge`: arbitra entre candidatos com critérios explícitos
- `memory`: único papel com acesso ao vector store; os demais pedem a ele
- `caching`: aconselha políticas de cache; coder implementa
- `codereviewer` é o agente de entrada padrão do enxame de especialistas

### Lições verificadas (deste projeto)

1. **Graphs não-compilados**: `create_swarm` retorna `StateGraph` não-compilado —
   sempre `.compile()` antes de `.invoke()` (bug real corrigido em tracing.py).
2. **Fakes precisam de AIMessage real**: ao invocar agentes LangChain v1 em
   testes, o modelo falso deve retornar `langchain_core.messages.AIMessage` —
   objetos ad-hoc falham na coerção de mensagens (`MESSAGE_COERCION_FAILURE`).
3. **Handoffs custam tokens**: cada transferência reinjeta o histórico; em
   cadeias longas (research→analyst→writer) o contexto cresce linearmente.
   Para 8 especialistas, prefira transferências diretas ao hub-and-spoke.
4. **Estado compartilhado**: o `SwarmState` (TypedDict) acumula
   `messages` + campos de trabalho; `total=False` permite atualização
   parcial por nó — mas ty exige `cast(Any, schema)` no StateGraph.
5. **Prompts de papel**: cada prompt deve declarar (a) a especialidade,
   (b) quando transferir, (c) quando responder direto — sem (b)/(c) o
   enxame oscila entre agentes ou nunca transfere.
6. **Backend do modelo é uniforme**: todos os 11 agentes (3 do enxame
   base + 8 especialistas) aceitam qualquer um dos 20 provedores via
   `get_chat_model` — modelos diferentes por papel são um kwargs de distância.

### Métricas de multiagentes (o que observar)

- **Hop count**: número de transferências por tarefa (proxy de custo/latência)
- **Cobertura de papel**: fração de tarefas que tocam cada especialista
- **Tempo por agente**: exposto via LangSmith tracing (`swarmagents/tracing.py`)
- **Contexto acumulado**: tamanho final de `messages` (token budget)

### Referências do ecossistema

- `langgraph-swarm` 0.1.0: `create_swarm(agents, default_active_agent=...)`
- `langgraph.prebase`/`StateGraph`: `add_node`, `add_edge`,
  `add_conditional_edges`, `compile`
- LangChain v1 `create_agent(model, tools, system_prompt, name)`: assinatura
  estável usada por todos os agentes deste repo

## 11. Pesquisa externa: multiagentes em produção (2026-10-18)

Fontes: DeepWiki (langchain-ai/langgraph), web research (Anthropic
engineering, surveys 2025-2026). Achados aplicados e verificados:

### Números de referência (Anthropic, medidos)

- Sistema multiagente consome **~15x os tokens** de um chat single-agent
  (single agent ~4x). Uso de tokens explica **80% da variância** de
  performance em research evals.
- Sistema multiagente Opus-4-líder + Sonnet-4 subagents superou o
  single-agent Opus 4 em **90.2%** na avaliação de research deles.
- Implicação: paralelismo de agentes compensa em pesquisa/exploração,
  não em tarefas lineares.

### Os 4 alavancas de produção (Paiteq/Anthropic, convergentes)

1. **Resumo rolante** no supervisor em vez de histórico completo
2. **JSON estruturado** dos subagentes em vez de prosa livre
3. **Modelo mais barato competente por papel** (router barato, worker médio,
   planner forte)
4. **Turn cap duro** para loops fugitivos — IMPLEMENTADO nesta sessão:
   `SWARM_RECURSION_LIMIT` (default 50) aplicado via `recursion_limit` em
   `tracing.py` e `parallel.py`

### Topologias dominantes em produção (2026)

- Supervisor e pipeline sequencial dominam; hierárquico
  (supervisor-de-supervisores) para fluxos genuinamente complexos;
  swarm/blackboard brilham em research-and-summarise paralelo.
- Frameworks são camada fina: outer loop LangGraph/CrewAI em volta de
  inner calls de vendor SDKs (Anthropic/OpenAI Agents SDK).
- O padrão swarm converge no formato: planner decompõe → fan-out
  paralelo de research workers → um writer sintetiza — exatamente o que
  `parallel.fan_out` + `judge_over_candidates` implementam.

### Gotchas de handoff (LangGraph DeepWiki, verificado no código)

- **History bleed-through**: o agente receptor vê todo o histórico do
  transmissor a menos que filtrado — usar input filter / state explícito.
- **Guardrails de saída**: nenhum agente "possui" a resposta final em um
  swarm; guardrails devem rodar em TODOS os agentes que podem responder
  ao usuário.
- **State isolation**: subgrafos com checkpointer precisam de nomes de nó
  distintos ou config explícita, senão contaminam estado entre si.
- `create_react_agent` está deprecado → `create_agent` (já usamos o correto).
- Interrupts em subgrafos borbulham ao pai; retome com `Command(resume=...)`.

### Decisão design-vs-framework (Anthropic, "Building effective agents")

Workflows (caminhos pré-definidos) primeiro; agentes (fluxo dirigido pelo
modelo) só quando o fluxo é genuinamente desconhecido; multiagente só
quando single-agent fica aquém — o custo é ~4x adicional em tokens.
