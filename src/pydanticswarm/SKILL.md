# SKILL.md — PydanticSwarm

Criar orquestrações de agentes (swarm) com **Pydantic** puro — sem LangChain/LangGraph.
Este skill descreve o processo de criação de agentes, paralelismo e divisão de tarefas.

---

## Fluxograma do workflow

```
                              ┌──────────────────────────┐
                              │        USER GOAL         │
                              │   (texto / instruction)  │
                              └────────────┬─────────────┘
                                           │
                                           ▼
                              ┌──────────────────────────┐
                              │      PLANNER (LLM)       │
                              │  decompõe o objetivo em  │
                              │      subtasks[]         │
                              └────────────┬─────────────┘
                                           │
                                           ▼
                          ┌────────────────────────────────┐
                          │        TASK QUEUE  (FIFO)      │
                          │   subtasks aguardando workers  │
                          └───────┬──────────────────┬───┘
                                  │                  │
                     ┌────────────▼───┐        ┌──────▼──────────┐
                     │  DISPATCHER    │        │   DISPATCHER    │
                     │ (round-robin / │  ...   │  (capacity-     │
                     │  skill match)  │        │   aware)        │
                     └────────────┬───┘        └──────┬──────────┘
                                  │                   │
                     ┌────────────▼───────────────────▼──────────┐
                     │            PARALLEL WORKERS               │
                     │  ┌─────────┐ ┌─────────┐ ┌─────────┐     │
                     │  │ Agent A │ │ Agent B │ │ Agent C │ ... │
                     │  │ (async) │ │ (async) │ │ (async) │     │
                     │  └────┬────┘ └────┬────┘ └────┬────┘     │
                     └───────┼───────────┼───────────┼──────────┘
                             │           │           │
                             ▼           ▼           ▼
                     ┌────────────────────────────────────────┐
                     │            RESULT BUS (asyncio.Queue)  │
                     └───────────────────────┬────────────────┘
                                             │
                                             ▼
                     ┌────────────────────────────────────────┐
                     │           AGGREGATOR / REDUCER        │
                     │  valida (Pydantic), mescla, resume     │
                     └───────────────────────┬────────────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                   ┌─────────────────────┐        ┌──────────────────┐
                   │  goal satisfeito?   │  NÃO   │  re-enfileira    │
                   │  (critério/LLM      ├───────►│  subtasks novas  │──► TASK QUEUE
                   │   judge)            │        │  (refinamento)   │
                   └──────────┬──────────┘        └──────────────────┘
                              │ SIM
                              ▼
                   ┌─────────────────────┐
                   │     FINAL ANSWER   │
                   └─────────────────────┘
```

---

## Os 5 estágios do processo

1. **Criação de agentes** — cada agente é um modelo Pydantic (`AgentSpec`) com
   `name`, `skills`, `model` e um handler async. Validação garante agentes bem formados.
2. **Divisão de tarefas** — um `Planner` decompõe o objetivo em `Subtask`s
   independentes (map). Subtasks dependentes viram grafo DAG opcional.
3. **Paralelismo** — um `Dispatcher` distribui subtasks por uma workers
   pool (`asyncio.Queue` + N corotinas worker). Cada worker roda de forma concorrente.
4. **Agregação** — resultados chegam por um result bus e passam pelo
   `Aggregator`, que valida cada saída com um modelo Pydantic (`T`) antes de
   aceitar (reduce). Falha de validação → retry ou re-enfileiramento.
5. **Loop de refinamento** — um critério de conclusão (heurístico ou LLM judge)
   decide se encerra ou gera novas subtasks.

---

## API rápida

```python
import asyncio
from pydanticswarm import AgentSpec, Swarm, Subtask

agents = [
    AgentSpec(name="researcher", skills=["search", "summarize"]),
    AgentSpec(name="coder", skills=["code", "test"]),
    AgentSpec(name="writer", skills=["write", "edit"]),
]

swarm = Swarm(agents=agents)

result = asyncio.run(
    swarm.run(
        goal="Write a blog post about Pydantic",
        plan=[
            Subtask(
                id="1",
                description="Research Pydantic features",
                required_skills=["search"],
            ),
            Subtask(
                id="2", description="Write code examples", required_skills=["code"]
            ),
            Subtask(id="3", description="Compose the post", required_skills=["write"]),
        ],
    )
)
print(result.summary)
```

## O que a implementação garante

- Cada resultado é validado por Pydantic antes de entrar na agregação.
- Workers são intercambiáveis: qualquer agente com a skill exigida pode pegar a subtask.
- O dispatcher é capacity-aware: não pega mais subtasks do que workers disponíveis.
- Falhas ficam isoladas por subtask (uma subtask falhar não derruba o swarm).

## Referência

- Implementação: `src/pydanticswarm/`
- Exemplo completo: `examples/` (em inglês no código, docs em português no README da skill)
- Testes: `tests/test_swarm.py`
