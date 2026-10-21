# SKILL.md — PydanticSwarm v0.2

Criar orquestrações de agentes (swarm) com **Pydantic puro** — sem LangChain/LangGraph.
Skill completa: criação de agentes, matriz de decisão, comunicação, self-learning,
afinidade e os 3 modos de divisão de tarefas.

Análise dos 5 frameworks (PydanticAI, Google ADK, OpenAI SDK, LangGraph, CrewAI)
e o design derivado: **`Agents/AGENTS.md`**.

---

## Fluxograma do workflow

```
                              ┌──────────────────────────┐
                              │        USER GOAL         │
                              │   (texto / instruction)  │
                              └────────────┬─────────────┘
                                           │
                                           ▼
                          ┌────────────────────────────────┐
                          │   DECISION MATRIX (determin.)  │
                          │  REJECT / DIRECT / FAN-OUT-*  │
                          │  / SPECIALIZE / AFFINITY       │
                          └──────┬───────────┬────────────┘
                                 │           │
                    ═══ SAME task ══   ═══ different/divisible ══
                                 │           │
              ┌──────────────────▼──┐   ┌────▼──────────────────┐
              │ FAN-OUT-SAME        │   │ PLANNER decomposes    │
              │ N agents, same task │   │ into Subtasks (DAG)   │
              │ vote (majority) ou  │   └────┬──────────────────┘
              │ race (first valid)  │        │
              └──────────┬──────────┘        ▼
                         │            ┌─────────────────────────┐
                         │            │ TASK QUEUE (FIFO)       │
                         │            └────┬──────────────┬────┘
                         │                 │              │
                         ▼                 ▼              ▼
              ┌────────────────┐  ┌────────────────────────────────┐
              │ REDUCER        │  │ DISPATCHER (affinity-aware)    │
              │ vote / race    │  │ argmax(score) among qualified  │
              └───────┬────────┘  └────┬───────────────────────────┘
                      │               ▼
                      │   ┌────────────────────────────────────┐
                      │   │ PARALLEL WORKERS (asyncio)         │
                      │   │ AgentSpec + skill match + capacity  │
                      │   └────┬───────────────────────────┬───┘
                      │        │                           │
                      ▼        ▼                           ▼
              ┌──────────────────────────────────────────────────┐
              │      MESSAGE BUS (typed Envelope, audit log)     │
              │      REQUEST / RESULT / BROADCAST / HANDOFF     │
              └───────────────────────────┬──────────────────────┘
                                          ▼
              ┌──────────────────────────────────────────────────┐
              │ AGGREGATOR: valida com Pydantic antes de aceitar │
              └───────────────────────────┬──────────────────────┘
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
              ┌─────────────────────┐          ┌─────────────────────┐
              │ MEMORY STORE        │          │ goal satisfeito?    │
              │ Outcome por         │          │ SIM → FINAL ANSWER  │
              │ (agent, skill)      │          │ NÃO → novas subtasks│
              │ + Lessons (JSON)    │          └─────────────────────┘
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ AFFINITY ROUTER     │──► alimenta o DISPATCHER
              │ score = coverage    │     (aprendizado fast loop)
              │ + success_rate     │
              │ − busyness          │
              └─────────────────────┘
```

---

## API (v0.2)

Tudo exposto no top-level `pydanticswarm`:

| Import | Papel |
|---|---|
| `AgentSpec`, `Subtask`, `Swarm`, `SwarmOutcome`, `AgentResult` | núcleo: agentes declarativos + run loop |
| `Orchestrator` | matriz de decisão + roteamento por afinidade |
| `divide_same`, `divide_multiple`, `divide_different`, `DivisionMode` | 3 modos de divisão (vote/race, fan-out, especialização) |
| `MessageBus`, `Envelope`, `MessageKind`, `Blackboard` | comunicação tipada e auditável |
| `MemoryStore`, `Outcome`, `Lesson` | self-learning (fast + slow loop) |
| `AffinityTracker` | score aprendido por (agente, skill) |

## Uso rápido

```python
import asyncio
from pydanticswarm import (
    AgentSpec,
    DivisionMode,
    MemoryStore,
    Orchestrator,
    Subtask,
    Swarm,
)


async def handler(agent, subtask):
    return f"[{agent.name}] {subtask.description}: done"


agents = [
    AgentSpec(name="coder", skills=["code"]),
    AgentSpec(name="writer", skills=["write"]),
    AgentSpec(name="reviewer-a", skills=["review"]),
    AgentSpec(name="reviewer-b", skills=["review"]),
]
swarm = Swarm(agents=agents, handler=handler)
orch = Orchestrator(agents=agents, swarm=swarm, memory=MemoryStore())

outcome = asyncio.run(
    orch.run(
        goal="ship feature",
        subtasks=[
            Subtask(id="a", description="write docs", required_skills=["write"]),
            Subtask(id="b", description="code it", required_skills=["code"]),
            Subtask(
                id="c",
                description="review",
                required_skills=["review"],
                division=DivisionMode.SAME,
            ),  # consenso: 2 agentes + vote
        ],
    )
)
```

## Design guarantees

- **Determinístico primeiro**: a matriz de decisão roda sem LLM; afinidade só desempata
- **Validação Pydantic em toda fronteira**: envelopes, outcomes, results
- **Falha isolada por subtask**: um agente flaky não derruba o swarm (demo incluso)
- **Aprendizado sem fine-tuning**: stats de sucesso afetam roteamento imediatamente (fast loop); `Lesson`s persistem em JSON (slow loop)
- **Comunicação auditável**: todo tráfego passa pelo bus com log append-only

## Exemplos e testes

- `examples/agents_folder_demo.py` — end-to-end: matriz, afinidade, divisão, bus, blackboard
- `examples/pydantic_swarm_demo.py` — demo DAG com dependências
- `tests/test_swarm.py` + `tests/test_swarm_capabilities.py` — 16 testes
- `Agents/AGENTS.md` — análise dos 5 frameworks e design rationale
