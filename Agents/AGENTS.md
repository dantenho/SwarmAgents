# AGENTS.md — SwarmAgents Architecture

## Working rule

**Search before you code.** Any code touching an external API, framework, or
version-sensitive surface must be preceded by a web search verifying the
current API shape. Example caught by this rule: google-genai exposes a native
async client (`client.aio.models.generate_content`) — wrapping the sync client
in `asyncio.to_thread` is unnecessary and slower.

Analysis of the 5 frameworks and the design of **our own swarm system**,
built on plain Pydantic (no LangChain/LangGraph dependency).

---

## 1. Analysis of the 5 frameworks

| Capability | PydanticAI | Google GenAI/ADK | OpenAI Agents SDK | LangGraph Swarm | CrewAI |
|---|---|---|---|---|---|
| **Core unit** | `Agent` (typed output) | `LlmAgent` + workflow agents | `Agent` (name, tools, handoffs) | node in `StateGraph` | `Agent` (role, goal, backstory) |
| **Task division** | delegation via tools | Sequential/Parallel/Loop agents | handoffs + agents-as-tools | graph edges + conditional routing | sequential / hierarchical process |
| **Agent→agent comms** | parent calls child, gets result back | session state + AgentTool | handoff carries full context | messages in shared State | manager delegates, validates results |
| **Self-learning** | none (stateless agents) | session state only | sessions (history) | checkpointed memory | crew memory (optional) |
| **Afinidade / routing** | manual wiring | LLM `transfer` | LLM picks handoff target | LLM picks handoff | manager LLM reassigns |
| **Guardrails** | output via Pydantic models | eval tooling | input+output guardrails | none built-in | none built-in |
| **Weakness** | no built-in handoff lists yet | Google-centric | vendor lock-in | every agent must know all others | hierarchical mode misroutes in practice |

**Key insights we adopt:**
- From **PydanticAI**: every result validated by a Pydantic model before acceptance.
- From **OpenAI SDK**: input/output guardrails at the swarm boundary; handoff loops need a hard stop.
- From **LangGraph**: swarm outperforms supervisor when agents > distractors
  (their benchmark); we support both topologies.
- From **ADK**: deterministic workflow agents (sequential/parallel/loop) for
  predictable pipelines, LLM routing only when needed.
- From **CrewAI**: role-based agents and a manager that *validates outcomes*.

---

## 2. Workflow path decision matrix

Every incoming task is routed by this matrix (deterministic first, LLM never needed):

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TASK INTAKE                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
                  ┌─────────────────────────────┐
                 │ Q1: skills required known?   │
                 └───────┬───────────────┬───────┘
                    YES  │               │  NO
                          ▼               ▼
          ┌───────────────────┐   ┌──────────────────┐
         │ Q2: how many      │   │ PLANNER LLM      │
         │ qualified agents? │   │ decomposes goal  │
          └──┬─────┬─────┬───┘   │ into Subtasks    │
             │     │     │       └────────┬─────────┘
     1 agent  │     │     │ ≥2 agents      │
             ▼     │     ▼                ▼
     ┌──────────┐ │  ┌────────────────────────────┐
    │ DIRECT   │ │  │ Q3: task divisible?          │
     │ EXECUTE  │ │  └──────┬──────────────┬───────┘
     └──────────┘ │    YES  │              │ NO
                  ▼         ▼              ▼
        ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
        │ Q4: need    │ │ FAN-OUT      │ │ AFFINITY     │
        │ consensus?  │ │ (parallel    │ │ ROUTER: pick │
        └──┬─────┬────┘ │  split of    │ │ best agent   │
      YES  │     │ NO   │  same task)  │ │ by history   │
           ▼     ▼      └──────────────┘ └──────────────┘
    ┌──────────┐ ┌───────────┐
    │ VOTE /   │ │ FAN-OUT   │
    │ REDUCE   │ │ (divide)  │
    └──────────┘ └───────────┘
```

**Decision table (the matrix, evaluated in order):**

| # | Condition | Path | Implementation |
|---|-----------|------|----------------|
| 1 | No agent has the required skills | `REJECT` (or planner decomposes further) | `Swarm.dispatch` |
| 2 | Exactly one qualified agent | `DIRECT` | `Worker.execute` |
| 3 | Same task, consensus needed | `FAN-OUT-SAME` → vote | `DivisionMode.SAME` |
| 4 | Same task, speed needed | `RACE` → first valid result wins | `DivisionMode.SAME` + `RaceReducer` |
| 5 | Task decomposable, independent parts | `FAN-OUT-MULTI` (parallel subtasks) | `DivisionMode.MULTIPLE` |
| 6 | Task decomposable, dependent parts | `DAG` (topological waves) | `Subtask.depends_on` |
| 7 | Different tasks, different skills | `SPECIALIZE` (per-skill routing) | `DivisionMode.DIFFERENT` |
| 8 | ≥2 qualified agents, no consensus need | `AFFINITY-ROUTE` (best history wins) | `AffinityTracker.pick` |

---

## 3. Agent communication

In-process `asyncio` message bus (agents never call each other directly —
all traffic flows through typed messages, so it is replayable and testable):

```
┌────────┐  Envelope   ┌─────────┐  Envelope   ┌────────┐
│ Agent A│────────────►│MessageBus│───────────►│ Agent B│
└────────┘             └────┬────┘             └────────┘
   ▲                        │  every envelope is a Pydantic model:
   │      ┌─────────────────┘  Message{from,to,kind,payload,ts}
┌─┴──────┴──┐
│Blackboard │  shared key-value state, append-only log
└───────────┘
```

Message kinds: `REQUEST` (delegate), `RESULT`, `BROADCAST` (blackboard write),
`ERROR`, `HANDOFF` (control transfer, swarm-style).

---

## 4. Self-learning

Every completed subtask appends a validated `Outcome` to the `MemoryStore`.
Two loops:

- **Fast loop (affinity, per-run):** success-rate per (agent, skill) updates
  routing decisions immediately.
- **Slow loop (skill extraction, post-run):** failed subtasks whose retry
  succeeded are mined into reusable `Lesson` records (what prompt/pattern
  worked), persisted as JSON and injected into future agent system prompts.

No fine-tuning, no external service — just outcome stats and lesson
records, both Pydantic-validated and JSON-serializable.

## 5. Agent affinity

`AffinityTracker.score(agent, subtask) = skill_match
+ success_rate(agent, skill) - failure_penalty(agent, skill) - busyness(agent)`

The router picks argmax(score) among qualified agents. Affinity is
*learned*: an agent that succeeds at "code" tasks gets routed more code
tasks; one that fails gets routed fewer until it redeems itself
(exponential decay keeps scores adaptive).

## 6. Task division modes

| Mode | Semantics | Failure isolation | Used for |
|---|---|---|---|
| `SAME` | N agents get the identical task; reducer votes/races | full (reducer picks best valid) | consensus, QA, flaky LLMs |
| `MULTIPLE` | one task split into K independent subtasks, fan-out | per-subtask | parallelizable work |
| `DIFFERENT` | unrelated tasks routed by skill | per-task | batch pipelines |

## 7. Map to the code

- `src/pydanticswarm/swarm.py` — core run loop (already present)
- `src/pydanticswarm/communication.py` — message bus + blackboard
- `src/pydanticswarm/learning.py` — memory, outcomes, lessons
- `src/pydanticswarm/affinity.py` — affinity router
- `src/pydanticswarm/division.py` — SAME/MULTIPLE/DIFFERENT dividers + reducers
- `tests/test_division.py`, `tests/test_affinity.py` — coverage
- `examples/agents_folder_demo.py` — end-to-end demo

All models are Pydantic; all I/O is validated; the swarm degrades to a
supervisor-less queue when affinity data is cold.
