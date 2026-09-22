# AI Alarm System

A prototype AI-driven home security system built for a take-home challenge.

## Architecture

A sensor event (simulated CCTV/mic) reaches a deterministic **Controller**, which starts or forwards it to a
**Situation Interpreter** — one LangGraph workflow instance (one checkpointed thread) per ongoing situation:

```
sensors (simulated) 
    -> Controller 
        -> Situation Interpreter (LangGraph)
            -> fans out in parallel to agents (object detector, person identifier, behavioural/noise/weather interpreter)
        <- merges results into a situation summary
    -> applies deterministic thresholds sends warning / alarm 
        -> Communication Unit + log
            -> Human 
                -> Can elevate/dismiss warnings
        -> Speaker 
            -> Warns on unpermitted entry
```

Only the Controller's fixed rules (suspicion/danger thresholds, permission checks) can raise a warning or alarm never an agent output or free text.

## Frameworks and their roles

- **LangGraph** (`workflow.py`) implements the Situation Interpreter as a typed `StateGraph`
  (`intake -> detect -> {people, animals, weather} -> merge -> report -> observe`, looping back to `intake` or on
  to `resolve`). It owns state transitions, the parallel fan-out to agents, and checkpointing.
- **Pydantic** (`signals.py`) defines every inter-component message as a schema-validated model
- **Flask** (`api.py`) is a thin HTTP layer with no business logic; every route reads the DB or delegates to the
  Controller/System.
- **SQLAlchemy + SQLite** (`db/models.py`) is the knowledge base and event/situation/warning/alarm log

## Quickstart

** Run the backend **
```bash
pip install -e ".[dev]"                 # or use the pre-built .venv/
python scripts/run_server.py            # API at :5000/api, serves frontend/dist/ if built
```

** Run the frontend **

```bash
cd frontend && npm install && npm run build
```

The frontend features several example scenarios. You may run them by choosing one in the "Simulation" tab.

## Human-in-the-loop (HITL)

HITL is implemented, via a warning that pauses the workflow (a LangGraph interrupt) until a human answers:

- A warning is colour-coded **yellow** or **orange** by suspicion (orange above 0.75, lowered to **0.5 at night**). The user elevates (`{elevate}`) or dismisses (`{dismiss}`) it in the
  frontend.
- **Fallback policy:** if a warning goes unanswered within a fixed time frame or nobody could be
  reached, it resolves automatically — yellow is dismissed, orange is elevated to an alarm. During the night, this threshold is lowered, meaning that more situations may trigger an alarm.
  This is done to account for the higher chance of missed notifications during the night.
- An alarm caused by strong suspicion (score > 0.85) or a top-danger animal (e.g. a bear) bypasses HITL entirely and fires directly —
  those cases don't benefit from waiting on a human.

## Checkpointing, error handling, idempotency

- **Checkpointing:** each situation's LangGraph state is persisted after every step (`langgraph-checkpoint-sqlite`),
  keyed by the situation ID as thread ID. A crash or restart resumes the same situation from its last checkpoint.
- **Error handling:** every agent/service call goes through a timeout wrapper (`WorkflowConfig.agent_timeout_s`); a
  timeout, connection failure, or a response that fails its Pydantic schema is all treated identically as "not
  available" (`agents/base.py:InvalidResponse`). Depending on the failed agent, the fallback policy differs, e.g.
  - A person is labelled as `not_decidable`, if the person indentifier fails or is unresponsive. An alarm may still be triggered upon suspicious behaviour
  - A possible weather event does not trigger an alarm, if the weather forecast cannot be fetched.
- **Idempotency:** every warning/alarm/speaker signal has a key derived from `<situation_id>:<cause>`; the
  Controller checks a persistent store before sending and no-ops on a repeat. This makes retries, a resume from
  checkpoint, or multiple situations reaching the same conclusion safe from double-firing.

## AI-based vs. deterministic parts

- **AI-based (agents):** object detector, person identifier, behavioural/noise/weather interpreter — stateless,
  answer only in a validated structured schema, never call each other or trigger side effects directly.
- **Deterministic:** the Controller (thresholds, permission checks, idempotency, alarm/warning decisions), the
  Communication Unit (warning colour/night logic, fallback policy).
- **Human:** can elevate/dismiss a warning.

## AI tools used

For the agents, there is currently no real LLM/vision-model backend wired in: `agents/mock.py`'s
`ScriptedModel` returns canned, per-demo answers (see `demos.py`) against the same `Model` protocol
(`agents/base.py`) a real backend would implement.
Swapping in a real model means providing a `Model` that turns `(system prompt, request, JSON schema)` into a
schema-valid JSON answer — no change to the agents, workflow, or controller is required.

## Known limitations

- No real vision/LLM model connected (mocked, see above)
- no authentication
- no editing of people/areas/rules
- GDPR/personal-data handling is explicitly out of scope for the challenge
- SQLite is a challenge-scope stand-in for a production database.