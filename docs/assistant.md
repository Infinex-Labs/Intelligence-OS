# The AI Assistant

Ask questions in English, get answers backed by keyframes. The interesting part
is what the model is *not* allowed to do.

## It is not a chatbot over your footage

The hard constraint is: **the model may not assert anything it cannot point at.**
That is enforced by construction, not by prompting.

```
   "who was near the door yesterday afternoon?"
                    │
                    ▼
        ┌───────────────────────┐
        │  LLM: parse to query  │   ← the ONLY thing the model does
        └───────────┬───────────┘
                    │  {start, end, zone, entity_label, predicate_contains}
                    ▼
        ┌───────────────────────┐
        │  SQL over observations│   ← every fact comes from here
        └───────────┬───────────┘
                    ▼
     arrivals, departures, durations, keyframes, rule events
```

The LLM's only job is to turn the English question into a structured query:
a time window, a zone, an entity label, a predicate filter. It is given the
current time, the known zone names and the known entity labels, and it must
choose from those lists.

Everything in the answer — who arrived, when they left, how long they stayed,
which rules fired, which keyframes to show — is **aggregated deterministically
from the observation rows that query returns**. There is no free-text generation
in the answer path, so there is nothing to hallucinate. If the query returns
nothing, the answer is that nothing was seen.

**The query itself is returned with the answer.** Every reply in the UI has an
inspectable trace: expand it and you see exactly what was asked of the database.
If an answer looks wrong, the trace tells you whether the model mis-parsed the
question or memory genuinely doesn't contain what you expected — two very
different problems.

## Follow-ups

Earlier turns are passed to the planning step, so "what about yesterday?" or
"and was anyone with them?" resolve into a complete query on their own. History
is clamped to the last six turns.

The history comes from the **server's** copy of the thread, not from the
browser. A client cannot fabricate a conversation to steer the planner.

## Chat history

Threads live in SQLite (`conversations`, `chat_turns`), which buys three things
localStorage wouldn't:

**They survive.** Reload, different browser, different machine — same threads.

**Reopening replays; it does not re-run.** Each turn stores the evidence payload
its answer was rendered from. Open a thread from last week and you see what
memory said *then*. A record of what the system reported must not silently
change when memory moves on — the same reason `reports` stores its body rather
than a query.

**The KPIs are aggregates, not tallies.** Threads, questions, entities surfaced,
observations scanned, keyframes cited, rule events, average answer time: every
figure is a `SUM` over stored turns, denormalised onto `chat_turns` at write
time so the strip is one query rather than a JSON scan over every answer ever
given. The browser keeps no counter of its own to drift.

Per-answer counts appear under each reply, so a single answer can be audited on
its own, and **Export** writes the open thread to Markdown with its traces
included.

## Ownership

Threads are scoped to the signed-in user, and the scope is a `WHERE` clause on
every read *and* mutation in `store.py` — knowing a thread id is not authority
to read, rename, delete or append to it. A thread that isn't yours reads as
**absent (404)**, not forbidden, so an id cannot be probed for existence.

Threads created from the CLI carry no owner and stay visible to anyone signed
in; that is deliberate, and tested.

## Using it

**In the UI** — the AI Assistant pane. Suggestions on an empty thread, Enter to
send, Shift+Enter for a newline.

**From the CLI** — no server needed:

```bash
python -m intelligence_os.ask "who was near the door this afternoon?"
```

**Over HTTP** — see [api.md](api.md):

```bash
curl -b jar -X POST localhost:8000/api/ask \
     -H 'Content-Type: application/json' \
     -d '{"question": "who was in the loading bay yesterday?"}'
```

The response carries `answer`, `entities` (with their keyframes and rule
events), `total_observations`, the query trace, `conversation_id`, `turn_id`,
`counts` and refreshed `stats`. Omitting `conversation_id` starts a new thread;
passing one that isn't yours quietly starts a new thread rather than appending
to a stranger's.

## Requirements and limits

- **Needs `ANTHROPIC_API_KEY`** — this is the one pane that does. Everything
  else (detection, tracking, zones, rules, distillation, the timeline) works
  without one.
- **It can only answer from what was observed.** If no zone was drawn, nothing
  has a location; if identity is off, people are not the same person across
  days. The assistant will tell you it saw nothing rather than guess.
- **Question shapes are limited** to what the query tool can express: time
  window, zone, entity, predicate substring. Counting and comparison questions
  are on the [roadmap](../ROADMAP.md).
