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
                    │  {intent, start, end, zones[], cameras[],
                    │   entity_labels[], entity_type, text, order,
                    │   limit, group_by, ...}
                    ▼
        ┌───────────────────────┐
        │  SQL over memory      │   ← every fact comes from here
        │  observations ·       │
        │  habits · snapshots   │
        │  words + meaning      │
        └───────────┬───────────┘
                    ▼
                    │
                    ▼  nothing found?
        ┌───────────────────────┐
        │  loosen ONE thing,    │   ← and say which
        │  retry, up to 3x      │
        └───────────┬───────────┘
                    ▼
     arrivals, departures, durations, keyframes, rule events,
     recurrence, who-was-there-too, distilled connections
```

The LLM's only job is to turn the English question into a structured query: a
time window, some places, some cameras, some people (or some people to leave
out), what kind of thing, the words to look for, an order and a cap. It is given
the current time, the known zone names, the known camera names and the known
entity labels, and it must choose from those lists — a name that is not on a
list does not exist in this memory, and a filter that matches nothing returns
nothing rather than quietly widening to everything.

The query also carries an `intent`, which filters nothing. It says which part of
the result is the answer: "how many sightings at the bay" and "who was at the
bay" run the identical query, and only one of them is asking for a number.

Three intents go further and read a different part of memory altogether.
`how_often` reaches the mined habits, `who_with` reaches the settled scene
inventories, and `relations` reaches the distilled edges. These change *who the
answer is about* — "who was with Priya" names Priya and answers with somebody
else — so the names in the query select the anchor, not the answer.

Recurrence is counted from the sightings that matched and corroborated by the
mined habit, which are two different claims and both are shown. Six sightings
means it happened six times; six sightings under a confirmed habit means it is
what that subject *does*. Habits are bucketed on the deployment's local clock —
the same one answers are rendered in — so "most Tuesdays around 2" means two in
the afternoon where the cameras are.

## Finding the right words, and finding the right meaning

The words in a question go to two indexes at once. One matches **terms**: it
stems, so "cigarettes" finds "cigarette", and it is exact about it. The other
matches **meaning**: it finds "standing around, waiting" when you asked about
loitering, which no amount of stemming ever could — the two share not one
character sequence.

The two lists are then fused by rank rather than by score, because a term score
and a meaning score are not on a common scale and never can be. Fusion only ever
*adds* candidates: a row the term index found cannot be pushed out of an answer
by a meaning ranker that disagrees with it.

Neither index is allowed to widen the question. The window, the zones, the
cameras and the people are decided in SQL first, and both indexes only ever rank
what those already permitted. A question about Tuesday cannot return Wednesday
because the ranker liked it.

**Every hit says which index found it** — `lexical`, `semantic`, or `both` — and
that is the point rather than a detail. A row found by meaning alone used
different words from the ones you typed, so the answer tells you what was
actually recorded instead of repeating your phrasing back at you.

Meaning-based search needs a local model (~120MB, downloaded once, run entirely
on your machine — no keys, no network). Without it, search is term-matching
alone: narrower, never wrong.

## Why it will not search your footage by appearance

The obvious next feature is asking for *"the red van"* and having the system look
at the pictures. It does not, and the reason is worth stating because the
capability half-exists.

An image model can be asked which stored frame is closest to a phrase, and it
answers — always. What it cannot do is decline. Measured on this system's own
retained frames, a dim indoor room containing a dog and a sofa, the phrase *"a
hospital bed"* scored higher than nine of the twelve things genuinely in shot.
No threshold separates the two: at the point where three-quarters of real matches
have been discarded, false ones are still getting through.

A search that always returns its closest frame would answer *"was there a red
van?"* with a photograph of somebody in a white shirt. That is worse than
answering nothing, because a keyframe reads as proof, and it would undo the one
thing the section above is for — if every question returns something, *"I could
not find it"* and *"it did not happen"* become the same reply again.

So the image model is used only where it cannot make that mistake: **ordering
results the word and meaning indexes already found.** Ask about the dog and the
frame where the dog is actually the subject sorts to the front. It cannot add a
result, cannot remove one, and cannot make an empty answer non-empty. It is off
unless switched on (`visual_reranking: true`), costs ~350MB of weights, and
buys the order of an answer rather than its reach.

## When nothing matches, it says where else it looked

"I could not find it" and "it did not happen" used to be the same reply, which
is the worst thing this system can do, because the difference is invisible.

So when a question returns nothing, the assistant loosens **one** constraint,
tries again, and stops at the first thing it finds. It widens the time window
first — the commonest near miss is asking about "after six" when the sighting
was at 17:52 — then drops the place, then the person, then falls back to
searching by meaning alone. At most three attempts.

Three things it will not do, and they are what make widening safe rather than
convenient:

- **It only widens a question that found nothing.** A question that got an
  answer is untouched, so nothing that worked can start returning something
  else.
- **It only drops a filter that named something real.** Ask about someone who
  has never been seen here and the name is *not* dropped — abandoning it would
  answer about a stranger, which is not a wider answer to your question, it is a
  confident answer to a different one. The same goes for a place that isn't a
  place.
- **It never overrides an exclusion.** "Anyone except the courier" comes back
  without the courier, however far it has to widen.

**A widened answer always says it was widened** — in the prose, in the trace,
and in the CLI output. An answer that quietly answered an easier question would
be worse than the empty result it replaced. And when the widening finds nothing
either, it tells you what it tried, which is the difference between *"nothing
found"* and *"nothing found, and here is everywhere else I looked"*.

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
- **A widened answer is a weaker claim, and reads like one.** It answers a
  question adjacent to yours and says which one. If you would rather a miss
  stayed a miss, set `widen_empty_searches: false` in `config.yaml` — the
  answers you already get do not change, because widening only ever runs on a
  question that found nothing.
- **Question shapes are limited** to what the query tool can express: a window,
  places, cameras, people (or people to leave out), a kind of thing, words to
  look for, an order, a cap, and one of eight intents. Counting, recurrence,
  co-presence and connections are supported; open-ended comparison ("was it
  busier than last week?") is not.
- **Paraphrase needs the optional model.** With `sentence-transformers`
  installed, `"loitering"` reaches a stored `"standing around, waiting"`. Without
  it, matching is term-based only and different vocabulary is not reached.
  Build the index with `python -m intelligence_os.semantic`.
- **Appearance is not searchable, only re-orderable.** If nobody described the
  van as red, no question finds it by being red — see above for the measurement
  that settled that. Visual re-ranking changes which of the results you already
  had comes first; it never changes which results you get.
- **Meaning cannot read negation.** Asked about an open gate, a meaning match
  scores `"gate — closed"` about as highly as `"gate — open"`; the words *open*
  and *closed* are what tell them apart, and that is the term index's job. Both
  run, which is why both exist.
- **Recurrence answers, and meaning search, lag by a distillation pass.** The
  counts are live. The mined habit that corroborates them, and the vector that
  makes a newly described scene findable by meaning, both appear after the next
  pass — a minute in the live configuration. Term search is immediate.
