# Search & Retrieval Architecture Plan

**Status:** Phases 0–6 landed; 7–8 still proposal. See
[`search-baseline.md`](search-baseline.md) for the scored trajectory.
**Scope:** the query/retrieval layers of Intelligence OS — `ask.py`, the search
surface of `store.py`, the memory-write path in `run.py`/`vlm.py`, and the
recurrence tables in `distill.py`.
**Non-scope:** vision models, capture, tracking, identity matching, alerting.
Perception is not the problem this plan solves.

---

## 0. Thesis

> Storage is ahead of retrieval.

Intelligence OS records three kinds of memory — sightings, habits, and
who-was-present-together — all traceable to a keyframe. Search reaches **one** of
the three, by exact substring match, over a database that **discards roughly half
of what the vision model actually reported**.

The result is a memory that answers questions phrased the way the model happened
to write them down, and returns `0 observations` — indistinguishable from *"it
never happened"* — for everything else.

Nothing in this plan requires a better vision model. It is an indexing and
retrieval problem, and the largest fixes need **no new dependencies**.

---

## 1. Where we are today (audit)

### 1.1 The current query path

`ask.ask()` → `plan_query()` → `execute()` → `narrate()`.

**`plan_query()`** (`ask.py:74-102`) makes one LLM call that fills a five-slot
form (`ask.py:24-45`):

| slot | type | expressiveness |
|---|---|---|
| `start` | epoch or null | one window |
| `end` | epoch or null | one window |
| `zone` | string or null | **one** place |
| `entity_label` | string or null | **one** person |
| `predicate_contains` | string or null | **case-folded substring test** |

**`execute()`** (`ask.py:105-168`) then:

1. `store.observations(since=start)` — loads **every row** from `start` onward
   into Python (`ask.py:120`). With `start=None`, that is the whole table.
2. Filters `end`, zone, and predicate in a Python loop (`ask.py:124-128`) —
   despite `store.observations()` already supporting `until` and the DB having an
   index on `timestamp` (`store.py:187`).
3. Calls `store.get_entity()` **inside the loop** (`ask.py:133`). An entity
   rejected by the label filter is never cached, so it is re-fetched on every
   subsequent row it appears in — an unbounded N+1.
4. Sorts by `first_seen`, truncates to `keyframes[:4]` and `states[:8]`
   (`ask.py:163-168`). There is **no relevance ranking anywhere** in the system.

### 1.2 Confirmed gaps

| # | Gap | Evidence |
|---|---|---|
| **G1** | Search is exact-substring, not lexical or semantic. `"smoking"` does not match the stored `state:having a cigarette`; `"on the phone"` does not match `state:on phone`. | `ask.py:128` |
| **G2** | `SceneDescription.states()` keeps only `people.state`, `context_analysis` and `notable`. **`locations` and `objects` are dropped entirely**, and `desc.raw` is never persisted. | `vlm.py:137-147`, `run.py:291-305` |
| **G3** | `ask.py` never queries `relations` (habits) or `scene_snapshots` (co-presence). Two of three memory kinds have **no** search path. | `ask.py` imports only `observations`, `get_entity`, `locations`, `list_entities` |
| **G4** | No relaxation, no re-plan, no reflection. First miss is the final answer, and "not found" is reported identically to "did not happen". | `ask.py:240` |
| **G5** | Query form cannot express: multiple zones, camera, entity type, negation, ordering, limit, or count/recurrence intent. | `ask.py:24-45` |
| **G6** | Full-table scan into Python per question; degrades linearly with footage forever. | `ask.py:120-162` |
| **G7** | No search-quality test set. No change to ranking can be shown to help, and regressions are silent. | `intelligence_os/tests/` |
| **G8** | Habits are bucketed by **hour-of-day only** (`present_around_14h`) with **no weekday dimension**, and bucketed in **UTC** (`time.gmtime`) while answers are rendered in **local time** (`time.localtime`). Recurrence questions involving weekdays are unanswerable, and hour buckets are wrong for any non-UTC deployment. | `distill.py:148`, `ask.py:194` |
| **G9** | `mine_habits`/`mine_relations` scan the entire observations table with no time bound, and `mine_relations` issues `n` separate writes per relation in a `for _ in range(n)` loop. | `distill.py:140`, `distill.py:186`, `distill.py:196` |

### 1.3 What is already right, and must survive

These are load-bearing properties. Every phase below is constrained by them.

- **The model cannot assert anything it cannot point at.** The LLM writes the
  query and phrases the result; every fact is aggregated from real rows.
- **Every claim carries a keyframe.**
- **Cross-camera identity holds** — one person stays one entity.
- **The query trace is returned with the answer**, so any result is auditable.
- **It degrades, never breaks, with no API key** (`ask.py:79-82`, `vlm.py:182`).

---

## 2. What we are borrowing from memories.ai

Their Large Visual Memory Model is described as six cooperating stages: a **Query**
model (cue → searchable request), a **Full-Modal Indexing** model, a **Retrieval**
model (coarse recall), a **Selection** model (fine-grained extraction), a
**Reflection** model (monitors its own recall), and a **Reconstruction** model
(assembles the memory into an answer).

The transferable insight is **not** the models. It is the shape:

> **Index everything. Cast wide. Then narrow. Then check your own work.**

We currently do the inverse — one narrow exact guess, and whatever it catches is
the answer.

| their stage | ours today | plan |
|---|---|---|
| Query | 5-slot form | **Phase 4** — intent + plural fields + free-text cue |
| Indexing | discards half, indexes none | **Phase 1–2** — persist all, FTS5 index |
| Retrieval (coarse) | absent | **Phase 2 + 6** — lexical + semantic recall |
| Selection (fine) | absent | **Phase 6** — fusion, rerank, dedup, diverse keyframes |
| Reflection | absent | **Phase 7** — deterministic relaxation ladder |
| Reconstruction | ✅ present, and stronger than theirs | keep unchanged |

### 2.1 Explicitly NOT borrowed

- **"Unlimited video context" / feeding footage to a model to answer from.** This
  is precisely what our architecture refuses to do, and refusing is why our
  answers cannot be fabricated. Our Reconstruction stage stays as-is.
- **On-device token compression (MARC), wearable/robotics form factors.** Solves
  a battery-powered-camera-on-a-face problem. We are fixed cameras with a real
  machine and a SQLite file.

---

## 3. Target architecture

```
question
   │
   ▼
┌──────────────────────────────────────────────────┐
│ QUERY PLAN  (1 LLM call, Phase 4)                │
│  intent · time · zones[] · cameras[] · labels[]  │
│  · entity_type · exclude[] · text cue · limit    │
└──────────────────────────────────────────────────┘
   │
   ├─── intent = how_often ──────► habits + row counts (Phase 5)
   ├─── intent = who_with ───────► snapshots table     (Phase 5)
   ├─── intent = relations ──────► distilled edges     (Phase 5)
   │
   ▼  intent = who / when / count / timeline / last
┌──────────────────────────────────────────────────┐
│ HARD FILTERS  (SQL, Phase 3)                     │
│  time · zone · camera · entity · origin · conf   │
│  — these MUST match. Never relaxed silently.     │
└──────────────────────────────────────────────────┘
   │  candidate rows
   ▼
┌──────────────────────────────────────────────────┐
│ COARSE RECALL  (Phase 2 + 6)                     │
│  ├ lexical   FTS5 bm25   "smok*"                 │
│  └ semantic  cosine      "having a cigarette"    │
└──────────────────────────────────────────────────┘
   │  two ranked lists
   ▼
┌──────────────────────────────────────────────────┐
│ SELECTION  (Phase 6)                             │
│  RRF fuse → dedup near-identical → rank          │
│  → diverse keyframe pick                         │
└──────────────────────────────────────────────────┘
   │
   ▼  empty?
┌──────────────────────────────────────────────────┐
│ REFLECTION  (Phase 7)                            │
│  relax one step, retry, RECORD WHICH STEP        │
└──────────────────────────────────────────────────┘
   │
   ▼
 RECONSTRUCTION (unchanged) — narrate over real rows only
```

**The invariant that makes this safe:** structured constraints (time, zone,
camera, entity) are **hard filters**. Wording is a **ranking signal**. A question
about Tuesday can never return Wednesday because the ranker liked it.

---

## 4. The phases

Each phase is independently shippable, leaves tests green, and is reversible.

---

### Phase 0 — Measurement first ✅ DONE

**Why.** G7. Without a scored test set, every phase after this is faith-based:
we cannot show that ranking improved, and we cannot detect when it silently
regresses. Most cases fail against today's code, and that failure list *is* the
baseline.

**Measured baseline** — see [`search-baseline.md`](search-baseline.md):

| metric | today |
|---|---|
| cases passing | **9 / 32** |
| **silent failures** | **22 / 32** — an answer existed; nothing was returned, and nothing said so |
| perception retention | **46.2%** — `objects` **0/6**, `locations.contents` **2/10** |
| recall@1 / @5 / MRR | 62.5% / 62.5% / 0.625 |
| p50 / p95 latency @ 100k obs | **427 ms / 1113 ms** |

Two findings sharper than the audit predicted:

- **`objects` retention is 0/6, not "partial".** Every object the vision model
  described is lost. The two `locations.contents` facts that *do* survive only do
  so because another section of the same frame happened to repeat them.
- **G6 is already real, not theoretical.** p95 is over a second at 100k
  observations — a size a single busy camera reaches in weeks, not years.

**What was built.**

1. **`tests/fixtures/search_corpus.py`** — a deterministic memory anchored to a
   fixed UTC Tuesday (`ANCHOR = 1780408800`, asserted so it cannot drift). 7
   entities, 3 zones, 2 cameras, deliberately awkward stored wordings
   (`state:having a cigarette`, `state:on phone`, `state:standing around,
   waiting`), 8 van visits across 6 Tuesdays + 2 Thursdays for recurrence, and
   two co-presence snapshots.

   The load-bearing decision: the fixture writes VLM output by calling the **real
   `SceneDescription.states()`** and the real `run.py` subject-resolution rule,
   not a copy. So whatever production discards, the fixture discards — and when
   Phase 1 stops discarding, retention moves **without this file changing**.

2. **`tests/fixtures/search_cases.yaml`** — 32 cases, each tagged with the gap it
   exhibits, the phase expected to fix it, and its recorded baseline. Six are
   regression guards on behaviour that already works; three are true negatives
   that must *stay* empty (the guard against Phase 7 relaxing its way into
   inventing an answer).

3. **`tests/test_search_quality.py`** — scores with the planner **stubbed**, so
   this measures retrieval rather than how well the LLM parses English.

4. **`scripts/search_eval.py`** — the scorecard, and `--write-baseline`.

**Two design decisions worth keeping.**

- **An ignored plan field is a hard failure.** `execute()` silently drops fields
  it does not understand, which is not "no filter" — it returns *the whole
  table*. A case whose plan used `text:` or `intent:` would therefore have
  **spuriously passed** on unfiltered results. The runner declares
  `SUPPORTED_PLAN_FIELDS` and fails anything outside it. Each phase extends that
  set; the comments in it are the record of what the engine could do when.
- **The baseline is asserted in both directions.** A regression fails the build,
  and so does an *improvement*, until its `baseline:` is flipped. Nothing moves
  silently — including good news.

**Retention is scored per report, not globally.** A fact counts as retained only
if a row written from the *same keyframe* carries every content word of it.
Scoring against the whole database flattered the number — an object fact looked
"retained" because an unrelated frame's state mentioned the same noun. Matching
is token-superset rather than substring, so it survives Phase 1 choosing a
different separator when rendering `observations.text`.

**Risk.** Low — tests only, no production code touched.

**Done.** `python scripts/search_eval.py` runs offline with no API key. CI runs
the gate as a named step, plus the scorecard on every build so the numbers land
in the log even when nothing drifted. 114 tests pass (was 110); ruff clean.

---

### Phase 1 — Stop discarding what was perceived ✅ DONE

**Why.** G2. The vision model already reports `objects[]` (with descriptions) and
`locations[].contents`. `SceneDescription.states()` throws both away, and the
full report is never written anywhere. *"Was the gate left open?"* is unanswerable
because the answer was seen, described, and then deleted before the insert.

**Work.**

1. New table (additive, via `_migrate_v5_search()` following the existing
   `_migrate_*` try/except convention at `store.py:231-292`):

   ```sql
   CREATE TABLE IF NOT EXISTS scene_descriptions (
       description_id TEXT PRIMARY KEY,
       camera_id      TEXT,
       location_id    TEXT,
       timestamp      REAL NOT NULL,
       source_ref     TEXT,            -- keyframe, for provenance
       model          TEXT NOT NULL,   -- which VLM produced it
       raw            TEXT NOT NULL,   -- the COMPLETE report, verbatim JSON
       text           TEXT NOT NULL    -- flattened prose, for Phases 2 & 6
   );
   CREATE INDEX IF NOT EXISTS idx_desc_time ON scene_descriptions(timestamp);
   ```

2. `ALTER TABLE observations ADD COLUMN description_id TEXT` — links a sighting
   back to the full report it came from.

3. `ALTER TABLE observations ADD COLUMN text TEXT` — the human-readable rendering
   of the row, which is what Phases 2 and 6 index. `state:having a cigarette`
   → `having a cigarette`; an object row → `gate — open, unlatched`. Predicates
   stay untouched: they remain the machine contract; `text` is the search surface.

4. `vlm.py` — add `SceneDescription.flatten() -> str` and
   `SceneDescription.rows()` that emits **all four** report sections, not three:
   people states, context analysis, notable, **objects** (`label` + `description`),
   and **locations** (`location` + `contents`).

5. `run.py:291-305` — write the description row first, then observations carrying
   `description_id`. Object and area rows attach to a `scene` pseudo-subject when
   no entity owns them, so they stay queryable without faking entity identity.

**Outcome.** Nothing the vision model says is lost. Object and area questions
become answerable — from the ship date forward.

**Risk.** Low-medium. Row volume per VLM call rises (roughly 3× on busy frames).
Mitigation: cap emitted rows per description and reuse the existing keyframe
pruning (`distill.py:33`). No extra API calls — we are already paying for these
words.

**Explicit limit.** This is **not retroactive**. Footage already processed never
had objects/areas written; that data is gone. Old sightings still gain the index
and ranking of Phases 2–6.

**Done when.** A fixture frame's full report round-trips out of
`scene_descriptions`, and objects/areas appear as searchable rows.

**Result.**

| | before | after |
|---|---|---|
| perception retention | 46.2% (12/26) | **100.0% (26/26)** |
| `objects` | 0/6 — total loss | 6/6 |
| `locations.contents` | 2/10 | 10/10 |
| cases passing | 9/32 | 9/32 — see below |

Landed as specced, with three decisions worth recording:

- **The unowned subject is `scene:<location_id>`, not the first person in frame.**
  `store.scene_subject()` mints it and `ask.execute()` renders it as a
  pseudo-entity named for the zone. Attributing "the gate is open" to Dave
  because Dave was nearby would have been a fabricated claim about a person,
  which is the one thing this system is built not to do.
- **`predicate` and `text` are separate fields on purpose.** The predicate stays
  the machine contract that rules and distillation match on; `text` is the
  search surface. Rewording a row for searchability can now never break a rule.
- **One write path, not two.** `vlm.record_description()` holds the whole rule
  and both `run.py` and the Phase 0 fixture call it, so the fixture measures
  production rather than a copy of it that could drift.

**Case count did not move, and that is correct.** The four object/area cases
query with a `text` field the engine does not honour until Phase 2, and the
harness scores an unsupported plan field as a hard fail rather than let an
unfiltered result pass for the wrong reason. The data they need now exists;
Phase 2 makes it reachable.

**Also unchanged by design:** production VLM rows still carry no `location_id`
for people-facts — one VLM call covers a frame spanning several zones, so
attributing a person's state to one zone would be a guess. Area facts *do* get a
zone, from the name the model itself used. `p95` latency rose within measurement
noise (~1150 → ~1249 ms); Phase 3 is what addresses latency.

---

### Phase 2 — The lexical index (zero new dependencies) ✅ DONE

**Why.** G1. SQLite ships FTS5. We are hand-rolling `in` on a string
(`ask.py:128`) instead of using it. FTS5 gives word variants, phrases,
`NEAR`, and — critically — **bm25 ranking**, which turns a pass/fail gate into a
scored list.

**Work.**

1. External-content FTS table over the `observations.text` column added in
   Phase 1:

   ```sql
   CREATE VIRTUAL TABLE IF NOT EXISTS obs_fts USING fts5(
       text,
       content='observations',
       content_rowid='rowid',
       tokenize='porter unicode61'   -- porter stems: smoking→smok, gates→gate
   );
   ```
   Plus `desc_fts` over `scene_descriptions.text`.

2. `AFTER INSERT / UPDATE / DELETE` triggers on both tables to keep the index in
   sync. Backfill for existing databases via
   `INSERT INTO obs_fts(obs_fts) VALUES('rebuild')`, guarded and idempotent.

3. `Store.search_text(query, *, since, until, location_ids, camera_ids, limit)`
   → rows with a `bm25` score, filters applied **in the same SQL statement** so
   we never materialise the non-matching set.

**Design note — what does *not* go in the index.** Entity labels and zone names
stay **out**. They are structured fields with their own filters; indexing them as
text would let a renamed entity leave stale strings in the index, and would let a
name accidentally rank a row that the structured filter should have excluded.

**Outcome.** `"smoking"` matches `smoking`/`smoked`/`smokes` via stemming.
`"on the phone"` matches `on phone` (stopword-tolerant phrase handling). Results
arrive **ranked**. Paraphrase across different vocabulary still misses — that is
Phase 6.

**Risk.** Low. Purely additive; `predicate_contains` keeps working throughout.
FTS5 is present in every SQLite build we ship on — add a `PRAGMA
compile_options` assertion mirroring the existing thread-safety check in
`test_store_threads.py`, and degrade to substring matching if absent.

**Done when.** `search_corpus` proves stem and phrase matches, backfill works on
a pre-Phase-1 database copy, and Phase 0 recall improves measurably.

**Result.**

| | before | after |
|---|---|---|
| cases passing | 9/32 | **17/32** |
| silent failures | 22/32 | **14/32** |
| recall@1 / MRR | 62.5% / 0.625 | **72.7% / 0.727** |
| p50 / p95 @ 100k | 600 / 1853 ms | 609 / 1878 ms (paired, same session) |

Landed as specced — `_migrate_v6_fts()`, both indexes, all six triggers,
`Store.search_text` / `search_descriptions` — with four decisions worth recording:

- **A user's words are data, never syntax.** `Store.fts_query()` extracts word
  characters and quotes each term individually, so a question containing `"`,
  `*`, `:` or the word `OR` is a search, not a syntax error and not an operator.
  This is the whole safety boundary: everything past it is FTS5 grammar. `search_text`
  still catches `OperationalError` and returns nothing — a malformed query is an
  empty answer, never a 500.
- **Stopwords are dropped from the query, not required of the row.** The memory
  stores `on phone` and the question is *"on the phone"*; demanding every typed
  word fails on the one word carrying no meaning. A query that is *nothing but*
  stopwords keeps them, because dropping them all leaves a match-everything.
- **`predicate_contains` became a union, not a replacement.** It keeps its
  substring behaviour — rule rows carry no prose, so the index cannot see
  `rule_fired` at all — and gains the index on top. A phase that widens recall
  must never take an existing hit away. The new `text` field is the strict one:
  match, or you are not a result. Its planner description also had to be
  inverted: `'smok'` used to be the safe way to reach *"smoking"* and is now the
  way to miss it, since porter stems `smoking`→`smoke` but `smok`→`smok`.
- **An unscored hit reports `null`, not `0.0`.** A substring match was never
  seen by bm25, and calling that a zero would rank it as the worst match rather
  than the unranked one it is. Scored results sort first, then unscored, then by
  time; `ranked` in the payload says which ordering was used.

**Two things the numbers do not say.** Four of the eight flipped cases are
Phase 1's work collected late — the facts were already written and `text` merely
became reachable. And bm25's magnitude is meaningless at this corpus size: with
few documents the IDF term collapses and scores land around 1e-6. It orders
results; it must never gate them, which is why there is no score threshold
anywhere.

**Not done here, deliberately.** `QUERY_TOOL` still has its five slots — the
widened query form is Phase 4. Phase 2 reaches production through
`predicate_contains`, which the planner already emits.

---

### Phase 3 — Push filtering into the database ✅ DONE

**Why.** G6. Every question loads the table into Python. The indexes to avoid
this already exist (`store.py:186-192`); `execute()` just does not use them.

**Work.**

1. Extend `Store.observations()` (`store.py:660`) with `location_ids`,
   `predicate_prefixes`, `exclude_predicates`, `origins`, `min_confidence`,
   `entity_ids`, `order`, `limit` — all composed into the `WHERE` clause.
2. Composite indexes:
   ```sql
   CREATE INDEX IF NOT EXISTS idx_obs_time_loc  ON observations(timestamp, location_id);
   CREATE INDEX IF NOT EXISTS idx_obs_subj_time ON observations(subject_entity_id, timestamp);
   ```
3. Rewrite `execute()` to consume pre-filtered rows. **Fix the N+1**: batch-load
   entities with one `WHERE entity_id IN (...)`, and cache label-rejected ids so
   they are not re-fetched (`ask.py:133`).
4. Pass `until` through — it already exists and is simply never used.
5. Do counting in SQL (`COUNT`, `GROUP BY`) rather than Python accumulation.

**Outcome.** Flat latency as footage grows. Prerequisite for recurrence
questions, which are aggregate queries and unaffordable as Python loops.

**Risk.** Medium — this rewrites the aggregation core. Mitigation: Phase 0's
suite plus existing `test_ask.py` must produce byte-identical output for every
currently-passing case. Land as pure refactor, behaviour frozen.

**Done when.** p95 on a 100k-row DB is sub-second, and all pre-existing tests
pass unchanged.

**Result.**

| | before | after |
|---|---|---|
| p50 @ 100k | 609 ms | **0.49 ms** |
| p95 @ 100k | 1878 ms | **66.73 ms** |
| cases passing | 17/32 | 17/32 — unchanged, deliberately |
| tests | 161 | 202 |

Landed as specced, plus one thing the spec did not anticipate. Four decisions
worth recording:

- **The output was frozen by diffing it, not by hoping.** All 32 case results
  were dumped before the change and after it and compared field by field. The
  first diff was real and would never have been caught by the pass/fail score:
  two states listed under one entity had swapped places. Rows sharing a
  timestamp — which *every* multi-fact frame produces, since one report writes
  several rows at one instant — were being ordered by whatever index the plan
  happened to pick, so adding a filter reshuffled equal rows. `ORDER BY
  timestamp, rowid` makes the order total: ties break by insertion order, which
  is the order the report was written in. That bug predates this phase; pushing
  filters down is what made it observable.
- **`None` is not an empty list.** Every sequence filter reads `None` as "no
  filter" and `[]` as "a filter with no permitted values, matching nothing".
  Collapsing them would mean that asking about an empty set of zones returns the
  entire table — a search that silently widens at the moment it was asked to
  narrow. `exclude_predicates` is the one inverted filter, so an empty exclusion
  excludes nothing; same rule read from the other side.
- **A name is not a column.** The label filter was the last full scan (562 ms of
  the 1878), but it filters on the *displayed* name, which is derived: an
  unlabeled entity shows as `Person 4f2a91`, and a fact nobody owns shows as the
  zone it happened in — and a `scene:` subject may be keyed by a camera, or by
  nothing, when there is no zone. So `_label_for()` is now the single definition
  of that name, used both to resolve a name to an id set for SQL and to display
  it in the loop. Two definitions would mean filtering on one meaning and
  showing another. SQL narrows; the loop still decides.
- **Counting stayed in Python, contrary to item 5.** `total_observations` counts
  rows that survived *entity resolution* — a row whose subject no longer exists
  is not an answer — and SQL cannot see that. `Store.count_observations()` was
  added and used where a count is genuinely all that is wanted (`run.py`,
  `operator.py`, `phase1.py`, and the merge ranking in `store.py`, all of which
  built a full list of row objects and kept only its length).

**The N+1 was worse than "one query per row" suggests.** It was one
`get_entity()` per row *including rejected ones*, which were never cached, so
asking about Priya paid a query for every row belonging to everyone else. The
cost of a question scaled with how long other people stood in front of the
camera. It is now one batched `get_entities()` for the whole result.

**Not done here, deliberately.** The `LIKE '%…%'` half of the word union still
scans (~70 ms at 100k) and did not get an index, because Phase 4 retires
`predicate_contains` and rule matching becomes an indexed prefix lookup. The new
`predicate_prefixes`, `exclude_predicates`, `origins`, `min_confidence`, `order`
and `limit` filters are built and tested but not yet reachable from a question —
that is Phase 4's job, and this is the layer it needs underneath it.

**Still scanning elsewhere.** `distill.py:147,194` and `rules.py:228` iterate
`store.observations()` and filter in Python, exactly as `execute()` used to.
They are on the write path rather than the query path, so they were out of scope
here, but they are where the same treatment pays next.

---

### Phase 4 — Widen the query form (the Query stage) ✅ DONE

**Why.** G5. The five-slot form cannot express most real investigative questions.

**Work.** Replace `QUERY_TOOL` (`ask.py:24-45`) with:

| field | purpose |
|---|---|
| `intent` | `who \| when \| count \| how_often \| who_with \| timeline \| last` — **routes the query** |
| `start`, `end` | unchanged |
| `zones[]` | several places |
| `cameras[]` | which camera saw it (recorded since M6, never searchable) |
| `entity_labels[]` | several people |
| `entity_type` | `person \| object \| any` |
| `text` | **free-text cue** — the semantic/lexical query, replacing `predicate_contains` |
| `exclude_predicates[]` | "anyone except the delivery guy" |
| `min_confidence` | trust threshold |
| `order`, `limit` | "the last five" |

Also: `PLAN_SYSTEM` gains the known camera list; `_history_messages()`
(`ask.py:60`) is unchanged — follow-ups already work.

**Compatibility.** The planner is a single tool-call boundary, so the old shape
can be accepted and up-converted for one release. Stored chat turns keep their
rendered payload (`web.py:968`), so history replays without re-running queries.

**Outcome.** Multi-zone, multi-person, negation, ordering, limits — and the
`intent` field, which is what makes Phase 5 reachable at all.

**Risk.** Medium. A bigger schema means more planner mistakes. Mitigation: the
trace is already returned with every answer (`ask.py:267`); extend Phase 0 with
plan-accuracy cases run against a live key, marked `slow`.

**Done when.** All Phase 0 structured cases plan correctly, and old-shape plans
still execute.

**Result.** Landed as specified, plus two things the spec did not ask for.

| | before | after |
|---|---|---|
| cases passing | 17/32 | **23/32** |
| silent failures | 14/32 | **5/32** |
| tests | 209 | **266** |
| the 21 cases Phase 3 could run, p95 | 468.97 ms | **430.71 ms** |

All six G5 cases flip. The other twenty-six answer **byte-identically** to
before — diffed field by field, same harness as Phase 3, so "we only changed
what we meant to change" is a check rather than a claim.

Recorded decisions:

- **`predicate_contains` was not retired, only removed from the schema.** The
  spec says `text` replaces it. It does for the *planner* — the tool no longer
  offers the field — but `normalize_plan` still honours it, because they are not
  synonyms: `text` is the lexical index alone, while `predicate_contains` also
  keeps a literal substring test, and that is the machine contract `rule_fired`
  is matched on. Three currently-passing cases go through it. Folding it into
  `text` would have silently dropped half of a filter, which is the exact class
  of bug this phase exists to remove.
- **`limit` caps subjects, not rows.** SQL's `LIMIT` would be faster and wrong:
  an entity's `first_seen`, duration and states are aggregates over *all* of its
  rows, so truncating rows returns subjects with a quietly truncated history
  rather than fewer subjects. What did get optimised is finding *which*
  subjects: a bounded walk in the requested order takes the first *n* distinct
  ones, which is exact because a subject's `last_seen` is where it first appears
  reading backwards. So "the last two sightings" no longer aggregates the whole
  memory to throw all but two away.
- **A zone name that matches nothing now matches nothing.** Previously an
  unrecognised `zone` left the filter unset, and unset means *anywhere* — so
  asking about a place that does not exist returned everyone, everywhere,
  confidently. Every new filter is asserted to fail closed
  (`TestNarrowingNeverWidens`), because a wider form is only an improvement if
  narrowing can still come back empty.
- **`intent` routes `count` only.** `how_often`, `who_with` and `relations` are
  in the enum and parse, so the planner's vocabulary will not have to change
  again, but they reach their aggregates in Phase 5. Their cases now *run* and
  fail on the missing `expect_result_key` rather than being rejected before
  execution — which is why silent failures fell by nine while the case count
  moved by six.
- **`group_by` was deliberately left unsupported.** It is Phase 5's, and listing
  a field the engine ignores is the silent widening the harness exists to catch.

Not in the spec, done anyway:

- **The ranked shortlist is drawn from the filtered pool.** `search_text` caps
  its bm25 list, so filtering *after* it cost recall on exactly the narrow
  questions the new filters serve. It now takes the zone, camera and entity
  filters directly. This is why the shared-case p95 improved.
- **`entity_type: scene`.** A fact nobody owns — an open gate, a spill — is
  recorded against the place, and `person | object | any` gave no way to ask for
  those specifically.

**Still scanning elsewhere.** `entity_type_objects_only` and `limit_last_two`
take ~5 s at 100k rows: both aggregate every padding row because the fixture
puts all 100,000 on one subject, and a per-entity aggregate spans all of that
subject's rows (~3.2 s materialising, ~2.3 s in the loop — nothing wasteful).
Removing it means computing aggregates in SQL rather than in the Python loop,
which is the same lever Phase 5 needs. `distill.py:147,194` and `rules.py:228`
still scan and filter in Python, unchanged since Phase 3.

**Not done here.** The plan-accuracy suite against a live key is still not
written; the widened schema is exercised by 57 unit tests against stubbed plans,
which measures the engine and not the planner. That gap is unchanged from Phase
0 and is now larger, since there is more schema to get wrong.

---

### Phase 5 — Wire up habits and co-presence ✅ DONE

**Why.** G3, G8. Two of three memory kinds are unreachable. The README's
*"How often does that van come by?"* and *"Who was the person my wife let in on
Tuesday?"* are **currently unanswerable despite the data existing**.

**Work.**

1. **Fix habit granularity (G8).** `distill.mine_habits()` buckets by hour only
   (`present_around_14h`) and in **UTC** (`time.gmtime`, `distill.py:148`) while
   answers render in local time (`ask.py:194`). Add a weekday dimension and make
   the timezone explicit and consistent:
   - keep `present_around_14h` (any day, hour bucket)
   - add `present_tue_around_14h` (weekday + hour)
   - bucket in the deployment's local timezone, matching how answers are read
   Without this, *"most Tuesdays"* cannot be expressed at all.
2. `Store.habits(entity_ids, location_ids, min_weight, status)` — reads
   `relations` where `kind='habit'`, returning weight, `last_reinforced_at`, and
   the `supporting_observation_ids` provenance list that is already stored.
3. `Store.co_presence(entity_id, since, until)` — joins `scene_snapshots`
   (`store.py:707`) to find entities sharing a location and time bucket.
4. Route in `ask.execute()`: `intent=how_often` → habits;
   `intent=who_with` → co-presence. Both render with the **same** grounded
   evidence contract — every claim keeps its supporting observation ids and
   keyframes.
5. **Bound distillation (G9).** Give `mine_habits`/`mine_relations` a time window
   instead of scanning all history, and collapse the `for _ in range(n)`
   repeated-write loops (`distill.py:186`, `distill.py:196`) into a single
   weighted write.

**Outcome.** Recurrence and co-presence answers, with provenance:

```
The van has been by 14 times in the last 6 weeks — most often Tuesdays
around 14:00 (11 of 14). Last seen Tue 15:12.
  habit: present_tue_around_14h   weight 0.86   14 supporting observations
```

**Risk.** Medium. Habit predicates change shape, so `digest.py` and any
predicate-string consumer need checking. New predicates go through the existing
`register_predicate` contract (`store.py:821`); old habits keep working, and
weekday habits accumulate from the ship date.

**Done when.** Both README questions pass in Phase 0, with provenance attached.

**Result.** Both pass, and so do the other three.

| | Phase 4 | Phase 5 |
|---|---|---|
| cases passing | 23/32 | **28/32** |
| silent failures | 5/32 | **4/32** |
| recall@1 / recall@5 / MRR | 0.623 / 0.737 / 0.697 | **0.767 / 0.850 / 0.850** |
| tests | 266 | **312** |
| p95, the 21 cases Phase 3 can also run | — | 79.23 → **76.47 ms** |

Every remaining failure is a `paraphrase_*` case, and every one of them is
Phase 6's.

**Decisions recorded.**

- **The timezone question is answered: local.** It was left open in §8.2 and is
  now closed in favour of the deployment's clock, because that is the clock the
  answers already render in — a habit reported in a timezone nobody works in is
  not a pattern anybody recognises. `time.localtime` reads `TZ`, so a deployment
  states its timezone the way every other unix service does. One definition
  (`distill.bucket_hour` / `bucket_weekday` / `bucket_day`), three callers.
- **A recurrence is counted from the rows *and* corroborated by the habits.**
  The spec routes `how_often` to the habits table. Habits alone would answer
  "how often?" with nothing until the next nightly pass — over the window
  someone asking is most likely to mean — so the counts come from the matched
  observations and the mined habit rides along with its weight, status and
  supporting ids. Counts say it happened; the habit says it is a routine.
- **`who_with` and `relations` re-aim the subject filter.** Both name an anchor
  and answer with something else, so once the companions or the related things
  are resolved they replace the label filter entirely. Exclusions survive the
  swap, because "who was with her, apart from the courier" constrains the answer
  rather than the anchor.
- **The far end of a relation is built from the edge's own evidence.** A
  forklift is never the *subject* of an observation — it is only ever what
  somebody was near — so the row path cannot reach it however the filters are
  set. Its entry comes from `supporting_observation_ids`, which is the same
  provenance the graph UI walks. The window and zone filters reach the belief
  through that evidence, which is the only end they can reach it from.
- **The eval corpus now runs a mining pass.** A six-week-old memory has mined
  edges; a fixture without them is one where the scheduler never ran. The two
  miners are called directly rather than `Distiller.run()`, which ends in
  `prune_old_keyframes()` and would delete files out of the real `FRAMES_DIR`.

**Not in the spec, done anyway.**

- `intent=relations` was in the enum but had no route. It has one, including
  location-valued edges, so "where does she go" is answered by a `frequents`
  edge with the zone as the far end.
- `group_by` gained `hour` and `day` alongside `weekday`, and is defaulted in
  `normalize_plan` so the trace records which buckets actually ran.
- Two tests were asserting the timezone rather than the behaviour — `test_pipeline`
  on the literal `09h`, `test_digest` on `unusual_hour:…:03`. Both were only ever
  right on a machine in UTC. They ask for the bucket now.

**Measured, for G9.** One weighted write took **0.11 s** on the fixture's
worst-case edge (100,000 rows on one entity at one location). The loop it
replaces took 227 s for the first 2,000 of its 100,000 iterations, because each
one re-read and rewrote the whole provenance list — about three hours,
extrapolated, for one edge.

**Still scanning elsewhere.** `rules.py:228` is unchanged. So is the
aggregate-in-the-loop that makes `entity_type_objects_only` and `limit_last_two`
cost ~950 ms at 100k rows; Phase 5 did not need it, and it is still the lever
that would remove them.

**Not done here.** The plan-accuracy suite against a live key is still not
written. `intent` now decides which *table* an answer comes out of, so a planner
that picks the wrong one returns a confidently wrong shape rather than a
confidently wrong filter — the gap is worth more than it was.

---

### Phase 6 — Semantic retrieval and fusion (Retrieval + Selection) ✅ DONE

**Why.** Phase 2 fixes word *variants*. It does not fix different *vocabulary*:
`"loitering"` will never lexically match `"standing around, waiting"`. This is
the last big slice of G1.

**Dependency decision — answered: yes, optional.** Needs a local
sentence-embedding model via `sentence-transformers`. **torch is already
installed via ultralytics**, so marginal install weight is small and it runs
fully offline — the no-cloud-keys promise holds. It goes in the *optional*
dependency block in `requirements.txt` alongside `insightface`, and search
degrades to Phase 2 behaviour when absent.

**Work.**

1. Reuse the existing float32-blob convention (`store._pack`/`_unpack`,
   `store.py:526-532`):
   ```sql
   CREATE TABLE IF NOT EXISTS text_embeddings (
       embedding_id TEXT PRIMARY KEY,
       kind         TEXT NOT NULL,   -- observation | description
       ref_id       TEXT NOT NULL,
       dim          INTEGER NOT NULL,
       vec          BLOB NOT NULL,   -- float32, L2-normalised
       model        TEXT NOT NULL,   -- so a model swap is detectable
       created_at   REAL NOT NULL
   );
   CREATE INDEX IF NOT EXISTS idx_txtemb_ref ON text_embeddings(kind, ref_id);
   ```
2. `intelligence_os/semantic.py` — lazy-loaded encoder, batch backfill script,
   incremental embedding on write. Same "no dep → no-op" pattern as `vlm.py:182`.
3. Cosine top-k in numpy over the **hard-filtered** candidate set — the same
   approach `auto_merge_people` (`store.py:913`) already uses. Brute force is
   correct to ~10⁵ rows; revisit only if the eval shows latency pressure.
4. **Fusion** — Reciprocal Rank Fusion, `score = Σ 1/(60 + rank_i)` over the
   lexical and semantic lists. Rank-based, so it needs no score calibration
   between bm25 and cosine.
5. **Selection** — collapse near-identical rows (same entity + predicate within
   N seconds) into one hit with a count; pick **temporally diverse** keyframes
   instead of the first four (`ask.py:166`).
6. Return per-hit `match_reason` (`lexical` / `semantic` / `both` + score) so the
   trace explains *why* a row surfaced.

**Outcome.** `"loitering"` finds `"standing around, waiting"`. Ranked, deduped
results with the strongest evidence first and the reason visible.

**Risk.** Medium-high — the only phase with a new dependency, and semantic
matching can surface plausible-but-wrong rows. Mitigations: hard filters are
never relaxed by the ranker; a similarity floor below which semantic hits are
dropped; `match_reason` on every hit; the whole layer is behind a config flag
and off if the dep is missing.

**Done when.** Paraphrase cases pass, silent-failure rate drops sharply, p95
holds sub-second, and the suite still passes with the dependency uninstalled.

**Result.** All four met. `28/32 → 32/32`, silent failures `4 → 0`, recall@1
`76.7% → 91.7%`, recall@5 `85.0% → 100%`, MRR `0.850 → 1.000`, tests `312 → 352`.
p95 328 ms at 100k, well under the bar. With the dependency uninstalled the
scorecard is Phase 5's to the decimal and the whole suite is green — asserted on
every run, not checked once.

Decisions taken, and why:

1. **The model is `all-MiniLM-L12-v2`, not the L6 named above.** Rejected on
   measurement: L6 puts a real answer (`loitering` → `standing around, waiting`,
   0.276) and pure noise (`dog`, 0.248) 0.028 apart, so no floor separates them
   — either the paraphrase fails or an absent thing gets an answer. L12 opens
   that gap to 0.142 for 30MB more. `bge-small` was also tried and is worse for
   this: it scores everything high (0.494 for `dog`) because it is trained to
   rank, and a ranker has nothing to say about an absolute yes/no. Full table in
   the baseline doc.
2. **The floor gates; bm25 still only orders.** `min_similarity = 0.30`, in the
   middle of that gap. Cosine is an angle and comparable across queries, so it
   *can* be a threshold — and must be, or every question returns the nearest
   sentence in memory whether or not anything answers it.
3. **Vectors are built by the distillation pass, not on insert.** The spec said
   "incremental embedding on write"; the observation write path runs per frame,
   per camera, all day, and a matrix multiply does not belong there. The cost is
   that a row is lexically searchable at once and semantically searchable one
   tick later — the same lag the mined habits already have, and now documented.
4. **`text` goes to both indexes; `predicate_contains` goes only to the lexical
   one.** What is left of that field is the machine contract `rule_fired:` is
   matched on. Asking a meaning ranker which predicates *feel like* an
   identifier is not a question with an answer.
5. **Fusion is unconditional.** RRF over one list is order-preserving, so a
   deployment with no index gets exactly Phase 2's ranking through the same code
   path — rather than a branch under which the ranking silently becomes a
   different algorithm.

**Not in the spec, done anyway:**

- `cascade_delete` now removes a subject's vectors. An embedding of "holding the
  gate open" is derived data about a person; a §11 privacy removal that left it
  behind would delete them by name and leave the sentence searchable by meaning.
- Query vectors are cached (LRU 256). The forward pass is ~70 ms and is the
  entire cost of a semantic search; questions repeat far more than they look
  like they do. Cleared on encoder swap, since the cache is keyed on text alone.
- `INTELLIGENCE_OS_NO_SEMANTIC=1` and a two-baseline case format, so "does this
  still work without the dependency?" is answered on every run instead of once,
  on a laptop, by uninstalling things.
- `UNIQUE(kind, ref_id, model)` on the index. The backfill is a batch job that
  will be interrupted and re-run nightly forever; without it each run adds
  another copy of every vector and quietly weights those rows higher.

**Known limitation, stated rather than hidden:** a meaning index cannot read
negation. Asked for `"gate open"` the model ranks `"gate — closed"` (0.760) above
`"gate — open, unlatched"` (0.711). This is a property of sentence embeddings,
and a large part of why fusion is not a replacement for the lexical index — bm25
on the word *open* does tell those apart. Every hit carries `match_reason` so a
reader can see which index believed what.

**Not done here:** near-duplicate collapsing exists but is not applied to
`states`, whose shape is a public contract several callers read; it surfaces as
a parallel `hits` list instead. The `description` embedding kind is defined in
the schema and unused — nothing reads whole-report vectors yet.

---

### Phase 7 — Reflection: never let a dead end be the answer

**Why.** G4. Today, empty is final, and *"I could not find it"* is
indistinguishable from *"it did not happen"* — the worst failure mode, because it
is invisible.

**Work.** A deterministic relaxation ladder in `ask.ask()`. **No second LLM
call** — reflection is a policy, not a model, which keeps it testable and free:

| step | relax | keeps |
|---|---|---|
| 1 | drop zone constraint | time, entity, text |
| 2 | widen window ×4 (bounded) | zone, entity, text |
| 3 | drop entity label | time, zone, text |
| 4 | text-only semantic, hard time bound | text |

Rules: at most **N=3** relaxations; **stop at the first non-empty result**;
record every step in the trace; **never** relax an explicit constraint the user
stated without saying so. `narrate()`'s system prompt gains one instruction — if
the answer came from a relaxed query, **say which constraint was loosened**.

**Outcome.**

```
Nothing in the loading bay after six. Widening to all zones —
two sightings at the side gate, 18:20 and 18:47.

trace: zone filter relaxed (loading_bay → any) after 0 hits
```

**Risk.** Low-medium. Risk is over-relaxing into noise. Mitigations: hard cap,
stop-on-first-hit, mandatory disclosure, and Phase 0 cases asserting that a
genuinely absent event still returns empty after the full ladder.

**Done when.** No Phase 0 case returns a bare empty result where an answer
exists, and a true-negative case still returns empty **with** the ladder recorded.

---

### Phase 8 — Visual search (optional, gated, later)

**Why.** Some questions describe appearance nobody ever wrote down: *"the red
van"*, *"the yellow hard hat"*. No text index can find what no text describes.

**Work.** CLIP embeddings over stored keyframes into the same `text_embeddings`
table (`kind='keyframe'`, shared joint space), added as a third RRF input.

**Cost.** ~350MB of weights and a third index to keep in sync.

**Recommendation.** Defer. Ship 0–7, read the eval, and only build this if the
scorecard shows a real appearance-query gap. **Off by default** if built.

---

## 5. Sequencing

| phase | new deps | risk | reversible | unlocks |
|---|---|---|---|---|
| 0 · Measurement | none | low | trivially | proof any of this works |
| 1 · Stop data loss | none | low-med | yes (additive) | object/area questions |
| 2 · Lexical index | none | low | yes (additive) | ranked word search |
| 3 · SQL pushdown | none | **medium** | yes (refactor) | flat latency, aggregates |
| 4 · Query form | none | medium | yes (compat shim) | real questions + `intent` |
| 5 · Habits + who-with | none | medium | yes (additive) | **2 README questions** |
| 6 · Semantic + fusion | **~120MB** | med-high | yes (flag) | paraphrase |
| 7 · Reflection | none | low-med | yes (flag) | no silent failures |
| 8 · Visual | ~350MB | high | yes (flag) | appearance queries |

**Phases 0–5 need no new dependencies** and fix the data loss, the unreachable
habits, the latency ceiling, and most guesswork. Phase 6 is the only one gated on
a dependency decision, and it is optional rather than required: without it the
scorecard is Phase 5's exactly, which is the property that made the decision
easy to take.

Recommended: **0 → 1 → 2 → 3** as foundations (measurable, no deps, no behaviour
change you rely on), then **4 → 5** for the largest visible win, then **6 → 7**.
0–6 are done; **7 is next**, and it needs nothing installed.

---

## 6. Success criteria

### 6.1 Question-level

| question | today | target |
|---|---|---|
| "anyone at the loading bay after six?" | ✅ | ✅ faster |
| "anyone **smoking** out back?" | ❌ empty (G1) | ✅ ranked |
| "anyone **loitering**?" | ❌ empty (G1) | ✅ Phase 6 |
| "was the **gate left open**?" | ❌ discarded (G2) | ✅ Phase 1+ |
| "**how often** does that van come by?" | ❌ no path (G3) | ✅ Phase 5 |
| "most **Tuesdays**?" | ❌ no weekday bucket (G8) | ✅ Phase 5 |
| "**who was with her** Tuesday?" | ❌ no path (G3) | ✅ Phase 5 |
| "the bay **or** the side gate" | ❌ (G5) | ✅ Phase 4 |
| "anyone **except** the delivery guy" | ❌ (G5) | ✅ Phase 4 |
| "**last five** times" | ❌ (G5) | ✅ Phase 4 |
| 6 months of footage | ⚠️ degrading (G6) | ✅ sub-second |

**Three of the README's headline questions used to fail. All three pass as of
Phase 5.**

### 6.2 Metrics

Targets are set against the Phase 0 baseline once it exists, rather than guessed
now. Tracked in `docs/search-baseline.md`, one row per phase:

- **silent-failure rate** — the headline. Empty results where an answer exists.
- **recall@1 / @5 / @20**
- **MRR**
- **plan accuracy** — structured fields correct, live-key, `slow`-marked
- **p50 / p95 latency** on a 100k-observation DB

### 6.3 Invariants — must hold at every phase

1. **No fabrication.** Retrieval changes *what is found*, never what is asserted.
2. **Every claim keeps its keyframe.**
3. **The trace stays truthful** — and now also explains ranking and relaxation.
4. **No API key still works**, degrading to structured + lexical search.
5. **Missing optional deps still works**, degrading to Phase 2 behaviour.
6. **Hard filters are never relaxed silently.**

---

## 7. Honest limits

- **Not retroactive.** Objects and areas from already-processed footage were never
  written and cannot be recovered. Phase 1 applies forward; Phases 2–6 improve
  retrieval over old rows.
- **Still retrospective.** Answers what happened; does not stop anything in
  progress. Unchanged, deliberately.
- **Perception bounds everything.** If the camera was dark and the model saw
  nothing, better search finds nothing. This plan fixes retrieval, not sight.
- **Paraphrase improves, it is not solved.** Ranked semantic search is a large
  improvement, not a guarantee. Phase 7 is what keeps the residual misses
  *honest* rather than silent.
- **Semantic search can surface plausible-but-wrong rows.** Bounded by hard
  filters, a similarity floor, and visible `match_reason` — not eliminated.

---

## 8. Open questions

1. ~~**Phase 6 dependency**~~ — **answered: yes, optional**, taken in Phase 6.
   ~120MB (`all-MiniLM-L12-v2`, not the L6 originally costed — see the Phase 6
   result for why L6 could not gate). It sits in the optional block beside
   `insightface`; with it absent the scorecard is Phase 5's to the decimal and
   the suite is green, asserted on every run via
   `INTELLIGENCE_OS_NO_SEMANTIC=1`.
2. ~~**Habit timezone**~~ — **answered: local**, taken in Phase 5 on the stated
   recommendation, since "most Tuesdays around 2" is a local-time claim and
   local is the clock the answers already render in. `time.localtime` reads
   `TZ`. One definition in `distill.py`; `digest.py` and `ask.py` call it.
3. **Row-volume ceiling** — cap observations emitted per VLM description? Suggest
   a configurable cap, default generous. *Phase 1 tuning, not a blocker.*
4. **Phase 8** — defer pending the scorecard. Recommend yes, defer.
