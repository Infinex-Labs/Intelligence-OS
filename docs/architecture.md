# Architecture

The organising idea: **everything is a cascade.** Cheap stages gate expensive
ones, so the only paid, slow stage — the VLM — fires on meaningful change rather
than on every frame.

The second organising idea: **the memory graph is the product.** The vision
models are commodity glue and are meant to be replaceable. What is not
replaceable is the guarantee that the same person stays one entity across days
and cameras, that every claim traces to a frame, and that an operator can
correct the graph when it is wrong.

```
                 ┌─────────────┐
   camera  ─────▶│  A  motion  │  MOG2 gate — still frames cost nothing
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  B  detect  │  YOLO + ByteTrack
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  C identity │  face embedding → match-or-mint (opt-in)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  D  scene   │  which zone is each box in, over time
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  E  trigger │  did the scene change enough to pay for a VLM call?
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  F describe │  structured VLM description  ← the only paid stage
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  G  memory  │  grounded observations, each with a keyframe
                 └──────┬──────┘
                        ▼
             ┌──────────┴──────────┐
             ▼                     ▼
      ┌─────────────┐       ┌─────────────┐
      │ H  distill  │       │ I  ask /    │
      │ habits,     │       │ operator    │
      │ relations   │       │             │
      └─────────────┘       └─────────────┘
```

| Stage | What happens | Module |
|---|---|---|
| A | MOG2 motion gate — skip still frames entirely | `capture.py` |
| B | YOLO + ByteTrack detection & tracking, object persistence | `detect.py` |
| C | Face embedding (InsightFace) → **match-or-mint** against a persistent gallery | `identity.py` |
| D | Assign each box to a per-camera zone; snapshot the zone inventory over time | `scene_state.py` |
| E | Decide whether the scene changed enough to spend a VLM call | `vlm.py` |
| F | Structured VLM description, primed with what memory already knows | `vlm.py` |
| G | Write grounded observations (subject, predicate, object, location, keyframe) | `observe.py`, `store.py` |
| H | Roll repeated observations into durable relations / habits / events | `distill.py` |
| I | Query, name, merge, split, cascade-delete | `operator.py`, `ask.py` |

Alongside the cascade: `rules.py` (compiled rules with their own zone → dwell →
verify → cooldown gate chain), `digest.py` (diff the graph against learned
habits to produce a "what broke pattern" briefing) and `deliver.py` (email,
webhook, chat delivery).

## Why a cascade and not just "run the model"

Running a VLM on every frame of every camera is the obvious design and it is
wrong on three counts: it costs money linearly with time, it produces a torrent
of near-identical descriptions that bury the one that mattered, and it makes the
system's accuracy entirely a function of somebody else's model.

Each gate answers a cheaper question first:

- **Is anything moving?** No → stop. (free)
- **Is it a class we care about?** No → stop. (cheap, local)
- **Have we seen this exact scene state already?** Yes → stop. (free, a dict lookup)
- **Only then:** describe it. (paid)

The gates also compose with `TriggerConfig.vlm_cooldown_seconds`, a hard floor
between calls, so a pathological scene cannot run up a bill.

## Multiple cameras, one memory

`run.py` runs **one thread per camera**, sharing **one SQLite store** and **one
set of models** (`_SharedModels`) — loading YOLO once per camera would cost N
times the memory for identical weights.

Per camera: the motion gate, the settle detector, and the zones (frame geometry
is camera-specific).

Global: **identity.** A face seen on `front_door` is scored against *every*
signature in the gallery, so it resolves to the same entity when it appears on
`back_yard`.

That shared gallery is the **only** mechanism fusing cameras. There is no
camera-to-camera handoff logic and no geometric calibration — which means that
with face matching off, you have several independent camera timelines sharing a
database, not cross-camera identity. Faces are the sole long-term signal; body
and clothing embeddings are same-day only, because people change clothes.

## The data model

SQLite, WAL mode. The schema string at the top of `store.py` **is** the
contract; it is `executescript`'d on every open with `CREATE TABLE IF NOT
EXISTS`, so it self-migrates and there is no migrations tool.

| Table | What it holds |
|---|---|
| `entities` | A person or object. Anonymous by default (`entity_N`); gains a `label` when an operator names them. |
| `signatures` | L2-normalised face vectors per entity — the gallery. Capped and pruned to capture pose/lighting variation without storing near-duplicates. |
| `observations` | The raw transcript: `subject → predicate → object` at a `location` and `timestamp`, tagged with `origin` (detector / vlm / rule), a `source_ref` keyframe, and a `camera_id`. Predicates are **open strings**, not an enum. |
| `locations` | Zones, scoped to one camera's frame. |
| `scene_snapshots` | Who was present in a zone at a point in time. |
| `relations` | Distilled `relation` / `habit` / `event` rows with a `weight`, a `candidate → confirmed` lifecycle, decay, and `supporting_observation_ids` for provenance. |
| `conversations`, `chat_turns` | The assistant's threads, each turn keeping the evidence payload it was rendered from. |
| `users`, `sessions` | Dashboard auth (scrypt password hashes). |
| `cases`, `reports` | Operator workflow — grouping evidence, and generated briefings. |

### The invariant

Every higher-level claim — a habit, an alert, an answer — must be walkable back
down to the observations and keyframes that produced it. `relations` carry
`supporting_observation_ids`; observations carry `source_ref`; the assistant
stores the evidence payload each answer was rendered from.

If you add a new kind of claim, add its provenance at the same time. A claim
without evidence is the one thing this design does not permit.

### Migration rules

Additive only: new tables, or new nullable columns, created on open. Never
rename or drop a column in a released version — an existing `memory.db` has to
keep opening. Follow the `camera_id` migration in `store.py` for the pattern.

## Ownership and multi-user

Ownership predicates live in the SQL `WHERE` clause of every read **and**
mutation, not in the request handler (`_OWNED` in `store.py`). A handler that
forgets to pass `user_id` fails closed rather than quietly widening access, and
an object that isn't yours reads as *absent* (404) rather than *forbidden*, so
an id cannot be probed for existence.

## Deliberate limits

- **Retrospective, not preventive.** It answers what happened. It does not stop
  anything in progress and must not be presented as a safety interlock.
- **Keyframes, not video.** Keyframes older than `raw_retention_days` (default
  7) are pruned at startup. This is not an archival system.
- **Privacy defaults.** Face recognition ships off. Deletion cascades. Neither
  is a feature flag to flip for convenience.
