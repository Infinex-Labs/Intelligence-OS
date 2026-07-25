# Intelligence OS — a memory system that happens to have eyes

> **Layout:** the repo is `Intelligence-OS/`; the Python package inside it is
> `intelligence_os/`. Every command below is run from the repo root, so the
> package resolves as `python -m intelligence_os.<module>`.

**From pixels to intelligence.** Intelligence OS ingests camera streams, runs cheap
detection/tracking filters, and builds a **persistent, correctable,
evidence-weighted memory graph** of who and what appeared — where, and when. You
then *ask questions* and get answers backed by keyframes, instead of scrubbing
video.

The differentiator is **not** the vision models (those are commodity glue). It's
the memory: the same person stays one entity across days and across cameras,
every claim traces to a frame, and you can correct the graph (name, merge, split,
delete) when it's wrong.

> **Scope:** deliberately *retrospective*, not a safety interlock. It answers
> "what happened," it does not stop anything in progress. It runs useful with **no
> cloud keys** and upgrades gracefully when you add an Anthropic key for VLM
> descriptions.

---

## Table of contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Running it](#running-it)
- [Configuration](#configuration)
- [Codebase map](#codebase-map-for-contributors)
- [Data model](#data-model)
- [Contributing](#contributing)
- [Development](#development)
- [Tests](#tests)
- [FAQ](#faq)

---

## How it works

Everything is a **cascade**: cheap stages gate expensive ones, so the VLM (the
only paid, slow stage) fires only on meaningful change.

```
camera ─▶ [A motion gate] ─▶ [B detect + track] ─▶ [C identity / re-id]
       ─▶ [D scene-state] ─▶ [E VLM trigger] ─▶ [F VLM describe] ─▶ [G memory write]
                                  └──▶ [H distillation] ──▶ [I operator / ask]
```

| Stage | What happens | Module |
|---|---|---|
| A | MOG2 motion gate — skip still frames entirely | `capture.py` |
| B | YOLO + ByteTrack detection & tracking | `detect.py` |
| C | Face embedding (InsightFace) → **match-or-mint** against a persistent gallery | `identity.py` |
| D | Assign each box to a per-camera zone; snapshot the zone inventory over time | `scene_state.py` |
| E | Decide whether the scene *changed enough* to spend a VLM call | `vlm.py` |
| F | Structured VLM description, primed with what memory already knows | `vlm.py` |
| G | Write grounded observations (subject, predicate, object, location, keyframe) | `observe.py`, `store.py` |
| H | Roll repeated observations into durable **relations / habits / events** | `distill.py` |
| I | Query, name, merge, split, cascade-delete | `operator.py`, `ask.py` |

### Multiple cameras, one memory

`run.py` runs **one thread per camera**, but every thread shares **one SQLite
store** and **one set of models** (`_SharedModels`). Each camera keeps its own
motion gate, settle detector, and zones (frame geometry is camera-specific), but
**identity is global**: a face seen on `front_door` is scored against *every*
signature in the gallery, so it resolves to the same entity when it later appears
on `back_yard`. That shared face gallery is the *only* thing that fuses cameras
into one picture — there is no explicit camera-to-camera handoff logic. Faces are
the sole long-term signal; body/clothing is same-day only (clothing changes).

---

## Quick start

Requires **Python 3.10+** and a webcam or video file.

```bash
git clone <this-repo> && cd Intelligence-OS
python -m venv .venv && source .venv/bin/activate

# there is no requirements.txt yet — this is what the code imports
pip install "numpy<2" "opencv-python-headless==4.10.*" pyyaml psutil ultralytics
pip install insightface onnxruntime       # optional — face identity (stage C)
pip install anthropic                     # optional — VLM descriptions (stage F)

# optional — enables the VLM description stage
echo 'ANTHROPIC_API_KEY=sk-ant-...' > intelligence_os/.env
```

`config.yaml` is optional and lives at the repo root: with none present the app
boots on webcam 0, and the UI writes the file for you when you save settings.

The key is read from **`intelligence_os/.env`** (next to the package, not the repo root
— see `_load_dotenv` in `config.py`); a plain `export ANTHROPIC_API_KEY=...` works
too.

Models download on first use: YOLO weights (`yolo26s.pt`) and InsightFace
`buffalo_l` (~280 MB), so the first boot is slow — that's a download, not a hang.
`numpy` is held at 1.x on purpose — the torch/opencv/onnxruntime wheels are
built against the 1.x ABI.

Then launch the web UI:

```bash
python -m intelligence_os.web --webcam 0
# → http://localhost:8000
```

First visit prompts you to create a login. The dashboard streams the live feed
and exposes Alerts, Timeline, the AI Assistant, Rules, and Reports.

---

## Running it

You can drive the system from the web UI **or** the CLI. Both take the same
source flags (`--webcam N`, `--video path.mp4`), and both fall back to
`config.yaml`, then to webcam 0, when you pass nothing.

```bash
# Web app (pipeline + dashboard)
python -m intelligence_os.web --webcam 0 [--port 8000]

# Headless full pipeline (no UI)
python -m intelligence_os.run --video clip.mp4 --zones intelligence_os/data/zones.json

# Just the identity foundation — accumulate & tune entities (do this FIRST)
python -m intelligence_os.phase1 run  --webcam 0
python -m intelligence_os.phase1 tune --video enroll_clip.mp4   # tune the match threshold

# Distill observations → relations / habits / events
python -m intelligence_os.distill

# Operator: inspect and correct memory
python -m intelligence_os.operator list
python -m intelligence_os.operator inspect <entity_id>
python -m intelligence_os.operator name    <entity_id> "Raj"
python -m intelligence_os.operator merge   <src> <dst>     # fuse two fragments
python -m intelligence_os.operator split   <entity_id>     # undo a bad merge
python -m intelligence_os.operator delete  <entity_id>     # cascade delete = privacy removal

# Ask a question from the CLI
python -m intelligence_os.ask "who was near the door this afternoon?"
```

### The AI Assistant pane

The same grounded query surface, as a chat. Threads are stored server-side, so
they survive a reload and follow-ups ("and was anyone with them?") resolve
against the thread the server has, not a client-side array:

- **Chat history** in the left rail, grouped Today / Yesterday / Previous 7 days,
  searchable, renameable, deletable (two clicks — no browser confirm dialog).
- **Reopening a thread replays the stored answer**, not a fresh query. A report
  of what memory said then must not silently change when memory moves on.
- **KPI strip** — threads, questions, entities surfaced, observations scanned,
  keyframes cited, rule events, average answer time. Every figure is a `SUM`
  over stored turns, not a tally the browser keeps.
- **Per-answer counts** under each reply, plus the query trace, so an answer can
  be audited on its own.
- **Export** writes the open thread to Markdown, answers and traces included.

Threads are scoped to the signed-in user, and that scope is a `WHERE` clause on
every read *and* mutation in `store.py` — knowing a thread id is not authority to
read, rename, delete or append to it. Endpoints: `GET/POST /api/chats`,
`GET /api/chats/<id>`, `POST /api/chats/<id>/rename`, `POST /api/chats/<id>/delete`,
and `POST /api/ask` (takes `conversation_id`, appends the turn, returns the
evidence, the counts and the refreshed stats).

### Multi-camera setup

Point it at multiple streams via `config.yaml`:

```yaml
cameras:
  - { name: front_door, source: "rtsp://user:pass@10.0.0.5/stream1" }
  - { name: back_yard,  source: "rtsp://user:pass@10.0.0.6/stream1" }
  - { name: webcam,     source: 0 }
```

Camera names must be unique (duplicates are a boot error, not a silent
overwrite). Zones are drawn per-camera in the UI or supplied via a zones JSON.

---

## Configuration

Two layers:

- **`config.yaml`** (repo root) — sources and coarse toggles the UI writes back,
  e.g. `face_matching: true` and the `cameras:` list. Settings saved in the UI
  survive a restart.
- **`intelligence_os/config.py`** — every *tunable*, as typed dataclasses. The ones you
  will actually touch:

| Knob | Default | Why it matters |
|---|---|---|
| `IdentityConfig.face_match_threshold` | `0.45` | **The #1 risk.** Too low fuses two people; too high fragments one person into many. **Tune on your real people in your real lighting** before trusting anything above it. |
| `IdentityConfig.enabled` | `false` | Face recognition is **opt-in** by design (privacy). |
| `TriggerConfig.sensitivity` | `lazy` | `lazy` / `balanced` / `eager` — how eagerly the VLM fires. Web defaults to `balanced`. |
| `TriggerConfig.vlm_cooldown_seconds` | `30` | Floor between VLM calls (cost control). |
| `DistillConfig.decay_half_life_days` | `14` | How fast an un-reinforced habit fades. |
| `VLMConfig.model` | `claude-opus-4-8` | VLM used for descriptions; only active when `ANTHROPIC_API_KEY` is set. |
| `raw_retention_days` | `7` | Keyframes older than this are pruned at startup. |

---

## Codebase map (for contributors)

~6.3k lines of Python, no framework — the stdlib `http.server` runs the web
layer, SQLite *is* the database, and the schema is the contract. Start with
`store.py` (the product) and `run.py` (how it all wires together).

| File | Role | LOC |
|---|---|---|
| `store.py` | **The product.** SQLite data model + weights/decay + merge/split/cascade-delete + assistant threads. The schema string at the top is the contract. | 943 |
| `run.py` | Pipeline orchestrator — one thread per camera over shared models; the full cascade lives in `_camera_loop`. | 448 |
| `web.py` | `http.server`-based dashboard + JSON API (auth, live MJPEG, alerts, timeline, ask + chat history, rules, reports). | 1778 |
| `identity.py` | Face embedding + match-or-mint against the persistent gallery. The one guarantee everything else rests on. | 128 |
| `detect.py` | YOLO + ByteTrack; object persistence / re-id. | 150 |
| `capture.py` | Webcam/RTSP/video source + MOG2 motion gate + settled-keyframe detector. | 156 |
| `scene_state.py` | Per-location (zone) inventory over time. | 100 |
| `observe.py` | Grounded observation writer (present / near) + cooldown. | 110 |
| `vlm.py` | VLM trigger decision + structured describer (Anthropic). | 251 |
| `rules.py` | Rule compiler + cascade filter (zone/dwell/cooldown → optional VLM verify → alert). | 413 |
| `distill.py` | Change detection + event/habit/relation mining with decay. | 244 |
| `digest.py` | Graph diff vs. learned habits → "what broke pattern" briefing. | 262 |
| `deliver.py` | Delivery engine for briefings and real-time alerts (email/webhook/chat). | 536 |
| `ask.py` | The grounded natural-language query surface (plan → execute → narrate). | 284 |
| `operator.py` | CLI to query and correct memory. | 135 |
| `auth.py` | Users, sessions, password hashing for the web UI. | 109 |
| `config.py` | Every tunable as typed dataclasses + `cameras:` resolution. | 226 |
| `phase1.py` | Standalone identity accumulation + threshold tuning harness. | 269 |

All paths above are relative to the `intelligence_os/` package. The `§`-numbered
spec sections cited throughout the docstrings refer to the POC/PRD documents,
which are **not** in this repo — the code comments are the authority until they
are checked in.

---

## Data model

Core tables (full schema in `store.py`):

- **`entities`** — a person or object. Anonymous by default (`entity_N`), given a
  `label` when you name them.
- **`signatures`** — L2-normalized face vectors per entity (the gallery). Capped
  and pruned so we capture pose/lighting variation without storing near-duplicates.
- **`observations`** — the raw transcript: `subject → predicate → object` at a
  `location` and `timestamp`, tagged with `origin` (detector/vlm), a
  `source_ref` keyframe, and a `camera_id`. Predicates are **open strings**, not
  an enum.
- **`locations`** — zones, scoped to one camera's frame.
- **`scene_snapshots`** — who was present in a zone at a point in time.
- **`relations`** — distilled `relation` / `habit` / `event` rows with a
  `weight`, a `candidate → confirmed` lifecycle, decay, and
  `supporting_observation_ids` for provenance.
- **`conversations` / `chat_turns`** — the assistant's threads. Each turn keeps
  the question, the answer, the evidence payload it was rendered from, and the
  counts denormalized out of that payload so the KPI strip is one aggregate
  query instead of a JSON scan.

Every higher-level claim (a habit, an alert, an answer) can be walked back down
to the observations and keyframes that produced it. That traceability is the
point.

---

## Contributing

This is a working prototype, not a frozen product — the surface area for
contribution is wide. Good places to start, roughly by required depth:

**Good first issues**
- More `object_classes` in `DetectConfig` and predicate schema entries.
- UI polish in `intelligence_os/static/` (`home.html`, `login.html`, `style.css`).
- More delivery channels in `deliver.py` (Slack, Discord, generic webhook).
- Docs: worked examples, tuning write-ups, sample `zones.json` per camera setup.

**Meatier**
- Identity robustness: better gallery pruning, per-lighting thresholds, an
  auto-merge heuristic that's harder to fool (`identity.py`, `store.auto_merge_people`).
- Distillation: smarter habit mining and decay (`distill.py`).
- Query planning in `ask.py` — more question shapes, better narration.
- A pluggable detector/VLM backend so it isn't hard-wired to YOLO + Anthropic.

**Before you send a PR**
1. Trace the whole flow your change touches — the cascade is interdependent.
2. Keep the schema the contract: migrations in `store.py` must be *additive*
   (see the `camera_id` migration for the pattern).
3. Run the acceptance tests below; they are the correctness gates.
4. Match the surrounding style — small, explicit, tunable. New non-trivial logic
   should leave one runnable check behind.

There is currently **no `LICENSE` file** — add one (or ask the maintainers) before
depending on this in anything you ship.

---

## Development

The dev loop is: edit → run one test module → restart the web app. There is no
migrations tool — the schema migrates itself on `Store()` open. The Python side
has no build step; the front end is served from `intelligence_os/static/`, where
`dist/` holds a **prebuilt** bundle whose source is not in this repo, so editing
it means editing the built asset (or wiring the source back in).

Run everything from the repo root — `web.py` resolves `intelligence_os/static`
relative to the working directory:

```bash
source .venv/bin/activate
python -m intelligence_os.web --video clip.mp4 --port 8000   # no camera needed
```

`--video` replays a file through the identical pipeline, so you can develop
without a webcam and get reproducible runs. There is no auto-reload: restart the
process after a Python edit.

**State lives in three places**, all gitignored, all safe to delete:

```bash
rm intelligence_os/data/memory.db*      # all three files — SQLite is in WAL mode
rm -rf intelligence_os/data/frames/*    # retained keyframes
rm config.yaml                    # UI-written settings (cameras, face_matching)
```

Deleting `memory.db` resets entities, observations, rules **and the login** — the
next page visit prompts you to create a user again. That is the fastest way back
to a clean slate.

**Env knobs for dev** (all read at import, `config.py`):

| Var | Use |
|---|---|
| `INTELLIGENCE_OS_DB` | Point at a scratch DB so your real memory graph isn't touched. |
| `INTELLIGENCE_OS_DATA` | Relocate the whole data dir (DB + frames + rules). |
| `INTELLIGENCE_OS_YOLO` | Use a specific weights file instead of the auto-download. |
| `INTELLIGENCE_OS_CONFIG` | Alternate `config.yaml` — handy for a multi-camera fixture. |
| `IDENTITY_ENABLED` | Lets rules that reference *named people* compile (`rules.py`). Separate from the face matcher itself — that's `face_matching` in `config.yaml`. |

Face recognition is **off by default**. Turn it on with `face_matching: true` in
`config.yaml` (or the UI toggle), then tune the threshold on your own footage —
`python -m intelligence_os.phase1 tune --video enroll_clip.mp4` — before trusting
anything identity-dependent.

There is no Dockerfile in the repo yet — the venv above is the only supported way
to run it today.

---

## Tests

Plain `unittest`, no pytest, no fixtures. Offline — no camera or API key required.

The two acceptance gates:

```bash
python -m intelligence_os.tests.test_foundation   # identity + correctness machinery (Phase-1 gate)
python -m intelligence_os.tests.test_pipeline     # scene-state → distillation memory pipeline
```

`test_foundation` is the gate that matters: if identity is unstable, every layer
above it narrates a fictional world.

The full suite (~24 modules — auth, rules, delivery, multicam, web payloads):

```bash
python -m unittest discover -s intelligence_os/tests -t .
```

Every module is also runnable on its own (`python -m intelligence_os.tests.test_rules`)
— that's the tight loop while you work on one area. Tests build their own
temporary DBs, so they never touch `intelligence_os/data/memory.db`.

---

## FAQ

**Do I need an Anthropic API key?** No. Without it, detection, tracking,
identity, zones, observations, rules, and distillation all work. The key only
enables the VLM *description* stage (stage F) — natural-language scene summaries.

**Does it store video?** No — it keeps **keyframes + the graph**, and prunes
keyframes older than `raw_retention_days` (default 7). It is not a continuous
archival system.

**How do I remove someone for privacy?** `operator delete <entity_id>` — it
cascades to their signatures, observations, and relations.

**RTSP won't connect.** The pinned `opencv-python-headless==4.10` ships
`FFMPEG:YES` (required for RTSP). Don't jump to 5.x, and check the stream URL
resolves with `ffmpeg`/`ffprobe` first.

