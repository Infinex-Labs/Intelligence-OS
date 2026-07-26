# Development

The dev loop is: **edit → run one test module → restart the app.** No build
step, no migration tool, no auto-reload.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the house rules and the PR
process. This page is the mechanics.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt     # 3 packages — enough for the whole suite
pip install -e .
```

Install `-r requirements.txt` on top only when you need to actually run
detection.

## The loop

```bash
python -m intelligence_os.web --video clip.mp4 --port 8000
```

`--video` replays a file through the identical pipeline, so runs are
reproducible and you don't need a camera. Restart the process after a Python
edit; front-end edits just need a refresh.

Point the app at a scratch database while you work:

```bash
INTELLIGENCE_OS_DB=/tmp/scratch.db python -m intelligence_os.web --video clip.mp4
```

## Codebase map

~6.3k lines of Python, no framework: the stdlib `http.server` runs the web
layer and SQLite **is** the database. Start with `store.py` (the product) and
`run.py` (how it wires together).

| File | Role |
|---|---|
| `store.py` | **The product.** Data model, weights and decay, merge/split/cascade-delete, assistant threads. The schema string at the top is the contract. |
| `run.py` | Pipeline orchestrator — one thread per camera over shared models. The whole cascade is in `_camera_loop`. |
| `web.py` | Dashboard + JSON API: auth, MJPEG, alerts, timeline, ask, chat history, rules, reports. |
| `identity.py` | Face embedding + match-or-mint against the persistent gallery. The guarantee everything else rests on. |
| `detect.py` | YOLO + ByteTrack; object persistence and re-id. |
| `capture.py` | Webcam/RTSP/video source, MOG2 motion gate, settled-keyframe detector. |
| `scene_state.py` | Per-zone inventory over time. |
| `observe.py` | Grounded observation writer (present / near) with cooldown. |
| `vlm.py` | VLM trigger decision + structured describer. |
| `rules.py` | Rule compiler (trust boundary) + cascade engine. |
| `distill.py` | Change detection, event/habit/relation mining, decay. |
| `digest.py` | Graph diff vs. learned habits → "what broke pattern". |
| `deliver.py` | Delivery for briefings and alerts (email / webhook / chat). |
| `ask.py` | The grounded query surface: plan → execute → narrate. |
| `operator.py` | CLI to query and correct memory. |
| `auth.py` | Users, sessions, scrypt password hashing. |
| `config.py` | Every tunable as typed dataclasses + camera resolution. |
| `phase1.py` | Standalone identity accumulation and threshold-tuning harness. |

All paths are relative to `intelligence_os/`. Some docstrings cite `§`-numbered
spec sections from POC/PRD documents that are **not** in this repo — where they
disagree with the code, the code wins.

## Tests

Plain `unittest`. Offline: no camera, no key, no network. Each test builds its
own temporary database, so your real `memory.db` is never touched.

```bash
python -m unittest discover -s intelligence_os/tests -t .    # the whole suite
python -m intelligence_os.tests.test_rules                   # one module
```

### The two acceptance gates

| Gate | Protects |
|---|---|
| `test_foundation` | Identity and the correctness machinery. If identity is unstable, every layer above it narrates a fictional world. |
| `test_pipeline` | Scene state → distillation: that observations actually become durable memory. |

If either goes red, stop and fix it rather than working around it. CI runs both
by name before the rest, so a failure says what broke instead of scrolling past
a hundred dots.

### Patterns worth copying

- **Handlers are driven directly**, not over HTTP:
  `object.__new__(web.RequestHandler)` with `send_json` / `send_error` /
  `_read_json` stubbed. See `test_chat_history.py`.
- **Heavy deps are stubbed only if absent** (`tests/_stubs.py`) —
  `sys.modules.setdefault` alone was wrong once the venv actually had cv2, since
  whichever module imported first won for the whole discovery run.
- **Redirect the store with `INTELLIGENCE_OS_DB`** when the code under test
  opens its own `Store()`.

## Linting

```bash
ruff check .
```

The rule set (`pyproject.toml`, `[tool.ruff]`) is a floor, not a style bible:
undefined names, unused imports, syntax errors, whitespace. It is green on
`main`. Tightening steps (`B`, `UP`, `E7`) are listed in a comment there — each
is a large mechanical diff and belongs in its own PR.

Don't reformat files you aren't otherwise touching.

## Schema changes

There is no migration tool by design. The schema string at the top of `store.py`
is `executescript`'d on every open with `CREATE TABLE IF NOT EXISTS`, so new
tables and new nullable columns appear by themselves.

The rule: **additive only.** Never rename or drop a column in a released
version — an existing `memory.db` has to keep opening. Follow the `camera_id`
migration for the pattern, and declare the change in your PR and in
`CHANGELOG.md`.

## Resetting state

```bash
rm intelligence_os/data/memory.db*      # all three files — WAL mode
rm -rf intelligence_os/data/frames/*
rm config.yaml
```

Deleting `memory.db` also deletes the login; the next page visit asks you to
create one.

## Packaging

```bash
python -m build && twine check dist/*
```

CI additionally asserts the wheel contains `static/home.html`, `login.html` and
`style.css`. A wheel that ships without the UI it serves builds perfectly and
then 404s on its own dashboard at runtime — cheap check, expensive bug.
