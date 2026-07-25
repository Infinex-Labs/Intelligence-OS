# Contributing to Intelligence OS

Thanks for being here. This is a working prototype with a wide contribution
surface — the cascade has eight stages and most of them have obvious next steps.

Everything below assumes you have read [the architecture](docs/architecture.md).
The one thing to internalise before you change anything: **the memory graph is
the product.** The vision models are replaceable glue. A change that makes
detection 5% better but breaks the guarantee that every claim traces back to a
keyframe is a regression.

---

## Setting up

Requires **Python 3.10+**. No Docker, no build step, no migration tool.

```bash
git clone https://github.com/Infinex-Labs/Intelligence-OS.git
cd Intelligence-OS
python -m venv .venv && source .venv/bin/activate

pip install -r requirements-dev.txt    # enough to run the tests (3 packages)
pip install -e .                       # editable install + the `intelligence-os` script
```

`requirements-dev.txt` is deliberately smaller than `requirements.txt`. The test
suite is offline and stubs the heavy optional dependencies, so you do **not**
need torch, ultralytics, insightface or an API key to work on most of the
codebase. Install `-r requirements.txt` when you need to actually run the
pipeline against a camera.

### Running it while you work

```bash
python -m intelligence_os.web --video examples/quickstart/clip.mp4 --port 8000
```

`--video` replays a file through the identical pipeline, so you get a
reproducible run with no camera attached. There is **no auto-reload** — restart
the process after a Python edit. Front-end edits (`intelligence_os/static/`)
just need a browser refresh.

### Getting back to a clean slate

State lives in three places, all gitignored, all safe to delete:

```bash
rm intelligence_os/data/memory.db*     # all three files — SQLite is in WAL mode
rm -rf intelligence_os/data/frames/*   # retained keyframes
rm config.yaml                         # UI-written settings
```

Deleting `memory.db` also deletes the login; the next page visit asks you to
create one again.

---

## Tests

Plain `unittest`. No pytest, no fixtures, no network, no camera.

```bash
# the whole suite — this is what CI runs
python -m unittest discover -s intelligence_os/tests -t .

# one module, the tight loop while you work
python -m intelligence_os.tests.test_rules
```

Two of them are **acceptance gates** rather than unit tests:

| Gate | What it protects |
|---|---|
| `test_foundation` | Identity and the correctness machinery. If identity is unstable, every layer above it narrates a fictional world. |
| `test_pipeline` | Scene-state → distillation. That observations actually become durable memory. |

If either of those goes red, stop and fix it — don't work around it.

Tests build their own temporary databases, so they never touch your real
`memory.db`. Handlers are driven directly rather than over HTTP (see
`test_chat_history.py` for the pattern: `object.__new__(web.RequestHandler)`
with `send_json` / `_read_json` stubbed).

**New non-trivial logic should leave one runnable check behind.** Not a
coverage target — one test that would have caught the bug you just fixed.

## Linting

```bash
ruff check .          # config in pyproject.toml under [tool.ruff]
```

The rule set is a **floor, not a style bible**: undefined names, unused imports,
syntax errors, whitespace. It is green today and should stay green. There are
deliberate tightening steps listed in a comment in `pyproject.toml` (`B`, `UP`,
`E7`) — each is a large mechanical diff, so they are their own PRs, not
something to smuggle into a feature change.

Please don't reformat files you aren't otherwise touching. A diff that is 90%
whitespace is a diff nobody can review.

---

## Where to start

**Good first issues**
- More `object_classes` in `DetectConfig`, more predicate schema entries.
- UI polish in `intelligence_os/static/` (`home.html`, `login.html`, `style.css`).
- More delivery channels in `deliver.py` (Slack, Discord, generic webhook).
- Docs: worked examples, tuning write-ups, a sample `zones.json` for a real setup.

**Meatier**
- Identity robustness — gallery pruning, per-lighting thresholds, an auto-merge
  heuristic that is harder to fool (`identity.py`, `store.auto_merge_people`).
- Smarter habit mining and decay (`distill.py`).
- More question shapes and better narration in `ask.py`.
- A pluggable detector/VLM backend so it isn't hard-wired to YOLO + Anthropic.

Issues tagged [`good first issue`][gfi] are scoped to be completable without
reading the whole cascade first.

[gfi]: https://github.com/Infinex-Labs/Intelligence-OS/labels/good%20first%20issue

---

## House rules for changes

**1. Trace the whole flow your change touches.** The cascade is interdependent:
a motion-gate tweak changes how often the VLM fires, which changes what gets
written, which changes what distillation learns. Say in the PR what downstream
stages you checked.

**2. Schema migrations must be additive.** `store.py` has no migration tool —
the schema string at the top is `executescript`'d with `CREATE TABLE IF NOT
EXISTS` on every open, so new tables and new nullable columns appear by
themselves. Follow the `camera_id` migration for the pattern. Never rename or
drop a column in a released version; an existing `memory.db` has to keep
opening.

**3. Scope ownership in SQL, not in the handler.** Multi-user reads *and*
mutations put the ownership check in the `WHERE` clause (see `_OWNED` in
`store.py`). A handler that forgets to pass `user_id` should fail closed rather
than quietly widen access. Knowing an object's id is never authority over it.

**4. Keep the traceability guarantee.** Every higher-level claim — a habit, an
alert, an answer — must be walkable back down to the observations and keyframes
that produced it. If you add a new kind of claim, add its provenance column at
the same time.

**5. Match the surrounding style.** Small, explicit, tunable. Comments explain
*why*, not *what*. New constants belong in `config.py` as typed dataclass
fields, not as magic numbers at the call site.

**6. Privacy is a default, not a feature.** Face recognition ships **off**
(`IdentityConfig.enabled = false`). Keyframes are pruned at
`raw_retention_days`. `operator delete` cascades. Don't add anything that
retains more than it needs to, or that turns an opt-in on for people.

---

## Pull requests

- Branch from `main`: `git checkout -b fix/rtsp-reconnect`.
- Prefix names with `fix/`, `feat/`, `docs/`, `test/`, `chore/`.
- Commits in the imperative mood, subject under ~72 characters, and a body that
  says why when the change isn't obvious.
- Keep one PR to one concern. Two concerns is two PRs.
- Fill in the PR template — particularly *how you tested it*. "Ran the suite" is
  fine for a refactor; a pipeline change needs a clip and what you saw.
- CI must be green (tests on 3.10/3.11/3.12, plus lint). A red CI on a PR is
  yours to fix, not the reviewer's to interpret.

Discussion before code is welcome and encouraged for anything large — open an
issue or a [Discussion][disc] and describe the approach first. Nobody enjoys
rejecting a finished 800-line PR that took the wrong path.

[disc]: https://github.com/Infinex-Labs/Intelligence-OS/discussions

---

## Reporting a security issue

Do **not** open a public issue. See [SECURITY.md](SECURITY.md).

## Code of conduct

Participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Licensing of contributions

Contributions are accepted under the [Apache License 2.0](LICENSE), the same
licence as the project. By opening a PR you confirm you have the right to submit
the code under it. There is no CLA.
