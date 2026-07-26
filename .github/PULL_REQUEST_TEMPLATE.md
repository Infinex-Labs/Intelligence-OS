<!--
Thanks for the PR. Keep one PR to one concern — two concerns is two PRs.
Delete any section that genuinely doesn't apply; don't delete them all.
-->

## What this changes

<!-- One paragraph. What did you change, and why is it the right change? -->

Fixes #

## Which stages does it touch?

<!--
The cascade is interdependent: a motion-gate tweak changes how often the VLM
fires, which changes what gets written, which changes what distillation learns.
Tick what you changed, and say below what you checked downstream of it.
-->

- [ ] A — motion gate (`capture.py`)
- [ ] B — detection / tracking (`detect.py`)
- [ ] C — identity / re-id (`identity.py`)
- [ ] D — zones / scene state (`scene_state.py`)
- [ ] E/F — VLM trigger or description (`vlm.py`)
- [ ] G — memory write (`observe.py`, `store.py`)
- [ ] H — distillation (`distill.py`)
- [ ] I — ask / operator (`ask.py`, `operator.py`)
- [ ] Rules, alerting, delivery
- [ ] Web UI / API (`web.py`, `static/`)
- [ ] Docs, packaging, CI only

**Downstream effects I checked:**

## How I tested it

<!--
"Ran the suite" is enough for a refactor or a docs change.
A pipeline change needs the clip you ran it on and what you actually saw.
-->

- [ ] `python -m unittest discover -s intelligence_os/tests -t .` passes
- [ ] `ruff check .` passes
- [ ] Ran the pipeline against: <!-- clip / webcam / RTSP; how long -->

## Schema

- [ ] No schema change
- [ ] Additive only — new table, or new nullable column, created by
      `CREATE TABLE/ALTER ... IF NOT EXISTS` on open
- [ ] **Breaking** — an existing `memory.db` will not open unchanged
      *(say why this is unavoidable, and what a user with existing data should do)*

## Checklist

- [ ] New non-trivial logic leaves one runnable check behind — a test that
      would have caught the bug I just fixed
- [ ] Ownership checks for anything multi-user live in the SQL `WHERE` clause,
      not in the handler
- [ ] Any new kind of claim (alert, inference, answer) can still be traced back
      to the observations and keyframes that produced it
- [ ] New constants are typed fields in `config.py`, not magic numbers at the
      call site
- [ ] I did not reformat files I wasn't otherwise touching
- [ ] `CHANGELOG.md` updated under `## [Unreleased]` if this is user-visible
- [ ] No credentials, RTSP URLs, API keys or real footage in the diff

## Screenshots

<!-- UI changes only. Before and after, and dark mode if you touched styling. -->
