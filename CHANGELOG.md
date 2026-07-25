# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because the memory graph is the product, one extra rule applies: **any change
that alters the SQLite schema is called out explicitly**, with whether an
existing `memory.db` keeps opening. The schema self-migrates on `Store()` open,
and migrations are required to be additive — if that ever stops being true for a
release, it will say so here in bold.

## [Unreleased]

No release has been tagged yet. Everything below is what `main` contains and
what `0.1.0` will be.

### Added

- **AI Assistant chat history.** Threads are stored server-side
  (`conversations`, `chat_turns`), so they survive a reload and follow-up
  questions resolve against the thread the server has rather than a client-side
  array. Sidebar grouped Today / Yesterday / Previous 7 days, searchable,
  renameable, deletable, exportable to Markdown.
  *Schema: two new tables. Existing databases open unchanged.*
- **Replay, not re-run.** Reopening a thread renders the evidence payload stored
  with each turn. A record of what memory said then must not silently change
  when memory moves on.
- **Assistant KPI strip** — threads, questions, entities surfaced, observations
  scanned, keyframes cited, rule events, average answer time. Every figure is a
  `SUM` over stored turns; the browser keeps no tally of its own.
- **Packaging.** `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`,
  and an `intelligence-os` console script. `pip install -e .` then
  `intelligence-os --webcam 0` now works from any directory.
- **Project documentation** — `docs/`, runnable `examples/`, `CONTRIBUTING.md`,
  `SECURITY.md`, `ROADMAP.md`, issue and PR templates.
- **CI** — the test suite on Python 3.10/3.11/3.12 plus a `ruff` lint gate.
- **Apache 2.0 licence.** The repository previously had no `LICENSE` file at
  all, which meant nobody could legally depend on it.

### Fixed

- **Thread ownership is enforced in SQL, not in the handler.** Listing and
  reading were scoped to the signed-in user, but rename, delete, and the
  `conversation_id` path of `POST /api/ask` were keyed on the thread id alone —
  any signed-in operator could rename, delete or append turns to another
  operator's thread if they knew its id. The ownership predicate now lives in
  the `WHERE` clause of every read *and* mutation, so a caller that forgets to
  pass `user_id` fails closed. Threads that are not yours read as absent (404),
  not forbidden, so an id cannot be probed for existence.
- **The web UI no longer depends on the working directory.** Static assets
  resolve against the package, not `os.getcwd()`, so the server serves its own
  dashboard when started from anywhere.
- A failed question no longer leaves an empty untitled thread in the sidebar.
- Lint: removed unused imports and trailing whitespace across the package. No
  behaviour change.

### Changed

- `.gitignore` no longer ignores every directory named `docs` or `data` at any
  depth. The runtime data directory is now matched by path
  (`intelligence_os/data/`).

## [0.1.0] — unreleased

The first tagged release will capture the system as it stands: the eight-stage
cascade (motion gate → detect/track → identity → scene state → VLM trigger →
describe → memory write → distillation), the operator CLI for correcting
memory, rules and alerting, scheduled digests, and the web dashboard.

[Unreleased]: https://github.com/Infinex-Labs/Intelligence-OS/commits/main
