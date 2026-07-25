# Roadmap

What we are building next, and — just as usefully — what we are deliberately
not building. Dates are intentions, not commitments; a checked box means it is
on `main`.

The organising principle for every line below: **the memory graph is the
product.** Work that makes the graph more correct, more correctable, or more
traceable comes before work that makes the vision fancier.

Want something that isn't here? Open a [Discussion][disc] — the roadmap is
supposed to change.

[disc]: https://github.com/Infinex-Labs/Intelligence-OS/discussions

---

## Now — 2026 Q3

The theme is *someone else can actually run this.*

- [x] Server-side assistant chat history, replayed from stored evidence
- [x] Ownership enforced in SQL on every read and mutation
- [x] Apache 2.0 licence, contribution guide, security policy
- [x] `pyproject.toml`, pinned requirements, `intelligence-os` console script
- [x] CI: full suite on 3.10–3.12 plus a lint gate
- [x] Runnable examples that need no camera and no API key
- [ ] **RTSP resilience** — retry with backoff in `run.py:_camera_loop`. Today a
      camera that refuses a connection at startup kills its thread for the life
      of the process, and the UI just shows a dead tile.
- [ ] **Dockerfile + compose** — one command to a running dashboard
- [ ] A 30-second demo GIF in the README that shows the ask-with-keyframes loop
- [ ] First tagged release, `v0.1.0`

## Next — 2026 Q4

The theme is *trust the graph.*

- [ ] **Identity robustness.** Per-lighting thresholds, better gallery pruning,
      and an auto-merge heuristic that is harder to fool. This is the single
      highest-leverage correctness work in the project: if identity is unstable,
      every layer above it narrates a fictional world.
- [ ] **A calibration harness** that reports, on your own footage, the false
      merge / false split rate at a range of thresholds — so tuning stops being
      a vibe.
- [ ] **Pluggable backends.** A detector interface and a VLM interface, so the
      system isn't hard-wired to YOLO + Anthropic. Local VLM support (Ollama,
      llava) is the first consumer.
- [ ] **Query planning in `ask.py`** — more question shapes (counting, "how
      often", comparisons between periods), better narration, and a way to say
      "I don't know" that is louder than a confident wrong answer.
- [ ] More delivery channels in `deliver.py`: Slack, Discord, generic webhook.
- [ ] Documentation site (mkdocs) built from `docs/`.

## Later — 2027 H1

- [ ] **Retention and consent tooling.** Per-zone retention windows, an
      export-everything-about-this-entity command, and a scheduled purge — the
      operator-facing half of "privacy is a default, not a feature".
- [ ] **Multi-operator roles.** Today every signed-in user is equal apart from
      thread ownership. Viewer / operator / admin, with the boundary enforced in
      SQL the way thread ownership is.
- [ ] **Distillation v2** — relationship inference between entities, not just
      per-entity habits, with the same provenance guarantee.
- [ ] **Cross-site memory.** Federating several installations into one query
      surface without shipping keyframes off-site.
- [ ] A real front-end build for `static/`, whose source lives in this repo.

---

## Explicitly not on the roadmap

Saying no is part of the design:

- **Real-time intervention.** This is a *retrospective* system. It answers what
  happened; it does not stop anything in progress, and it must not be sold as a
  safety interlock.
- **Continuous video archival.** We keep keyframes and the graph, and prune
  keyframes at `raw_retention_days`. Storing every frame is a different product.
- **A hosted SaaS with your footage on our servers.** Self-hosted is the point.
- **Emotion, intent, or demographic inference from faces.** Not reliable, not
  ethical, not happening.
- **Covert operation.** Nothing that helps a deployment hide from the people it
  is watching.

---

## Helping

Anything above is fair game. Items tagged [`help wanted`][hw] are ones we
actively want a hand with; [`good first issue`][gfi] items are scoped so you
don't have to read the whole cascade first. Start with
[CONTRIBUTING.md](CONTRIBUTING.md).

[hw]: https://github.com/Infinex-Labs/Intelligence-OS/labels/help%20wanted
[gfi]: https://github.com/Infinex-Labs/Intelligence-OS/labels/good%20first%20issue
