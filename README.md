<div align="center">

<img src="assets/logo.png" alt="Intelligence OS" width="120">

# Intelligence OS

**Turn camera streams into a memory you can question — not footage you have to scrub.**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Self-hosted](https://img.shields.io/badge/self--hosted-no%20cloud%20required-success.svg)](docs/installation.md)

[Quick start](#quick-start) · [Docs](docs/) · [Examples](examples/) · [Contributing](CONTRIBUTING.md)

</div>

---

## Why

You have cameras. What you actually want is an answer:

> *"Was anyone at the loading bay after six?"*
> *"How often does that van come by?"*
> *"Who was the person my wife let in on Tuesday?"*

Today you get that by scrubbing video, which nobody does, so the footage sits
there unwatched until something goes wrong. Object detectors don't fix it —
they hand you a firehose of `person 0.87` and leave the remembering to you.

**Intelligence OS remembers instead.** It watches the stream, spends compute
only when something changes, and writes what it saw into a persistent graph:
who and what appeared, where, and when. Then you ask questions of the graph.

The differentiator is **not** the vision models — those are commodity glue that
you should be able to swap out. It's the memory:

- **The same person stays one entity** across days and across cameras.
- **Every claim traces back to a frame.** A habit is not an opinion the model
  formed; it is a pointer to the observations and keyframes that produced it.
- **You can correct it.** Name, merge, split, delete. Because it *will* be
  wrong, and a memory you can't fix is a memory you can't trust.

> **Scope, stated plainly.** This is deliberately *retrospective*. It answers
> what happened; it does not stop anything in progress and must not be deployed
> as a safety interlock. It runs fully useful with **no cloud keys**, and
> upgrades gracefully when you add one.

---

## Features

| | |
|---|---|
| 🧠 **Persistent memory graph** | Entities, observations, zones and relations in SQLite. The schema is the contract. |
| 🔎 **Ask in English** | Grounded answers with keyframes. The model only writes the query — every fact is aggregated from observation rows, so there is nothing to hallucinate. |
| 👤 **Cross-camera identity** | One face gallery fuses several cameras into one picture. **Opt-in**, off by default. |
| 📐 **Zones and rules** | "Tell me if someone hangs around the loading bay." Compiled from English, gated by zone → dwell → verify → cooldown. |
| 🚫 **Rules that refuse** | "Alert me if someone looks stressed" is rejected at compile time, loudly, instead of firing noise forever. |
| 📊 **Habits and digests** | Repetition becomes durable memory with decay; a daily briefing tells you what *broke* pattern. |
| ✏️ **Correctable** | Name, merge, split, cascade-delete — from the UI or the CLI. |
| 💸 **Cheap by construction** | A cascade of gates means the paid VLM stage fires on meaningful change, with a hard cooldown floor. |
| 🔒 **Self-hosted** | Your machine, your disk. No account, no telemetry, no footage leaving the building unless you configure it to. |

---

## Quick start

Requires **Python 3.10+** and a webcam or a video file.

```bash
git clone https://github.com/Infinex-Labs/Intelligence-OS.git
cd Intelligence-OS
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

intelligence-os --webcam 0        # → http://localhost:8000
```

First visit asks you to create a login. The dashboard gives you the live feed,
Alerts, Timeline, the AI Assistant, Rules and Reports.

<details>
<summary><b>Only got two minutes? Skip the camera entirely.</b></summary>

```bash
pip install -r requirements-dev.txt          # 3 packages, seconds
python examples/01_memory_graph/run.py
```

Prints the whole thing end to end — observations accumulating, distillation
turning repetition into habits, a claim being walked back down to its evidence,
a bad merge being corrected, and a cascade delete emptying the graph. No camera,
no API key, no model weights.

</details>

<details>
<summary><b>Optional extras, and what each one unlocks</b></summary>

```bash
pip install -e ".[identity]"     # stage C: face re-id across days and cameras
pip install -e ".[vlm]"          # stage F: scene descriptions + the assistant

echo 'ANTHROPIC_API_KEY=sk-ant-...' > intelligence_os/.env
```

Note the key location — `intelligence_os/.env`, next to the package. Without a
key you still get motion gating, detection, tracking, zones, observations,
rules, alerting, distillation and the dashboard.

**The first boot is slow** because it downloads YOLO weights (~20 MB) and, with
identity on, InsightFace `buffalo_l` (~280 MB). It is downloading, not hanging.

</details>

---

## How it works

Everything is a **cascade**: cheap stages gate expensive ones, so the only paid,
slow stage fires on meaningful change rather than on every frame.

```
camera ─▶ [A motion gate] ─▶ [B detect + track] ─▶ [C identity / re-id]
       ─▶ [D scene state]  ─▶ [E VLM trigger]    ─▶ [F VLM describe]
       ─▶ [G memory write] ─▶ [H distillation]   ─▶ [I ask / operator]
```

| Stage | What happens | Module |
|---|---|---|
| A | MOG2 motion gate — still frames cost nothing | `capture.py` |
| B | YOLO + ByteTrack detection and tracking | `detect.py` |
| C | Face embedding → **match-or-mint** against a persistent gallery | `identity.py` |
| D | Assign each box to a per-camera zone; snapshot the inventory over time | `scene_state.py` |
| E | Did the scene change enough to be worth paying for a description? | `vlm.py` |
| F | Structured VLM description, primed with what memory already knows | `vlm.py` |
| G | Write grounded observations — subject, predicate, object, location, keyframe | `observe.py`, `store.py` |
| H | Roll repetition into durable relations / habits / events, with decay | `distill.py` |
| I | Query, name, merge, split, cascade-delete | `ask.py`, `operator.py` |

Several cameras run one thread each, sharing one store and one set of models.
Zones are per-camera; **identity is global** — and that shared face gallery is
the only thing fusing cameras into one picture.

→ **[Full architecture](docs/architecture.md)**

---

## Ask it something

```bash
python -m intelligence_os.ask "who was near the door this afternoon?"
```

The LLM's *only* job is turning that sentence into a structured query — a time
window, a zone, an entity, a predicate filter. Every fact in the answer is then
aggregated deterministically from the observation rows that query returns, and
the query itself comes back with the answer as an inspectable trace.

So when an answer looks wrong, you can tell which thing is wrong: the parse, or
the memory.

The dashboard's **AI Assistant** pane is the same surface as a chat — threads
stored server-side, grouped and searchable, replayed from stored evidence rather
than silently re-queried, with a KPI strip that is a `SUM` over stored turns and
per-answer counts you can audit.

→ **[How the assistant works](docs/assistant.md)** · **[HTTP API](docs/api.md)**

---

## Running it

```bash
# Web app (pipeline + dashboard)
intelligence-os --webcam 0 [--port 8000]
intelligence-os --video clip.mp4              # replay a file through the same pipeline

# Headless, no UI
python -m intelligence_os.run --video clip.mp4 --zones zones.json

# Identity foundation — accumulate and tune entities (do this FIRST)
python -m intelligence_os.phase1 run  --webcam 0
python -m intelligence_os.phase1 tune --video enroll_clip.mp4

# Distil observations into relations / habits / events
python -m intelligence_os.distill

# Inspect and correct memory
python -m intelligence_os.operator list
python -m intelligence_os.operator name   <entity_id> "Raj"
python -m intelligence_os.operator merge  <src> <dst>
python -m intelligence_os.operator split  <entity_id>
python -m intelligence_os.operator delete <entity_id>     # cascades — the privacy path
```

→ **[Configuration](docs/configuration.md)** · **[Tuning](docs/tuning.md)** · **[Deployment](docs/deployment.md)**

---

## Examples

| | | Needs |
|---|---|---|
| 1 | [**The memory graph**](examples/01_memory_graph/) — observations → habits → provenance → correction → deletion | 3 packages |
| 2 | [**The rule cascade**](examples/02_rules_engine/) — why rules get refused, how the dwell gate works | 3 packages |
| 3 | [**The video pipeline**](examples/03_video_pipeline/) — the real cascade on a clip, with the dashboard | Full install |
| 4 | [**Multiple cameras**](examples/04_multi_camera/) — one memory, several streams, per-camera zones | Two sources |

---

## The one number that matters

`IdentityConfig.face_match_threshold`, default `0.45`.

Too low and two people fuse into one entity — silently, and it corrupts history.
Too high and one person fragments into many — annoying, visible, fixable.

**Tune it on your real people in your real lighting before trusting anything
identity-dependent:**

```bash
python -m intelligence_os.phase1 tune --video enroll_clip.mp4
```

→ **[Tuning guide](docs/tuning.md)**

---

## Documentation

| | |
|---|---|
| [Installation](docs/installation.md) | Dependencies, optional extras, first-boot |
| [Architecture](docs/architecture.md) | The cascade, multi-camera, the data model |
| [Configuration](docs/configuration.md) | `config.yaml`, every tunable, env vars |
| [Tuning](docs/tuning.md) | Identity threshold, VLM cost, motion sensitivity |
| [The AI Assistant](docs/assistant.md) | Grounded answers and chat history |
| [HTTP API](docs/api.md) | Every endpoint the dashboard uses |
| [Development](docs/development.md) | Dev loop, tests, schema rules |
| [Deployment](docs/deployment.md) | Running it somewhere real |
| [Responsible use](docs/responsible-use.md) | What lands on you as the deployer, by jurisdiction |
| [Licensing](docs/licensing.md) | Apache-2.0 code, AGPL-3.0 detector — read before selling |
| [FAQ](docs/faq.md) | Keys, storage, RTSP, "why is nothing detected" |

---

## Tests

Plain `unittest`. Offline — no camera, no key, no network. Each test builds its
own temporary database.

```bash
python -m unittest discover -s intelligence_os/tests -t .
```

Two of them are acceptance gates rather than unit tests:

```bash
python -m intelligence_os.tests.test_foundation   # identity + correctness
python -m intelligence_os.tests.test_pipeline     # scene state → distillation
```

`test_foundation` is the one that matters most: if identity is unstable, every
layer above it narrates a fictional world.

---

## Contributing

The surface area is wide and the codebase is small (~6.3k lines, no framework).
[**CONTRIBUTING.md**](CONTRIBUTING.md) has the setup, the house rules and the
PR process; issues tagged [`good first issue`][gfi] are scoped so you don't have
to read the whole cascade first.

Good places to start: more tracked object classes, more delivery channels
(Slack, Discord, webhook), identity robustness, smarter habit mining, more
question shapes in `ask.py`, or a pluggable detector/VLM backend so this isn't
hard-wired to YOLO + Anthropic.

Four house rules worth knowing before you open a PR:

1. **Trace the whole flow you touch** — the cascade is interdependent.
2. **Schema migrations are additive** — an existing `memory.db` has to keep opening.
3. **Ownership checks live in SQL**, not in handlers, so a forgetful caller fails closed.
4. **Every new claim needs provenance.** No exceptions; it's the whole point.

[gfi]: https://github.com/Infinex-Labs/Intelligence-OS/labels/good%20first%20issue

---

## Privacy and responsible use

Not a footnote, a design constraint:

- **Face recognition is opt-in** and ships off. Most rules, zones and alerts
  work without it. Turning it on makes this a biometric system — with the legal
  weight that carries.
- **Keyframes, not video**, pruned at `raw_retention_days` (default 7).
- **Deletion cascades** and is meant to be irreversible.
- **No audio, ever.** Deliberate: wiretap law is far harsher than
  video-surveillance law, and a microphone would be the largest single increase
  in risk this project could take.
- **Nothing leaves the machine** unless you set an API key or configure a
  delivery channel. No telemetry, ever.

Intelligence OS is meant for **premises you are responsible for, with the
knowledge of the people who enter them.** If you run it against real people, you
are the data controller — the authors never see your footage and cannot comply
on your behalf. **[docs/responsible-use.md](docs/responsible-use.md)** has the
pre-deployment checklist and what applies where (GDPR Art. 9, the EU AI Act,
Illinois BIPA, India's DPDP Act).

Never use an output of this system as the sole basis for an accusation or a
decision affecting someone. Identity matching has a real error rate in both
directions, and it does not fail uniformly across demographic groups. The graph
is correctable precisely because it is expected to be wrong sometimes.

Things we will not build — covert operation, emotion inference, real-time
intervention — are listed with reasons in
[responsible use](docs/responsible-use.md#what-we-will-not-build).

---

## Community

- 💬 [Discussions](https://github.com/Infinex-Labs/Intelligence-OS/discussions) — questions, tuning advice, show and tell
- 🐛 [Issues](https://github.com/Infinex-Labs/Intelligence-OS/issues) — bugs and feature requests
- 🔒 [Security](SECURITY.md) — report privately, never in a public issue
- 📋 [Changelog](CHANGELOG.md)

## License

[Apache License 2.0](LICENSE) — commercial use, modification and distribution
permitted, with an express patent grant. No CLA.

One caveat worth knowing before you build a business on it: YOLO arrives via
`ultralytics`, which is **AGPL-3.0**, so a default deployment is an AGPL-3.0
combined work even though this repository's own source is Apache-2.0. Hosting
the dashboard for other users engages AGPL §13. See
**[docs/licensing.md](docs/licensing.md)** for the options — a permissively
licensed detector backend would remove the constraint entirely, and is the
highest-value contribution anyone could make here.
