# Configuration

Three layers, in precedence order:

1. **CLI flags** — `--webcam`, `--video`, `--zones`, `--sensitivity`, `--port`.
2. **`config.yaml`** at the repo root — what the UI writes back, and what you
   check into your own deployment repo (minus the credentials).
3. **`intelligence_os/config.py`** — every tunable, as typed dataclasses. Edit
   the defaults here; there is no separate settings file for these.

With none of the above, the app boots on webcam 0. That is deliberate: a first
run should require nothing.

---

## `config.yaml`

Lives at the repo root. Written by the UI when you save settings, so changes
made in the dashboard survive a restart. **Gitignored, because RTSP URLs carry
credentials.** A redacted sample is in
[`examples/04_multi_camera/config.yaml`](../examples/04_multi_camera/config.yaml).

```yaml
cameras:
  - { name: front_door, source: "rtsp://user:pass@10.0.0.5/stream1" }
  - { name: back_yard,  source: "rtsp://user:pass@10.0.0.6/stream1" }
  - { name: webcam,     source: 0 }        # an int, not a string

zones: examples/04_multi_camera/zones.json  # optional; or draw them in the UI
face_matching: false                        # stage C on/off — see tuning.md
sensitivity: balanced                       # lazy | balanced | eager
```

Camera names must be **unique** — a duplicate is a boot error rather than a
silent overwrite, because two cameras sharing a name would corrupt the graph's
sense of where things happened.

---

## Environment variables

All read at import time in `config.py`, so set them before starting the process.

| Var | Use |
|---|---|
| `ANTHROPIC_API_KEY` | Enables the VLM stage, the assistant, and rule compilation. Read from `intelligence_os/.env` or the environment. |
| `INTELLIGENCE_OS_DB` | Point at a scratch database so your real memory graph isn't touched. |
| `INTELLIGENCE_OS_DATA` | Relocate the whole data directory (DB + frames + rules). |
| `INTELLIGENCE_OS_YOLO` | Use a specific weights file instead of the auto-download. |
| `INTELLIGENCE_OS_CONFIG` | Alternate `config.yaml` — handy for a multi-camera fixture. |
| `IDENTITY_ENABLED` | Lets rules that reference *named people* compile (`rules.py`). Separate from the face matcher itself, which is `face_matching` in `config.yaml`. |
| `TELEGRAM_BOT_TOKEN` | Telegram delivery channel (`deliver.py`). |

---

## The tunables in `config.py`

### `IdentityConfig` — who is who

| Knob | Default | What it does |
|---|---|---|
| `enabled` | `false` | **Face recognition is opt-in.** Off, entities are stable within a session by tracking but not across days. |
| `face_match_threshold` | `0.45` | Cosine similarity above which a face is *the same person*. **The riskiest number in the project** — see [tuning](tuning.md). |
| `max_signatures_per_entity` | `12` | Cap on gallery vectors per person, to capture pose and lighting variation. |
| `signature_novelty_min` | `0.10` | A new signature is only stored if it is this different (1 − cos) from existing ones, so you keep variety rather than twelve near-duplicates. |
| `min_face_det_score` | `0.55` | Below this, a face is too uncertain to use for identity at all. |

### `MotionConfig` — stage A

| Knob | Default | What it does |
|---|---|---|
| `min_motion_fraction` | `0.002` | Fraction of pixels that must change to count as motion. Raise it if a swaying tree keeps the gate open; lower it if slow movement is being missed. |
| `mog2_history` | `200` | Frames of background history. |
| `mog2_var_threshold` | `32.0` | MOG2 sensitivity. |
| `diff_threshold` | `25` | Pixel-intensity delta in the frame-diff fallback. |

### `TriggerConfig` — when to spend money

| Knob | Default | What it does |
|---|---|---|
| `sensitivity` | `lazy` | `lazy` / `balanced` / `eager`. The web app defaults to `balanced` for a responsive live demo; `lazy` is the right bias for a long-running memory builder, where scene-diff is the safety net. |
| `vlm_cooldown_seconds` | `30.0` | Hard floor between VLM calls per scene signature. This is your cost ceiling. |
| `observation_cooldown_seconds` | `10.0` | Stops continuous presence from spamming memory with identical rows. |
| `settle_frames` | `3` | Frames of stillness that mark a keyframe as "settled" and worth keeping. |

### `DetectConfig` — stage B

| Knob | Default | What it does |
|---|---|---|
| `conf` | `0.35` | YOLO confidence floor. |
| `iou` | `0.5` | NMS IoU. |
| `object_classes` | chair, laptop, cell phone, bottle, cup, book, backpack, handbag, potted plant, tv, couch, bed | COCO classes tracked as objects of interest. Everything else relies on the VLM for open-vocabulary naming. **Adding to this list is a good first contribution.** |
| `yolo_weights` | auto | `yolo26s.pt` if present next to the repo, else downloaded. |
| `tracker_cfg` | `bytetrack.yaml` | Tracker config passed to ultralytics. |

### `DistillConfig` — stage H

| Knob | Default | What it does |
|---|---|---|
| `weight_increment` | `0.15` | How much one reinforcement moves an edge's weight. |
| `weight_cap` | `1.0` | Ceiling. |
| `confirm_weight` | `0.6` | Where `candidate` becomes `confirmed`. |
| `decay_half_life_days` | `14.0` | How fast an un-reinforced habit fades. Lower it for a fast-changing space. |
| `live_interval_s` | `60.0` | Distillation cadence while the app runs. `0` turns it off and leaves it to `python -m intelligence_os.distill`. |

### `VLMConfig` — stage F

| Knob | Default | What it does |
|---|---|---|
| `model` | `claude-opus-4-8` | Used for both the describer and the reasoner role. |
| `max_tokens` | `1500` | Per call. |
| `enabled` | auto | True when `ANTHROPIC_API_KEY` is set. Without it, the system falls back to a template describer rather than failing. |

### Retention

| Knob | Default | What it does |
|---|---|---|
| `raw_retention_days` | `7` | Keyframes older than this are pruned at startup. This is the setting that keeps the system from becoming a video archive. |

---

## Zones

Zones are polygons in **one camera's pixel space**, so a zone only means
something on the camera it was drawn on. Draw them in the Zones pane, or supply
a JSON file:

```json
{ "zones": [ { "name": "doorway", "polygon": [[180,120],[420,120],[420,400],[180,400]] } ] }
```

A box is assigned to the **first** zone containing it, so list specific zones
before catch-all ones.

Presence is only recorded *somewhere* once a zone exists — and **a rule whose
zone doesn't exist can never fire**. The rule engine warns about that at startup
rather than silently watching nothing.
