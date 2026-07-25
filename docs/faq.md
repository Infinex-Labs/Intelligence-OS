# FAQ

### Do I need an Anthropic API key?

No. Without one you still get motion gating, detection, tracking, identity,
zones, observations, rules, alerting, distillation, the timeline and the
dashboard. The key enables the VLM **description** stage (natural-language scene
summaries), the AI Assistant, and compiling rules from English.

### Does it store video?

No. It keeps **keyframes plus the graph**, and prunes keyframes older than
`raw_retention_days` (default 7). It is not a continuous archival system, and
turning it into one is something we've explicitly decided against — see
[what we will not build](responsible-use.md#what-we-will-not-build).

### Why are all my entities called `entity_4`?

Because nobody has named them. Entities are **anonymous by default** — that is
the intended state, not a failure. Name one with
`python -m intelligence_os.operator name <entity_id> "Raj"`, or from the UI.

### Why is the same person showing up as five different entities?

Either face matching is off (the default), or the threshold is too high. With
matching off, a person is stable within a session by tracking but not across
days or cameras. With it on, see [tuning](tuning.md) — and use
`operator merge` to fix the ones already in the graph.

### Two different people got merged into one. Now what?

`python -m intelligence_os.operator split <entity_id>` undoes a bad merge. Then
**raise** the threshold: fusing two people is the failure mode of a threshold
that is too *low*, and it is the more dangerous of the two failures because it
is silent. See [tuning](tuning.md).

### Nothing is being detected at all.

In order of likelihood:

1. The motion gate never opens (a static scene) — check the Live pane shows
   frames moving.
2. Nothing in frame is a class YOLO detects at `DetectConfig.conf` (0.35).
3. You are running the synthetic clip from
   [example 3](../examples/03_video_pipeline/), which contains no people on
   purpose.
4. No zone is drawn, so presence has nowhere to be recorded.

### RTSP won't connect.

The pinned `opencv-python-headless==4.10` ships `FFMPEG:YES`, which RTSP needs —
don't float it to 5.x without checking `cv2.getBuildInformation()`. Verify the
URL resolves with `ffprobe` first, and remember a camera that refuses at startup
currently takes its thread down for the life of the process.

### The first run is taking forever.

It is downloading YOLO weights and, if identity is on, InsightFace `buffalo_l`
(~280 MB). It is a download, not a hang, and it happens once.

### How do I remove someone, for privacy?

```bash
python -m intelligence_os.operator delete <entity_id>
```

It cascades to their signatures, observations and relations. It is meant to be
irreversible.

### Can I run it on a Raspberry Pi?

The memory layer, yes, easily. YOLO on a Pi CPU will be slow enough that you
want a smaller model, a low frame rate, or an accelerator. Nobody has published
numbers — if you try it, [tell us](https://github.com/Infinex-Labs/Intelligence-OS/discussions).

### Can I use a local model instead of Anthropic?

Not yet. A pluggable detector/VLM backend is wanted — local VLMs (Ollama,
llava) would be the first consumer — but it isn't built. Until then the VLM
stage is hard-wired to Anthropic, and remember it is optional.

### How many cameras can it handle?

Models load once and are shared across camera threads, so memory barely grows
with camera count — CPU does. One camera is comfortable on a laptop; four
realistically wants a GPU. Each camera keeps its own motion gate and zones;
identity is global. See [example 4](../examples/04_multi_camera/).

### Can it alert me in real time?

Rules fire in real time and deliver by email, webhook or chat. But the system is
**retrospective by design** — it tells you what happened, and it must not be
deployed as a safety interlock or anything a person's safety depends on.

### Is my data sent anywhere?

Only where you configure it to go:

- **`ANTHROPIC_API_KEY` set** — keyframes go to the Anthropic API for stage F
  descriptions, and questions go there for the assistant's query planning.
- **A delivery channel configured** — digests and alerts go to the SMTP server,
  webhook URL or Telegram bot you pointed them at (`deliver.py`).

With no key and no delivery channel, nothing leaves the machine. There is no
telemetry, ever.

### Can I ask it about things it never saw?

No, and that is the design. The model only translates your question into a
database query; every fact in the answer comes from observation rows. If nothing
was observed, the answer says so. See [assistant.md](assistant.md).

### Why SQLite and not Postgres?

Because the schema is the contract and the deployment is one machine watching
its own cameras. SQLite in WAL mode handles the write volume (observations are
gated, not per-frame), needs no separate service, and makes the whole system's
state a file you can copy. If you outgrow it, the schema is portable — but be
sure you have, rather than assuming you will.

### Why is there no `dist/` source in the repo?

`intelligence_os/static/dist/` is a prebuilt bundle whose source isn't checked
in. Editing it means editing the built asset. Bringing that source into the
repository is a known gap; the main dashboard (`home.html`) is hand-written and
directly editable.
