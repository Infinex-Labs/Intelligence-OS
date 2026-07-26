# Example 3 — The video pipeline

The real cascade, end to end, against a video file. `--video` replays a file
through the identical pipeline a live camera goes through, so this is also how
you get reproducible runs while developing.

**Needs the full dependency set.** The first run downloads YOLO weights
(~20 MB) and, if you enable identity, InsightFace `buffalo_l` (~280 MB). That
first boot is slow because it is downloading, not because it has hung.

```bash
pip install -r requirements.txt
```

## Option A — your own footage (recommended)

Any mp4 with people in it. Thirty seconds is plenty.

```bash
python -m intelligence_os.web --video /path/to/your/clip.mp4 --port 8000
# → http://localhost:8000
```

First visit asks you to create a login. Then:

1. **Live** — the replayed frames, with detections drawn on them.
2. **Timeline** — observations as they are written. This is the memory graph
   filling up.
3. **Zones** — draw a polygon on the frame and name it. Zones are per-camera
   because frame geometry is per-camera. Presence is only recorded *somewhere*
   once a zone exists.
4. **AI Assistant** — ask "who was in the frame?" once there are observations to
   answer from. Needs `ANTHROPIC_API_KEY`; everything else on this list does not.

## Option B — a synthetic clip, for smoke-testing

```bash
python examples/03_video_pipeline/make_clip.py     # writes clip.mp4 next to this file
python -m intelligence_os.web --video examples/03_video_pipeline/clip.mp4
```

**YOLO will find no people in it** — it is a moving rectangle. The memory graph
stays empty on purpose. What this clip does exercise is everything that has
nothing to do with the models: the capture loop, the MOG2 motion gate (the
rectangle stops moving three-quarters of the way through, and you can watch
frames stop getting through), the settled-keyframe detector, the MJPEG stream,
and the dashboard. It is the fastest way to confirm your install works.

## Headless, no dashboard

```bash
python -m intelligence_os.run --video clip.mp4 --zones zones.json --snapshot-every 30
```

Then inspect and correct what it built:

```bash
python -m intelligence_os.operator list
python -m intelligence_os.operator inspect <entity_id>
python -m intelligence_os.operator name <entity_id> "Raj"
python -m intelligence_os.distill                     # roll observations into habits
```

## What to expect on a first pass

- **Entities are anonymous.** `entity_3` is a successful outcome, not a failure.
  Naming is an operator action.
- **Face identity is off by default.** Without it, a person is re-identified
  within a session by tracking, but not across days. Turn it on with
  `face_matching: true` in `config.yaml`, then
  [tune the threshold](../../docs/tuning.md) on your own footage before trusting
  anything identity-dependent.
- **Nothing is described in natural language without an API key.** Detection,
  tracking, zones, observations, rules and distillation all work without one.
- **Keyframes are pruned** at `raw_retention_days` (default 7). This is not a
  video archive.
