# Example 4 — Multiple cameras, one memory

```bash
cp examples/04_multi_camera/config.yaml ./config.yaml   # then edit the URLs
python -m intelligence_os.web --port 8000
```

No CLI flag needed: with no `--webcam`/`--video`, the app reads `config.yaml`
from the repo root. Pass a flag and it wins over the file.

Files here:

- **`config.yaml`** — the camera list, zones path, and the two toggles that
  change behaviour most (`face_matching`, `sensitivity`).
- **`zones.json`** — the zone format, if you would rather check zones into a
  repo than draw them in the UI.

## How the cameras actually become one picture

`run.py` starts **one thread per camera**. Every thread shares **one SQLite
store** and **one set of models** — loading YOLO three times would cost three
times the memory for identical weights.

What is per-camera:

- the motion gate and settled-keyframe detector (each stream has its own idea of
  "nothing is happening"),
- zones, because frame geometry is per-camera.

What is global:

- **identity.** A face seen on `front_door` is scored against *every* signature
  in the gallery, so it resolves to the same entity when it turns up on
  `back_yard` later.

That shared face gallery is the **only** thing fusing cameras into one picture.
There is no camera-to-camera handoff logic, no geometric calibration, no
overlap assumptions — which also means:

> **With `face_matching: false`, you do not have multi-camera identity.** You
> have several independent camera timelines writing into one database. That is
> still useful — zones, rules, alerts, the timeline all work per-camera — but a
> person crossing from one camera to another will be two entities.

Faces are the sole long-term signal. Body and clothing embeddings are same-day
only, because people change clothes.

## Before you trust it

Turning on `face_matching` is the point at which this system starts making
claims about *who*, and the threshold that governs it is the single riskiest
knob in the project:

```yaml
face_matching: true
```

```bash
python -m intelligence_os.phase1 tune --video enroll_clip.mp4
```

`IdentityConfig.face_match_threshold` defaults to `0.45`. Too low fuses two
people into one entity; too high fragments one person into many. **Tune it on
your real people in your real lighting** — see [docs/tuning.md](../../docs/tuning.md).

## Gotchas

- **Camera names must be unique.** A duplicate is a boot error, deliberately.
- **A zone belongs to one camera.** Coordinates are pixels in that camera's
  frame; the same polygon on a differently-sized stream is meaningless.
- **A rule whose zone doesn't exist can never fire.** The engine warns at
  startup — believe the warning.
- **RTSP that won't connect.** The pinned `opencv-python-headless==4.10` ships
  `FFMPEG:YES`, which RTSP needs. Check the URL resolves with `ffprobe` first.
  A camera that refuses at startup currently takes its thread down for the life
  of the process — retry-with-backoff is on the [roadmap](../../ROADMAP.md).
- **Never commit your `config.yaml`.** It has credentials in it. The root one is
  gitignored; this sample is redacted.
