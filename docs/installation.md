# Installation

Requires **Python 3.10+**. macOS and Linux are what it is developed on.

If you only want to *run* this, **[Docker](docker.md)** is the shorter path —
`docker compose up -d` and nothing on this page applies. Everything below is for
a local Python environment, which is what you want if you intend to change the
code.

## The short version

```bash
git clone https://github.com/Infinex-Labs/Intelligence-OS.git
cd Intelligence-OS
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

intelligence-os --webcam 0        # → http://localhost:8000
```

First visit prompts you to create a login. There is no default password.

## Choosing what to install

| Install | Gets you | Cost |
|---|---|---|
| `pip install -r requirements-dev.txt` | The test suite and examples 1–2. Detection does **not** run. | 3 packages, seconds |
| `pip install -r requirements.txt` | The full pipeline: motion, detection, tracking, zones, observations, rules, distillation, dashboard. | Pulls torch — several hundred MB |
| `+ pip install -e ".[identity]"` | Stage C: face re-identification across days and cameras. | +InsightFace, +onnxruntime |
| `+ pip install -e ".[vlm]"` | Stage F: natural-language scene descriptions, the AI Assistant, rule compilation. | +anthropic, and an API key |
| `+ pip install -e ".[semantic]"` | Search by meaning: "loitering" finds a recorded "standing around, waiting". | +sentence-transformers, ~120 MB of weights |

The design point is that **the middle row is fully useful on its own**: no cloud
key, no biometrics, no account anywhere. The optional extras upgrade it; they
are not prerequisites.

## The API key

Only needed for stage F (descriptions), the assistant, and compiling rules from
English.

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > intelligence_os/.env
```

Note the location: **`intelligence_os/.env`**, next to the package — not the repo
root. See `_load_dotenv` in `config.py`. A plain `export ANTHROPIC_API_KEY=...`
works too.

## The first boot is slow, and that is a download

On the first run the pipeline fetches:

- YOLO weights (`yolo26s.pt`, ~20 MB), and
- if identity is enabled, InsightFace `buffalo_l` (~280 MB).

There is no progress bar on some of this. It is downloading, not hung. Give it a
few minutes on a first run and it will never happen again.

The semantic search model (`all-MiniLM-L12-v2`, ~120 MB) downloads the first
time anything asks for a vector — normally the first distillation pass after you
install it. It runs entirely on your machine afterwards, with no network and no
key. To fetch it and index existing memory up front:

```bash
python -m intelligence_os.semantic          # build the index
python -m intelligence_os.semantic --query "someone loitering"
```

If it is missing, search falls back to term matching and says so in the trace.
Nothing waits on it and nothing fails.

## Visual re-ranking (off, and needs no extra install)

Optional and **off by default**. It reuses the `[semantic]` extra — the same
`sentence-transformers` package carries CLIP — so there is no third dependency,
only ~350 MB more weights fetched on first use.

```bash
export INTELLIGENCE_OS_VISUAL=1              # or visual_reranking: true in config.yaml
python -m intelligence_os.visual             # index retained keyframes
python -m intelligence_os.visual --rank "a dog"
```

Read what it does narrowly, because the obvious reading is wrong. It **orders**
results the word and meaning indexes already found, putting the one whose
picture best fits your words first. It cannot find a result those indexes
missed, and asking for something never recorded still returns nothing.

`--rank` is spelled that way rather than `--search` for the same reason: it
prints the closest frames to a phrase and explicitly does not claim any of them
contains it. An image model always has a closest frame. See *What Phase 8 moved,
and what it refused to* in [search-baseline.md](search-baseline.md) for the
measurement, which is the whole argument for why this is a re-ranker.

## Pins you should not casually change

Both are in `requirements.txt` with the same reasoning:

- **`numpy<2`.** The torch / opencv / onnxruntime wheels are built against the
  numpy 1.x ABI. Installing numpy 2 gives you import errors or segfaults, not a
  deprecation warning.
- **`opencv-python-headless==4.10.*`.** This build ships `FFMPEG:YES`, which
  RTSP needs. Before floating it, check `cv2.getBuildInformation()` still shows
  FFMPEG support.

If you already have `opencv-python` (the non-headless build) installed, you do
not need both — headless is specified because the server has no display.

## Verifying the install

```bash
python -m unittest discover -s intelligence_os/tests -t .    # 110 tests, offline
python examples/01_memory_graph/run.py                       # the graph, no camera
```

Both work with only `requirements-dev.txt` installed. If the suite passes, the
memory layer — the part that matters — is sound, whatever the camera is doing.

## Where state lives

Three places, all gitignored, all safe to delete:

```bash
rm intelligence_os/data/memory.db*      # all three files — SQLite is in WAL mode
rm -rf intelligence_os/data/frames/*    # retained keyframes
rm config.yaml                          # UI-written settings
```

Deleting `memory.db` also deletes the login; the next page visit asks you to
create one again. It is the fastest way back to a clean slate.

Relocate the whole data directory with `INTELLIGENCE_OS_DATA=/path/to/dir` — see
[configuration](configuration.md).

## Troubleshooting

**`ModuleNotFoundError: No module named 'cv2'`** — you installed
`requirements-dev.txt` only, or the venv isn't activated.

**The dashboard 404s on its own page.** Fixed in 0.1.0: static assets now
resolve against the package rather than the working directory. If you are on an
older checkout, run from the repo root.

**RTSP connects with `ffprobe` but not here.** Check the FFMPEG pin above.

**`lap` fails to build.** ByteTrack's assignment solver needs a compiler.
On Debian/Ubuntu: `apt install build-essential python3-dev`. It is pinned in
`requirements.txt` because ultralytics imports it lazily and would otherwise
only fail at the first tracked frame.
