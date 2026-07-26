# Running in Docker

For people who want to **run** Intelligence OS rather than develop it. No Python
environment, no dependency resolution, no `numpy<2` ABI archaeology — one image
and one volume.

If you want to change the code, don't use this. Use
[the development setup](../CONTRIBUTING.md#setting-up); there is no hot reload
here and rebuilding the image for a one-line edit is a bad loop.

---

## Quick start

```bash
git clone https://github.com/Infinex-Labs/Intelligence-OS.git
cd Intelligence-OS
docker compose up -d
docker compose logs -f
```

Then open **http://localhost:8000** and create the first account.

The first boot downloads the YOLO weights (~40 MB) before the pipeline starts,
so give it a minute. The download lands in the volume, not the image, so it
happens once rather than on every container start.

Or without compose:

```bash
docker build -t intelligence-os .
docker run -d --name intelligence-os \
  -p 127.0.0.1:8000:8000 \
  -v io-data:/data \
  --restart unless-stopped \
  intelligence-os
```

## What's in the image

| | |
|---|---|
| Base | `python:3.11-slim-bookworm`, two-stage — the compiler toolchain `lap` needs doesn't ship |
| Size | ~1.6 GB. Torch is most of it |
| Torch | **CPU-only wheels**, installed from PyTorch's CPU index. The default PyPI wheels bundle ~2.5 GB of CUDA libraries that a CPU instance never loads |
| User | Non-root, uid `10001` |
| Stages included | Motion gate, detection, tracking, scene state, memory, distillation, the dashboard and the assistant |
| Stages **not** included | Face identity (stage C). `insightface` + `onnxruntime` are a large addition for a feature that ships off — see [below](#enabling-face-identity) |

## Configuration

Everything mutable is under `/data`, which is the volume:

```
/data
├── memory.db          # the graph — plus WAL and SHM files
├── frames/            # retained keyframes, pruned at raw_retention_days
├── config.yaml        # cameras and zones, written by the UI or by you
├── .ultralytics/      # ultralytics settings
└── yolo26s.pt         # weights, fetched on first run
```

The image sets `INTELLIGENCE_OS_CONFIG=/data/config.yaml` explicitly. Without
it, `config.yaml` resolves next to the installed package inside `site-packages`
and would be thrown away on every container recreate.

### Cameras

The default source is webcam 0, which does not exist in a container. Give it a
real source before it can do anything useful. Easiest path is the settings page
in the UI, which writes `config.yaml` for you. To do it by hand:

```bash
docker compose exec intelligence-os sh -c 'cat > /data/config.yaml' <<'YAML'
cameras:
  - name: front-door
    source: rtsp://user:pass@192.168.1.50:554/stream1
  - name: warehouse
    source: rtsp://user:pass@192.168.1.51:554/stream1
sensitivity: balanced
YAML
docker compose restart
```

RTSP needs nothing special — the container has `ffmpeg` and network access.
See [configuration](configuration.md) for the full schema and
[`examples/04_multi_camera/`](../examples/04_multi_camera/) for a worked file.

**A USB webcam** requires device passthrough, which only works on a **Linux**
host — uncomment the `devices:` block in `docker-compose.yml`. Docker Desktop on
macOS and Windows runs containers inside a VM with no USB passthrough, so use
RTSP or a video file there instead.

**A video file** — mount a directory and point at it:

```yaml
volumes:
  - ./media:/media:ro
```

```yaml
# /data/config.yaml
cameras:
  - name: replay
    source: /media/clip.mp4
```

### Scene descriptions (optional)

Stage F calls Anthropic and is off without a key. Put it in a `.env` file next
to `docker-compose.yml` — compose reads that automatically:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Don't pass it with `-e` on a `docker run` command line; it ends up in your shell
history and in `docker inspect`. Everything else works without it.

### Enabling face identity

Stage C isn't in the image because it ships **off**, and `insightface` plus
`onnxruntime` is a large addition for a disabled feature. Turning it on also
turns this into a biometric system with real legal weight attached — read
[responsible use](responsible-use.md) first, not after.

If you need it, extend the image:

```dockerfile
FROM intelligence-os:latest
USER root
RUN pip install --no-cache-dir "insightface>=0.7" "onnxruntime>=1.17"
USER app
```

Then set `enabled: true` under `identity` in your config.

## Deploying on a cloud instance

The container binds `0.0.0.0` **inside** its own network namespace — it has to,
or the published port reaches nothing. The isolation is done by the port
publish, and the default publishes to the host's loopback only:

```yaml
ports:
  - "127.0.0.1:8000:8000"
```

That means the dashboard is not on the instance's public interface. Reach it
over an SSH tunnel:

```bash
ssh -L 8000:localhost:8000 ubuntu@your-instance
# now http://localhost:8000 on your laptop
```

**Do not change that to `8000:8000` and call it done.** There is no TLS, no CSRF
token and no rate limiting on the login endpoint. If you need real access for
more than yourself, terminate TLS at a proxy and restrict by source — the nginx
config in [deployment](deployment.md#exposing-it-beyond-localhost) applies
unchanged, pointing at the published port.

Also worth doing on any cloud instance:

- **Security group**: don't open 8000. Open 22, or 443 to your proxy.
- **Sizing**: YOLO is the expensive stage. Roughly one camera per 2 vCPU for
  comfortable throughput — a `t3.large` handles one or two, four wants a GPU or
  a smaller model. The `deploy.resources.limits` in the compose file are a
  starting point, not a measurement of your workload.
- **Disk**: keyframes are pruned at `raw_retention_days` (default 7), but size
  the volume for your retention window and camera count, not for the default.
- **A GPU instance**: drop the `--extra-index-url` line from the Dockerfile,
  rebuild, and run with `--gpus all`. Nothing else changes.

## Operating it

```bash
docker compose logs -f                      # follow
docker compose restart                      # after a config change
docker compose down                         # stop; the volume survives
docker compose down -v                      # stop and DELETE the volume
```

That last one destroys the database, the keyframes and the login. There is no
confirmation prompt.

**Backups.** The volume is the whole system:

```bash
docker compose stop                         # WAL mode: a live copy can be torn
docker run --rm -v io-data:/data -v "$PWD":/backup alpine \
  tar czf /backup/io-backup-$(date +%F).tar.gz -C /data .
docker compose start
```

`memory.db` contains face embeddings if identity is enabled — biometric data.
Encrypt the backup and treat it like the original.

**Restore:**

```bash
docker compose down
docker run --rm -v io-data:/data -v "$PWD":/backup alpine \
  sh -c 'rm -rf /data/* && tar xzf /backup/io-backup-2026-07-26.tar.gz -C /data'
docker compose up -d
```

**A shell in the container**, for the operator CLI:

```bash
docker compose exec intelligence-os python -m intelligence_os.operator delete ent_...
```

**Health.** The image has a `HEALTHCHECK` against `/api/auth/status`, which is
unauthenticated by design — it's what the login page calls to decide whether to
show first-run setup:

```bash
docker inspect --format '{{.State.Health.Status}}' intelligence-os
```

`--start-period` is 180s because the first boot downloads weights before the
server is useful.

## Using a bind mount instead of a named volume

A named volume is the default because it avoids uid mismatches. If you want the
data somewhere you can see it, the container runs as uid `10001`, so:

```bash
mkdir -p ./data && sudo chown -R 10001:10001 ./data
```

```yaml
volumes:
  - ./data:/data
```

Skip the `chown` and the container fails to write its database on first boot.

## Troubleshooting

**Nothing is detected.** Almost always the camera source. Check the logs for a
frame-read failure, confirm the RTSP URL resolves from inside the container
(`docker compose exec intelligence-os ffprobe "$URL"`), and remember the default
webcam 0 doesn't exist here. [FAQ](faq.md) has the longer list.

**Container is `unhealthy`.** Give it three minutes on a first run — weights are
downloading. If it persists, `docker compose logs` will have the traceback.

**Permission denied on `/data`.** A bind mount without the `chown` above.

**Weights re-download on every start.** The volume isn't mounted at `/data`, so
each container gets a fresh one.

**Build fails compiling `lap`.** It needs numpy headers present *before* it
builds, which is why the Dockerfile installs numpy in its own step. If you've
reordered those layers, put it back.

## Licensing note

The image bundles `ultralytics`, which is **AGPL-3.0**. That makes a running
container an AGPL-3.0 combined work even though this project's own source is
Apache-2.0 — and because the dashboard is served over a network, AGPL §13
applies once other people use it. If you are deploying this as part of a product
or hosting it for anyone but yourself, read [licensing](licensing.md) before you
ship it.

The Dockerfile does not bake weights into the image, partly for size and partly
because redistributing Ultralytics' weights carries the same licence.
