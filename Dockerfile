# Intelligence OS — runtime image.
#
# For people who want to *run* this, not develop it. Build:
#
#   docker build -t intelligence-os .
#   docker run --rm -p 127.0.0.1:8000:8000 -v io-data:/data intelligence-os
#
# Two stages so the compiler toolchain that `lap` needs to build doesn't ship in
# the final image.

# ---------------------------------------------------------------- build stage
FROM python:3.11-slim-bookworm AS build

# lap has no universal wheel and compiles against numpy headers; git is here
# because a few ultralytics extras resolve from git refs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Everything lands in a venv we copy wholesale into the runtime stage.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Not cosmetic. Without it the build runs in `/`, and setuptools' package
# auto-discovery walks the entire root filesystem looking for `intelligence_os*`
# — including /proc and /sys, where it hangs indefinitely with no output.
WORKDIR /src

# Quote the specifiers. Unquoted, `setuptools>=77` is a shell redirection into a
# file called `=77` and the constraint silently vanishes.
#
# numpy goes in first and alone: lap's setup.py needs the headers *at build
# time*, and within one pip invocation there is no ordering guarantee.
RUN pip install --no-cache-dir --upgrade pip "setuptools>=77" wheel \
    && pip install --no-cache-dir "numpy<2"

# CPU-only torch, explicitly. The default PyPI wheels carry bundled CUDA
# libraries — roughly 2.5 GB of them — which is pure waste on the CPU instances
# most people will run this on. If you have a GPU box, drop the index URL and
# rebuild; nothing else in the image needs to change.
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch torchvision

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ultralytics depends on `opencv-python` — the GUI build — so pip installs it
# *over* the headless one we pinned. Both unpack into the same `cv2/` directory,
# so the winner is whichever landed last, and the pin quietly loses. Force the
# issue: remove both, reinstall headless alone. Verified below rather than
# assumed, because a silent regression here is a container that drags in GUI
# libraries it can never use.
RUN pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir "opencv-python-headless==4.10.*" \
    && python -c "import cv2, sys; \
v = cv2.__version__; \
ff = 'FFMPEG:                      YES' in cv2.getBuildInformation(); \
sys.exit(f'expected 4.10.x with FFMPEG, got {v} ffmpeg={ff}') if not (v.startswith('4.10.') and ff) else print(f'cv2 {v} ffmpeg=yes')"

# The package itself last, so edits to source don't invalidate the dependency
# layers above — those are the slow ones.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY intelligence_os ./intelligence_os
# --no-build-isolation: setuptools and wheel are already in this venv, so the
# isolated build env would be a pure round trip to PyPI for the same packages.
RUN pip install --no-cache-dir --no-deps --no-build-isolation .

# -------------------------------------------------------------- runtime stage
FROM python:3.11-slim-bookworm AS runtime

# opencv-python-headless still needs libGL's ABI present for some codec paths,
# and ffmpeg is what actually decodes RTSP. curl is for the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root. The uid is fixed so a bind-mounted host directory has a predictable
# owner to chown to (see docs/docker.md).
RUN useradd --create-home --uid 10001 app

# Everything mutable lives under /data, which is the only path that needs to be
# a volume: the database, retained keyframes, the UI-written config, and the
# YOLO weights that ultralytics fetches on first run.
#
# INTELLIGENCE_OS_CONFIG has to be set explicitly. Unset, config.yaml resolves
# next to the installed package inside site-packages, where it would be silently
# discarded on every container recreate.
ENV INTELLIGENCE_OS_DATA=/data \
    INTELLIGENCE_OS_CONFIG=/data/config.yaml \
    YOLO_CONFIG_DIR=/data/.ultralytics \
    MPLCONFIGDIR=/data/.cache/matplotlib \
    PYTHONUNBUFFERED=1

# WORKDIR is /data on purpose rather than for tidiness: ultralytics downloads
# weights relative to the working directory, so this is what makes the ~40 MB
# fetch happen once instead of on every container start.
WORKDIR /data
RUN mkdir -p /data && chown -R app:app /data
VOLUME ["/data"]

USER app
EXPOSE 8000

# Unauthenticated by design — it is what the login screen itself calls to decide
# whether to show first-run setup — so it works as a liveness probe without
# baking credentials into the image. start-period is generous because the first
# boot downloads model weights before the server is useful.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/auth/status || exit 1

# 0.0.0.0 is correct *here* and nowhere else: the container's network namespace
# is the boundary, and loopback inside the container would make the published
# port unreachable. Publish to 127.0.0.1 on the host unless a proxy fronts it.
ENTRYPOINT ["intelligence-os", "--host", "0.0.0.0", "--port", "8000"]
