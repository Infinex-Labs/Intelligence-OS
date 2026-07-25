"""Capture + Motion gate (§A, FR-1).

Frames come from a webcam or a video file (same interface, so the same pipeline
runs on a recorded clip for testing and a live camera in use). The motion gate is
the cheap, every-frame stage that kills most frames before any model runs — the
first link in the cost-control cascade.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np

from .config import CONFIG


@dataclass
class Frame:
    index: int
    timestamp: float          # seconds; wall-clock for webcam, stream-time for file
    image: np.ndarray         # BGR


def _is_stream(source: Union[int, str, Path]) -> bool:
    """A webcam index or a live network URL is a STREAM: a dropped read is a glitch
    to reconnect through, not an end. A local file path is finite: a failed read is
    EOF and ends the session (§10 — RTSP reconnect vs. file playback)."""
    if isinstance(source, int):
        return True
    s = str(source).lower()
    return s.startswith(("rtsp://", "http://", "https://", "udp://", "tcp://", "rtmp://"))


class Capture:
    """Unified webcam / RTSP / video-file source. Streams reconnect with backoff so a
    dropped camera never silently ends ingest; files stop at EOF. `last_frame_ts` and
    `down_since` are the ingest heartbeat the UI reads to tell 'camera down' from 'quiet'."""

    def __init__(self, source: Union[int, str, Path], realtime: bool = False):
        self.source = source
        self.realtime = realtime  # webcam/stream -> wall clock; file -> frame-derived time
        self.is_stream = _is_stream(source)
        self._src = source if isinstance(source, int) else str(source)
        self.last_frame_ts: Optional[float] = None   # wall clock of last good read
        self.down_since: Optional[float] = None      # wall clock ingest went dark, else None
        self.cap = None
        if not self._open():
            raise RuntimeError(f"Could not open capture source: {source!r}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def _open(self) -> bool:
        if isinstance(self._src, str) and self._src.lower().startswith('rtsp'):
            os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')
        self.cap = cv2.VideoCapture(self._src)
        return self.cap.isOpened()

    def frames(self) -> Iterator[Frame]:
        import time as _t
        idx = 0
        start = _t.time()
        misses = 0
        backoff = 1.0
        while True:
            ok, img = self.cap.read()
            if not ok:
                if not self.is_stream:
                    break                       # file EOF -> session done
                misses += 1
                if misses <= 10:
                    _t.sleep(0.05)              # transient hiccup: quick retries
                    continue
                # sustained drop: reconnect with backoff, keep the session alive (§10).
                # Silence is catastrophic — a down camera must not look like a quiet one.
                if self.down_since is None:
                    self.down_since = _t.time()
                print(f"[capture] source {self._src!r} down {int(_t.time() - self.down_since)}s "
                      f"— reconnecting in {backoff:.0f}s")
                self.release()
                _t.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                self._open()
                misses = 0
                continue
            if self.down_since is not None:
                print(f"[capture] source {self._src!r} reconnected after "
                      f"{int(_t.time() - self.down_since)}s")
                self.down_since = None
            misses = 0
            backoff = 1.0
            self.last_frame_ts = _t.time()
            ts = (_t.time() if self.realtime else start + idx / self.fps)
            yield Frame(idx, ts, img)
            idx += 1
        self.release()

    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


@dataclass
class MotionResult:
    present: bool
    fraction: float                       # fraction of pixels changed
    mask: Optional[np.ndarray] = None     # foreground mask (changed regions)


class MotionGate:
    """Background-subtraction motion gate (MOG2). Returns whether meaningful
    motion is present and the changed-region mask, so downstream stages can be
    skipped on static frames (no motion -> drop, no downstream cost)."""

    def __init__(self):
        cfg = CONFIG.motion
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=cfg.mog2_history, varThreshold=cfg.mog2_var_threshold,
            detectShadows=True)
        self.min_fraction = cfg.min_motion_fraction
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def update(self, frame_bgr: np.ndarray) -> MotionResult:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        fg = self.bg.apply(gray)
        # MOG2 marks shadows as 127; keep only hard foreground (255).
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
        fraction = float(np.count_nonzero(fg)) / fg.size
        return MotionResult(fraction >= self.min_fraction, fraction, fg)


class SettleDetector:
    """Detects a 'settled' keyframe: motion just stopped and the scene is steady
    for `settle_frames` consecutive frames (§8 keyframe selection)."""

    def __init__(self, settle_frames: Optional[int] = None):
        self.settle_frames = settle_frames or CONFIG.trigger.settle_frames
        self._still_streak = 0
        self._was_moving = False

    def update(self, motion_present: bool) -> bool:
        """Return True on the single frame where the scene becomes settled after
        having moved (rising edge of stillness)."""
        if motion_present:
            self._still_streak = 0
            self._was_moving = True
            return False
        self._still_streak += 1
        if self._was_moving and self._still_streak >= self.settle_frames:
            self._was_moving = False
            return True
        return False
