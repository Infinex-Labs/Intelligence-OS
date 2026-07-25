"""Acceptance gate for §10 ingest resilience: stream vs. file, and RTSP reconnect.

No camera, no network: cv2.VideoCapture is faked. Run:
    .venv/bin/python -m intelligence_os.tests.test_capture
"""
import time

import numpy as np

import intelligence_os.capture as C


def test_is_stream():
    assert C._is_stream(0) is True                       # webcam index
    assert C._is_stream("rtsp://cam/stream") is True     # the bug this fixes: RTSP != file
    assert C._is_stream("http://cam/mjpeg") is True
    assert C._is_stream("clip.mp4") is False             # local file -> EOF ends it
    assert C._is_stream("/data/a.mov") is False
    print("  _is_stream: webcam/RTSP/HTTP -> stream, path -> file OK")


class _FakeCap:
    def __init__(self, reads):
        self.reads = list(reads)
        self.opened = True
    def isOpened(self): return self.opened
    def read(self): return self.reads.pop(0) if self.reads else (False, None)
    def get(self, _): return 30.0
    def release(self): self.opened = False


def test_stream_reconnect():
    """A stream that drops reads long enough reconnects and keeps yielding — it does
    NOT end the session (that would make a down camera look like a quiet one)."""
    frame = np.zeros((4, 4, 3), np.uint8)
    # first capture drops forever; after reconnect the second one delivers a frame
    caps = [_FakeCap([(False, None)] * 12), _FakeCap([(True, frame)])]
    orig_vc, orig_sleep = C.cv2.VideoCapture, time.sleep
    C.cv2.VideoCapture = lambda src: caps.pop(0)
    time.sleep = lambda *a: None                          # don't actually wait on backoff
    try:
        cap = C.Capture("rtsp://x", realtime=True)        # opens caps[0]
        got = None
        for f in cap.frames():
            got = f
            break
        # the only frame lives behind a reconnect, so getting it proves reconnect worked
        assert got is not None, "expected a frame after reconnect"
        assert cap.down_since is None, "down_since should clear on recovery"
        assert cap.last_frame_ts is not None
    finally:
        C.cv2.VideoCapture, time.sleep = orig_vc, orig_sleep
    print("  stream reconnect after a sustained drop OK")


def test_file_stops_at_eof():
    frame = np.zeros((4, 4, 3), np.uint8)
    cap_obj = _FakeCap([(True, frame), (True, frame), (False, None)])
    orig_vc = C.cv2.VideoCapture
    C.cv2.VideoCapture = lambda src: cap_obj
    try:
        cap = C.Capture("clip.mp4", realtime=False)       # a file
        n = sum(1 for _ in cap.frames())
        assert n == 2, n                                   # EOF ends it, no reconnect
    finally:
        C.cv2.VideoCapture = orig_vc
    print("  file stops at EOF (no reconnect) OK")


if __name__ == "__main__":
    test_is_stream()
    test_stream_reconnect()
    test_file_stops_at_eof()
    print("test_capture: ALL PASS")
