"""Acceptance gate for §10 config foundation: resolve_source precedence.

CLI flag > config.yaml > default (webcam 0). No camera hardware, no network.
Run: .venv/bin/python -m intelligence_os.tests.test_config
"""
import tempfile
import types
from pathlib import Path

from intelligence_os import config as C
from intelligence_os.run import resolve_source


def _args(**kw):
    d = {"webcam": None, "video": None, "zones": None, "sensitivity": None}
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_precedence():
    tmp = Path(tempfile.mkdtemp()) / "config.yaml"
    tmp.write_text("camera: /clips/a.mp4\nzones: z.json\nsensitivity: eager\n")
    orig = C.APP_CONFIG_PATH
    C.APP_CONFIG_PATH = tmp
    try:
        # config.yaml fills when the CLI gave nothing
        a = _args(); resolve_source(a)
        assert a.video == "/clips/a.mp4" and a.webcam is None, a
        assert a.zones == "z.json" and a.sensitivity == "eager", a

        # CLI flag wins over config.yaml
        a = _args(webcam=2); resolve_source(a)
        assert a.webcam == 2 and a.video is None, a

        # an integer camera resolves to a webcam index, not a path
        tmp.write_text("camera: 1\n")
        a = _args(); resolve_source(a)
        assert a.webcam == 1 and a.video is None, a

        # no config file at all -> default webcam 0 (the ten-minute path)
        C.APP_CONFIG_PATH = tmp.parent / "nope.yaml"
        a = _args(); resolve_source(a)
        assert a.webcam == 0 and a.video is None, a
    finally:
        C.APP_CONFIG_PATH = orig
    print("  resolve_source precedence (CLI > config.yaml > webcam 0) OK")


if __name__ == "__main__":
    test_precedence()
    print("test_config: ALL PASS")
