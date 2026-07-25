"""Headless stubs for the heavy optional deps — import for the side effect.

`sys.modules.setdefault` on its own is wrong once the venv actually has numpy
and cv2 installed: whichever test module imports first wins for the entire
discovery run, so a stubbed cv2 leaked into test_multicam and broke its real
`cv2.imwrite`. Stub only what isn't installed.
"""
import sys
import types

for _m in ("numpy", "cv2", "ultralytics", "torch"):
    try:
        __import__(_m)
    except ImportError:
        sys.modules[_m] = types.ModuleType(_m)

if not hasattr(sys.modules["numpy"], "ndarray"):
    sys.modules["numpy"].ndarray = object
