"""V4-M2: pausing one camera must not touch the others (FR-LV-4, FR-LV-6).

The wall fans out per camera precisely because the bare /api/camera/toggle
flips *every* camera — this pins both behaviours so neither drifts.
"""
from __future__ import annotations

import unittest

from intelligence_os import web


def _handler(query):
    """A RequestHandler with no socket: only the toggle path is exercised."""
    h = object.__new__(web.RequestHandler)
    h.sent = []
    h.send_json = h.sent.append
    h._parse_qs = lambda: query
    h.send_error = lambda *a: h.sent.append({"error": a})
    return h


class TestCameraToggle(unittest.TestCase):
    def setUp(self):
        web.pipeline_state["cameras"] = {
            "front": {"paused": False, "last_frame_ts": 1.0},
            "back": {"paused": False, "last_frame_ts": 2.0},
        }

    def cams(self):
        return web.pipeline_state["cameras"]

    def test_named_camera_toggles_alone(self):
        h = _handler({"cam": ["front"]})
        h.serve_camera_toggle()
        self.assertEqual(h.sent[0], {"name": "front", "paused": True})
        self.assertTrue(self.cams()["front"]["paused"])
        self.assertFalse(self.cams()["back"]["paused"], "back must be untouched")

    def test_state_survives_a_re_read(self):
        _handler({"cam": ["front"]}).serve_camera_toggle()
        h = _handler({})
        h.serve_cameras()
        by_name = {c["name"]: c for c in h.sent[0]["cameras"]}
        self.assertTrue(by_name["front"]["paused"])
        self.assertFalse(by_name["back"]["paused"])

    def test_bare_toggle_flips_everything(self):
        """Why the UI never calls it without ?cam: it would resume the paused."""
        self.cams()["front"]["paused"] = True
        _handler({}).serve_camera_toggle()
        self.assertFalse(self.cams()["front"]["paused"])
        self.assertTrue(self.cams()["back"]["paused"])

    def test_unknown_camera_is_rejected(self):
        h = _handler({"cam": ["ghost"]})
        h.serve_camera_toggle()
        self.assertIn("error", h.sent[0])


if __name__ == "__main__":
    unittest.main()
