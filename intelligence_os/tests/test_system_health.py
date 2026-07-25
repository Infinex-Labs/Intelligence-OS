"""V4-M1: the rail-footer numbers (FR-SH-8) come from real system state."""
from __future__ import annotations

import time
import unittest

from intelligence_os.web import RequestHandler


class TestSystemHealth(unittest.TestCase):
    def test_counts_only_recently_seen_cameras_as_online(self):
        now = time.time()
        h = RequestHandler._system_health({
            "live": {"last_frame_ts": now},
            "stale": {"last_frame_ts": now - 500},   # >10s = offline
            "never": {},
        }, now_ts=now)
        self.assertEqual((h["cameras_online"], h["cameras_total"]), (1, 3))

    def test_resource_fractions_are_0_to_1(self):
        h = RequestHandler._system_health({}, now_ts=time.time())
        for k in ("cpu_pct", "memory_pct"):
            self.assertIsNotNone(h[k], k)
            self.assertTrue(0.0 <= h[k] <= 1.0, f"{k}={h[k]}")
        self.assertTrue(0.0 < h["storage"]["pct"] < 1.0)
        self.assertEqual(h["cameras_total"], 0)


if __name__ == "__main__":
    unittest.main()
