"""V4-M3: timeline window, event class and server-side buckets (FR-TL-1..4)."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

from intelligence_os import web                      # noqa: E402
from intelligence_os.store import Store              # noqa: E402

NOW = time.time()
HOUR = 3600.0


class TestTimeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db = Path(self.tmp.name) / "t.db"
        s = Store(db_path=db)
        # An entity created long ago: everything it does is routine.
        old = s.create_entity("person", label="Regular")
        s.conn.execute("UPDATE entities SET created_at=? WHERE entity_id=?",
                       (NOW - 30 * 24 * HOUR, old))
        s.conn.commit()
        s.add_observation(old, "seen_at", origin="detector", camera_id="front",
                          timestamp=NOW - 2 * HOUR, source_ref="/frames/a.jpg")
        s.add_observation(old, "rule_fired:loitering", origin="rule",
                          camera_id="front", timestamp=NOW - HOUR)
        # Created inside the window -> never seen before -> unusual.
        new = s.create_entity("person", label="Stranger")
        s.add_observation(new, "seen_at", origin="detector", camera_id="back",
                          timestamp=NOW - 30 * 60)
        # An object, and one event well outside the 24h window.
        box = s.create_entity("object", label="Crate")
        s.add_observation(box, "seen_at", origin="detector", timestamp=NOW - 10 * 60)
        s.add_observation(old, "seen_at", origin="detector",
                          timestamp=NOW - 40 * HOUR)
        s.close()
        self._real = web.Store
        web.Store = lambda *a, **k: self._real(db_path=db)

    def tearDown(self):
        web.Store = self._real
        self.tmp.cleanup()

    def _get(self, query=""):
        h = object.__new__(web.RequestHandler)
        h.sent = []
        h.send_json = h.sent.append
        h.send_error = lambda *a: h.sent.append({"error": a})
        h.path = "/api/observations" + query
        h.serve_observations()
        return h.sent[0]

    def test_default_window_is_24h_and_excludes_older_events(self):
        r = self._get()
        self.assertEqual(len(r["events"]), 4, "the 40h-old row is outside the window")

    def test_event_class_marks_rules_and_first_ever_sightings(self):
        by = {(e["entity"], e["predicate"]): e["class"] for e in self._get()["events"]}
        self.assertEqual(by[("Regular", "seen_at")], "routine")
        self.assertEqual(by[("Regular", "rule_fired:loitering")], "fired")
        self.assertEqual(by[("Stranger", "seen_at")], "unusual",
                         "an entity first created inside the window is not routine")

    def test_buckets_are_server_side_and_land_in_time_order(self):
        b = self._get("?bins=24")["buckets"]
        self.assertEqual(len(b), 24)
        self.assertEqual(sum(sum(x.values()) for x in b), 4, "every event is bucketed")
        self.assertEqual([i for i, x in enumerate(b) if x["fired"]], [23 - 1],
                         "the fired event sits one hour back")

    def test_type_filter_narrows_events_but_not_the_chip_counts(self):
        r = self._get("?type=alerts")
        self.assertEqual([e["rule"] for e in r["events"]], ["loitering"])
        # chips are one exclusive dimension — filtering must not zero the others
        self.assertEqual((r["counts"]["all"], r["counts"]["person"],
                          r["counts"]["object"]), (4, 3, 1))
        self.assertEqual(len(self._get("?type=person")["events"]), 3)

    def test_dashboard_card_ignores_the_filter_bar(self):
        """Filtering the timeline pane must not narrow the dashboard card."""
        r = self._get("?type=alerts")
        self.assertEqual(len(r["events"]), 1, "the filter really does bite")
        self.assertEqual(len(r["recent"]), 4, "recent is pre-filter")
        self.assertGreater(r["recent"][0]["timestamp"], r["recent"][-1]["timestamp"],
                           "newest first")

    def test_explicit_window_and_camera_filter(self):
        r = self._get(f"?since={NOW - 90 * 60}&until={NOW}&camera=front")
        self.assertEqual([e["camera"] for e in r["events"]], ["front"],
                         "only the fired event is both on front and inside 90m")

    def test_keyframe_is_a_basename_not_a_path(self):
        kf = [e["keyframe"] for e in self._get()["events"] if e["keyframe"]]
        self.assertEqual(kf, ["a.jpg"], "the client joins it onto /keyframe/")


if __name__ == "__main__":
    unittest.main()
