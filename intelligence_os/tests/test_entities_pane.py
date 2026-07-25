"""V4-M5: /api/entities carries the appearance stats the grid renders (FR-EN-1..4)."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

from intelligence_os import web                      # noqa: E402
from intelligence_os.store import Store              # noqa: E402

NOW = time.time()
DAY = 86400.0


class TestEntitiesPayload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db = Path(self.tmp.name) / "t.db"
        s = Store(db_path=db)
        # Seen 10x over 2 days -> 5/day -> high.
        self.hot = s.create_entity("person", label="Regular")
        s.conn.execute("UPDATE entities SET created_at=? WHERE entity_id=?",
                       (NOW - 2 * DAY, self.hot))
        s.conn.commit()
        for i in range(10):
            s.add_observation(self.hot, "seen_at", origin="detector",
                              timestamp=NOW - i * 3600,
                              source_ref=f"/frames/{i}.jpg")
        # Seen once in 10 days -> low.
        self.cold = s.create_entity("person", label="Rare")
        s.conn.execute("UPDATE entities SET created_at=? WHERE entity_id=?",
                       (NOW - 10 * DAY, self.cold))
        s.conn.commit()
        s.add_observation(self.cold, "seen_at", origin="detector",
                          timestamp=NOW - 9 * DAY)
        # Never observed at all.
        self.unseen = s.create_entity("object", label="Crate")
        s.close()
        self._real = web.Store
        web.Store = lambda *a, **k: self._real(db_path=db)

    def tearDown(self):
        web.Store = self._real
        self.tmp.cleanup()

    def _get(self):
        h = object.__new__(web.RequestHandler)
        h.sent = []
        h.send_json = h.sent.append
        h.path = "/api/entities"
        h.serve_entities()
        return {e["id"]: e for e in h.sent[0]}

    def test_appearance_stats_are_derived_not_stored(self):
        e = self._get()[self.hot]
        self.assertEqual(e["times_seen"], 10)
        self.assertAlmostEqual(e["last_seen"], NOW, delta=1)

    def test_frequency_bands_by_sightings_per_day(self):
        r = self._get()
        self.assertEqual(r[self.hot]["frequency"], "high")     # 5/day
        self.assertEqual(r[self.cold]["frequency"], "low")     # 0.1/day

    def test_an_entity_with_no_observations_still_ships(self):
        e = self._get()[self.unseen]
        self.assertEqual((e["times_seen"], e["last_seen"], e["keyframe"]),
                         (0, None, None), "the grid must not drop it")

    def test_keyframe_is_the_newest_basename(self):
        self.assertEqual(self._get()[self.hot]["keyframe"], "0.jpg",
                         "newest observation wins; client joins onto /keyframe/")

    def test_the_profile_ships_the_frame_behind_each_observation(self):
        """Investigators read the picture, not the predicate."""
        h = object.__new__(web.RequestHandler)
        h.sent = []
        h.send_json = h.sent.append
        h.path = f"/api/entity/{self.hot}"
        h.serve_entity(self.hot)
        self.assertEqual(h.sent[0]["recent_observations"][0]["keyframe"], "0.jpg")

    def test_an_unlabelled_entity_is_named_the_same_way_everywhere(self):
        """V4-M9: the profile said 'Unknown' while every list said 'Person abc123',
        so one entity read as two. Both fall back to type + id suffix now."""
        s = web.Store()
        eid = s.create_entity("person")
        s.close()
        h = object.__new__(web.RequestHandler)
        h.sent = []
        h.send_json = h.sent.append
        h.path = f"/api/entity/{eid}"
        h.serve_entity(eid)
        self.assertEqual(h.sent[0]["label"], f"Person {eid[-6:]}")
        self.assertEqual(self._get()[eid]["label"], None,
                         "the list ships null and lets the client name it")


if __name__ == "__main__":
    unittest.main()
