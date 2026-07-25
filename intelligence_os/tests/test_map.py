"""V4-M7 /api/map: zones per camera, one marker per entity at its last zone.

The map has no geometry of its own — everything it draws comes from this
payload, so this pins placement, not pixels.
"""
from __future__ import annotations

import tempfile
import time
import unittest
from unittest import mock

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

from intelligence_os import web                            # noqa: E402
from intelligence_os.store import Store                    # noqa: E402


class TestMap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(db_path=self.tmp.name + "/m.db")
        self.close = self.store.close
        self.store.close = lambda: None   # the handler closes what it opens
        self.now = time.time()

        self.dock = self.store.upsert_location(
            "north dock", {"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]},
            camera_id="Cam A")
        self.store.upsert_location(
            "parking", {"polygon": [[10, 10], [30, 10], [30, 30]]}, camera_id="Cam B")

        self.p = self.store.create_entity("person", label=None)
        self.car = self.store.create_entity("object", label="car")
        self.store.add_observation(self.p, "entered", location_id=self.dock,
                                   camera_id="Cam A", timestamp=self.now - 60)
        self.store.add_observation(self.car, "seen", camera_id="Cam B",
                                   timestamp=self.now - 30)

        self.h = object.__new__(web.RequestHandler)
        self.h.path = "/api/map"
        self.sent = {}
        self.h.send_json = lambda d: self.sent.update(d)
        self.h.send_error = lambda *a: self.sent.update({"error": a})
        self._patch = mock.patch.object(web, "Store", lambda *a, **k: self.store)
        self._patch.start()
        self._cams = mock.patch("intelligence_os.config.resolve_cameras",
                                lambda cfg=None: [{"name": "Cam A", "source": 0},
                                                  {"name": "Cam B", "source": 1}])
        self._cams.start()
        web.pipeline_state["cameras"] = {}

    def tearDown(self):
        self._patch.stop()
        self._cams.stop()
        self.close()
        self.tmp.cleanup()

    def get(self, qs=""):
        self.h.path = "/api/map" + qs
        self.sent.clear()
        self.h.serve_map()
        return self.sent

    def test_zones_land_on_their_own_camera_with_a_centroid(self):
        d = self.get()
        cams = {c["name"]: c for c in d["cameras"]}
        self.assertEqual([z["name"] for z in cams["Cam A"]["zones"]], ["north dock"])
        self.assertEqual([z["name"] for z in cams["Cam B"]["zones"]], ["parking"])
        z = cams["Cam A"]["zones"][0]
        self.assertEqual((z["cx"], z["cy"]), (50, 50))

    def test_one_marker_per_entity_kinded_and_placed(self):
        d = self.get()
        by = {m["entity_id"]: m for m in d["markers"]}
        self.assertEqual(len(d["markers"]), 2)
        self.assertEqual(by[self.p]["kind"], "person")
        self.assertEqual(by[self.p]["zone_name"], "north dock")
        # a car is an object in the store but a vehicle on the map
        self.assertEqual(by[self.car]["kind"], "vehicle")
        self.assertIsNone(by[self.car]["zone_name"])
        self.assertEqual(by[self.car]["camera"], "Cam B")

    def test_only_the_latest_sighting_places_an_entity(self):
        self.store.add_observation(self.p, "seen", camera_id="Cam B",
                                   timestamp=self.now - 10)
        marks = [m for m in self.get()["markers"] if m["entity_id"] == self.p]
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0]["camera"], "Cam B")
        self.assertIsNone(marks[0]["zone"])

    def test_the_window_is_what_hours_says(self):
        self.assertEqual(len(self.get("?hours=1")["markers"]), 2)
        for o in self.store.observations():
            self.store.conn.execute(
                "UPDATE observations SET timestamp=? WHERE observation_id=?",
                (self.now - 40 * 3600, o["observation_id"]))
        self.store.conn.commit()
        self.assertEqual(self.get("?hours=24")["markers"], [])
        self.assertEqual(len(self.get("?hours=168")["markers"]), 2)

    def test_an_unresolved_rule_adds_an_alert_marker_the_entity_does_not(self):
        self.store.add_observation(self.p, "rule_fired:after_hours",
                                   location_id=self.dock, camera_id="Cam A",
                                   origin="rule", timestamp=self.now - 5)
        d = self.get()
        alerts = [m for m in d["markers"] if m["kind"] == "alert"]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["label"], "after hours")
        self.assertEqual(alerts[0]["zone_name"], "north dock")
        # still exactly one person marker alongside it
        self.assertEqual(sum(1 for m in d["markers"] if m["kind"] == "person"), 1)

    def test_a_resolved_alert_leaves_the_map(self):
        oid = self.store.add_observation(self.p, "rule_fired:after_hours",
                                         camera_id="Cam A", origin="rule",
                                         timestamp=self.now - 5)
        self.store.set_alert_status(oid, "resolved")
        self.assertEqual([m for m in self.get()["markers"] if m["kind"] == "alert"], [])

    def test_legacy_rows_belong_to_the_only_camera_there_is(self):
        """camera_id is NULL on everything recorded before multi-camera."""
        self._cams.stop()
        self._cams = mock.patch("intelligence_os.config.resolve_cameras",
                                lambda cfg=None: [{"name": "Only Cam", "source": 0}])
        self._cams.start()
        self.store.add_observation(self.p, "seen", timestamp=self.now - 1)
        m = [m for m in self.get()["markers"] if m["entity_id"] == self.p][0]
        self.assertEqual(m["camera"], "Only Cam")


if __name__ == "__main__":
    unittest.main()
