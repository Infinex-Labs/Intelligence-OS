"""M6 tests: multi-camera config, schema, rules camera filter, observer camera_id.

Two synthetic sources feed two cameras. Observations carry their own camera_id.
A rule bound to cam_a never fires on cam_b's detections. resolve_cameras()
backward compat is verified.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Step 1: resolve_cameras backward compat
# ---------------------------------------------------------------------------
class TestResolveCameras(unittest.TestCase):
    def test_cameras_list(self):
        from intelligence_os.config import resolve_cameras
        cfg = {"cameras": [
            {"name": "front", "source": "rtsp://a"},
            {"name": "back", "source": 1},
        ]}
        cams = resolve_cameras(cfg)
        self.assertEqual(len(cams), 2)
        self.assertEqual(cams[0], {"name": "front", "source": "rtsp://a"})
        self.assertEqual(cams[1], {"name": "back", "source": 1})

    def test_single_camera_sugar(self):
        from intelligence_os.config import resolve_cameras
        cams = resolve_cameras({"camera": 0})
        self.assertEqual(len(cams), 1)
        self.assertEqual(cams[0], {"name": "default", "source": 0})

    def test_single_camera_string(self):
        from intelligence_os.config import resolve_cameras
        cams = resolve_cameras({"camera": "/path/to/vid.mp4"})
        self.assertEqual(cams[0], {"name": "default", "source": "/path/to/vid.mp4"})

    def test_empty_config(self):
        from intelligence_os.config import resolve_cameras
        cams = resolve_cameras({})
        self.assertEqual(len(cams), 1)
        self.assertEqual(cams[0], {"name": "default", "source": 0})

    def test_cameras_overrides_camera(self):
        from intelligence_os.config import resolve_cameras
        cfg = {"camera": 99, "cameras": [{"name": "a", "source": 0}]}
        cams = resolve_cameras(cfg)
        self.assertEqual(len(cams), 1)
        self.assertEqual(cams[0]["name"], "a")

    def test_duplicate_names_raises(self):
        from intelligence_os.config import resolve_cameras
        cfg = {"cameras": [
            {"name": "dup", "source": 0},
            {"name": "dup", "source": 1},
        ]}
        with self.assertRaises(ValueError):
            resolve_cameras(cfg)

    def test_auto_names(self):
        from intelligence_os.config import resolve_cameras
        cfg = {"cameras": [0, "rtsp://b"]}
        cams = resolve_cameras(cfg)
        self.assertEqual(cams[0]["name"], "cam_0")
        self.assertEqual(cams[1]["name"], "cam_1")

    def test_bool_source_normalised(self):
        from intelligence_os.config import resolve_cameras
        cams = resolve_cameras({"camera": True})
        self.assertEqual(cams[0]["source"], 0)


# ---------------------------------------------------------------------------
# Step 2: schema — camera_id columns
# ---------------------------------------------------------------------------
class TestSchemaM6(unittest.TestCase):
    def test_observation_has_camera_id(self):
        from intelligence_os.store import Store
        with tempfile.TemporaryDirectory() as d:
            s = Store(Path(d) / "test.db")
            eid = s.create_entity("person", "test")
            oid = s.add_observation(eid, "present", camera_id="cam_a")
            obs = s.observations()
            row = [o for o in obs if o["observation_id"] == oid][0]
            self.assertEqual(row["camera_id"], "cam_a")
            s.close()

    def test_observation_camera_id_null_by_default(self):
        from intelligence_os.store import Store
        with tempfile.TemporaryDirectory() as d:
            s = Store(Path(d) / "test.db")
            eid = s.create_entity("person", "test")
            oid = s.add_observation(eid, "present")
            obs = s.observations()
            row = [o for o in obs if o["observation_id"] == oid][0]
            self.assertIsNone(row["camera_id"])
            s.close()

    def test_observations_filter_by_camera(self):
        from intelligence_os.store import Store
        with tempfile.TemporaryDirectory() as d:
            s = Store(Path(d) / "test.db")
            eid = s.create_entity("person", "test")
            s.add_observation(eid, "present", camera_id="cam_a")
            s.add_observation(eid, "present", camera_id="cam_b")
            s.add_observation(eid, "present")  # no camera_id
            self.assertEqual(len(s.observations(camera_id="cam_a")), 1)
            self.assertEqual(len(s.observations(camera_id="cam_b")), 1)
            self.assertEqual(len(s.observations()), 3)
            s.close()

    def test_location_camera_id(self):
        from intelligence_os.store import Store
        with tempfile.TemporaryDirectory() as d:
            s = Store(Path(d) / "test.db")
            lid = s.upsert_location("door", {"polygon": []}, camera_id="cam_a")
            locs = s.locations(camera_id="cam_a")
            self.assertEqual(len(locs), 1)
            self.assertEqual(locs[0]["camera_id"], "cam_a")
            # locations(None) returns all
            self.assertEqual(len(s.locations()), 1)
            s.close()

    def test_wal_mode(self):
        from intelligence_os.store import Store
        with tempfile.TemporaryDirectory() as d:
            s = Store(Path(d) / "test.db")
            mode = s.conn.execute("PRAGMA journal_mode").fetchone()[0]
            self.assertEqual(mode.lower(), "wal")
            s.close()

    def test_migration_idempotent(self):
        """Calling _migrate_m6 twice doesn't crash."""
        from intelligence_os.store import Store
        with tempfile.TemporaryDirectory() as d:
            s = Store(Path(d) / "test.db")
            s._migrate_m6()  # second call
            s._migrate_m6()  # third call
            s.close()


# ---------------------------------------------------------------------------
# Step 3: observer — camera_id on ResolvedDetection flows to observations
# ---------------------------------------------------------------------------
class TestObserverCameraId(unittest.TestCase):
    def test_camera_id_flows(self):
        from intelligence_os.store import Store
        from intelligence_os.observe import Observer, ResolvedDetection
        from intelligence_os.detect import Detection
        with tempfile.TemporaryDirectory() as d:
            s = Store(Path(d) / "test.db")
            obs = Observer(s)
            eid = s.create_entity("person", "test")
            det = Detection(cls_name="person", bbox=(10, 10, 50, 50),
                            conf=0.9, track_id=1)
            rd = ResolvedDetection(eid, det, location_id=None, camera_id="cam_front")
            ids = obs.observe_frame([rd], 1000.0)
            self.assertTrue(len(ids) >= 1)
            row = s.observations(camera_id="cam_front")
            self.assertTrue(len(row) >= 1)
            self.assertEqual(row[0]["camera_id"], "cam_front")
            s.close()


# ---------------------------------------------------------------------------
# Step 4: rules — camera filter
# ---------------------------------------------------------------------------
class TestRulesCameraFilter(unittest.TestCase):
    def _make_rule(self, name="test_rule", camera=None, zone=None):
        return {
            "name": name,
            "source": "test",
            "trigger": {"class": "person", "zone": zone, "camera": camera},
            "gate": {"dwell_seconds": 0},
            "verify": None,
            "cooldown": {"per_track_seconds": 0},
            "needs_identity": False,
            "enabled": True,
        }

    def test_rule_camera_filter_blocks(self):
        from intelligence_os.store import Store
        from intelligence_os.rules import RuleEngine
        from intelligence_os.observe import ResolvedDetection
        from intelligence_os.detect import Detection
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            s = Store(Path(d) / "test.db")
            rule = self._make_rule(camera="cam_a")
            eng = RuleEngine(s, rules=[rule], frames_dir=Path(d))

            det = Detection(cls_name="person", bbox=(10, 10, 50, 50),
                            conf=0.9, track_id=1)
            # cam_b detection should NOT fire the cam_a rule
            rd_b = ResolvedDetection("ent_test123456", det, camera_id="cam_b")
            fired = eng.feed([rd_b], np.zeros((100, 100, 3), dtype=np.uint8), 1000.0)
            self.assertEqual(len(fired), 0)

            # cam_a detection SHOULD fire
            rd_a = ResolvedDetection("ent_test123456", det, camera_id="cam_a")
            fired = eng.feed([rd_a], np.zeros((100, 100, 3), dtype=np.uint8), 1001.0)
            self.assertEqual(len(fired), 1)
            self.assertEqual(fired[0].rule, "test_rule")
            s.close()

    def test_rule_no_camera_matches_all(self):
        from intelligence_os.store import Store
        from intelligence_os.rules import RuleEngine
        from intelligence_os.observe import ResolvedDetection
        from intelligence_os.detect import Detection
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            s = Store(Path(d) / "test.db")
            rule = self._make_rule(camera=None)  # no camera filter
            eng = RuleEngine(s, rules=[rule], frames_dir=Path(d))

            det = Detection(cls_name="person", bbox=(10, 10, 50, 50),
                            conf=0.9, track_id=1)
            rd = ResolvedDetection("ent_test123456", det, camera_id="cam_any")
            fired = eng.feed([rd], np.zeros((100, 100, 3), dtype=np.uint8), 1000.0)
            self.assertEqual(len(fired), 1)
            s.close()

    def test_apply_gates_camera_field(self):
        """The compiler output includes trigger.camera."""
        from intelligence_os.rules import _apply_gates
        out = {
            "feasible": True,
            "targets_individual": False,
            "name": "test",
            "trigger_class": "person",
            "trigger_zone": "",
            "trigger_camera": "front_door",
            "needs_verify": False,
            "dwell_seconds": 0,
            "cooldown_seconds": 300,
        }
        spec = _apply_gates(out, "test rule")
        self.assertEqual(spec["trigger"]["camera"], "front_door")

    def test_apply_gates_empty_camera(self):
        from intelligence_os.rules import _apply_gates
        out = {
            "feasible": True,
            "targets_individual": False,
            "name": "test",
            "trigger_class": "person",
            "trigger_zone": "",
            "trigger_camera": "",
            "needs_verify": False,
        }
        spec = _apply_gates(out, "test rule")
        self.assertIsNone(spec["trigger"]["camera"])


if __name__ == "__main__":
    unittest.main()
