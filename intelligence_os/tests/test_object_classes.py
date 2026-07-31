"""Animals and vehicles as first-class detections, and what may be said about them.

Three things are under test, and they are separable on purpose:
  1. the vocabulary itself (which class is what kind, what a config file may ask for)
  2. what repeated proximity is allowed to become per kind (the distiller's verb)
  3. that nothing animate is re-identified across restarts by colour alone
"""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

from intelligence_os import config as C                # noqa: E402
from intelligence_os.detect import Detection, ObjectRegistry   # noqa: E402
from intelligence_os.distill import Distiller          # noqa: E402
from intelligence_os.store import Store                # noqa: E402

NOW = time.time()


class TestVocabulary(unittest.TestCase):
    def test_kinds(self):
        for cls, kind in [("dog", "animal"), ("bird", "animal"), ("car", "vehicle"),
                          ("truck", "vehicle"), ("chair", "object"),
                          ("person", "person")]:
            self.assertEqual(C.kind_for_class(cls), kind, cls)

    def test_hot_dog_is_lunch(self):
        """A substring match on 'dog' would classify food as livestock."""
        self.assertEqual(C.kind_for_class("hot dog"), "object")
        self.assertEqual(C.kind_for_class("teddy bear"), "object")

    def test_every_coco_class_has_a_kind(self):
        self.assertEqual(len(C.COCO_CLASSES), 80)
        self.assertTrue(all(C.kind_for_class(c) in
                            {"person", "animal", "vehicle", "object"}
                            for c in C.COCO_CLASSES))

    def test_unknown_classes_are_reported_not_dropped(self):
        kept, rejected = C.normalize_object_classes(["dog", "dogs", "CAR", " cat "])
        self.assertEqual(kept, ["dog", "car", "cat"], "trimmed and lowercased")
        self.assertEqual(rejected, ["dogs"],
                         "a typo can never be detected — the caller must be told")

    def test_duplicates_collapse(self):
        kept, _ = C.normalize_object_classes(["dog", "dog", "Dog"])
        self.assertEqual(kept, ["dog"])


class TestConfigReachable(unittest.TestCase):
    """The list used to live only in the dataclass, so enabling a class meant
    editing source."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.yaml"
        self._real_path, C.APP_CONFIG_PATH = C.APP_CONFIG_PATH, self.path
        self._real_classes = list(C.CONFIG.detect.object_classes)

    def tearDown(self):
        C.APP_CONFIG_PATH = self._real_path
        C.CONFIG.detect.object_classes = self._real_classes
        self.tmp.cleanup()

    def test_config_yaml_overrides_the_default_list(self):
        self.path.write_text("object_classes: [dog, car, chair]\n")
        C.apply_app_config()
        self.assertEqual(C.CONFIG.detect.object_classes, ["dog", "car", "chair"])

    def test_unknown_class_is_skipped_and_the_rest_still_apply(self):
        self.path.write_text("object_classes: [dog, wombat]\n")
        C.apply_app_config()
        self.assertEqual(C.CONFIG.detect.object_classes, ["dog"])

    def test_person_is_not_a_togglable_class(self):
        """detect.py always adds 'person'; listing it would imply it could be removed."""
        self.path.write_text("object_classes: [person, dog]\n")
        C.apply_app_config()
        self.assertEqual(C.CONFIG.detect.object_classes, ["dog"])

    def test_an_empty_list_keeps_the_defaults(self):
        self.path.write_text("object_classes: []\n")
        C.apply_app_config()
        self.assertEqual(C.CONFIG.detect.object_classes, self._real_classes,
                         "an empty list is a mistake, not a request to see nothing")


class TestProximityVerb(unittest.TestCase):
    """§7: the distiller interprets, but only as far as the evidence reaches."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(db_path=Path(self.tmp.name) / "t.db")
        self.person = self.store.create_entity("person", label="Regular")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _mine(self, label):
        """Three 'near' rows against a thing labelled `label` -> the mined verb."""
        obj = self.store.create_entity("object", label=label)
        for i in range(3):
            self.store.add_observation(self.person, "near", object_entity_id=obj,
                                       origin="detector", timestamp=NOW - i * 60)
        Distiller(self.store).mine_relations()
        rows = [r for r in self.store.relations() if r["object_entity_id"] == obj]
        self.assertTrue(rows, f"no relation mined for {label}")
        return rows[0]["predicate"]

    def test_a_laptop_is_used(self):
        self.assertEqual(self._mine("laptop"), "uses")

    def test_a_dog_is_company_not_equipment(self):
        """'Person 03 uses Dog 07' reads as a reasoning failure even though it is
        only a wording one."""
        self.assertEqual(self._mine("dog"), "accompanied_by")

    def test_a_van_is_used(self):
        self.assertEqual(self._mine("truck"), "uses")

    def test_a_renamed_entity_falls_back_to_the_weaker_claim(self):
        """The class survives as the label; once a human renames it the kind is
        gone, and under-claiming is the safe direction."""
        self.assertEqual(self._mine("Rex"), "uses")


class TestAnimateAreNotReidentifiedByColour(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(db_path=Path(self.tmp.name) / "t.db")
        self.reg = ObjectRegistry(self.store)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_two_identically_coloured_dogs_stay_two_entities(self):
        """The appearance vector is an HS histogram. Two black dogs share one.
        Detector hands animate classes appearance=None so no merge can happen."""
        a = self.reg.resolve(Detection("dog", (0, 0, 10, 10), 0.9, track_id=1,
                                       appearance=None))
        b = self.reg.resolve(Detection("dog", (0, 0, 10, 10), 0.9, track_id=2,
                                       appearance=None))
        self.assertNotEqual(a, b, "merging them would assert an identity nobody saw")

    def test_the_same_track_is_still_one_entity_within_a_stream(self):
        d = Detection("dog", (0, 0, 10, 10), 0.9, track_id=7, appearance=None)
        self.assertEqual(self.reg.resolve(d), self.reg.resolve(d))

    def test_a_static_object_still_re_matches_across_restarts(self):
        """The behaviour animals lose must survive for the things it was built for."""
        import numpy as np
        sig = np.zeros(50, dtype=np.float32)
        sig[0] = 1.0
        first = self.reg.resolve(Detection("chair", (0, 0, 10, 10), 0.9,
                                           track_id=1, appearance=sig))
        fresh = ObjectRegistry(self.store)          # simulates a restart
        again = fresh.resolve(Detection("chair", (0, 0, 10, 10), 0.9,
                                        track_id=99, appearance=sig))
        self.assertEqual(first, again)


if __name__ == "__main__":
    unittest.main()
