"""V4-M6: adding a camera must not clobber the ones already configured.

The old /api/camera/source wrote a singular `camera:` key, so every "add"
replaced the whole rig. This pins the multi-camera list, the name collision,
and the removal path.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

from intelligence_os import config, web                # noqa: E402


class TestCameraConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._real = config.APP_CONFIG_PATH
        config.APP_CONFIG_PATH = Path(self.tmp.name) / "config.yaml"

    def tearDown(self):
        config.APP_CONFIG_PATH = self._real
        self.tmp.cleanup()

    def _post(self, body):
        h = object.__new__(web.RequestHandler)
        h.sent, h.errors = [], []
        h.send_json = h.sent.append
        h.send_error = lambda code, msg=None: h.errors.append((code, msg))
        h._read_json = lambda: body
        h.serve_camera_source()
        return h

    def test_second_camera_is_added_not_substituted(self):
        self._post({"name": "gate", "source": "rtsp://a"})
        h = self._post({"name": "yard", "source": "0"})
        names = [c["name"] for c in h.sent[0]["cameras"]]
        self.assertEqual(names, ["gate", "yard"])
        # and it survives a reload from disk
        self.assertEqual([c["name"] for c in config.resolve_cameras()], names)
        # "0" is a device index, not a filename
        self.assertEqual(config.resolve_cameras()[1]["source"], 0)

    def test_duplicate_name_repoints_instead_of_duplicating(self):
        """A re-used name is an edit, not an error: the handler repoints that
        camera's source rather than rejecting it or adding a second entry."""
        self._post({"name": "gate", "source": "rtsp://a"})
        h = self._post({"name": "gate", "source": "rtsp://b"})
        self.assertEqual(h.errors, [])
        cams = config.resolve_cameras()
        self.assertEqual(len(cams), 1)
        self.assertEqual(cams[0]["source"], "rtsp://b")

    def test_remove_leaves_the_others(self):
        self._post({"name": "gate", "source": "rtsp://a"})
        self._post({"name": "yard", "source": "rtsp://b"})
        self._post({"remove": "gate"})
        self.assertEqual([c["name"] for c in config.resolve_cameras()], ["yard"])

    def test_settings_persist_to_config_yaml(self):
        config.update_app_config(retention_days=30, face_matching=True)
        self.assertEqual(config.CONFIG.raw_retention_days, 30)
        self.assertTrue(config.CONFIG.identity.enabled)
        config.CONFIG.raw_retention_days, config.CONFIG.identity.enabled = 7, False
        config.apply_app_config()                     # as a restart would
        self.assertEqual(config.CONFIG.raw_retention_days, 30)
        self.assertTrue(config.CONFIG.identity.enabled)


if __name__ == "__main__":
    unittest.main()
