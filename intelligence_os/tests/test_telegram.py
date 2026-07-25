"""Telegram sink: the alert and its keyframe travel as one message.

urlopen is stubbed — this pins the request we build, not Telegram's uptime.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from intelligence_os.tests import _stubs   # noqa: F401  (stubs the deps we lack)

from intelligence_os import deliver                      # noqa: E402


class _Resp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class TestTelegram(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kf = Path(self.tmp.name) / "kf.jpg"
        self.kf.write_bytes(b"\xff\xd8jpegbytes\xff\xd9")
        self._env = os.environ.get("TELEGRAM_BOT_TOKEN")
        os.environ["TELEGRAM_BOT_TOKEN"] = "42:TOKEN"

    def tearDown(self):
        if self._env is None:
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        else:
            os.environ["TELEGRAM_BOT_TOKEN"] = self._env
        self.tmp.cleanup()

    def _send(self, *a, **k):
        with mock.patch.object(urllib.request, "urlopen",
                               return_value=_Resp()) as u:
            ok = deliver.send_telegram(*a, **k)
        return ok, u.call_args[0][0]

    def test_no_keyframe_is_a_plain_message(self):
        ok, req = self._send("999", "something happened")
        self.assertTrue(ok)
        self.assertTrue(req.full_url.endswith("/bot42:TOKEN/sendMessage"))
        self.assertEqual(json.loads(req.data)["chat_id"], "999")

    def test_keyframe_rides_along_as_the_photo_caption(self):
        ok, req = self._send("999", "rule fired", str(self.kf))
        self.assertTrue(ok)
        self.assertTrue(req.full_url.endswith("/sendPhoto"))
        self.assertIn("multipart/form-data; boundary=",
                      req.headers["Content-type"])
        self.assertIn(b"jpegbytes", req.data)
        self.assertIn(b'name="caption"', req.data)

    def test_missing_keyframe_degrades_to_text_rather_than_failing(self):
        ok, req = self._send("999", "rule fired", "/nope/gone.jpg")
        self.assertTrue(ok)
        self.assertTrue(req.full_url.endswith("/sendMessage"))

    def test_no_token_is_a_refusal_not_a_crash(self):
        del os.environ["TELEGRAM_BOT_TOKEN"]
        with mock.patch.object(urllib.request, "urlopen") as u:
            self.assertFalse(deliver.send_telegram("999", "hi"))
        u.assert_not_called()

    def test_caption_is_truncated_to_telegrams_cap(self):
        _, req = self._send("999", "x" * 5000, str(self.kf))
        self.assertIn(b"x" * 1024, req.data)
        self.assertNotIn(b"x" * 1025, req.data)


if __name__ == "__main__":
    unittest.main()
