"""Verification tests for Intelligence OS delivery engine (FinalPRD §9.5 — M9).

Uses mock smtplib and urllib.request layers to test scheduled briefings,
real-time alerts, inline attachments, and rate window scheduling correctness.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from intelligence_os.deliver import (
    deliver_digest,
    deliver_realtime_event,
    run_scheduler_tick,
    send_email,
    send_webhook,
)
from intelligence_os.rules import FiredEvent
from intelligence_os.store import Store


class TestDelivery(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_memory.db"
        self.store = Store(db_path=self.db_path)

        # Create a test user in DB
        with self.store.tx() as c:
            c.execute(
                "INSERT INTO users (user_id, username, password_hash, created_at, "
                "delivery_schedule, delivery_sink, delivery_destination, email) "
                "VALUES ('usr_test', 'test_user', 'pbkdf2:dummy', 12345.0, "
                "'daily', 'email', 'test@example.com', 'test@example.com')"
            )

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    @patch("urllib.request.urlopen")
    def test_send_webhook(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response

        payload = {"hello": "world"}
        success = send_webhook("http://example.com/webhook", payload)
        self.assertTrue(success)

        # Verify urllib.request.urlopen call argument
        args, kwargs = mock_urlopen.call_args
        req = args[0]
        self.assertEqual(req.full_url, "http://example.com/webhook")
        self.assertEqual(req.method, "POST")
        self.assertEqual(req.headers.get("Content-type"), "application/json")
        self.assertEqual(json.loads(req.data.decode("utf-8")), payload)

    @patch("smtplib.SMTP")
    def test_send_email_inline_attachments(self, mock_smtp):
        # Create a temp keyframe file inside the workspace
        kf_path = str(Path(__file__).parent / "temp_kf_test.jpg")
        with open(kf_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nfake png image bytes")

        try:
            # We mock the SMTP connection details
            smtp_instance = MagicMock()
            mock_smtp.return_value = smtp_instance

            # We change standard configurations in deliver module to force real email generation
            with patch("intelligence_os.deliver.SMTP_HOST", "smtp.test.com"):
                success = send_email(
                    to_email="user@test.com",
                    subject="Test Digest",
                    html_body="<p>Test digest with keyframe <img src='cid:test.jpg'></p>",
                    text_body="Test digest",
                    keyframe_paths=[kf_path]
                )
                self.assertTrue(success)

            # Check SMTP call logs
            self.assertTrue(smtp_instance.sendmail.called)
            from_addr, to_addrs, msg_str = smtp_instance.sendmail.call_args[0]
            self.assertEqual(to_addrs, ["user@test.com"])

            # Message should contain the inline CID header and the content-type boundary
            self.assertIn("Content-ID: <", msg_str)
            self.assertIn("Content-Disposition: inline", msg_str)
        finally:
            # Cleanup temp file
            if Path(kf_path).exists():
                Path(kf_path).unlink()

    @patch("intelligence_os.deliver.send_email")
    def test_deliver_digest_routing(self, mock_send_email):
        user = {
            "username": "test_user",
            "delivery_schedule": "daily",
            "delivery_sink": "email",
            "delivery_destination": "test@example.com",
            "email": "test@example.com"
        }
        # Verify deliver_digest routes to send_email
        mock_send_email.return_value = True
        success = deliver_digest(self.store, user, since=0, now=1000)
        self.assertTrue(success)
        self.assertTrue(mock_send_email.called)

    @patch("intelligence_os.deliver.send_email")
    def test_deliver_realtime_alert(self, mock_send_email):
        # Update user to realtime schedule
        with self.store.tx() as c:
            c.execute("UPDATE users SET delivery_schedule = 'realtime', delivery_sink = 'email'")

        event = FiredEvent(
            rule="test_rule",
            entity_id="ent_person_123",
            location_id="dock_cam",
            observation_id="obs_123",
            keyframe="",
            timestamp=9999.0
        )

        deliver_realtime_event(self.store, event)
        self.assertTrue(mock_send_email.called)

        # Verify call args
        args, kwargs = mock_send_email.call_args
        to_email, subject, html_body, text_body, kf = args
        self.assertEqual(to_email, "test@example.com")
        self.assertIn("ALERT", subject)
        self.assertIn("test_rule", subject)

    @patch("intelligence_os.deliver.deliver_digest")
    def test_scheduler_loop_ticks(self, mock_deliver_digest):
        mock_deliver_digest.return_value = True

        # 1. Test scheduler tick when time is not 08:00 AM local
        # mock local hour to 12:00 PM (tm_hour=12)
        local_time_mock = MagicMock()
        local_time_mock.tm_hour = 12
        with patch("time.localtime", return_value=local_time_mock):
            run_scheduler_tick(self.store, now=100000.0)
            self.assertFalse(mock_deliver_digest.called)

        # 2. Test scheduler tick when local hour is 08:00 AM local
        local_time_mock.tm_hour = 8
        with patch("time.localtime", return_value=local_time_mock):
            run_scheduler_tick(self.store, now=100000.0)
            self.assertTrue(mock_deliver_digest.called)

            # Check user table has last_delivered_at updated to now
            user = self.store.conn.execute("SELECT * FROM users").fetchone()
            self.assertEqual(user["last_delivered_at"], 100000.0)

        # 3. Test scheduler tick does not double-fire inside the same hour window
        mock_deliver_digest.reset_mock()
        with patch("time.localtime", return_value=local_time_mock):
            run_scheduler_tick(self.store, now=100060.0)  # 60s later, still tm_hour=8
            self.assertFalse(mock_deliver_digest.called)  # Should NOT fire (cooldown prevents it)
