"""Verification tests for Intelligence OS auth subsystem (FinalPRD §9.6 — M7).

Uses an in-memory mock request handler to test web server routing and auth checks
without requiring local TCP socket binding, ensuring compatibility with sandboxed
execution environments.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from intelligence_os.auth import (
    authenticate_user,
    count_users,
    create_session,
    create_user,
    delete_session,
    hash_password,
    verify_password,
    verify_session,
)
from intelligence_os.store import Store
from intelligence_os.web import RequestHandler


class TestAuthHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_auth.db"
        self.store = Store(self.db_path)

    def tearDown(self):
        self.store.close()
        self.tmp_dir.cleanup()

    def test_password_hash_and_verify(self):
        pwd = "secure_password_123"
        h = hash_password(pwd)
        self.assertTrue(verify_password(h, pwd))
        self.assertFalse(verify_password(h, "wrong_pwd"))
        self.assertFalse(verify_password("invalid_hash", pwd))

    def test_user_lifecycle(self):
        self.assertEqual(count_users(self.store), 0)
        uid = create_user(self.store, "admin", "admin_pwd_123")
        self.assertEqual(count_users(self.store), 1)

        # Authenticate successfully
        self.assertEqual(authenticate_user(self.store, "admin", "admin_pwd_123"), uid)

        # Authenticate fail (wrong pwd)
        self.assertIsNone(authenticate_user(self.store, "admin", "wrong_pwd"))

        # Authenticate fail (unknown user)
        self.assertIsNone(authenticate_user(self.store, "non_existent", "admin_pwd_123"))

    def test_session_lifecycle(self):
        uid = create_user(self.store, "user1", "some_password")
        sid = create_session(self.store, uid, duration_seconds=10)

        # Valid session
        self.assertEqual(verify_session(self.store, sid), uid)

        # Invalidate / Delete session
        delete_session(self.store, sid)
        self.assertIsNone(verify_session(self.store, sid))

    def test_session_expiration(self):
        uid = create_user(self.store, "user2", "some_password")
        # Expired session (-5 seconds remaining)
        sid = create_session(self.store, uid, duration_seconds=-5)
        self.assertIsNone(verify_session(self.store, sid))


class TestAuthWebServer(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "web_test.db"

        # Point INTELLIGENCE_OS_DB to our test database
        self.orig_db = os.environ.get("INTELLIGENCE_OS_DB")
        os.environ["INTELLIGENCE_OS_DB"] = str(self.db_path)

        # Initialize Store
        s = Store(self.db_path)
        s.close()

    def tearDown(self):
        self.tmp_dir.cleanup()
        if self.orig_db:
            os.environ["INTELLIGENCE_OS_DB"] = self.orig_db
        else:
            os.environ.pop("INTELLIGENCE_OS_DB", None)

    def make_handler(self, path: str, method: str = "GET", headers: dict | None = None,
                     body: dict | None = None) -> RequestHandler:
        headers = headers or {}
        body_bytes = json.dumps(body).encode("utf-8") if body else b""
        if body:
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body_bytes))

        # Subclass RequestHandler to bypass BaseHTTPRequestHandler init socket binding
        class TestHandler(RequestHandler):
            def __init__(self):
                pass

        handler = TestHandler()
        handler.request = MagicMock()
        handler.client_address = ("127.0.0.1", 12345)
        handler.server = MagicMock()
        handler.path = path
        handler.command = method
        handler.rfile = io.BytesIO(body_bytes)
        handler.wfile = io.BytesIO()

        # Capture response outputs
        handler.response_code = None
        handler.response_headers = {}

        def send_response(code, message=None):
            handler.response_code = code

        def send_header(keyword, value):
            handler.response_headers[keyword] = value

        def end_headers():
            pass

        handler.send_response = send_response
        handler.send_header = send_header
        handler.end_headers = end_headers

        # Mock headers.get
        handler.headers = MagicMock()
        handler.headers.get.side_effect = lambda k, default=None: headers.get(k, default)

        return handler

    def test_web_auth_lifecycle(self):
        # 1. Unauthenticated API returns 401
        h = self.make_handler("/api/rules", "GET")
        h.do_GET()
        self.assertEqual(h.response_code, 401)
        res = json.loads(h.wfile.getvalue().decode("utf-8"))
        self.assertIn("error", res)

        # 2. Check auth status on first run
        h = self.make_handler("/api/auth/status", "GET")
        h.do_GET()
        self.assertEqual(h.response_code, 200)
        data = json.loads(h.wfile.getvalue().decode("utf-8"))
        self.assertTrue(data["first_run"])
        self.assertFalse(data["authenticated"])

        # 3. Block login attempt on first run
        h = self.make_handler("/api/auth/login", "POST", body={"username": "admin", "password": "password123"})
        h.do_POST()
        self.assertEqual(h.response_code, 400)
        res = json.loads(h.wfile.getvalue().decode("utf-8"))
        self.assertIn("Onboarding required", res["error"])

        # 4. Register first admin user
        h = self.make_handler("/api/auth/register", "POST", body={"username": "admin", "password": "password123"})
        h.do_POST()
        self.assertEqual(h.response_code, 200)
        res = json.loads(h.wfile.getvalue().decode("utf-8"))
        self.assertTrue(res["ok"])
        cookie_header = h.response_headers.get("Set-Cookie")
        self.assertIsNotNone(cookie_header)
        session_cookie = cookie_header.split(";")[0]

        # 5. Check auth status (first_run should be false, authenticated should be true with cookie)
        h = self.make_handler("/api/auth/status", "GET", headers={"Cookie": session_cookie})
        h.do_GET()
        self.assertEqual(h.response_code, 200)
        data = json.loads(h.wfile.getvalue().decode("utf-8"))
        self.assertFalse(data["first_run"])
        self.assertTrue(data["authenticated"])

        # 6. Authenticated request to rules API works
        h = self.make_handler("/api/rules", "GET", headers={"Cookie": session_cookie})
        h.do_GET()
        self.assertEqual(h.response_code, 200)

        # 7. Unauthenticated UI request redirects to login
        h = self.make_handler("/", "GET")
        h.do_GET()
        self.assertEqual(h.response_code, 302)
        self.assertEqual(h.response_headers.get("Location"), "/static/login.html")

        # 8. Logout invalidates session
        h = self.make_handler("/api/auth/logout", "POST", headers={"Cookie": session_cookie})
        h.do_POST()
        self.assertEqual(h.response_code, 200)
        new_cookie = h.response_headers.get("Set-Cookie")
        self.assertIn("Max-Age=0", new_cookie)


if __name__ == "__main__":
    unittest.main()
