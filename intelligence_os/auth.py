"""Authentication and Session Management for Intelligence OS (FinalPRD §9.6 — M7).

Implements password hashing via stdlib hashlib.scrypt and session storage.
Provides check decorators/helpers for request routing.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Optional

from intelligence_os.store import Store, _uid


def hash_password(password: str) -> str:
    """Hash password using scrypt (stdlib) with a 16-byte random salt."""
    salt = os.urandom(16)
    # n=16384, r=8, p=1 are safe and standard parameters for interactive auth.
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1)
    return f"{salt.hex()}:{dk.hex()}"


def verify_password(hash_str: str, password: str) -> bool:
    """Verify standard scrypt password format: salt_hex:hash_hex."""
    if ":" not in hash_str:
        return False
    try:
        salt_hex, dk_hex = hash_str.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1)
        return dk.hex() == dk_hex
    except Exception:
        return False


def create_user(store: Store, username: str, password: str) -> str:
    """Create a new user with a hashed password. Returns user_id."""
    user_id = _uid("usr")
    pwd_hash = hash_password(password)
    with store.tx() as c:
        c.execute(
            "INSERT INTO users (user_id, username, password_hash, created_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, username.strip(), pwd_hash, time.time())
        )
    return user_id


def get_user_by_username(store: Store, username: str) -> Optional[dict]:
    """Retrieve user details by username."""
    row = store.conn.execute(
        "SELECT * FROM users WHERE username = ?", (username.strip(),)
    ).fetchone()
    return dict(row) if row else None


def get_user_by_id(store: Store, user_id: str) -> Optional[dict]:
    """Retrieve user details by user_id."""
    row = store.conn.execute(
        "SELECT * FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()
    return dict(row) if row else None


def count_users(store: Store) -> int:
    """Get the total number of users (used to detect first-run / onboarding state)."""
    row = store.conn.execute("SELECT COUNT(*) FROM users").fetchone()
    return row[0] if row else 0


def authenticate_user(store: Store, username: str, password: str) -> Optional[str]:
    """Verify username and password. Returns user_id if valid, else None."""
    u = get_user_by_username(store, username)
    if u and verify_password(u["password_hash"], password):
        return u["user_id"]
    return None


def create_session(store: Store, user_id: str, duration_seconds: int = 2592000) -> str:
    """Create a new session cookie token. Defaults to 30 days."""
    session_id = _uid("ses")
    expires_at = time.time() + duration_seconds
    with store.tx() as c:
        c.execute(
            "INSERT INTO sessions (session_id, user_id, expires_at) "
            "VALUES (?, ?, ?)",
            (session_id, user_id, expires_at)
        )
    return session_id


def verify_session(store: Store, session_id: str) -> Optional[str]:
    """Verify a session token. Returns user_id if valid/unexpired, else None."""
    row = store.conn.execute(
        "SELECT user_id, expires_at FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if not row:
        return None
    if time.time() > row["expires_at"]:
        delete_session(store, session_id)
        return None
    return row["user_id"]


def delete_session(store: Store, session_id: str) -> None:
    """Invalidate / delete a session from DB."""
    with store.tx() as c:
        c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
