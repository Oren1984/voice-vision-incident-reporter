"""Local accounts, sessions and CSRF protection for the POC.

- Passwords: PBKDF2-HMAC-SHA256, 600,000 iterations, 16-byte random salt (stdlib only).
- Sessions: 32-byte random token in an HttpOnly, SameSite=Strict cookie; only its sha256 is stored.
- CSRF: per-session random token, required on every state-changing request.
- Roles: reporter (submit, correct own transcript), reviewer (edit, approve, reject), admin (ops views
  and user management through the CLI). This is an identity mechanism for a single-machine POC, not a
  replacement for an organisation's identity provider.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .db import Database, now

ITERATIONS = 600_000
ROLES = ("reporter", "reviewer", "admin")
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,31}$")
MIN_PASSWORD = 12


@dataclass(frozen=True)
class User:
    id: int
    username: str
    role: str

    @property
    def can_review(self) -> bool:
        return self.role in ("reviewer", "admin")


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${dk.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        algo, it, salt, digest = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(it))
        return hmac.compare_digest(dk.hex(), digest)
    except (ValueError, TypeError):
        return False


def create_user(db: Database, username: str, role: str, password: str) -> User:
    username = username.strip().lower()
    if not USERNAME_RE.match(username):
        raise ValueError("username: 3-32 chars, lower-case letters, digits, . _ -")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"password must be at least {MIN_PASSWORD} characters")
    cur = db.execute(
        "INSERT INTO users (username, role, pw_hash, created_at) VALUES (?,?,?,?)",
        (username, role, hash_password(password), now()),
    )
    return User(cur.lastrowid, username, role)


def set_password(db: Database, username: str, password: str) -> bool:
    """Replace a user's password and end their open sessions. False if the user does not exist."""
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"password must be at least {MIN_PASSWORD} characters")
    row = db.one("SELECT id FROM users WHERE username = ?", (username.strip().lower(),))
    if row is None:
        return False
    db.execute("UPDATE users SET pw_hash = ? WHERE id = ?", (hash_password(password), row["id"]))
    db.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
    return True


class LoginThrottle:
    """In-memory: 5 failed attempts per username lock it for 5 minutes."""

    def __init__(self, limit: int = 5, window: float = 300.0) -> None:
        self.limit, self.window = limit, window
        self.failures: dict[str, list[float]] = {}

    def blocked(self, username: str) -> bool:
        t = time.monotonic()
        recent = [x for x in self.failures.get(username, []) if t - x < self.window]
        self.failures[username] = recent
        return len(recent) >= self.limit

    def fail(self, username: str) -> None:
        self.failures.setdefault(username, []).append(time.monotonic())

    def reset(self, username: str) -> None:
        self.failures.pop(username, None)


def authenticate(db: Database, username: str, password: str) -> User | None:
    row = db.one("SELECT * FROM users WHERE username = ?", (username.strip().lower(),))
    if row is None:
        check_password(password, hash_password("timing-equaliser"))  # similar cost for unknown users
        return None
    if not check_password(password, row["pw_hash"]):
        return None
    return User(row["id"], row["username"], row["role"])


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(db: Database, user: User, hours: float) -> tuple[str, str]:
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    db.execute(
        "INSERT INTO sessions (token_hash, user_id, csrf, created_at, expires_at) VALUES (?,?,?,?,?)",
        (_token_hash(token), user.id, csrf, now(), expires),
    )
    return token, csrf


def session_user(db: Database, token: str | None) -> tuple[User, str] | None:
    if not token:
        return None
    row = db.one(
        "SELECT u.id, u.username, u.role, s.csrf, s.expires_at FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ?",
        (_token_hash(token),),
    )
    if row is None or row["expires_at"] < now():
        return None
    return User(row["id"], row["username"], row["role"]), row["csrf"]


def end_session(db: Database, token: str | None) -> None:
    if token:
        db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


def csrf_ok(expected: str, given: str | None) -> bool:
    return bool(given) and hmac.compare_digest(expected, given)
