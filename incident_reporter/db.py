"""SQLite storage. One file, created on first start. All SQL uses bound parameters."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL CHECK (role IN ('reporter', 'reviewer', 'admin')),
    pw_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    csrf TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    created_by INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('DRAFT', 'NEEDS_REVIEW', 'APPROVED', 'REJECTED')),
    version INTEGER NOT NULL DEFAULT 1,
    audio_file TEXT, audio_sha256 TEXT, audio_meta TEXT,
    image_file TEXT, image_sha256 TEXT, image_meta TEXT,
    transcript_original TEXT, asr_meta TEXT,
    transcript_corrected TEXT,
    analysis TEXT, draft TEXT, flags TEXT NOT NULL DEFAULT '[]',
    decided_by INTEGER REFERENCES users(id), decided_at TEXT, decision_note TEXT
);
CREATE TABLE IF NOT EXISTS draft_revisions (
    id INTEGER PRIMARY KEY,
    incident_id TEXT NOT NULL REFERENCES incidents(id),
    version INTEGER NOT NULL,
    author TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    changes TEXT NOT NULL,
    draft TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approved_records (
    incident_id TEXT PRIMARY KEY REFERENCES incidents(id),
    approved_by INTEGER NOT NULL REFERENCES users(id),
    approved_at TEXT NOT NULL,
    record TEXT NOT NULL,
    record_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    request_id TEXT,
    incident_id TEXT,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    details TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS audit_incident ON audit_events(incident_id);
CREATE TRIGGER IF NOT EXISTS approved_records_immutable BEFORE UPDATE ON approved_records
BEGIN SELECT RAISE(ABORT, 'approved records are immutable'); END;
CREATE TRIGGER IF NOT EXISTS approved_records_nodelete BEFORE DELETE ON approved_records
BEGIN SELECT RAISE(ABORT, 'approved records are immutable'); END;
CREATE TRIGGER IF NOT EXISTS audit_append_only_u BEFORE UPDATE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_append_only_d BEFORE DELETE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
"""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)

    def execute(self, sql: str, params: tuple | dict = ()):
        with self._lock:
            return self.conn.execute(sql, params)

    def one(self, sql: str, params: tuple | dict = ()):
        with self._lock:
            return self.conn.execute(sql, params).fetchone()

    def all(self, sql: str, params: tuple | dict = ()):
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def transaction(self) -> "_Tx":
        return _Tx(self)

    def close(self) -> None:
        self.conn.close()


class _Tx:
    """BEGIN IMMEDIATE ... COMMIT/ROLLBACK, re-entrant within one thread."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.outer = False

    def __enter__(self) -> Database:
        self.db._lock.acquire()
        self.outer = not self.db.conn.in_transaction
        if self.outer:
            self.db.conn.execute("BEGIN IMMEDIATE")
        return self.db

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.outer:
                self.db.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.db._lock.release()
        return False


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def loads(s: str | None, default=None):
    return json.loads(s) if s else default
