"""Append-only, hash-chained audit log.

Each event stores sha256(prev_hash + canonical event). `verify()` recomputes the chain, so an edited
or deleted row is detectable; SQLite triggers also refuse UPDATE and DELETE on the table. Details carry
ids, hashes, lengths and changed field names — not transcripts, summaries or other raw report content.
This is tamper-evidence for a local POC, not protection against someone who can rewrite the whole file.
"""

from __future__ import annotations

import hashlib

from .db import Database, dumps, loads, now

GENESIS = "0" * 64


def _digest(prev: str, ts: str, request_id, incident_id, actor: str, action: str, details: str) -> str:
    payload = dumps([prev, ts, request_id, incident_id, actor, action, details])
    return hashlib.sha256(payload.encode()).hexdigest()


def record(db: Database, actor: str, action: str, incident_id: str | None = None, request_id: str | None = None, **details) -> None:
    ts = now()
    d = dumps(details)
    with db.transaction():
        row = db.one("SELECT hash FROM audit_events ORDER BY seq DESC LIMIT 1")
        prev = row["hash"] if row else GENESIS
        h = _digest(prev, ts, request_id, incident_id, actor, action, d)
        db.execute(
            "INSERT INTO audit_events (ts, request_id, incident_id, actor, action, details, prev_hash, hash) VALUES (?,?,?,?,?,?,?,?)",
            (ts, request_id, incident_id, actor, action, d, prev, h),
        )


def events(db: Database, incident_id: str | None = None, limit: int = 500) -> list[dict]:
    if incident_id:
        rows = db.all("SELECT * FROM audit_events WHERE incident_id = ? ORDER BY seq", (incident_id,))
    else:
        rows = db.all("SELECT * FROM audit_events ORDER BY seq DESC LIMIT ?", (limit,))
    return [{**dict(r), "details": loads(r["details"], {})} for r in rows]


def verify(db: Database) -> dict:
    prev, n = GENESIS, 0
    for r in db.all("SELECT * FROM audit_events ORDER BY seq"):
        n += 1
        expected = _digest(prev, r["ts"], r["request_id"], r["incident_id"], r["actor"], r["action"], r["details"])
        if r["prev_hash"] != prev or r["hash"] != expected:
            return {"ok": False, "events": n, "broken_at_seq": r["seq"]}
        prev = r["hash"]
    return {"ok": True, "events": n}
