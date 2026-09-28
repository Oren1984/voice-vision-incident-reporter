"""Create demo accounts and run the demo scenarios through the real local models.

  python tools/seed_demo.py [--data-dir var/demo] [--scenarios s1_flood_consistent,s2_conflict]

  python tools/seed_demo.py --data-dir var/demo --accounts-only --reset-password

Accounts: demo.reporter (reporter), demo.reviewer and demo.reviewer2 (reviewers), local and demo-only. The password
comes from VVIR_DEMO_PASSWORD, or is generated at random; a generated password is written to
<data-dir>/demo-credentials.txt (git-ignored, like all of var/). Existing accounts are kept unless --reset-password
is given, which sets a new password on the three demo accounts only. Prints a summary per scenario.
All inputs are illustrative (see demo/CREDITS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from incident_reporter.config import Settings  # noqa: E402
from incident_reporter.db import Database  # noqa: E402
from incident_reporter.media import MediaStore  # noqa: E402
from incident_reporter.models import ModelSuite  # noqa: E402
from incident_reporter.observability import Metrics, setup_logging  # noqa: E402
from incident_reporter import audit  # noqa: E402
from incident_reporter.security import User, create_user, set_password  # noqa: E402
from incident_reporter.workflow import IncidentService  # noqa: E402

ACCOUNTS = (("demo.reporter", "reporter"), ("demo.reviewer", "reviewer"), ("demo.reviewer2", "reviewer"))


def ensure_accounts(db: Database, reset: bool = False) -> str | None:
    """Create missing demo accounts (and with reset=True, re-password existing ones). Returns a generated password."""
    pw = os.environ.get("VVIR_DEMO_PASSWORD")
    generated = None
    for name, role in ACCOUNTS:
        exists = db.one("SELECT 1 FROM users WHERE username = ?", (name,))
        if exists and not reset:
            continue
        if not pw:
            pw = generated = secrets.token_urlsafe(12)
        if exists:
            set_password(db, name, pw)
            audit.record(db, "cli", "demo_password_reset", username=name)
        else:
            create_user(db, name, role, pw)
    return generated


def write_credentials(path: Path, password: str) -> None:
    lines = [
        "Local demo accounts - Voice + Vision Incident Reporter (demo-only; git-ignored; never reuse this password)",
        "URL:      http://127.0.0.1:8000/login",
        f"Password: {password}  (same for all three accounts)",
        *(f"User:     {name}  ({role})" for name, role in ACCOUNTS),
        f"Reset:    python tools/seed_demo.py --data-dir {path.parent} --accounts-only --reset-password",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "var" / "demo"))
    ap.add_argument("--scenarios", default="")
    ap.add_argument("--accounts-only", action="store_true")
    ap.add_argument("--reset-password", action="store_true", help="set a new password on the three demo accounts")
    a = ap.parse_args()
    settings = Settings(data_dir=Path(a.data_dir))
    setup_logging(settings.log_path)
    db = Database(settings.db_path)
    generated = ensure_accounts(db, reset=a.reset_password)
    if generated:
        creds = settings.data_dir / "demo-credentials.txt"
        write_credentials(creds, generated)
        print(f"demo password written to {creds} (local, git-ignored)")
    if a.accounts_only:
        return
    spec = json.loads((ROOT / "demo" / "scenarios.json").read_text(encoding="utf-8"))
    wanted = set(filter(None, a.scenarios.split(",")))
    row = db.one("SELECT id, username, role FROM users WHERE username = 'demo.reporter'")
    reporter = User(row["id"], row["username"], row["role"])
    svc = IncidentService(db, MediaStore(settings.media_dir), ModelSuite(settings), settings, Metrics())
    for s in spec["scenarios"]:
        if wanted and s["id"] not in wanted:
            continue
        audio = (ROOT / "demo" / s["audio"]).read_bytes()
        image = (ROOT / "demo" / s["image"]).read_bytes()
        iid = svc.create(reporter, audio, image)
        inc = svc.get(reporter, iid)
        an = inc["analysis"]
        views = {k: (v["label"], round(v["confidence"], 3)) if v.get("available") else None for k, v in an["views"].items()}
        likely = [(o["concept"], o["score"]) for o in an["image"]["observations"] if o["band"] != "not_detected"]
        print(json.dumps({
            "scenario": s["id"], "incident": iid, "status": inc["status"], "flags": inc["flags"], "heads": an["heads"],
            "transcript": inc["transcript_original"], "views": views, "image_obs": likely,
            "counts": {k: len(v) for k, v in an["comparison"].items()},
            "contradictions": [c["text"] for c in an["comparison"]["contradictions"]],
            "questions": [q["text"] for q in inc["draft"]["unresolved_questions"]],
        }, indent=1))


if __name__ == "__main__":
    main()
