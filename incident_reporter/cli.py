"""Command line: serve the app, manage local users, verify the audit chain, pre-load models.

  python -m incident_reporter serve [--host 127.0.0.1] [--port 8000]
  python -m incident_reporter create-user USERNAME --role reporter|reviewer|admin [--password-env VAR]
  python -m incident_reporter verify-audit
  python -m incident_reporter warmup
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys

from .config import Settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m incident_reporter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the local web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-warmup", action="store_true", help="load models on first request instead of at start")
    u = sub.add_parser("create-user", help="add a local account")
    u.add_argument("username")
    u.add_argument("--role", required=True, choices=("reporter", "reviewer", "admin"))
    u.add_argument("--password-env", help="read the password from this environment variable instead of prompting")
    sub.add_parser("verify-audit", help="recompute the audit hash chain")
    sub.add_parser("warmup", help="download/load the local models once")
    a = ap.parse_args(argv)
    settings = Settings()

    if a.cmd == "serve":
        import uvicorn

        from .web import create_app

        if a.host not in ("127.0.0.1", "localhost", "::1"):
            print("warning: binding beyond localhost exposes uploaded incident media to your network; "
                  "set VVIR_SECURE_COOKIES=1 behind HTTPS", file=sys.stderr)
        app = create_app(settings)
        if not a.no_warmup:
            print("loading local models (first run downloads them)…", file=sys.stderr)
            app.state.models.warmup()
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning", access_log=False)
        return 0

    from .db import Database

    db = Database(settings.db_path)
    if a.cmd == "create-user":
        from .security import create_user

        if a.password_env:
            pw = os.environ.get(a.password_env)
            if not pw:
                print(f"environment variable {a.password_env} is empty", file=sys.stderr)
                return 2
        else:
            pw = getpass.getpass("Password (min 12 chars): ")
            if pw != getpass.getpass("Repeat: "):
                print("passwords differ", file=sys.stderr)
                return 2
        try:
            user = create_user(db, a.username, a.role, pw)
        except Exception as e:  # ValueError for policy, IntegrityError for duplicates
            print(f"could not create user: {e}", file=sys.stderr)
            return 2
        from . import audit

        audit.record(db, "cli", "user_created", username=user.username, role=user.role)
        print(f"created {user.role} '{user.username}'")
        return 0
    if a.cmd == "verify-audit":
        from . import audit

        res = audit.verify(db)
        print(json.dumps(res))
        return 0 if res["ok"] else 1
    if a.cmd == "warmup":
        from .models import ModelSuite

        m = ModelSuite(settings)
        m.warmup()
        print(json.dumps(m.info(), indent=2, default=str))
        return 0
    return 1
