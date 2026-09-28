"""Run the test suite and record the outcome in results/test_summary.json (used by the site and docs).

Usage: python tools/record_tests.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-W", "ignore::DeprecationWarning"],
                       cwd=ROOT, capture_output=True, text=True)
    lines = p.stdout.strip().splitlines()
    tail = next((l for l in reversed(lines) if re.search(r"\d+ (passed|failed)", l)), lines[-1] if lines else "")
    counts = {k: int(v) for v, k in re.findall(r"(\d+) (passed|failed|errors?|skipped)", tail)}
    out = {"command": "python -m pytest", "exit_code": p.returncode, "summary": tail, "passed": counts.get("passed", 0),
           "failed": counts.get("failed", 0), "seconds": round(time.time() - t0, 1),
           "recorded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (ROOT / "results" / "test_summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(tail)
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
