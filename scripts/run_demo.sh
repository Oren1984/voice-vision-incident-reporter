#!/usr/bin/env sh
# Start the local demo: creates demo accounts (if missing) in var/demo and serves http://127.0.0.1:8000
# Effects: writes only under var/demo. Models load from .cache/hf/hub (downloaded on first run).
#   scripts/run_demo.sh          # accounts + server
#   scripts/run_demo.sh --seed   # also run the five illustrative scenarios through the real models
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
py="$root/.venv/bin/python"; [ -x "$py" ] || py="$root/.venv/Scripts/python.exe"
[ -x "$py" ] || { echo "create the virtual environment first (see README: Install)"; exit 1; }
export VVIR_DATA_DIR="$root/var/demo"
if [ "${1:-}" = "--seed" ]; then "$py" "$root/tools/seed_demo.py" --data-dir "$VVIR_DATA_DIR"
else "$py" "$root/tools/seed_demo.py" --data-dir "$VVIR_DATA_DIR" --accounts-only; fi
echo "users: demo.reporter (reporter), demo.reviewer / demo.reviewer2 (reviewers); password: var/demo/demo-credentials.txt; open http://127.0.0.1:8000/login"
exec "$py" -m incident_reporter serve
