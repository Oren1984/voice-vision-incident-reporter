# Start the local demo: creates demo accounts (if missing) in var\demo and serves http://127.0.0.1:8000
# Effects: writes only under var\demo (SQLite DB, uploaded media, logs). Models load from .cache\hf\hub
# (downloaded on first run). Nothing is sent to any external service at run time.
#
#   .\scripts\run_demo.ps1              # accounts + server
#   .\scripts\run_demo.ps1 -Seed        # also run the five illustrative scenarios through the real models
#
# Password: $env:VVIR_DEMO_PASSWORD, or a random one written to var\demo\demo-credentials.txt (git-ignored).
# Reset: .venv\Scripts\python tools\seed_demo.py --data-dir var\demo --accounts-only --reset-password
param([switch]$Seed, [int]$Port = 8000)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Create the virtual environment first (see README: Install)." }
$env:VVIR_DATA_DIR = Join-Path $root "var\demo"
Write-Host "data directory: $env:VVIR_DATA_DIR"
if ($Seed) { & $py (Join-Path $root "tools\seed_demo.py") --data-dir $env:VVIR_DATA_DIR }
else { & $py (Join-Path $root "tools\seed_demo.py") --data-dir $env:VVIR_DATA_DIR --accounts-only }
Write-Host "users: demo.reporter (reporter), demo.reviewer / demo.reviewer2 (reviewers); password: var\demo\demo-credentials.txt"
Write-Host "open http://127.0.0.1:$Port  — demo inputs: demo\audio, demo\images (illustrative, see demo\CREDITS.md)"
& $py -m incident_reporter serve --port $Port
