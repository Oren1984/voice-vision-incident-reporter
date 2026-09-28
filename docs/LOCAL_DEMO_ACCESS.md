# Local demo access

Everything here runs on your own machine. The app listens on `127.0.0.1` only, and the demo accounts exist only in
your local database. Install first: [setup.md](setup.md).

## 1. Start the demo

From the repository root:

```powershell
.\scripts\run_demo.ps1 -Seed      # Windows: demo accounts + five illustrative scenarios, then the server
```

```sh
scripts/run_demo.sh --seed        # macOS / Linux
```

Leave out `-Seed` / `--seed` to start with no incidents. Wait until the models have loaded; the first analysis takes
longer.

## 2. Open the login page

**http://127.0.0.1:8000/login**

## 3. Demo accounts

| Username | Role | What it can do |
|---|---|---|
| `demo.reporter` | reporter | Upload a voice report and a photo, correct the transcript, submit for review. Sees only its own incidents. |
| `demo.reviewer` | reviewer | See all incidents, edit the draft, answer open questions, approve or reject, open the Operations page. |
| `demo.reviewer2` | reviewer | Same as `demo.reviewer`. A second reviewer, because nobody may approve an incident they created. |

All three are local demo accounts with the same password. They work only with the demo data: fictional reports in a
synthetic voice and public-domain photos of unrelated events ([demo/CREDITS.md](../demo/CREDITS.md)).

## 4. Read the password

The password is generated on your machine when the accounts are created and saved in a local file that Git ignores.
It is not in the repository or in any document.

```powershell
Get-Content var\demo\demo-credentials.txt
```

(macOS / Linux: `cat var/demo/demo-credentials.txt`.) If you set `VVIR_DEMO_PASSWORD` before the accounts were
created, that value is the password and no file is written.

## 5. Short walkthrough

1. Sign in as `demo.reporter` → **New report**. Audio: `demo/audio/s1_flood_report.wav`; image:
   `demo/images/flood_street_fema_34508.jpg` → **Analyse report**.
2. Read down the page: transcript, three inference views, source comparison, draft. Fix the transcript if needed,
   then **Submit for review** and **Sign out**.
3. Sign in as `demo.reviewer`, open the incident, answer the open question, tick the confirmation and **Approve as
   record**. The approved record and the audit trail appear on the same page.

For a conflict, repeat step 1 with `demo/audio/s2_fire_report.wav` and the same flood photo. More scenarios:
[demo-guide.md](demo-guide.md).

## 6. Reset and troubleshooting

**New password** for the three demo accounts. This ends their open sessions and rewrites the credentials file; other
accounts are untouched. It also works while the server is running.

```powershell
.venv\Scripts\python tools\seed_demo.py --data-dir var\demo --accounts-only --reset-password
```

**Start from scratch** (deletes all local demo incidents, uploads, the audit log and the accounts): stop the server,
delete `var\demo`, and run `.\scripts\run_demo.ps1` again.

| Problem | Fix |
|---|---|
| Username or password rejected, but the password is right | The server may use a different data folder. `python -m incident_reporter serve` on its own uses `var\`, which has no demo accounts. Start with `run_demo.ps1`, or set `$env:VVIR_DATA_DIR = "$PWD\var\demo"` first. |
| "Too many attempts" | Five wrong passwords lock a username for five minutes. Wait, or restart the server. |
| `var\demo\demo-credentials.txt` is missing | The accounts were created before this file existed, or with `VVIR_DEMO_PASSWORD`. Run the reset command above. |
| Classifications differ from the screenshots | A fresh clone uses zero-shot heads; see [setup.md](setup.md#classifier-heads). |

Keep it local: never commit `var/` (check with `git check-ignore -v var/demo/demo-credentials.txt`), never reuse the
demo password, and don't bind the server beyond `127.0.0.1` with these accounts.
