# Demonstration guide

About ten minutes, entirely local. All inputs are illustrative: fictional reports read by a synthetic voice, and
public-domain photos of unrelated real events ([demo/CREDITS.md](../demo/CREDITS.md)). No pairing is real incident data.

## 0. Start

```powershell
.\scripts\run_demo.ps1            # add -Seed to pre-run all five scenarios
```

The password is in `var/demo/demo-credentials.txt` (git-ignored; or set `$env:VVIR_DEMO_PASSWORD` beforehand).
Open http://127.0.0.1:8000. Sign-in details and password reset: [LOCAL_DEMO_ACCESS.md](LOCAL_DEMO_ACCESS.md).
The first analysis after start takes longer while models load.

## 1. Consistent report (scenario s1)

1. Sign in as **demo.reporter** → **New report**.
2. Audio: `demo/audio/s1_flood_report.wav`; image: `demo/images/flood_street_fema_34508.jpg`. The page previews both and
   checks size and duration in the browser; the server repeats every check.
3. **Analyse report.** Expect status `DRAFT` (or `NEEDS_REVIEW` if a flag fires with your heads).
4. Walk down the page: inputs → transcript with highlighted terms → three views → comparison (agreement on flood water;
   address, time and the stranded couple only in the report; possible power lines as uncertainty) → draft with a source
   chip on every claim → unresolved questions.
5. Optional: edit the transcript (e.g. change "Mill" to "Mills") and **Save correction** — the analysis reruns, the
   original transcript stays under "Original machine transcript", and the audit trail records the edit count.
6. **Submit for review.** Sign out.

## 2. Review and approval

1. Sign in as **demo.reviewer** → open the incident.
2. In **Human review**: change the category if needed, answer the question(s), **Save draft changes**.
3. Tick the confirmation (and the acknowledgement if questions remain) → **Approve as record**. The page shows the
   approved record with its sha256; `record.json` downloads it; the audit trail lists every step with request ids.
4. Try the boundary: sign in as the reporter and try to approve (refused, HTTP 403); submit an incident as
   demo.reviewer and try to approve it yourself (refused, four-eyes) — demo.reviewer2 can.

## 3. Conflict (s2)

Upload `s2_fire_report.wav` (a fire at a warehouse) with the **flood** photo. Expect `NEEDS_REVIEW` with
`sources_conflict`: the report's fire/smoke versus the image's flood water, and (usually) a category disagreement
between the text and image views. Both sides are listed; the draft asks the reviewer which account is correct.

## 4. Unclear inputs (s4)

Upload `s4_flood_report_noisy.wav` (the s1 text with heavy noise) with `flood_street_fema_34508_blurred_dark.jpg`.
Expect image-quality flags (low contrast, blurry) and a transcript with recognition errors — in our run "stuck" became
"set" and "14 Mill Street" became "14 years", which removed the location and added a "where exactly?" question.
Note that the recogniser's confidence stayed high: the transcript check by a person is what catches this.

## 5. Other checks

- s3 (collapse): "Nobody seems to be hurt, but two people are trapped" — the negated and asserted terms are both shown.
- Upload a text file renamed `.wav`, a truncated JPEG or a 0.5-second clip → a clear validation message; nothing stored.
- **Operations** (reviewer): counts by state, flags, review outcomes, failure counters, p50/p95 durations, audit chain.
- `python -m incident_reporter verify-audit` → `{"ok": true, ...}`.

## Regenerating media

`python tools/make_demo_media.py` (demo voices), `python tools/capture_demo.py screenshots`,
`python tools/capture_demo.py video` then `python tools/make_video.py` (site video with captions and music; the music
file must be downloaded first — see [media-licenses.md](media-licenses.md)).
