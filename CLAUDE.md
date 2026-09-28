# CLAUDE.md — voice-vision-incident-reporter

Project router for agents working in this repository. Self-contained: everything referenced here lives in this repo.

## Identity and goal

Local proof of concept: a spoken incident report + an image → transcription, image analysis, text-only / image-only / combined inference, source comparison, a traceable incident draft, and **explicit human approval** before anything becomes a record. Research question (text+image vs single source on the same examples) is answered on CrisisMMD; the speech part is our own extension, evaluated separately. Delivery model: bounded POC → docs → handover. No deployment.

## Rules (hard constraints)

- Human approval boundary: only `IncidentService.approve()` may set `APPROVED`; it requires a reviewer, explicit confirmation, the current version, acknowledgement of open questions and (default) a reviewer who did not create the incident. Never add a path that approves automatically.
- Fail safe: model errors, invalid output, missing input, conflicts and low confidence land in `NEEDS_REVIEW` with a flag.
- The comparison never decides which conflicting source is true.
- CrisisMMD terms: research on humanitarian computing only; contents confidential. Never commit tweets, images, per-example features/predictions, or models trained on it (`.cache/`, `artifacts/` are git-ignored). Committed results are aggregate.
- No claims of reproducing a paper result. Keep the text/image research result and the speech extension clearly separated in code, docs, site and video.
- Logs and audit details: ids, hashes, sizes, durations, field names — no transcripts, summaries, passwords or tokens.
- No secrets in the repo; no paid services; no commit, push or publication without the owner's explicit approval.
- Protocol changes after results are seen go into `docs/experiment-design.md` § Deviations.

## Module selection (workspace Blueprint)

| Module | Decision | Where |
|---|---|---|
| 01 Context, 03 Architecture | baseline | `docs/architecture.md`, `README.md` |
| 02 Rules, 04 Scope | baseline | this file, `README.md` § Boundaries |
| 06 Commands | baseline | below, `docs/demo-guide.md` |
| 09 Testing | baseline | below, `docs/testing.md` |
| 10 Security | YES — file uploads, local accounts, sensitive media, confidential dataset | `docs/security-privacy-governance.md` |
| 14 Handover | YES — bounded POC delivery | `README.md`, `docs/setup.md` (install, reproduce, environment notes, open decisions), `docs/LOCAL_DEMO_ACCESS.md` |
| 11 Schemas | NO — the draft JSON is consumed by this app only; its shape is covered by tests | — |
| 05, 07, 08, 12, 13, 15, 16 | NO — standard single-agent flow; no recurring multi-service protocol; docs covered by `docs/` | — |

## Commands

```powershell
python -m venv .venv; .venv\Scripts\pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest                                  # tests (stub models, ~15 s)
.venv\Scripts\python -m incident_reporter create-user NAME --role reviewer
.venv\Scripts\python -m incident_reporter serve                 # http://127.0.0.1:8000
.venv\Scripts\python -m experiments.prepare; .venv\Scripts\python -m experiments.extract_features
.venv\Scripts\python -m experiments.run_rq1 select; .venv\Scripts\python -m experiments.run_rq1 test
```

Full list and the speech evaluation: `docs/experiment-results.md` § Reproduce. On this machine, set `VVIR_USE_OS_TRUSTSTORE=1` for model downloads (antivirus TLS inspection) and keep scipy/pandas at the pinned versions (Windows Application Control blocks some newer DLLs).

## Testing — definition of done

`python -m pytest` passes; the approval-boundary, fail-safe, validation and traceability tests are never weakened to make a change pass. Numbers quoted anywhere must come from `results/*.json`.
