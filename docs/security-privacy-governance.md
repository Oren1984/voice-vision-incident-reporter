# Security, privacy and governance

Scope: a local, single-machine proof of concept. Controls are sized for that; the gaps that matter before any
shared or production use are listed at the end.

## Human approval (governance boundary)

| Control | Where | Test |
|---|---|---|
| Only `approve()` sets `APPROVED`; processing never does | `workflow.py` | `test_processing_never_sets_approved` |
| Approver must have the reviewer (or admin) role | `workflow.approve` | `test_reporter_cannot_approve_or_edit`, `test_full_flow_over_http` |
| Four-eyes: the creator cannot approve their own incident (`VVIR_FOUR_EYES=1`, default) | `workflow.approve` | `test_four_eyes_creator_cannot_approve_own_incident` |
| Explicit confirmation checkbox; open questions must be answered or explicitly acknowledged | `workflow.approve`, `incident.html` | `test_approval_requires_explicit_confirmation_and_ack_of_open_questions` |
| Draft must be in review before approval | `workflow.approve` | `test_draft_cannot_be_approved_before_review` |
| Optimistic concurrency: stale version → 409 | `workflow._check_open` | `test_stale_version_is_refused` |
| Approved/rejected are terminal; approved record immutable (SQLite triggers) with sha256 | `workflow`, `db.py` | `test_approved_and_rejected_are_terminal` |
| Rejection requires a reason | `workflow.reject` | `test_reject_needs_reason_and_is_audited` |
| Who changed what: every reviewer edit stored as a revision with field-level changes and author | `workflow.edit_draft` | `test_reviewer_edit_is_recorded_with_changes` |

**Identity.** Local accounts (`python -m incident_reporter create-user`), PBKDF2-HMAC-SHA256 with 600,000
iterations and a random salt; minimum password length 12; the password is typed at a hidden prompt or read from
an environment variable, never from the command line. Sessions: 32-byte random tokens, stored as sha256 only,
HttpOnly + SameSite=Strict cookie, 8-hour expiry, `Secure` flag via `VVIR_SECURE_COOKIES=1` behind HTTPS.
Five failed logins lock a username for five minutes (in memory). This is appropriate for a POC on one machine; a
real deployment should use the organisation's identity provider (OIDC/SAML) and MFA.

**Roles.** `reporter` — create incidents, see and correct *their own* incidents, submit. `reviewer` — see all,
edit drafts, answer questions, approve/reject, operations page. `admin` — same as reviewer (user management is
CLI-only). Reporters get 404 (not 403) for others' incidents so ids cannot be probed.

## Fail-safe behaviour

| Situation | Result | Flag |
|---|---|---|
| ASR crashes or returns malformed output (missing text, non-finite scores) | transcript not used; rest of the analysis runs | `model_failure:asr` |
| No speech recognised | text views unavailable | `no_speech` |
| Recogniser unsure (duration-weighted mean log-prob < −0.8 or no-speech prob > 0.5) | reviewer asked to check the transcript | `asr_low_confidence` |
| Embedding or observation model error / invalid output | that view unavailable | `model_failure:clip_*`, `model_failure:view_*` |
| Classifier probabilities not a valid distribution (NaN, wrong length, not summing to 1) | view unavailable | `model_failure:view_*` |
| Audio or image missing | missing view; question added | `missing_audio`, `missing_image` |
| Blurry, dark, over-exposed, low-contrast or low-resolution image | uncertainty item | `image_quality` |
| Text and image conflict | contradiction with both sides + question | `sources_conflict` |
| Any view below 50% confidence | uncertainty item | `low_confidence` |
| Unexpected error anywhere in the analysis step (a bug, not a model error) | what already worked is kept; incident parked in review; `analysis_failed` audit event | `model_failure:analysis` |

Any of these sends the incident to `NEEDS_REVIEW`. None of them can approve it. Covered by the
`test_*_goes_to_review`, `test_invalid_*` and conflict tests in `tests/test_workflow.py`.

## Input validation and media handling

- Request bodies larger than the two file limits + 256 KB are refused (413) before multipart parsing; each
  file is also read with a hard byte limit.
- **Audio:** ≤ 15 MB, 1–120 s, format by magic bytes (WAV, FLAC, OGG, WebM, M4A, MP3), fully decoded with PyAV
  (damaged or truncated files fail), level/silence/clipping measured.
- **Image:** ≤ 10 MB, ≤ 40 MP checked before decoding (decompression bombs), JPEG/PNG/WebP by magic bytes and by
  Pillow's format, `verify()` + full `load()`, ≥ 64 px, EXIF orientation applied, then **re-encoded without
  metadata** — GPS coordinates and device data in the upload are not stored.
- Stored under server-generated ids (`inc_` + 12 hex) and fixed file names; ids are regex-validated and resolved
  paths are checked to stay inside the media root. Media is served only to authorised users, with a fixed
  content type, `nosniff` and `no-store`.
- All SQL uses bound parameters. Templates are autoescaped; transcript highlighting is built as data, not HTML.
- CSRF token required on every POST; CSP `default-src 'self'` with no inline script or style; `frame-ancestors
  'none'`, `Referrer-Policy: no-referrer`. The server binds to 127.0.0.1 by default and warns otherwise.

Tests: `tests/test_media.py`, `tests/test_web.py`.

## Secrets

No API keys are needed: every model runs locally. There are no secrets in the repository or in config
defaults. Demo accounts are created by `scripts/run_demo.ps1` with a password taken from `VVIR_DEMO_PASSWORD` or
generated at random and written to the git-ignored `var/demo/demo-credentials.txt`
([LOCAL_DEMO_ACCESS.md](LOCAL_DEMO_ACCESS.md) covers reset). `.env`, `var/`, `.cache/` and `artifacts/` are
git-ignored.

## Privacy

- Minimise: image metadata dropped on upload; logs and audit details contain ids, hashes, sizes, durations,
  states and field names — never transcripts, summaries, answers, reasons, passwords or tokens
  (`test_audit_details_carry_no_raw_transcript`, `test_reviewer_edit_is_recorded_with_changes`).
- The raw audio, image and transcripts are stored because a reviewer needs them; they live only in `var/` on
  the local machine.
- Voice is biometric personal data in many jurisdictions. A real deployment would need a lawful basis, a
  retention period and deletion tooling; this POC has none of the three (deletion is manual: remove
  `var/media/<id>` and the rows — approved records and audit events are deliberately protected by triggers).
- CrisisMMD content is confidential under its terms and never leaves `.cache/`/`artifacts/` (see
  [`dataset.md`](dataset.md)).

## Observability

- JSON logs (`var/logs/app.jsonl` + console) with `request_id`, route, status, duration and, for workflow
  events, `incident_id`, flags and stage durations. Every response carries `X-Request-ID`; audit events store the
  same id, so a log line, an audit event and a user report can be joined.
- `/healthz` (public, minimal): DB reachability, which models are loaded, which heads are active.
- `/ops` and `/api/metrics` (reviewers): incidents by state, flag counts, approvals/rejections/rejected uploads/
  transcription failures/corrections/edits, approvals that followed a reviewer edit, p50/p95/max durations per
  route and pipeline stage, failure counters, audit-chain verification.
- `python -m incident_reporter verify-audit` recomputes the audit hash chain (exit code 1 if broken).

## Audit log integrity

Each event stores `sha256(previous hash + event)`. SQLite triggers refuse UPDATE and DELETE on the audit table
and on approved records. Tampering that bypasses the triggers (e.g. dropping them) is detected by `verify()`
(`test_audit_is_append_only_and_tamper_evident`). This is tamper-*evidence* on a local file: someone who can
rewrite the whole database file could rebuild the chain. Anchoring the head hash externally would close that gap.

## Known gaps (before any shared use)

- No HTTPS termination, no organisation SSO/MFA, in-memory login throttling only.
- The login form itself carries no CSRF token (SameSite=Strict cookies mitigate login CSRF; all other POSTs are protected).
- Synchronous model inference inside the request (fine for one user; a queue would be needed for many).
- No retention policy, no subject-access or deletion workflow, no encryption at rest for `var/`.
- Starlette spools large multipart parts to temporary files; the Content-Length guard does not cover chunked
  uploads without a length header.
- The image observation thresholds and ASR confidence thresholds are uncalibrated design choices.
- The rule-based transcript extractor misses paraphrases and can be fooled; it is a transparency aid, not an
  understanding of the report.
