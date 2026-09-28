# Testing

`python -m pytest` runs the whole suite with a **stub model suite** (`tests/conftest.py::StubModels`) that has the same
methods as the real `ModelSuite` and can be told to return specific transcripts, probabilities and observations, or to
fail in specific ways. That keeps the tests fast, deterministic and focused on the application's own logic. The real
models are exercised by `tools/seed_demo.py`, the screenshot/video capture and the experiment scripts.

| File | What it proves |
|---|---|
| `tests/test_workflow.py` | Happy path DRAFT → NEEDS_REVIEW → APPROVED with audit trail; **approval boundary** (reporter cannot approve or edit; four-eyes; explicit confirmation; open questions answered or acknowledged; no approval before review; stale version 409; approved/rejected terminal; approved records immutable at DB level; reject needs reason); reviewer edits recorded with author and changed fields but no raw text in the audit; reporters see only their incidents; transcript correction keeps the original and re-analyses; **fail-safe states** for missing image, missing audio, no input, no speech, low ASR confidence, ASR crash, three kinds of invalid ASR output, NaN/malformed probabilities, embedding and observation failures, low confidence, dark/blurry image; hazard, negation and category **conflicts** reported with both sides; "not detected" treated as complementary, not contradiction; **traceability** (every transcript source quote equals the transcript slice; image and model sources match the analysis); processing never produces APPROVED; audit append-only and tamper-evident |
| `tests/test_media.py` | Audio decode/duration/quality; duration limits; type sniffing ignores the claimed type; corrupt audio; size limit while reading; image re-encoded **without EXIF/GPS**; PNG/WebP; truncated, tiny, too-many-pixel, oversized and non-image files; media store rejects unsafe ids and names and writes inside its root |
| `tests/test_web.py` | Every page requires login; wrong password + lockout; HttpOnly/SameSite cookie; CSP, nosniff, frame-deny, request id headers; POST without CSRF refused; full HTTP flow including a reporter's refused approval and the reviewer's approval + record JSON; upload errors shown as messages; oversized request 413 before parsing; bad ids and media kinds 404; ops/metrics reviewer-only; `/healthz` public and minimal; logout ends the session; password reset ends sessions and the old password; model scores never shown as 0% or 100% |
| `tests/test_evidence.py` | Concept spans, longest-term matching, negation within a clause, location/time/people fields, missing-information questions, text cleaning, WER normalisation |
| `tests/test_experiment.py` | Test stage refuses to run before selection; selection on dev then single test scoring on synthetic features; H1 verdict and head export; head files are label-checked and loaded without pickle; fast Macro F1 equals scikit-learn's; bootstrap deterministic; McNemar; the S3 head's attention actually depends on its input |

**Latest run:** `88 passed` — exact command output and timestamp in `results/test_summary.json`
(written by `python tools/record_tests.py`).

**Not covered by automated tests:** real-model accuracy (that is what `experiments/` measures), browser rendering
(checked with headless Chromium screenshots at 1280 px and 390 px), and audio playback in real browsers.
