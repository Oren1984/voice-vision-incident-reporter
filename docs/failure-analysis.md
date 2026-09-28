# Failure analysis

Where the system fails, how we found out, and what the workflow does about it. Numbers come from `results/*.json`
(rendered in [experiment-results.md](experiment-results.md)).

## 1. Classification (RQ1)

**Main confusions.** For all three views, the largest error mass sits between `not_humanitarian` and
`other_relevant_information` (e.g. the combined model: 33 not_humanitarian test posts called "other relevant", 23 the
other way). These classes overlap in meaning — a weather map or a news link can be either — and the labels were assigned
by crowd workers per modality. `rescue_volunteering_or_donation_effort` is also often confused with `not_humanitarian`
(donation appeals mixed with unrelated promotions).

**The rare class drives the headline gap.** `affected_individuals` has 9 test examples. The combined model gets 6 right,
image-only 3, text-only 4. Three examples move Macro F1 by several points, which is most of the combined model's
advantage over the image (post-hoc 4-class check: +0.011).

**Class weighting confound.** Dev selection gave balanced class weights only to the combined model. Giving them to every
view widens the combined model's margin, but only because it hurts the single-source models — a reminder that "fusion
helps" claims are sensitive to the training setup.

**Disagreement is the useful signal.** In 191 of 955 test pairs the text-only and image-only views disagree; there the
combined view is right 62.8% of the time versus 93.7% when they agree. The application therefore flags a category
disagreement as a contradiction (both views ≥ 55% confident) or an uncertainty, instead of silently trusting the combined
view.

**Near-duplicates.** 208 test images are near-duplicates of training images (reposted photos, screenshots). Removing them
(and one text duplicate) changes no conclusion, but it shows how much disaster-Twitter data repeats itself.

**Calibration.** The classifier scores are not probabilities: expected calibration error on test is 0.046 (text),
0.014 (image) and 0.069 (combined), and the combined model often outputs > 99% on the demo inputs. The UI labels them
"model score" and says they are not calibrated.

## 2. Speech

**Numbers vs words.** Whisper writes "twenty" as "20", "exit twelve" as "exit 12", "nine pm" as "9 p.m.". Our pre-registered
WER normalisation leaves numbers as written, so these count as errors even when the meaning is intact; part of the
clean-speech WER and the missed "critical terms" are this formatting effect.

**Spelling variants.** British voices produce "Harbour", "centre". Harmless for meaning, but a literal rule looking for
"Harbor Road" would miss it.

**Casing breaks rules.** Whisper wrote "on station road" in lower case, so the first version of the location rule (which
required capitalised street names) lost the location. Found in SP2; the rule now also accepts lower-case street names,
while refusing a bare "the road". SP2 was re-run after the change.

**Confidently wrong.** In the noisy demo clip (0 dB SNR) Whisper heard "Three cars are stuck … at 14 Mill Street" as
"Three cars are set … at 14 years". The mean log-probability stayed at −0.35, well above our low-confidence threshold
(−0.8), so no `asr_low_confidence` flag fired. The downstream effect was visible, though: the location field disappeared
and the draft gained a "where exactly is the incident?" question. Recogniser confidence alone cannot be trusted; the
transcript is always shown next to the audio for a person to correct.

**Hallucinated tail.** On one voice Whisper appended a stray "You" to an otherwise perfect transcript — a known Whisper
artefact on trailing silence.

**Heavy noise changes drafts.** At 0 dB SNR, WER rises to about 25%, incident-critical words are recalled 68.4% of the
time, 31 of 96 clips lose or gain a concept, and 51 of 96 drafts gain or lose a missing-information question (SP2).

**Optimistic setting.** All synthetic voices speak clearly and at an even pace. Real callers are stressed, interrupted,
accented and on phone lines. These results are an upper bound.

## 3. Image observations

**Overconfidence (fixed).** The first scoring (softmax of each concept prompt against five "ordinary scene" prompts) put
most concepts near 1.0 — "power lines 95%" on a flood photo, "screenshot 73%" on a wildfire photo. That would have put
false "may be visible" claims into drafts. It was replaced by a margin score before any screenshots or video were made.

**Still weak.** On CrisisMMD dev, concept scores rank images of the related class above others with AUROC 0.64–0.90:
useful, not reliable. The flood photo still gets "possible power lines" (43%) — there is in fact a utility pole in the
frame, which the demo reviewer notes; this is the intended use: a hint for a person, not a finding.

**Absence is not evidence.** A concept "not detected" never produces a contradiction; only positive evidence on both
sides does (e.g. the report denies injuries but the image likely shows injured people).

## 4. Workflow

| Failure | Handling | Test |
|---|---|---|
| ASR crash / malformed ASR output | transcript not used, `model_failure:asr`, rest of the analysis still runs, `NEEDS_REVIEW` | `test_asr_crash_is_contained`, `test_invalid_asr_output_is_not_used` |
| NaN or malformed probabilities | view marked unavailable, `model_failure:view_*` | `test_invalid_or_failed_model_output_goes_to_review` |
| Missing audio or image | missing view, question, `NEEDS_REVIEW` | `test_missing_*_goes_to_review` |
| Silent audio | `no_speech` | `test_no_speech_goes_to_review` |
| Blurry / dark image | `image_quality` + questions | `test_unclear_image_is_flagged` |
| Two reviewers at once | version check → 409 | `test_stale_version_is_refused` |
| Someone edits the audit table | triggers refuse; if bypassed, `verify()` reports the broken event | `test_audit_is_append_only_and_tamper_evident` |

**Rule-based evidence limits.** Paraphrases ("the place is on fire" works; "everything is ablaze" does not), unusual
place descriptions ("behind the old mill") and counts expressed indirectly are missed. "cars" is linked to the vehicle
concept even when the cars are only stuck, which is why that concept is labelled "vehicles involved or damaged".
