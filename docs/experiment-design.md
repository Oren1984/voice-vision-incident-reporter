# Experiment design (pre-registered protocol)

**Protocol version 1 — frozen 2026-09-27, before any feature extraction, training or evaluation.**
This file states what will be measured and what counts as success or failure. Results are reported
separately in [`experiment-results.md`](experiment-results.md). If anything below had to change after
results were seen, the change is listed in the *Deviations* section at the end, with the reason.

## 1. Research question

> **RQ1.** On the *same* held-out examples, does combining a post's text and its image improve
> humanitarian-category classification over text alone and over image alone?

The papers this project builds on ([Munia et al., 2025](https://arxiv.org/abs/2507.05165);
[Pranesh, 2022](https://aclanthology.org/2022.wnut-1.6/)) study **text + image** posts from Twitter. Neither
paper, nor CrisisMMD, contains speech. The voice part of this project is our own extension and is
evaluated separately (§ 7). No experiment here pairs audio with CrisisMMD images, and no synthetic
audio is presented as a real incident report.

## 2. Data

| Item | Value |
|---|---|
| Dataset | CrisisMMD **v2.0** (Alam et al., 2018), official source https://crisisnlp.qcri.org/crisismmd |
| Split | Official **agreed-label** split of the humanitarian task (Ofli et al., 2020): `crisismmd_datasplit_agreed_label.zip` |
| Unit | One (tweet text, image) pair. The split keeps only pairs whose text label and image label agree, so each pair has one label |
| Train / dev / test | 6,126 / 998 / 955 pairs (train contains 5,263 distinct tweets; some tweets carry several images) |
| Terms | Research on humanitarian computing only; contents kept confidential. See [`dataset.md`](dataset.md) |

**Labels (5 classes, fixed):** `affected_individuals`, `infrastructure_and_utility_damage`,
`not_humanitarian`, `other_relevant_information`, `rescue_volunteering_or_donation_effort`.

Test-set class counts: not_humanitarian 504 · other_relevant_information 235 ·
rescue_volunteering_or_donation_effort 126 · infrastructure_and_utility_damage 81 · affected_individuals **9**.
The `affected_individuals` class has only 9 test examples (71 in train); its per-class F1 is
reported but is statistically unreliable, and it has a large weight in Macro F1 (1/5).

**Leakage checks (run before this protocol was frozen, on the split files only):** no tweet id and no
image path occurs in more than one split; after normalising text (lower-case, URLs/mentions/`RT`
removed) one dev and one test tweet duplicate a training tweet. A perceptual-hash check for
near-duplicate *images* across splits is run once images are available (§ 6).

## 3. Representations

- **Backbone:** OpenAI CLIP ViT-B/32 (`openai/clip-vit-base-patch32`, Hugging Face revision
  `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`), **frozen**. This is the same checkpoint the authors'
  public code loads (`models_clip.py`). Chosen because it runs on a CPU laptop; the local application uses
  the same backbone.
- **Text input:** tweet text with URLs removed, a leading `RT @user:` removed, other `@mentions` replaced by
  `@user`, whitespace collapsed; truncated to CLIP's 77-token limit.
- **Image input:** the tweet image through the standard CLIP processor (resize 224, centre crop).
- **Features:** CLIP projected embeddings (512-d), L2-normalised. Extracted once and cached outside
  the repository.

## 4. Models

### Primary comparison — one learner, three inputs

The same learner is used for all three views so that the only difference is the input:
multinomial **logistic regression** (scikit-learn, `lbfgs`, `max_iter=5000`).

| View | Input |
|---|---|
| Text only | text embedding (512) |
| Image only | image embedding (512) |
| **Combined (primary)** | concatenation `[text ; image]` (1,024) — early fusion |

**Model selection (per view, on dev only):** `C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}` ×
`class_weight ∈ {None, "balanced"}`, choosing the highest dev Macro F1 (ties → smaller `C`, then `None`).
The chosen configuration is refit on train only. The test split is scored **once**, after every choice
above is frozen; the evaluation script refuses to run on test before selection results are written.

### Secondary comparisons (reported, not used to answer RQ1)

- **S1 Late fusion:** `p = w·p_text + (1−w)·p_image` from the two selected unimodal models,
  `w ∈ {0.0, 0.1, …, 1.0}` chosen on dev.
- **S2 Zero-shot CLIP (no training data):** class prompts (fixed in code before evaluation); text view =
  similarity of the tweet embedding to prompt embeddings; image view = image-to-prompt similarity;
  combined = mean of the two probability vectors. This is also the fallback the application uses when
  no trained heads are available.
- **S3 Paper-inspired fusion head (ours):** a small head over the same frozen CLIP embeddings that
  implements the *ideas* of the primary paper — guided cross-modal gating (each modality produces a
  sigmoid mask for the other) followed by differential attention (difference of two softmax attention
  maps with a learnable λ) over the two modality tokens. Adam, lr 1e-3, batch 64, ≤ 60 epochs, early
  stopping on dev Macro F1 (patience 8), class-weighted cross-entropy, seeds {13, 42, 7}; mean ± sd over
  seeds is reported. **It is not the authors' model and not a reproduction of their numbers**
  (see [`paper-mapping.md`](paper-mapping.md)).

## 5. Metrics and success criteria

- **Primary metric:** Macro F1 over the 5 classes on the official test split (n = 955).
- **H1 (success):** `MacroF1(combined) − max(MacroF1(text), MacroF1(image)) > 0` **and** the 95%
  paired-bootstrap confidence interval of that difference (10,000 resamples of test examples,
  seed 2026) has a lower bound above 0.
- **Failure outcomes (reported as such):** CI includes 0 → "no reliable improvement over the best single
  source"; point estimate ≤ 0 → "combined is not better". Any class with F1 = 0 is called out.
- **Secondary metrics:** accuracy, weighted F1, per-class precision/recall/F1, confusion matrices,
  exact McNemar test (combined vs. best unimodal, on correctness), expected calibration error
  (15 bins), Macro F1 by disaster event, and the difference of combined vs. *each* unimodal view with
  bootstrap CIs.
- **Sensitivity analysis:** the primary comparison repeated on the test subset that excludes examples
  whose normalised text duplicates a training text or whose image is a near-duplicate
  (64-bit average-hash Hamming distance ≤ 4) of a training image.

## 6. Reproducibility

- Seeds: 13 for model fitting and data handling; 2026 for bootstrap; {13, 42, 7} for S3.
- All package versions, model revisions, selected hyperparameters and dataset file hashes are written
  to `results/run_manifest.json` by the pipeline.
- Committed outputs are **aggregate only** (counts, metrics, confusion matrices). Per-example texts,
  images, features and predictions stay in the git-ignored `.cache/` / `artifacts/` directories because
  the dataset terms require confidentiality.

## 7. Speech workflow evaluation (our extension, separate from RQ1)

There is no authentic, labeled dataset of spoken incident reports paired with incident images, so the
speech part is evaluated on its own:

- **SP1 — ASR on real human speech.** A fixed sample of LibriSpeech `test-clean` utterances (read
  audiobook speech, CC BY 4.0; 100 utterances, deterministic sample, seed 13). Word error rate (WER) and
  character error rate (CER) against the official transcripts. Measures the recogniser on real voices;
  the content is not about incidents.
- **SP2 — Scripted incident reports (authored for this project).** Short reports written by us with
  reference transcripts, synthesised with an open-weights TTS model (Kokoro-82M, Apache-2.0) in several
  voices, under three conditions: clean, added white noise at 10 dB SNR, and at 0 dB SNR ("unclear
  speech"). Metrics: WER/CER, recall of incident-critical terms (hazard words, numbers, place names), and
  whether the downstream text view's category or the extracted evidence changes. Labelled **synthetic**
  everywhere.
- **SP3 — Voice channel on the RQ1 test set.** The 955 test tweet texts (cleaned as in § 3, hashtags
  spoken as words) synthesised with the same TTS, transcribed, and passed to the **already selected**
  text-only and combined models (no retraining). Compared on the same 955 examples: Macro F1 with the
  original text vs. the ASR transcript, WER, and the rate of predictions that change. This measures how
  much a transcription step costs; it is synthetic speech of written posts, **not** real voice
  reports. Audio and transcripts stay in `.cache/` (dataset confidentiality).
- **ASR:** faster-whisper `base.en` (CTranslate2 conversion of OpenAI Whisper base.en, revision
  `3d3d5dee26484f91867d81cb899cfcf72b96be6c`), int8 on CPU, beam size 5, English. WER uses a documented
  normalisation (lower-case, punctuation removed, numbers left as written).
- **Pre-stated expectations (descriptive, not hypothesis tests):** clean-speech WER ≤ 15% is treated
  as usable for this POC; a drop in text-only Macro F1 of ≤ 0.03 from original text to clean-speech
  transcripts is treated as "robust". Results outside these bounds are reported as they are.

## 8. Workflow robustness scenarios (application tests)

Missing audio, missing image, silent or unclear audio, blurred/dark/tiny image, conflicting text and
image, invalid model output (injected faults: NaN or malformed probabilities, empty transcript, model
exception), and low-confidence predictions. Expected behaviour for each: the incident lands in
`NEEDS_REVIEW` with an explicit flag, never in `APPROVED`, and nothing is silently dropped. Implemented as
automated tests in `tests/`.

## Deviations

None yet (updated if the protocol changes after results are seen).
