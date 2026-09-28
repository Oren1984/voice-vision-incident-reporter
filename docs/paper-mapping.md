# From the papers to this implementation

What each source contributes, what we took from it, what we changed, and what we did **not** do.
Nothing in this repository reproduces a number from either paper.

## Primary paper — Munia et al. (2025)

**Nusrat Munia, Junfeng Zhu, Olfa Nasraoui, Abdullah-Al-Zubaer Imran. *Differential Attention for Multimodal
Crisis Event Analysis.*** CVPR 2025 Workshops (MMFM3). arXiv:2507.05165, v1 submitted 2025-07-07.
https://arxiv.org/abs/2507.05165 · authors' code: https://github.com/Munia03/Multimodal_Crisis_Event
(inspected at commit `7182e3b`, 2025-10-02; the repository states no license, so no code was copied).

**What the paper does.** Classifies CrisisMMD posts on three tasks (informativeness, humanitarian category,
damage severity) from text and image. It uses frozen CLIP encoders, enriches the text with LLaVA-generated
image descriptions (or Wikipedia knowledge), and fuses modalities with *Guided Cross Attention* (each modality
produces sigmoid masks for the other) and *Differential Attention* (the difference of two softmax attention
maps with a learnable λ). It reports that CLIP features plus guided fusion beat DenseNet/Electra baselines;
the best humanitarian-task macro F1 in its Table 2 is 71.04 (CLIP + Wiki + Guided CA + Diff Attn).
Its unimodal rows use DenseNet (image) and Electra + Wiki (text); it reports **no CLIP text-only or CLIP
image-only baseline**, so its tables do not isolate how much the fusion adds over each CLIP modality alone.
That gap is what our RQ1 measures (for our split and learner).

| Paper element | In this repository | Difference / reason |
|---|---|---|
| Frozen CLIP encoders | `incident_reporter/ml/clip_backbone.py` — `openai/clip-vit-base-patch32`, frozen, pinned revision | Same checkpoint the authors' code loads (`models_clip.py:111`). Chosen for CPU runnability |
| Text + image fusion beats single source | RQ1 compares text-only, image-only and combined on identical test examples ([`experiment-design.md`](experiment-design.md)) | We test the claim with one fixed learner (logistic regression) for all three views, so only the input differs |
| Guided Cross Attention | S3 head: `gate_from_t` / `gate_from_i` in `experiments/run_s3_diffattn.py` | Gates are computed per sample from the other modality's embedding |
| Differential Attention | S3 head: `diff_attn()` — two softmax maps, `λ = exp(λq1·λk1) − exp(λq2·λk2) + λ_init`, GroupNorm, `(1 − λ_init)` scaling | Runs over **two tokens** (image, text) so the attention maps vary with the input; see code observations below |
| LLaVA / Wikipedia text enrichment | **Not implemented** | Needs a 7B-class VLM per image; out of scope for a CPU POC. Our "image text" is limited to CLIP concept scores shown to the reviewer, not fed to the classifier |
| Tasks 1 and 3 (informativeness, damage severity) | **Not implemented** | One pre-registered primary task keeps the comparison clean; the humanitarian task is the closest to "what kind of incident is this" |
| Their data split (`crisismmd_datasplit_settingA`, 3,802 humanitarian pairs, 8 raw labels merged to 6) | **Not used** | That split is not part of the official download. We use the official agreed-label split (8,079 pairs, 5 classes), so numbers are not comparable |
| SGD, lr 1e-3, 50 epochs, 3 runs | S3 uses Adam lr 1e-3, early stopping on dev, 3 seeds | Small head on cached features; dev-based stopping avoids test-set selection |

**Observations from reading the authors' public code** (our reading of commit `7182e3b`; offered as notes,
not as claims about the paper's reported experiments):

1. `apply_self_attention` (`models_clip.py:88–98`, `222–232`) multiplies a `(batch, dim)` matrix by its
   transpose, so each sample attends over *the other samples in the mini-batch*. A prediction can therefore
   depend on which other posts share its batch.
2. `MultiheadDiffAttn.forward` reshapes its input to a sequence of length 1 (`multihead_diffattn.py:74`,
   `src_len = 1`). With one token, each softmax equals 1, so the differential map at line 120 is the constant
   `1 − λ`, and the block acts as a learned linear projection with normalisation rather than as attention over
   tokens. Our S3 head uses two tokens to keep the mechanism meaningful.

These notes are why S3 is described as *paper-inspired*, not as a re-implementation.

## Secondary paper — Pranesh (2022)

**Raj Ratn Pranesh. *Exploring Multimodal Features and Fusion Strategies for Analyzing Disaster Tweets.***
Proceedings of the Eighth Workshop on Noisy User-generated Text (W-NUT 2022), pp. 62–68, Gyeongju, Korea. ACL.
https://aclanthology.org/2022.wnut-1.6/

**What it does.** Fine-tunes ImageNet image models (VGG19, ResNet-50, AlexNet) and language models (BERT,
RoBERTa, ALBERT) on CrisisMMD, following Ofli et al. (2020) in merging humanitarian labels to five classes, then
compares intra-modal (self-, relation-, transformer-attention) and cross-modal (factorised bilinear pooling and
compact bilinear pooling variants) fusion. It reports that fusion — best with transformer attention + FBP —
outperforms unimodal models.

**How we use it.** Background and justification: (a) it motivates evaluating unimodal views against fusion on
the Ofli et al. five-class humanitarian setup, which we adopt through the official agreed-label split; (b) it
shows that fusion strategy matters, which is why we report a simple early-fusion baseline (primary), a late-fusion
variant (S1) and an attention-based head (S3) rather than a single fused model. We implement none of its fusion
modules. Its paper labels the humanitarian and informativeness tasks in a way that is easy to mix up
(its "task_1"/"task_2" naming), so we cite it for the approach, not for specific numbers.

## Dataset papers

Alam et al. (ICWSM 2018) for CrisisMMD; Ofli et al. (ISCRAM 2020) for the agreed-label benchmark split. See
[`dataset.md`](dataset.md).

## What the papers do **not** cover — our extensions

None of these papers or CrisisMMD contain speech. Everything below is original to this project and is not
supported by their results:

- speech transcription (faster-whisper) and transcription-error analysis (SP1–SP3);
- rule-based transcript evidence with character spans, and CLIP concept observations with explicit uncertainty bands;
- the source comparison (agreement / complementary / contradiction / missing / uncertainty) and the rule that a
  missing image detection is not evidence of absence;
- the incident draft with claim-level traceability;
- the human review workflow, approval boundary, audit chain and fail-safe states.

See [`contributions.md`](contributions.md).
