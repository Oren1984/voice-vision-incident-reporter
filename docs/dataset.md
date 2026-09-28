# Dataset: origin, version, labels, splits and terms

Verified on 2026-09-27 against the official source.

## CrisisMMD

| Item | Value |
|---|---|
| Name | CrisisMMD: Multimodal Twitter Datasets from Natural Disasters |
| Publisher | Qatar Computing Research Institute (CrisisNLP) |
| Official page | https://crisisnlp.qcri.org/crisismmd |
| Version used | **v2.0** (16,058 tweets, 18,082 images, seven 2017 disasters). v2.0 maps "not relevant or can't judge" to `not_humanitarian`, drops "don't know" answers and removes duplicates relative to v1.0 |
| Archive | `CrisisMMD_v2.0.tar.gz`, 1,902,053,684 bytes, sha256 `351b9be6123a5d5074407d792bfdddc1f470d4909f65ddd429b584d24607d43f` (server `Last-Modified` 2021-10-16) |
| Split used | `crisismmd_datasplit_agreed_label.zip`, humanitarian task — the benchmark split of Ofli, Alam & Imran (ISCRAM 2020) |
| Split file hashes | see [`results/data_checks.json`](../results/data_checks.json) |

**Why this split.** It contains only pairs whose text annotation and image annotation agree, so a single
label applies to both modalities and text-only, image-only and combined models can be scored on the *same*
examples against the *same* label — exactly what the research question needs. It is also the split used by
Ofli et al. (2020) and, following them, by Pranesh (2022).

**Labels (5).** `affected_individuals`, `infrastructure_and_utility_damage`, `not_humanitarian`,
`other_relevant_information`, `rescue_volunteering_or_donation_effort`. In this split the original
fine-grained categories were already merged by the dataset authors (e.g. injured/dead and missing/found
people into `affected_individuals`; vehicle damage into infrastructure).

**Counts.**

| Split | Pairs | Distinct tweets | affected | infrastructure | not_hum. | other_rel. | rescue |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6,126 | 5,263 | 71 | 612 | 3,252 | 1,279 | 912 |
| dev | 998 | 998 | 9 | 80 | 521 | 239 | 149 |
| test | 955 | 955 | 9 | 81 | 504 | 235 | 126 |

The classes are heavily imbalanced; `affected_individuals` has 9 test examples, so its F1 is unstable and it
still counts as one fifth of Macro F1.

**Leakage checks** ([`results/data_checks.json`](../results/data_checks.json)): no tweet id and no image path
shared between splits; 1 dev and 1 test tweet duplicate a training tweet after text normalisation; **208 test
images (21.8%) are near-duplicates of a training image** (64-bit average hash, Hamming distance ≤ 4) —
reposted news photos and screenshots are common in disaster Twitter. The pre-registered sensitivity analysis
re-scores all views without these examples.

## Terms of use

The CrisisNLP terms of use (https://crisisnlp.qcri.org/terms-of-use.html; the split's `Readme.txt` also
points to a Dataverse copy that could not be reached on the verification date) require, in summary:

- use of the data **only for research on humanitarian computing**;
- keeping the **contents confidential** and deleting them on request or when the research ends (tweet ids may be kept);
- **citing** the dataset papers in derived publications;
- no defamation, harassment, infringement or privacy violations.

How this repository complies:

- Tweets, images, per-example features, per-example predictions and models trained on CrisisMMD are stored only
  in the git-ignored `.cache/` and `artifacts/` directories and are never committed, shown on the site or in
  the video.
- Only aggregate numbers (counts, metrics, confusion matrices) are published.
- The application's demo uses public-domain photos and fictional synthetic-voice reports, never CrisisMMD
  content. A fresh clone therefore runs the app with **zero-shot** heads; the CrisisMMD-trained heads appear
  only after you download the dataset yourself and run the experiment.
- Whether publishing trained weights would be compatible with the confidentiality clause is not settled here;
  they are withheld by default.

## Citations

- Firoj Alam, Ferda Ofli, Muhammad Imran. *CrisisMMD: Multimodal Twitter Datasets from Natural Disasters.* ICWSM 2018.
- Ferda Ofli, Firoj Alam, Muhammad Imran. *Analysis of Social Media Data using Multimodal Deep Learning for Disaster Response.* ISCRAM 2020. arXiv:2004.11838.

## Other data

| Data | Use | License / terms |
|---|---|---|
| LibriSpeech `test-clean` (Panayotov et al., ICASSP 2015), https://www.openslr.org/12 | SP1: ASR accuracy on real read speech, 100-utterance sample | CC BY 4.0 |
| `eval_data/speech/scripted_reports.json` | SP2: 24 fictional incident reports written for this project | project-authored, CC BY 4.0 |
| Demo photos and voices | application demo, site, video | see [`demo/CREDITS.md`](../demo/CREDITS.md) |
