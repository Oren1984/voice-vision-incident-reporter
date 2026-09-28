# Original contributions (and what is borrowed)

| Area | Status | What exactly |
|---|---|---|
| Text + image classification on CrisisMMD | **borrowed task, own experiment** | Task, dataset and split come from Alam et al. (2018) and Ofli et al. (2020); the idea that fusion helps comes from Munia et al. (2025) and Pranesh (2022). The controlled comparison — one learner, three inputs, identical test examples, pre-registered criteria, paired bootstrap, near-duplicate sensitivity analysis, CLIP unimodal baselines the primary paper does not report — is ours |
| Paper-inspired fusion head (S3) | **own implementation of published ideas** | Guided gating + differential attention over two modality tokens, on cached frozen CLIP features. Not the authors' code or model |
| Speech transcription and its evaluation | **our extension** | faster-whisper in the workflow; SP1 (real read speech), SP2 (authored reports, synthetic voices, noise), SP3 (voice channel cost on the RQ1 test set) |
| Transcript correction with retained original | **our extension** | Original and corrected transcripts stored; correction re-runs the analysis; word-edit counts audited |
| Image observations with visible uncertainty | **our extension** | CLIP concept scores against ordinary-scene prompts, three bands, quality checks, "not detected ≠ absent" |
| Source comparison | **our extension** | Agreement, complementary, contradiction, missing, uncertainty; contradictions need positive evidence on both sides; no automatic truth decision |
| Traceable incident draft | **our extension** | Each claim lists transcript spans, image observations or model outputs; reviewer edits are marked |
| Human review workflow and governance | **our extension** | States, approval boundary, four-eyes, acknowledgement of open questions, immutable approved records, hash-chained audit, fail-safe flags |

The papers and CrisisMMD establish nothing about voice reports. No number in this repository should be read as a
voice-reporting result except the SP1–SP3 measurements, which are explicitly limited to synthetic speech or
non-incident read speech.
