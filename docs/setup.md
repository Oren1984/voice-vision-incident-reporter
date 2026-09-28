# Install, test and reproduce

Everything runs locally on the CPU. Tested on Windows 11 with Python 3.12, 32 GB RAM; the models need about 2 GB of disk.

## Install

```powershell
python -m venv .venv
.venv\Scripts\pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements-dev.txt        # runtime + experiment + tests
.venv\Scripts\python -m incident_reporter warmup           # downloads Whisper base.en + CLIP ViT-B/32 once
```

macOS/Linux: the same with `.venv/bin/...`. After the first download, `VVIR_OFFLINE=1` keeps every model call local.

Then start the demo: [LOCAL_DEMO_ACCESS.md](LOCAL_DEMO_ACCESS.md).

## Classifier heads

A fresh clone uses **zero-shot CLIP heads**, which are weaker: on the test set, the combined view reaches Macro F1
0.593 with zero-shot heads, against 0.816 with trained heads. The trained heads come from running the experiment
below on CrisisMMD. They are written to the git-ignored `artifacts/` folder and are **not distributed**, because the
dataset's terms keep its contents confidential. The incident page and `/healthz` show which heads are active. The
screenshots and the demo video were made with locally trained heads, so a fresh clone can classify the same demo
inputs differently.

## Tests

```powershell
.venv\Scripts\python -m pytest              # stub models, about 15-20 s
.venv\Scripts\python tools\record_tests.py  # also writes results/test_summary.json (the count shown on the site)
```

What the tests cover: [testing.md](testing.md).

## Reproduce the experiment

Download CrisisMMD v2.0 and the agreed-label split from the official page into `.cache/crisismmd/`
([dataset.md](dataset.md)), then run the commands in [experiment-results.md § Reproduce](experiment-results.md#reproduce).
Seeds, package versions, model revisions and dataset hashes are recorded there and in `results/run_manifest.json`.
Protocol: [experiment-design.md](experiment-design.md).

## Screenshots, video and site

| Output | Command |
|---|---|
| `docs/img/*.png` | `python tools/capture_demo.py screenshots` |
| `site/media/demo.mp4` + captions, poster | `python tools/capture_demo.py video`, then `python tools/make_video.py` |
| `docs/experiment-results.md` | `python tools/render_results.py` |
| `site/index.html` | `python tools/build_site.py [--repo-url URL]`, then `python tools/check_site.py` |

The capture tools drive the real app on port 8765 with throw-away accounts in `var/capture`. `make_video.py` accepts
only the licensed music file, checked by its sha256 ([media-licenses.md](media-licenses.md)).

The site links to `../docs/*.md` and `../README.md`. Those links work when the site is opened from inside the
repository; if only `site/` is hosted, build it with `--repo-url` and point the links at the repository.

## Environment notes (Windows machine used for the recorded run)

- **Windows Application Control** blocked some newer scipy/pandas DLLs, so `requirements*.txt` pin scipy 1.17.1 and
  pandas 3.0.1.
- **TLS-inspecting antivirus** broke Python's TLS to Hugging Face. Set `VVIR_USE_OS_TRUSTSTORE=1` for model downloads
  only; the machine's TLS settings were left unchanged.
- **Playwright** uses the full Chromium build (`channel="chromium"`); the headless-shell build was not installed.
- **Run CPU-heavy jobs one at a time.** CLIP feature extraction and speech synthesis running together oversubscribe
  the CPU badly.

## Open decisions before sharing

- Publishing the repository and the site needs the owner's approval.
- The CrisisMMD-trained heads stay unpublished unless the owner decides otherwise under the dataset terms.
