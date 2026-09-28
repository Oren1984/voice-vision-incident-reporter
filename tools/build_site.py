"""Render site/index.html from site/templates/index.html.j2 and results/*.json; copy screenshots into site/media.

  python tools/record_tests.py     # writes results/test_summary.json (count shown on the site)
  python tools/build_site.py [--repo-url URL]

The site is published on its own (only site/ is copied to the web server), so every link to the README or docs/
points at the file on GitHub under --repo-url; the page never links outside site/ with a relative path.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/Oren1984/voice-vision-incident-reporter"
SITE = ROOT / "site"
R = ROOT / "results"

STEPS = [
    ("upload.png", "Upload a voice report and a photo",
     "Type, size and duration are checked in the browser and again on the server, where both files are fully decoded; "
     "image metadata such as GPS is removed before anything is stored.",
     "New incident report form with an audio file and a flood photo selected, both previewed"),
    ("transcript.png", "Transcribe — and let a person correct it",
     "Whisper base.en runs locally. Terms linked to incident concepts are highlighted. Corrections re-run the analysis; "
     "the original machine transcript is kept for audit and evaluation.",
     "Transcript with highlighted terms such as flooded and water is still rising, and a correction box"),
    ("views.png", "Three inference views",
     "Text only, image only, and combined text + image — the same classifier family over frozen CLIP embeddings, "
     "so the views can be compared directly. Scores are model outputs, not calibrated probabilities.",
     "Three cards showing text-only, image-only and combined class scores"),
    ("comparison.png", "Compare the sources, don't merge them",
     "Agreement, complementary details, contradictions, missing information and uncertainty — each with the transcript "
     "words, image scores or model outputs behind it. Weak image signals are marked as uncertain.",
     "Comparison panel: agreement on flood water, complementary location and time, one uncertainty, image observation scores"),
    ("conflict.png", "Conflicts are shown, not resolved",
     "A report of a warehouse fire with a photo of a flooded street: both sides are listed and the draft asks the reviewer "
     "which account is correct. The incident goes straight to review.",
     "Contradictions panel listing fire in the report versus flood water in the image, and a category disagreement"),
    ("draft.png", "A draft where every claim has a source",
     "Proposed category, summary, claims with source chips and unresolved questions. Nothing here is a record yet.",
     "Incident draft with claims, each followed by transcript or image source chips, and open questions"),
    ("review.png", "Human review",
     "A reviewer edits the category or summary, answers open questions, confirms the evidence and approves — or rejects with "
     "a reason. The reporter cannot approve their own incident.",
     "Review panel with edit form, approval confirmation checkboxes and reject form"),
    ("approved.png", "An approved, immutable record",
     "Approval creates a hashed record naming the reviewer; every step is in an append-only, hash-chained audit log.",
     "Approved record box with reviewer name, category and sha256"),
]


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default=REPO_URL)
    a = ap.parse_args()
    if not a.repo_url.startswith("https://github.com/"):
        raise SystemExit("--repo-url must be the repository's https://github.com/... URL")
    repo_url = a.repo_url.rstrip("/")
    media = SITE / "media"
    media.mkdir(parents=True, exist_ok=True)
    steps = []
    for img, title, text, alt in STEPS:
        src = ROOT / "docs" / "img" / img
        shutil.copyfile(src, media / img)
        w, h = Image.open(src).size
        steps.append({"img": img, "title": title, "text": text, "alt": alt, "w": w, "h": h})
    rq = load("rq1_results.json")
    video_meta = media / "demo.json"
    video = json.loads(video_meta.read_text()) if video_meta.exists() else {"duration_s": "—"}
    env = Environment(loader=FileSystemLoader(str(SITE / "templates")), autoescape=True, undefined=StrictUndefined)
    env.globals["doc"] = lambda path: f"{repo_url}/blob/main/{path}"
    html = env.get_template("index.html.j2").render(
        rq=rq, res=rq["results"], dc=load("data_checks.json"), s3=load("s3_diffattn_results.json"), ex=load("exploratory.json"),
        sp1=load("speech_librispeech.json"), sp2=load("speech_scripted.json"), sp3=load("speech_sp3.json"),
        tests_passed=load("test_summary.json")["passed"], steps=steps, video=video, repo_url=repo_url,
    )
    (SITE / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote site/index.html ({len(html):,} bytes), {len(steps)} screenshots")


if __name__ == "__main__":
    main()
