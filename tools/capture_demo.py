"""Drive the real local application with Playwright: screenshots for the docs/site and the demo video.

  python tools/capture_demo.py screenshots     # -> docs/img/*.png
  python tools/capture_demo.py video           # -> .cache/video/raw.webm + segments.json (then tools/make_video.py)

Starts its own server on 127.0.0.1:8765 with a fresh data dir (var/capture), creates throw-away accounts with a random
password, and uses only the illustrative demo inputs in demo/. Narration segments are synthesised first (Kokoro-82M) so
that each on-screen step lasts at least as long as its narration; segment start times are written for captions.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
DATA = ROOT / "var" / "capture"
VIDEO_DIR = ROOT / ".cache" / "video"
IMG = ROOT / "docs" / "img"
PY = sys.executable


def start_server(pw: str) -> subprocess.Popen:
    if DATA.exists():
        shutil.rmtree(DATA)
    env = {**os.environ, "VVIR_DATA_DIR": str(DATA), "VVIR_OFFLINE": "1", "VVIR_DEMO_PASSWORD": pw}
    subprocess.run([PY, str(ROOT / "tools" / "seed_demo.py"), "--data-dir", str(DATA), "--accounts-only"], env=env, check=True, capture_output=True)
    proc = subprocess.Popen([PY, "-m", "incident_reporter", "serve", "--port", str(PORT)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(600):
        try:
            socket.create_connection(("127.0.0.1", PORT), timeout=1).close()
            return proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("server did not start")


def login(page, user: str, pw: str) -> None:
    page.goto(f"{BASE}/login")
    page.fill("#username", user)
    page.fill("#password", pw)
    page.click("button[type=submit]")
    page.wait_for_url(f"{BASE}/")


def upload(page, scenario: dict) -> str:
    page.goto(f"{BASE}/incidents/new")
    page.set_input_files("#audio", str(ROOT / "demo" / scenario["audio"]))
    page.set_input_files("#image", str(ROOT / "demo" / scenario["image"]))
    page.wait_for_timeout(800)
    page.click("#submit-btn")
    page.wait_for_url(f"{BASE}/incidents/inc_*", timeout=300_000)
    return page.url.rsplit("/", 1)[1]


def scenarios() -> dict:
    return {s["id"]: s for s in json.loads((ROOT / "demo" / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]}


def logout(page) -> None:
    page.click("form[action='/logout'] button")
    page.wait_for_url(f"{BASE}/login")


def shot(page, name: str, selector: str | None = None, full: bool = False) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    if selector:
        page.locator(selector).first.screenshot(path=str(IMG / name))
    else:
        page.screenshot(path=str(IMG / name), full_page=full)
    print("saved", name)


def approve(page, iid: str, answer: str) -> None:
    page.goto(f"{BASE}/incidents/{iid}#review")
    for inp in page.locator("input[id^=answer_]").all()[:1]:
        inp.fill(answer)
    page.click("form[action$='/draft'] button[type=submit]")
    page.wait_for_load_state()
    page.check("input[name=confirm]")
    ack = page.locator("input[name=acknowledge_open]")
    if ack.count():
        ack.check()
    page.click("form[action$='/approve'] button[type=submit]")
    page.wait_for_load_state()


# ------------------------------------------------------------------ screenshots


def screenshots() -> None:
    from playwright.sync_api import sync_playwright

    pw = secrets.token_urlsafe(12)
    proc = start_server(pw)
    sc = scenarios()
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chromium", args=["--lang=en-US"])
            page = b.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1, locale="en-US")
            login(page, "demo.reporter", pw)
            page.goto(f"{BASE}/incidents/new")
            page.set_input_files("#audio", str(ROOT / "demo" / sc["s1_flood_consistent"]["audio"]))
            page.set_input_files("#image", str(ROOT / "demo" / sc["s1_flood_consistent"]["image"]))
            page.wait_for_timeout(1200)
            shot(page, "upload.png")
            page.click("#submit-btn")
            page.wait_for_url(f"{BASE}/incidents/inc_*", timeout=300_000)
            s1 = page.url.rsplit("/", 1)[1]
            s2 = upload(page, sc["s2_conflict"])
            s4 = upload(page, sc["s4_unclear"])
            page.goto(f"{BASE}/incidents/{s1}")
            shot(page, "transcript.png", "#transcript")
            shot(page, "views.png", "section[aria-labelledby=views-h]")
            shot(page, "comparison.png", "section[aria-labelledby=cmp-h]")
            shot(page, "draft.png", "#draft")
            page.click("form[action$='/submit'] button")
            page.wait_for_load_state()
            logout(page)
            login(page, "demo.reviewer", pw)
            page.goto(f"{BASE}/incidents/{s2}")
            shot(page, "conflict.png", "section[aria-labelledby=cmp-h]")
            page.goto(f"{BASE}/incidents/{s4}")
            shot(page, "unclear.png", "section[aria-labelledby=flags-h]")
            shot(page, "unclear-transcript.png", "#transcript")
            page.goto(f"{BASE}/incidents/{s1}#review")
            shot(page, "review.png", "#review")
            approve(page, s1, "A utility pole is visible on the left; utility company notified.")
            page.goto(f"{BASE}/incidents/{s1}")
            shot(page, "approved.png", "section[aria-labelledby=rec-h]")
            shot(page, "audit.png", "section[aria-labelledby=audit-h]")
            page.goto(f"{BASE}/")
            shot(page, "dashboard.png")
            page.goto(f"{BASE}/ops")
            shot(page, "ops.png", full=True)
            page.set_viewport_size({"width": 390, "height": 844})
            page.goto(f"{BASE}/incidents/{s1}")
            shot(page, "mobile-incident.png")
            b.close()
    finally:
        proc.kill()


# ------------------------------------------------------------------ video

SEGMENTS = [
    ("intro", "This is Voice plus Vision Incident Reporter, a local proof of concept. A spoken report and a photo go in; "
              "a structured incident draft comes out, and nothing becomes a record until a person approves it."),
    ("login", "A field reporter signs in. Every account and every action is logged."),
    ("upload", "The reporter uploads a short voice report and a photo of the scene. The server checks the file type, size, "
               "duration and whether the files decode, and strips location data from the image."),
    ("process", "Everything runs on this laptop's CPU: Whisper transcribes the audio, and CLIP analyses the image."),
    ("audio", "Here is the report being played back."),
    ("transcript", "The transcript is shown with the incident terms it found. If the recogniser misheard something, "
                   "the reporter corrects it here, and the original is kept for the audit trail."),
    ("views", "The report is classified three ways: text only, image only, and text and image combined."),
    ("compare", "Then the two sources are compared. They agree on flood water. The street address, the time and the stranded couple "
                "come only from the report. A weak image signal, possible power lines, is marked as uncertain and becomes a question."),
    ("draft", "The draft proposes a category and a summary. Every claim links back to the words or image evidence it came from."),
    ("conflict", "When the sources contradict each other, like a report of a fire with a photo of a flood, "
                 "the system lists both sides and does not decide which one is true."),
    ("review", "A reviewer, who is not the reporter, answers the open question, confirms the evidence, and approves."),
    ("approved", "Only now is there an approved record, with a hash, the reviewer's name, and every change in the audit log."),
    ("result", "Measured on 955 held-out CrisisMMD posts, combining text and image scored 0.82 macro F1, versus 0.79 for "
               "the image alone and 0.74 for the text alone. The gain over the image alone was not statistically reliable."),
    ("limit", "One honest limit: there is no real dataset of spoken incident reports, so speech was tested on synthetic "
              "voices and audiobook recordings, not on real callers."),
    ("outro", "Voice plus Vision Incident Reporter. Code, results and credits are in the repository."),
]


def slide_html(title: str, lines: list[str], small: str = "") -> str:
    items = "".join(f"<li>{l}</li>" for l in lines)
    return f"""<html><head><style>
    body{{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;background:#08111F;color:#F8FAFC;
    font-family:Inter,'Segoe UI',sans-serif;background-image:linear-gradient(rgba(59,130,246,.05) 1px,transparent 1px),
    linear-gradient(90deg,rgba(59,130,246,.05) 1px,transparent 1px);background-size:60px 60px}}
    .c{{max-width:980px;padding:40px}} .k{{font-family:Consolas,monospace;color:#22D3EE;letter-spacing:.12em;font-size:15px;text-transform:uppercase}}
    h1{{font-size:50px;letter-spacing:-.03em;margin:.3em 0;background:linear-gradient(135deg,#60A5FA,#A78BFA 50%,#22D3EE);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent}} li{{font-size:25px;color:#A9B8CC;margin:.45em 0}}
    ul{{padding-left:1.1em}} .s{{color:#74869E;font-size:17px;margin-top:28px;line-height:1.5}} b{{color:#F8FAFC}}
    </style></head><body><div class="c"><div class="k">Voice + Vision Incident Reporter</div><h1>{title}</h1><ul>{items}</ul>
    <div class="s">{small}</div></div></body></html>"""


def video() -> None:
    from playwright.sync_api import sync_playwright

    from experiments.speech_eval import TTS, save_wav

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    tts = TTS()
    durations = {}
    for key, text in SEGMENTS:
        x = tts.speak(text, "af_heart")
        save_wav(VIDEO_DIR / f"nar_{key}.wav", x)
        durations[key] = len(x) / 16000
    rq = json.loads((ROOT / "results" / "rq1_results.json").read_text())
    pw = secrets.token_urlsafe(12)
    proc = start_server(pw)
    sc = scenarios()
    marks: list[dict] = []
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(channel="chromium", args=["--lang=en-US"])
            # pre-create the conflict incident so it can be shown without waiting on camera
            prep = b.new_page(locale="en-US")
            login(prep, "demo.reporter", pw)
            conflict_id = upload(prep, sc["s2_conflict"])
            prep.close()
            ctx = b.new_context(locale="en-US", viewport={"width": 1280, "height": 720}, record_video_dir=str(VIDEO_DIR / "raw"),
                                record_video_size={"width": 1280, "height": 720})
            page = ctx.new_page()
            t0 = time.monotonic()

            def seg(key: str, action=None, min_extra: float = 0.6):
                start = time.monotonic() - t0
                if action:
                    action()
                target = durations[key] + min_extra
                left = target - (time.monotonic() - t0 - start)
                if left > 0:
                    page.wait_for_timeout(int(left * 1000))
                marks.append({"key": key, "start": round(start, 3), "end": round(time.monotonic() - t0, 3)})

            def smooth(selector: str):
                page.locator(selector).first.scroll_into_view_if_needed()
                page.evaluate("s => document.querySelector(s).scrollIntoView({behavior:'smooth', block:'start'})", selector)
                page.wait_for_timeout(900)

            def slide(*args):
                # about:blank first: set_content on an app page would inherit its CSP, which blocks the inline <style>
                page.goto("about:blank")
                page.set_content(slide_html(*args))

            slide("From a spoken report to an approved record",
                  ["Voice report + photo → transcript and image analysis",
                   "Text-only, image-only and combined inference",
                   "Source comparison and a traceable draft",
                   "<b>Human review and explicit approval</b>"],
                  "Local proof of concept · synthetic narration (Kokoro-82M) · illustrative inputs: fictional report, public-domain FEMA photo of an unrelated event")
            seg("intro", min_extra=1.0)

            def do_login():
                page.goto(f"{BASE}/login")
                page.fill("#username", "demo.reporter")
                page.wait_for_timeout(300)
                page.fill("#password", pw)
                page.wait_for_timeout(300)
                page.click("button[type=submit]")
                page.wait_for_url(f"{BASE}/")
            seg("login", do_login)

            def do_upload():
                page.goto(f"{BASE}/incidents/new")
                page.wait_for_timeout(700)
                page.set_input_files("#audio", str(ROOT / "demo" / sc["s1_flood_consistent"]["audio"]))
                page.wait_for_timeout(1200)
                page.set_input_files("#image", str(ROOT / "demo" / sc["s1_flood_consistent"]["image"]))
                page.wait_for_timeout(900)
                page.evaluate("window.scrollTo({top: 200, behavior: 'smooth'})")
            seg("upload", do_upload)

            def do_process():
                page.click("#submit-btn")
                page.wait_for_url(f"{BASE}/incidents/inc_*", timeout=300_000)
            seg("process", do_process, min_extra=0.3)
            iid = page.url.rsplit("/", 1)[1]

            audio_len = 0.0
            import soundfile as sf
            info = sf.info(str(ROOT / "demo" / sc["s1_flood_consistent"]["audio"]))
            audio_len = info.frames / info.samplerate

            def do_audio():
                smooth("section[aria-labelledby=inputs-h]")
                page.evaluate("document.querySelector('audio').play()")
            start_audio = time.monotonic() - t0
            seg("audio", do_audio, min_extra=0.2)
            page.wait_for_timeout(int(max(0.0, audio_len - (time.monotonic() - t0 - start_audio) + 0.8) * 1000))
            marks.append({"key": "playback", "start": round(start_audio + durations["audio"] + 0.1, 3), "end": round(time.monotonic() - t0, 3)})
            seg("transcript", lambda: smooth("#transcript"))
            seg("views", lambda: smooth("section[aria-labelledby=views-h]"))

            def do_compare():
                smooth("section[aria-labelledby=cmp-h]")
                page.wait_for_timeout(int(durations["compare"] * 450))
                page.evaluate("window.scrollBy({top: 330, behavior: 'smooth'})")
            seg("compare", do_compare)

            def do_draft():
                smooth("#draft")
                page.wait_for_timeout(int(durations["draft"] * 500))
                page.evaluate("window.scrollBy({top: 280, behavior: 'smooth'})")
            seg("draft", do_draft)

            def do_conflict():
                page.goto(f"{BASE}/incidents/{conflict_id}")
                page.wait_for_timeout(500)
                smooth("section[aria-labelledby=cmp-h]")
            seg("conflict", do_conflict)

            def do_review():
                page.goto(f"{BASE}/incidents/{iid}")
                page.click("form[action$='/submit'] button")
                page.wait_for_load_state()
                page.click("form[action='/logout'] button")
                page.wait_for_url(f"{BASE}/login")
                page.fill("#username", "demo.reviewer")
                page.fill("#password", pw)
                page.click("button[type=submit]")
                page.wait_for_url(f"{BASE}/")
                page.goto(f"{BASE}/incidents/{iid}#review")
                page.wait_for_timeout(600)
                smooth("#review")
                first = page.locator("input[id^=answer_]").first
                if first.count():
                    first.click()
                    first.type("A utility pole is visible on the left; utility company notified.", delay=25)
                page.click("form[action$='/draft'] button[type=submit]")
                page.wait_for_load_state()
                smooth("#review")
                page.wait_for_timeout(500)
                page.check("input[name=confirm]")
                ack = page.locator("input[name=acknowledge_open]")
                if ack.count():
                    page.wait_for_timeout(400)
                    ack.check()
                page.wait_for_timeout(700)
                page.click("form[action$='/approve'] button[type=submit]")
                page.wait_for_load_state()
            seg("review", do_review, min_extra=0.3)

            def do_approved():
                page.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
                page.wait_for_timeout(int(durations["approved"] * 550))
                smooth("section[aria-labelledby=audit-h]")
            seg("approved", do_approved)

            r = rq["results"]
            h1 = rq["h1"]
            slide("One measured result",
                  [f"CrisisMMD v2.0 test set, <b>n = {rq['test_n']}</b> text–image posts, 5 classes",
                   f"Macro F1 — combined <b>{r['combined']['macro_f1']:.3f}</b> · image only <b>{r['image']['macro_f1']:.3f}</b> · text only <b>{r['text']['macro_f1']:.3f}</b>",
                   f"Combined − image: {h1['point']:+.3f}, 95% CI [{h1['ci95'][0]:+.3f}, {h1['ci95'][1]:+.3f}] → <b>not reliable</b>",
                   "Combined − text: reliably better"],
                  "Pre-registered protocol · frozen CLIP ViT-B/32 + logistic regression · text/image research result, not a speech result")
            seg("result", min_extra=1.2)
            slide("One honest limitation",
                  ["No authentic dataset pairs spoken incident reports with photos",
                   "Speech tested separately: synthetic voices (Kokoro) and LibriSpeech read speech",
                   "Real callers — accents, stress, phone lines — are untested"],
                  "The research comparison uses genuine text–image pairs; audio and images are never artificially paired as data")
            seg("limit", min_extra=1.0)
            slide("Credits",
                  ["Music: “Dreamer” Kevin MacLeod (incompetech.com), licensed under Creative Commons: By Attribution 4.0 — "
                   "creativecommons.org/licenses/by/4.0 · excerpt, trimmed, volume lowered and ducked under the narration, fades added",
                   "Narration and demo voices: synthetic, Kokoro-82M (Apache-2.0)",
                   "Photos: FEMA / Jocelyn Augustino (flood, 2008) — public domain",
                   "Papers: Munia et al. 2025 (arXiv:2507.05165) · Pranesh 2022 (W-NUT) · Data: CrisisMMD (Alam et al. 2018)"],
                  "Oren Salami · AI Systems Portfolio")
            seg("outro", min_extra=2.5)
            ctx.close()
            raw = next((VIDEO_DIR / "raw").glob("*.webm"))
            final_raw = VIDEO_DIR / "raw.webm"
            if final_raw.exists():
                final_raw.unlink()
            raw.rename(final_raw)
            b.close()
    finally:
        proc.kill()
    (VIDEO_DIR / "segments.json").write_text(json.dumps({"segments": marks, "texts": dict(SEGMENTS), "durations": durations,
                                                         "demo_audio": str(ROOT / "demo" / sc["s1_flood_consistent"]["audio"])}, indent=1))
    print(json.dumps(marks, indent=1))


if __name__ == "__main__":
    {"screenshots": screenshots, "video": video}[sys.argv[1]]()
