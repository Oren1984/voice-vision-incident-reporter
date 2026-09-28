"""Assemble the demo video: screen recording + synthetic narration + demo audio + captions + ducked background music.

Inputs (from `python tools/capture_demo.py video`): .cache/video/raw.webm, segments.json, nar_*.wav.
Music: .cache/music/Dreamer.mp3 — refuses any file whose sha256 differs from the verified one (docs/media-licenses.md).
Outputs: site/media/demo.mp4 (captions burned in), demo.en.vtt, demo-poster.png, demo.json.
Usage: python tools/make_video.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
V = ROOT / ".cache" / "video"
OUT = ROOT / "site" / "media"
MUSIC = ROOT / ".cache" / "music" / "Dreamer.mp3"
MUSIC_SHA = "615e65c9ca131c11f8e12970644ed74ef822ec4f643b1d9d0085a223600bcdfb"
RATE = 48000
NARRATION_DELAY = 0.15


def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_duration(path: Path) -> float:
    p = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", p.stderr)
    if m and m.group(0) != "Duration: N/A":
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    # webm from the browser may lack a duration header: decode to count
    p = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True)
    t = re.findall(r"time=(\d+):(\d+):(\d+\.\d+)", p.stderr)[-1]
    return int(t[0]) * 3600 + int(t[1]) * 60 + float(t[2])


def read_wav(path: Path) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    g = np.gcd(sr, RATE)
    return resample_poly(x, RATE // g, sr // g).astype(np.float32)


def chunks(text: str, limit: int = 84) -> list[str]:
    """Split narration into caption lines at sentence/clause boundaries, ≤ limit characters each."""
    parts = re.split(r"(?<=[.!?])\s+|(?<=[,;:])\s+(?=\S)", text)
    out, cur = [], ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > limit:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur)
    final = []
    for c in out:  # hard-wrap anything still too long
        while len(c) > limit:
            cut = c.rfind(" ", 0, limit)
            final.append(c[:cut])
            c = c[cut + 1:]
        final.append(c)
    return final


def ts(t: float, sep: str = ".") -> str:
    h, rem = divmod(max(0.0, t), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}{sep}{int(round((s - int(s)) * 1000)):03d}"


def main() -> None:
    if not MUSIC.exists() or hashlib.sha256(MUSIC.read_bytes()).hexdigest() != MUSIC_SHA:
        raise SystemExit("music file missing or not the verified recording (see docs/media-licenses.md)")
    meta = json.loads((V / "segments.json").read_text())
    raw = V / "raw.webm"
    dur = probe_duration(raw)
    n = int(dur * RATE) + RATE
    voice = np.zeros(n, np.float32)
    cues: list[tuple[float, float, str]] = []
    texts, durations = meta["texts"], meta["durations"]
    for seg in meta["segments"]:
        key = seg["key"]
        if key == "playback":
            x = read_wav(Path(meta["demo_audio"]))
            s = int(seg["start"] * RATE)
            voice[s:s + len(x)] += 0.9 * x[: max(0, n - s)]
            spoken = json.loads((ROOT / "demo" / "scenarios.json").read_text(encoding="utf-8"))["scenarios"][0]["text"]
            lines = chunks(spoken, 70)
            span = len(x) / RATE
            tot = sum(len(l) for l in lines)
            t = seg["start"]
            for i, l in enumerate(lines):
                d = span * len(l) / tot
                cues.append((t, t + d, ("[Report, synthetic voice] " if i == 0 else "") + l))
                t += d
            continue
        x = read_wav(V / f"nar_{key}.wav")
        start = seg["start"] + NARRATION_DELAY
        s = int(start * RATE)
        voice[s:s + len(x)] += x[: max(0, n - s)]
        lines = chunks(texts[key])
        tot = sum(len(l) for l in lines)
        t = start
        for l in lines:
            d = durations[key] * len(l) / tot
            cues.append((t, t + d, l))
            t += d
    peak = float(np.abs(voice).max())
    voice = (voice / peak * 0.89).astype(np.float32) if peak > 0 else voice
    import soundfile as sf

    sf.write(str(V / "voice.wav"), voice[: int(dur * RATE)], RATE, subtype="PCM_16")

    OUT.mkdir(parents=True, exist_ok=True)
    vtt = ["WEBVTT", ""]
    srt = []
    for i, (a, b, text) in enumerate(cues, 1):
        vtt += [str(i), f"{ts(a)} --> {ts(b)}", text, ""]
        srt += [str(i), f"{ts(a, ',')} --> {ts(b, ',')}", text, ""]
    (OUT / "demo.en.vtt").write_text("\n".join(vtt), encoding="utf-8")
    (V / "captions.srt").write_text("\n".join(srt), encoding="utf-8")

    fade_out = max(0.0, dur - 4.0)
    style = "FontName=Segoe UI,FontSize=17,PrimaryColour=&H00FFFFFF,BackColour=&H99000000,BorderStyle=4,Outline=0,Shadow=0,MarginV=26,Alignment=2"
    filt = (
        f"[0:v]subtitles=captions.srt:force_style='{style}',format=yuv420p[v];"
        f"[2:a]atrim=0:{dur:.2f},asetpts=PTS-STARTPTS,volume=0.22,afade=t=in:d=3,afade=t=out:st={fade_out:.2f}:d=4,aformat=channel_layouts=stereo[m];"
        "[1:a]aformat=channel_layouts=stereo,asplit=2[vo][sc];"
        "[m][sc]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=600[duck];"
        "[vo][duck]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]"
    )
    cmd = [ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", "raw.webm", "-i", "voice.wav", "-i", str(MUSIC),
           "-filter_complex", filt, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "22",
           "-r", "25", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-t", f"{dur:.2f}", str(OUT / "demo.mp4")]
    subprocess.run(cmd, cwd=V, check=True)
    subprocess.run([ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-ss", "2.5", "-i", str(OUT / "demo.mp4"),
                    "-frames:v", "1", str(OUT / "demo-poster.png")], check=True)
    info = {
        "file": "site/media/demo.mp4", "duration_s": round(probe_duration(OUT / "demo.mp4"), 1),
        "sha256": hashlib.sha256((OUT / "demo.mp4").read_bytes()).hexdigest(), "captions": len(cues),
        "source": "screen recording of the local application (Playwright, Chromium) — not a mock-up",
        "narration": "synthetic, Kokoro-82M (Apache-2.0), voice af_heart",
        "music": {"track": "Dreamer", "creator": "Kevin MacLeod", "source": "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Dreamer.mp3",
                  "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/", "sha256_of_source": MUSIC_SHA,
                  "changes": "excerpt from the start, trimmed to video length, volume 0.22, side-chain ducked under narration, 3 s fade-in, 4 s fade-out, AAC"},
        "inputs": "demo scenarios s1 (flood report + FEMA flood photo) and s2 (fire report + same photo) — illustrative, see demo/CREDITS.md",
    }
    (OUT / "demo.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
