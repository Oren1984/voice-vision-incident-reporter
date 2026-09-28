"""Synthesise the demo reports in demo/scenarios.json with Kokoro-82M (Apache-2.0).

Output: demo/audio/*.wav (16 kHz mono). The noisy variant adds white noise at the stated SNR with a
fixed seed, so the files are reproducible. Requires .cache/kokoro/ model files (see docs/demo-guide.md).
Usage: python tools/make_demo_media.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.speech_eval import TTS, add_noise, save_wav  # noqa: E402


def main() -> None:
    spec = json.loads((ROOT / "demo" / "scenarios.json").read_text(encoding="utf-8"))
    tts = TTS()
    for s in spec["scenarios"]:
        x = tts.speak(s["text"], s["voice"])
        x = add_noise(x, s.get("noise_snr_db"), seed=2026)
        out = ROOT / "demo" / s["audio"]
        save_wav(out, x)
        print(f"{out.relative_to(ROOT)}  {len(x) / 16000:.1f} s")


if __name__ == "__main__":
    main()
