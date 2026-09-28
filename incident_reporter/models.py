"""Model suite: speech transcription, image analysis and the three classification views.

`ModelSuite` loads the real local models lazily. Tests substitute a stub with the same methods.
Every model output passes through a validator here; anything malformed raises `ModelOutputError`,
which the workflow turns into a NEEDS_REVIEW flag instead of a result.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .concepts import BACKGROUND_PROMPTS, CONCEPTS, LIKELY, POSSIBLE
from .labels import LABELS

log = logging.getLogger("vvir")

# ASR confidence thresholds (faster-whisper segment statistics; Whisper's own defaults use
# avg_logprob < -1.0 and no_speech_prob > 0.6 to detect failed decodes — we flag a little earlier).
ASR_LOW_LOGPROB = -0.8
ASR_HIGH_NO_SPEECH = 0.5
VIEW_LOW_CONFIDENCE = 0.50
VIEW_CONFIDENT = 0.55


class ModelOutputError(RuntimeError):
    """A model returned something the application must not use."""


# ---------------------------------------------------------------- validators


def validate_transcript(out) -> dict:
    if not isinstance(out, dict) or not isinstance(out.get("text"), str):
        raise ModelOutputError("transcriber returned no text field")
    segs = out.get("segments")
    if not isinstance(segs, list):
        raise ModelOutputError("transcriber returned no segment list")
    for s in segs:
        for k in ("start", "end", "avg_logprob", "no_speech_prob"):
            v = s.get(k)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ModelOutputError(f"segment field {k} missing or not finite")
    return out


def validate_proba(p, view: str) -> np.ndarray:
    try:
        a = np.asarray(p, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as e:
        raise ModelOutputError(f"{view}: probabilities are not numeric") from e
    if a.shape != (len(LABELS),):
        raise ModelOutputError(f"{view}: expected {len(LABELS)} probabilities, got {a.shape}")
    if not np.isfinite(a).all() or (a < -1e-6).any() or abs(a.sum() - 1.0) > 1e-3:
        raise ModelOutputError(f"{view}: probabilities are not a valid distribution")
    return a


def validate_observations(obs) -> list[dict]:
    if not isinstance(obs, list):
        raise ModelOutputError("image analyzer returned no observation list")
    known = {c.id for c in CONCEPTS}
    for o in obs:
        if o.get("concept") not in known:
            raise ModelOutputError("image analyzer returned an unknown concept")
        s = o.get("score")
        if not isinstance(s, (int, float)) or not (0.0 <= s <= 1.0) or not math.isfinite(s):
            raise ModelOutputError("image observation score out of range")
    return obs


# Observation score: margin of the best concept prompt over the best "ordinary scene" prompt, in cosine
# units, squashed with a logistic curve (margin 0.04 -> 0.5). The curve's centre and width were set by hand on
# the demo photos and checked for ranking quality on CrisisMMD dev (experiments/observation_check.py). It is a
# display score, not a calibrated probability.
OBS_CENTER = 0.04
OBS_WIDTH = 0.02


def observation_scores(image_vec: np.ndarray, prompt_emb: np.ndarray) -> list[dict]:
    """prompt_emb rows: every concept prompt in CONCEPTS order, then BACKGROUND_PROMPTS."""
    sims = prompt_emb @ np.asarray(image_vec, dtype=np.float32).reshape(-1)
    bg = float(sims[-len(BACKGROUND_PROMPTS):].max())
    out, k = [], 0
    for c in CONCEPTS:
        if not c.prompts:
            continue
        j = int(np.argmax(sims[k:k + len(c.prompts)]))
        margin = float(sims[k + j]) - bg
        score = 1.0 / (1.0 + math.exp(-(margin - OBS_CENTER) / OBS_WIDTH))
        out.append({"concept": c.id, "label": c.label, "score": round(score, 3), "margin": round(margin, 4),
                    "band": band(score), "prompt": c.prompts[j]})
        k += len(c.prompts)
    return out


def band(score: float) -> str:
    return "likely" if score >= LIKELY else "possible" if score >= POSSIBLE else "not_detected"


def image_quality(img: Image.Image) -> dict:
    """Brightness, contrast and blur (variance of a Laplacian) on a 512-px greyscale copy."""
    g = img.convert("L")
    g.thumbnail((512, 512))
    a = np.asarray(g, dtype=np.float32)
    lap = a[1:-1, 1:-1] * -4 + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    q = {
        "brightness": round(float(a.mean()), 1),
        "contrast": round(float(a.std()), 1),
        "sharpness": round(float(lap.var()), 1),
    }
    flags = []
    if q["brightness"] < 40:
        flags.append("too_dark")
    if q["brightness"] > 225:
        flags.append("overexposed")
    if q["contrast"] < 18:
        flags.append("low_contrast")
    if q["sharpness"] < 40:
        flags.append("blurry")
    if min(img.size) < 224:
        flags.append("low_resolution")
    q["flags"] = flags
    return q


# ---------------------------------------------------------------- real models


@dataclass
class ViewResult:
    view: str
    proba: list[float]
    label: str
    confidence: float
    available: bool = True
    reason: str | None = None


class ModelSuite:
    """Local models: faster-whisper (ASR) + CLIP ViT-B/32 (embeddings, observations, heads)."""

    def __init__(self, settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._asr = None
        self._clip = None
        self._heads = None
        self._obs_text = None

    # lazy loading -------------------------------------------------------
    def _load_clip(self):
        from .ml.clip_backbone import ClipBackbone
        from .ml.heads import HeadSet, zero_shot_headset

        with self._lock:
            if self._clip is None:
                self._clip = ClipBackbone(cache_dir=str(self.settings.model_cache))
                path = self.settings.heads_path
                if path.exists():
                    self._heads = HeadSet.load(path)
                    log.info("heads_loaded", extra={"kind": "trained"})
                else:
                    self._heads = zero_shot_headset(self._clip)
                    log.info("heads_loaded", extra={"kind": "zero_shot"})
                prompts = [p for c in CONCEPTS for p in c.prompts] + list(BACKGROUND_PROMPTS)
                self._obs_text = self._clip.embed_texts(prompts)
        return self._clip

    def _load_asr(self):
        from faster_whisper import WhisperModel

        from .ml.clip_backbone import maybe_use_os_truststore

        with self._lock:
            if self._asr is None:
                maybe_use_os_truststore()
                self._asr = WhisperModel(
                    self.settings.asr_model, device="cpu", compute_type="int8",
                    download_root=str(self.settings.model_cache),
                    local_files_only=os.environ.get("VVIR_OFFLINE") == "1",
                )
        return self._asr

    def info(self) -> dict:
        return {
            "asr": f"faster-whisper {self.settings.asr_model} (int8, CPU)",
            "clip": "openai/clip-vit-base-patch32 (frozen)",
            "heads": self._heads.kind if self._heads else "not loaded",
            "heads_meta": (self._heads.meta if self._heads else {}),
            "loaded": {"asr": self._asr is not None, "clip": self._clip is not None},
        }

    def warmup(self) -> None:
        self._load_clip()
        self._load_asr()

    # inference ----------------------------------------------------------
    def transcribe(self, samples: np.ndarray) -> dict:
        model = self._load_asr()
        segments, info = model.transcribe(samples, language="en", beam_size=5, vad_filter=False, condition_on_previous_text=False)
        segs = [{
            "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
            "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3),
            "compression_ratio": round(s.compression_ratio, 2),
        } for s in segments]
        return {"text": " ".join(s["text"] for s in segs).strip(), "segments": segs, "model": f"faster-whisper {self.settings.asr_model}"}

    def embed_text(self, text: str) -> np.ndarray:
        return self._load_clip().embed_texts([text])

    def embed_image(self, img: Image.Image) -> np.ndarray:
        return self._load_clip().embed_images([img])

    def observations(self, image_emb: np.ndarray) -> list[dict]:
        self._load_clip()
        return observation_scores(image_emb[0], self._obs_text)

    def classify(self, view: str, text_emb, image_emb) -> list[float]:
        self._load_clip()
        return self._heads.predict(view, text_emb, image_emb).tolist()

    @property
    def heads_kind(self) -> str:
        return self._heads.kind if self._heads else "unknown"
