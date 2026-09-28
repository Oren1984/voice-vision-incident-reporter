"""Shared fixtures: a stub model suite, synthetic media and an isolated app instance per test."""

from __future__ import annotations

import io
import math
import struct
import wave
from dataclasses import dataclass, field

import numpy as np
import pytest
from PIL import Image, ImageDraw

from incident_reporter.concepts import CONCEPTS
from incident_reporter.config import Limits, Settings
from incident_reporter.db import Database
from incident_reporter.labels import LABELS
from incident_reporter.media import MediaStore
from incident_reporter.observability import Metrics
from incident_reporter.security import create_user
from incident_reporter.workflow import IncidentService

PW = "correct-horse-battery"


def onehot(label: str, p: float = 0.9) -> list[float]:
    rest = (1 - p) / (len(LABELS) - 1)
    return [p if l == label else rest for l in LABELS]


@dataclass
class StubModels:
    """Same methods as ModelSuite. Tests set attributes to shape the outputs or inject faults."""

    transcript: str = "There is a fire on Main Street. Two people are hurt and smoke is still spreading since 10 am."
    asr_logprob: float = -0.2
    asr_no_speech: float = 0.01
    obs: dict = field(default_factory=lambda: {"fire_smoke": 0.9})
    proba: dict = field(default_factory=lambda: {v: onehot("infrastructure_and_utility_damage") for v in ("text", "image", "combined")})
    fail: set = field(default_factory=set)  # {"asr", "embed_image", "observations", "classify:text", ...}
    raw_transcribe: dict | None = None  # return this dict verbatim (invalid output tests)
    heads_kind: str = "stub"
    calls: list = field(default_factory=list)

    def transcribe(self, samples):
        self.calls.append("transcribe")
        if "asr" in self.fail:
            raise RuntimeError("asr crashed")
        if self.raw_transcribe is not None:
            return self.raw_transcribe
        return {"text": self.transcript, "model": "stub-asr", "segments": [] if not self.transcript else [
            {"start": 0.0, "end": 3.0, "text": self.transcript, "avg_logprob": self.asr_logprob, "no_speech_prob": self.asr_no_speech, "compression_ratio": 1.2}]}

    def embed_text(self, text):
        if "embed_text" in self.fail:
            raise RuntimeError("text encoder crashed")
        v = np.ones((1, 512), np.float32)
        return v / np.linalg.norm(v)

    def embed_image(self, img):
        if "embed_image" in self.fail:
            raise RuntimeError("image encoder crashed")
        v = np.full((1, 512), 0.5, np.float32)
        return v / np.linalg.norm(v)

    def observations(self, emb):
        if "observations" in self.fail:
            return [{"concept": "fire_smoke", "label": "x", "score": 3.0, "band": "likely"}]
        return [{"concept": c.id, "label": c.label, "score": float(self.obs.get(c.id, 0.05)), "band": "?", "prompt": ""} for c in CONCEPTS if c.prompts]

    def classify(self, view, text_emb, image_emb):
        if f"classify:{view}" in self.fail:
            return [float("nan")] * len(LABELS)
        if f"shape:{view}" in self.fail:
            return [0.5, 0.5]
        return self.proba[view]

    def info(self):
        return {"asr": "stub", "clip": "stub", "heads": "stub", "loaded": {"asr": True, "clip": True}}

    def warmup(self):
        pass


def make_wav(seconds: float = 3.0, freq: float = 220.0, rate: int = 16000, amp: float = 0.3) -> bytes:
    n = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(struct.pack("<h", int(amp * 32767 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n))
        w.writeframes(frames)
    return buf.getvalue()


def make_image(size=(640, 480), fmt="JPEG", exif_gps: bool = False, sharp: bool = True, brightness: int = 140) -> bytes:
    im = Image.new("RGB", size, (brightness, brightness - 20, brightness - 40))
    if sharp:
        d = ImageDraw.Draw(im)
        for x in range(0, size[0], 16):
            d.line([(x, 0), (x, size[1])], fill=(20, 20, 20), width=3)
        for y in range(0, size[1], 24):
            d.line([(0, y), (size[0], y)], fill=(240, 240, 240), width=2)
    buf = io.BytesIO()
    kw = {}
    if exif_gps:
        ex = Image.Exif()
        ex[0x010F] = "TestCamera"  # Make
        ex[0x8825] = {1: "N", 2: (32.0, 5.0, 0.0), 3: "E", 4: (34.0, 46.0, 0.0)}  # GPS IFD
        kw["exif"] = ex
    im.save(buf, format=fmt, **kw)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def fast_password_hashing(monkeypatch):
    """Production uses 600k PBKDF2 iterations; tests use fewer so the suite stays fast."""
    import incident_reporter.security as sec

    monkeypatch.setattr(sec, "ITERATIONS", 1000)


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "var", heads_path=tmp_path / "none.npz", limits=Limits())


@pytest.fixture
def stub():
    return StubModels()


@pytest.fixture
def env(settings, stub):
    db = Database(settings.db_path)
    svc = IncidentService(db, MediaStore(settings.media_dir), stub, settings, Metrics())
    users = {
        "reporter": create_user(db, "rita.reporter", "reporter", PW),
        "reporter2": create_user(db, "otto.reporter", "reporter", PW),
        "reviewer": create_user(db, "vera.reviewer", "reviewer", PW),
        "reviewer2": create_user(db, "rene.reviewer", "reviewer", PW),
    }
    yield svc, db, users, stub
    db.close()
