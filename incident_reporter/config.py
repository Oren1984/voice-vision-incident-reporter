"""Runtime settings, read from environment variables (prefix VVIR_). No secrets have defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"VVIR_{name}", default)


@dataclass(frozen=True)
class Limits:
    audio_max_bytes: int = 15 * 1024 * 1024
    audio_min_seconds: float = 1.0
    audio_max_seconds: float = 120.0
    image_max_bytes: int = 10 * 1024 * 1024
    image_max_pixels: int = 40_000_000
    image_min_side: int = 64
    transcript_max_chars: int = 5000
    text_field_max_chars: int = 4000


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "var"))))
    model_cache: Path = field(default_factory=lambda: Path(_env("MODEL_CACHE", str(ROOT / ".cache" / "hf" / "hub"))))
    heads_path: Path = field(default_factory=lambda: Path(_env("HEADS", str(ROOT / "artifacts" / "heads" / "crisismmd_heads.npz"))))
    asr_model: str = field(default_factory=lambda: _env("ASR_MODEL", "base.en"))
    session_hours: float = field(default_factory=lambda: float(_env("SESSION_HOURS", "8")))
    secure_cookies: bool = field(default_factory=lambda: _env("SECURE_COOKIES", "0") == "1")
    four_eyes: bool = field(default_factory=lambda: _env("FOUR_EYES", "1") == "1")
    limits: Limits = field(default_factory=Limits)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "incidents.sqlite3"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "logs" / "app.jsonl"
