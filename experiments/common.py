"""Paths and data loading for the CrisisMMD experiment (protocol: docs/experiment-design.md)."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
CRISISMMD = CACHE / "crisismmd"
SPLIT_DIR = CRISISMMD / "crisismmd_datasplit_agreed_label"
IMAGE_ROOT = CRISISMMD / "CrisisMMD_v2.0"
WORK = CACHE / "work"  # per-example derived data: confidential, never committed
ARTIFACTS = ROOT / "artifacts"  # trained heads: derived from CrisisMMD, never committed
RESULTS = ROOT / "results"  # aggregate results only: committed
HF_CACHE = CACHE / "hf" / "hub"
SPLITS = ("train", "dev", "test")
SEED = 13
BOOTSTRAP_SEED = 2026


def split_file(split: str) -> Path:
    return SPLIT_DIR / f"task_humanitarian_text_img_agreed_lab_{split}.tsv"


def load_split(split: str):
    import pandas as pd

    df = pd.read_csv(split_file(split), sep="\t", dtype={"tweet_id": str, "image_id": str})
    return df


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=_default) + "\n", encoding="utf-8")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def environment() -> dict:
    import importlib.metadata as md

    pkgs = {}
    for p in ("numpy", "scikit-learn", "torch", "transformers", "faster-whisper", "ctranslate2", "pillow", "kokoro-onnx", "onnxruntime", "pandas", "scipy"):
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pkgs}


def load_examples():
    """The example table written by experiments.prepare (JSON Lines, no pickle)."""
    import pandas as pd

    return pd.read_json(WORK / "examples.jsonl", orient="records", lines=True, dtype={"tweet_id": str, "image_id": str})
