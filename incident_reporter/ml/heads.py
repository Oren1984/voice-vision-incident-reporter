"""Classifier heads over CLIP embeddings, stored as plain arrays (.npz, no pickle).

A head is softmax(x @ W.T + b). The experiment exports its selected logistic-regression models in
this form, and the zero-shot fallback builds W from prompt embeddings, so the application runs the
same arithmetic whichever head is loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..labels import LABELS, ZERO_SHOT_PROMPTS

VIEWS = ("text", "image", "combined")


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class LinearHead:
    W: np.ndarray  # (n_classes, dim)
    b: np.ndarray  # (n_classes,)

    def proba(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return softmax(x @ self.W.T + self.b)


@dataclass
class HeadSet:
    """Text-only, image-only and combined heads plus provenance metadata."""

    kind: str  # "trained" | "zero_shot"
    heads: dict[str, LinearHead]
    meta: dict = field(default_factory=dict)
    combine_mean: bool = False  # zero-shot combined view = mean of text and image probabilities

    def predict(self, view: str, text_emb: np.ndarray | None, image_emb: np.ndarray | None) -> np.ndarray:
        if view == "text":
            return self.heads["text"].proba(text_emb)[0]
        if view == "image":
            return self.heads["image"].proba(image_emb)[0]
        if view == "combined":
            if self.combine_mean:
                return 0.5 * (self.heads["text"].proba(text_emb)[0] + self.heads["image"].proba(image_emb)[0])
            return self.heads["combined"].proba(np.concatenate([text_emb, image_emb], axis=-1))[0]
        raise ValueError(f"unknown view {view!r}")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for v, h in self.heads.items():
            arrays[f"{v}_W"], arrays[f"{v}_b"] = h.W.astype(np.float32), h.b.astype(np.float32)
        arrays["meta"] = np.array(json.dumps({"kind": self.kind, "labels": list(LABELS), **self.meta}))
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "HeadSet":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("labels") != list(LABELS):
                raise ValueError("head file labels do not match this build")
            heads = {v: LinearHead(z[f"{v}_W"], z[f"{v}_b"]) for v in VIEWS if f"{v}_W" in z}
        if set(heads) != set(VIEWS):
            raise ValueError("head file must contain text, image and combined heads")
        return cls(kind=meta.pop("kind"), heads=heads, meta=meta)


def zero_shot_headset(backbone) -> HeadSet:
    """Protocol S2: prompt-embedding classifier, no training data needed."""
    W = []
    for label in LABELS:
        e = backbone.embed_texts(list(ZERO_SHOT_PROMPTS[label])).mean(axis=0)
        W.append(e / np.linalg.norm(e))
    W = np.stack(W) * backbone.logit_scale
    b = np.zeros(len(LABELS), np.float32)
    head = LinearHead(W, b)
    return HeadSet(
        kind="zero_shot",
        heads={"text": head, "image": head, "combined": head},
        meta={"source": "CLIP zero-shot prompts (labels.ZERO_SHOT_PROMPTS)"},
        combine_mean=True,
    )
