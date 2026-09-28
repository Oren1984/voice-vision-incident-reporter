"""Frozen CLIP ViT-B/32 embeddings — the single backbone used by the experiment and the app."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def maybe_use_os_truststore() -> None:
    """Opt-in (VVIR_USE_OS_TRUSTSTORE=1): verify TLS against the OS certificate store.

    Needed only on machines whose antivirus inspects TLS with a root CA that Python's bundled
    store does not know. Certificates are still verified; nothing is disabled.
    """
    if os.environ.get("VVIR_USE_OS_TRUSTSTORE") == "1":
        import truststore

        truststore.inject_into_ssl()


class ClipBackbone:
    def __init__(self, cache_dir: str | None = None, device: str = "cpu") -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        maybe_use_os_truststore()
        kw = dict(revision=CLIP_REVISION, cache_dir=cache_dir)
        if os.environ.get("VVIR_OFFLINE") == "1":
            kw["local_files_only"] = True
        self._torch = torch
        torch.manual_seed(0)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL_ID, **kw).eval().to(device)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, **kw)
        self.device = device

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)

    @staticmethod
    def _as_array(out) -> np.ndarray:
        # transformers >= 5 may return a ModelOutput; the projected embedding is pooler_output.
        if hasattr(out, "pooler_output") and not hasattr(out, "shape"):
            out = out.pooler_output
        return out.float().cpu().numpy()

    def embed_texts(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        outs = []
        with self._torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = [t if t.strip() else "." for t in texts[i : i + batch_size]]
                inp = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77)
                inp = {k: v.to(self.device) for k, v in inp.items()}
                outs.append(self._as_array(self.model.get_text_features(**inp)))
        return self._unit(np.concatenate(outs)) if outs else np.zeros((0, 512), np.float32)

    def embed_images(self, images: Sequence, batch_size: int = 32) -> np.ndarray:
        """images: PIL.Image objects (RGB)."""
        outs = []
        with self._torch.inference_mode():
            for i in range(0, len(images), batch_size):
                inp = self.processor(images=list(images[i : i + batch_size]), return_tensors="pt")
                outs.append(self._as_array(self.model.get_image_features(pixel_values=inp["pixel_values"].to(self.device))))
        return self._unit(np.concatenate(outs)) if outs else np.zeros((0, 512), np.float32)

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.exp().item())
