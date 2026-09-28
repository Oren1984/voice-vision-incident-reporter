"""Step 2: frozen CLIP ViT-B/32 embeddings for every example (protocol § 3).

Writes .cache/work/features.npz (confidential, per-example). Usage: python -m experiments.extract_features
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from incident_reporter.ml.clip_backbone import ClipBackbone

from .common import HF_CACHE, IMAGE_ROOT, WORK, load_examples

Image.MAX_IMAGE_PIXELS = 60_000_000


def _load(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def main(batch: int = 64) -> None:
    import pandas as pd

    data: pd.DataFrame = load_examples()
    bb = ClipBackbone(cache_dir=str(HF_CACHE))
    t0 = time.time()
    text = bb.embed_texts(data["text_clean"].tolist())
    print(f"text embeddings {text.shape} in {time.time() - t0:.0f}s", flush=True)
    img = np.zeros((len(data), 512), np.float32)
    ok = np.zeros(len(data), bool)
    paths = data["image"].tolist()
    t0 = time.time()
    for i in range(0, len(paths), batch):
        ims, idx = [], []
        for j in range(i, min(i + batch, len(paths))):
            try:
                ims.append(_load(IMAGE_ROOT / paths[j]))
                idx.append(j)
            except Exception:
                pass  # counted via ok=False and reported by train_eval
        if ims:
            img[idx] = bb.embed_images(ims, batch_size=batch)
            ok[idx] = True
        if (i // batch) % 20 == 0:
            print(f"images {i + len(ims)}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    np.savez(WORK / "features.npz", text=text.astype(np.float32), image=img, image_ok=ok,
             image_id=np.array(data["image_id"].tolist()))
    print(f"done: {ok.sum()} images embedded, {len(ok) - ok.sum()} failed")


if __name__ == "__main__":
    main()
