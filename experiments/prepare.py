"""Step 1: verify CrisisMMD files, build the example table, run leakage checks.

Writes per-example data to .cache/work/ (confidential) and aggregate checks to results/data_checks.json.
Usage: python -m experiments.prepare
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

from incident_reporter.labels import LABELS
from incident_reporter.text_clean import clean_post

from .common import IMAGE_ROOT, RESULTS, SPLITS, WORK, load_split, sha256_file, split_file, write_json

Image.MAX_IMAGE_PIXELS = 60_000_000


def norm_text(t: str) -> str:
    t = clean_post(t).lower().replace("@user", "")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return " ".join(t.split())


def ahash(path) -> int:
    """64-bit average hash (8x8 greyscale)."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32)
    bits = (g > g.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def main() -> None:
    import pandas as pd

    WORK.mkdir(parents=True, exist_ok=True)
    frames, checks = [], {"files": {}, "splits": {}}
    for s in SPLITS:
        df = load_split(s)
        checks["files"][split_file(s).name] = sha256_file(split_file(s))
        assert set(df["label"]) <= set(LABELS), f"unexpected labels in {s}"
        assert (df["label_text"] == df["label_image"]).all(), "agreed split must have equal text/image labels"
        df["split"] = s
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["text_clean"] = data["tweet_text"].map(clean_post)
    data["text_norm"] = data["tweet_text"].map(norm_text)

    missing, unreadable, hashes = [], [], []
    for p in data["image"]:
        path = IMAGE_ROOT / p
        if not path.exists():
            missing.append(p)
            hashes.append(-1)
            continue
        try:
            hashes.append(ahash(path))
        except Exception:  # corrupt image: excluded from nothing, just reported
            unreadable.append(p)
            hashes.append(-1)
    data["ahash"] = hashes

    for s in SPLITS:
        d = data[data.split == s]
        checks["splits"][s] = {
            "rows": int(len(d)),
            "distinct_tweets": int(d.tweet_id.nunique()),
            "class_counts": {k: int(v) for k, v in d.label.value_counts().reindex(LABELS, fill_value=0).items()},
            "event_counts": {k: int(v) for k, v in d.event_name.value_counts().items()},
        }
    checks["missing_images"] = len(missing)
    checks["unreadable_images"] = len(unreadable)

    # Leakage checks against train
    tr = data[data.split == "train"]
    tr_ids, tr_imgs, tr_text = set(tr.tweet_id), set(tr.image), set(tr.text_norm)
    tr_hash = np.array([h for h in tr.ahash if h >= 0], dtype=np.uint64)
    leak = {}
    near_dup_flags = np.zeros(len(data), dtype=bool)
    for s in ("dev", "test"):
        idx = data.index[data.split == s]
        d = data.loc[idx]
        text_dup = d.text_norm.isin(tr_text).to_numpy()
        img_dup = np.zeros(len(d), dtype=bool)
        for k, h in enumerate(d.ahash):
            if h < 0:
                continue
            dist = _popcount(np.bitwise_xor(tr_hash, np.uint64(h)))
            img_dup[k] = bool((dist <= 4).any())
        near_dup_flags[idx] = text_dup | img_dup
        leak[s] = {
            "tweet_id_overlap": int(len(set(d.tweet_id) & tr_ids)),
            "image_path_overlap": int(len(set(d.image) & tr_imgs)),
            "normalised_text_duplicates_of_train": int(text_dup.sum()),
            "near_duplicate_images_of_train_ahash_le4": int(img_dup.sum()),
            "excluded_in_sensitivity_subset": int((text_dup | img_dup).sum()),
        }
    data["near_dup_of_train"] = near_dup_flags
    data["ahash"] = [format(h, "016x") if h >= 0 else "" for h in hashes]  # 64-bit values do not fit int64 JSON
    checks["leakage_vs_train"] = leak

    data.to_json(WORK / "examples.jsonl", orient="records", lines=True)  # local cache only
    write_json(RESULTS / "data_checks.json", checks)
    print(f"examples: {len(data)} | missing images: {len(missing)} | unreadable: {len(unreadable)}")
    print("leakage:", leak)


def _popcount(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    c = np.zeros(x.shape, dtype=np.uint64)
    for _ in range(64):
        c += x & np.uint64(1)
        x = x >> np.uint64(1)
    return c


if __name__ == "__main__":
    main()
