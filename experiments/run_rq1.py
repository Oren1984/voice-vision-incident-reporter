"""Step 3: RQ1 — text only vs image only vs combined, on identical test examples.

Two stages, enforced in this order (protocol § 4):
  python -m experiments.run_rq1 select   # model selection on dev only -> results/rq1_selection.json
  python -m experiments.run_rq1 test     # single scoring of the frozen models on test
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from incident_reporter.labels import LABELS
from incident_reporter.ml.heads import HeadSet, LinearHead, zero_shot_headset

from .common import ARTIFACTS, HF_CACHE, RESULTS, SEED, WORK, environment, load_examples, sha256_file, write_json
from .metrics import bootstrap_diffs, macro_f1, mcnemar_exact, summary

C_GRID = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)
CW_GRID = (None, "balanced")
VIEWS = ("text", "image", "combined")
SELECTION = RESULTS / "rq1_selection.json"
HEADS_PATH = ARTIFACTS / "heads" / "crisismmd_heads.npz"


def load():
    import pandas as pd

    data = load_examples()
    z = np.load(WORK / "features.npz")
    assert (z["image_id"] == data["image_id"].to_numpy()).all(), "features out of sync with examples"
    if not z["image_ok"].all():
        raise SystemExit(f"{(~z['image_ok']).sum()} images failed to embed; fix before evaluation")
    y = data["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    X = {"text": z["text"], "image": z["image"], "combined": np.concatenate([z["text"], z["image"]], 1)}
    return data, X, y


def fit(X, y, C, cw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return LogisticRegression(C=C, class_weight=cw, max_iter=5000, random_state=SEED).fit(X, y)


def zero_shot_probas(text_emb: np.ndarray, image_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Protocol S2: prompt-embedding classifier (loads CLIP for the prompt embeddings)."""
    from incident_reporter.ml.clip_backbone import ClipBackbone

    zs = zero_shot_headset(ClipBackbone(cache_dir=str(HF_CACHE)))
    return zs.heads["text"].proba(text_emb), zs.heads["image"].proba(image_emb)


def select() -> None:
    data, X, y = load()
    tr, dv = (data.split == "train").to_numpy(), (data.split == "dev").to_numpy()
    out = {"protocol": "docs/experiment-design.md v1", "grid": {"C": C_GRID, "class_weight": [str(c) for c in CW_GRID]}, "views": {}}
    dev_proba = {}
    for v in VIEWS:
        scores = []
        best = None
        for cw in CW_GRID:
            for C in C_GRID:
                m = fit(X[v][tr], y[tr], C, cw)
                p = m.predict_proba(X[v][dv])
                f = macro_f1(y[dv], p.argmax(1))
                scores.append({"C": C, "class_weight": str(cw), "dev_macro_f1": f})
                # ties -> smaller C, then None (grid order already encodes None first per C sweep)
                if best is None or f > best[0] + 1e-12 or (abs(f - best[0]) <= 1e-12 and C < best[1]):
                    best = (f, C, cw, p)
        out["views"][v] = {"selected": {"C": best[1], "class_weight": str(best[2]), "dev_macro_f1": best[0]}, "all": scores}
        dev_proba[v] = best[3]
        print(f"{v:9s} selected C={best[1]} cw={best[2]} dev MacroF1={best[0]:.4f}")
    # S1 late-fusion weight on dev
    ws = []
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        p = w * dev_proba["text"] + (1 - w) * dev_proba["image"]
        ws.append({"w_text": float(w), "dev_macro_f1": macro_f1(y[dv], p.argmax(1))})
    bw = max(ws, key=lambda r: (r["dev_macro_f1"], -abs(r["w_text"] - 0.5)))
    out["late_fusion"] = {"selected_w_text": bw["w_text"], "dev_macro_f1": bw["dev_macro_f1"], "all": ws}
    out["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(SELECTION, out)
    print(f"late fusion w_text={bw['w_text']} dev MacroF1={bw['dev_macro_f1']:.4f} -> {SELECTION}")


def test() -> None:
    if not SELECTION.exists():
        raise SystemExit("run `select` first: the test split is only scored after selection is frozen")
    sel = json.loads(SELECTION.read_text())
    data, X, y = load()
    tr, te = (data.split == "train").to_numpy(), (data.split == "test").to_numpy()
    yt = y[te]
    proba, models = {}, {}
    for v in VIEWS:
        s = sel["views"][v]["selected"]
        cw = None if s["class_weight"] == "None" else s["class_weight"]
        models[v] = fit(X[v][tr], y[tr], s["C"], cw)
        proba[v] = models[v].predict_proba(X[v][te])
    w = sel["late_fusion"]["selected_w_text"]
    proba["late_fusion"] = w * proba["text"] + (1 - w) * proba["image"]

    # S2 zero-shot (no training)
    zt, zi = zero_shot_probas(X["text"][te], X["image"][te])
    proba.update({"zero_shot_text": zt, "zero_shot_image": zi, "zero_shot_combined": 0.5 * (zt + zi)})

    pred = {k: p.argmax(1) for k, p in proba.items()}
    res = {k: summary(yt, p) for k, p in proba.items()}
    f = {k: r["macro_f1"] for k, r in res.items()}
    best_uni = "text" if f["text"] >= f["image"] else "image"

    boot = bootstrap_diffs(
        yt,
        {k: pred[k] for k in ("text", "image", "combined", "late_fusion")},
        {
            "combined_minus_best_unimodal": ("combined", ("max", "text", "image")),
            "combined_minus_text": ("combined", "text"),
            "combined_minus_image": ("combined", "image"),
            "late_fusion_minus_best_unimodal": ("late_fusion", ("max", "text", "image")),
        },
    )
    h1 = boot["combined_minus_best_unimodal"]
    verdict = (
        "SUCCESS: combined > best single source, 95% CI above 0" if h1["point"] > 0 and h1["ci95"][0] > 0
        else "NOT BETTER: combined <= best single source" if h1["point"] <= 0
        else "NO RELIABLE IMPROVEMENT: point estimate > 0 but 95% CI includes 0"
    )

    # Per-event Macro F1 and the leakage-sensitivity subset
    ev = data.loc[te, "event_name"].to_numpy()
    by_event = {e: {"n": int((ev == e).sum()), **{v: macro_f1(yt[ev == e], pred[v][ev == e]) for v in VIEWS}} for e in sorted(set(ev))}
    keep = ~data.loc[te, "near_dup_of_train"].to_numpy()
    sens = {"n": int(keep.sum()), "excluded": int((~keep).sum()), **{v: macro_f1(yt[keep], pred[v][keep]) for v in VIEWS}}
    sens["combined_minus_best_unimodal"] = sens["combined"] - max(sens["text"], sens["image"])

    # Agreement structure between single-source views (feeds the app's comparison design)
    agree = {
        "text_right_image_right": int(((pred["text"] == yt) & (pred["image"] == yt)).sum()),
        "text_right_image_wrong": int(((pred["text"] == yt) & (pred["image"] != yt)).sum()),
        "text_wrong_image_right": int(((pred["text"] != yt) & (pred["image"] == yt)).sum()),
        "both_wrong": int(((pred["text"] != yt) & (pred["image"] != yt)).sum()),
        "views_disagree": int((pred["text"] != pred["image"]).sum()),
        "accuracy_when_views_agree": float((pred["combined"][pred["text"] == pred["image"]] == yt[pred["text"] == pred["image"]]).mean()),
        "accuracy_when_views_disagree": float((pred["combined"][pred["text"] != pred["image"]] == yt[pred["text"] != pred["image"]]).mean()),
    }

    out = {
        "protocol": "docs/experiment-design.md v1",
        "labels": list(LABELS),
        "test_n": int(te.sum()),
        "test_class_counts": np.bincount(yt, minlength=len(LABELS)).tolist(),
        "results": res,
        "h1": {"best_unimodal_on_test": best_uni, **h1, "verdict": verdict},
        "bootstrap": boot,
        "mcnemar_combined_vs_best_unimodal": mcnemar_exact(yt, pred["combined"], pred[best_uni]),
        "by_event": by_event,
        "sensitivity_no_near_duplicates": sens,
        "view_agreement": agree,
        "selection_file_sha256": sha256_file(SELECTION),
    }
    write_json(RESULTS / "rq1_results.json", out)

    # Export the selected heads for the application (derived from CrisisMMD -> kept out of git)
    heads = {v: LinearHead(models[v].coef_.astype(np.float32), models[v].intercept_.astype(np.float32)) for v in VIEWS}
    HeadSet(kind="trained", heads=heads, meta={
        "source": "CrisisMMD v2.0 agreed-label humanitarian train split, logistic regression on frozen CLIP ViT-B/32",
        "selection": {v: sel["views"][v]["selected"] for v in VIEWS},
        "test_macro_f1": {v: f[v] for v in VIEWS},
    }).save(HEADS_PATH)

    manifest_path = RESULTS / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "rq1": {"completed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed": SEED, "bootstrap_seed": 2026},
        "clip": {"model": "openai/clip-vit-base-patch32", "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"},
        "environment": environment(),
        "commands": [
            "python -m experiments.prepare",
            "python -m experiments.extract_features",
            "python -m experiments.run_rq1 select",
            "python -m experiments.run_rq1 test",
        ],
    })
    write_json(manifest_path, manifest)
    for k in ("text", "image", "combined", "late_fusion", "zero_shot_text", "zero_shot_image", "zero_shot_combined"):
        r = res[k]
        print(f"{k:20s} MacroF1={r['macro_f1']:.4f} acc={r['accuracy']:.4f} wF1={r['weighted_f1']:.4f}")
    print("H1:", verdict, h1["point"], h1["ci95"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("select", "test"))
    a = ap.parse_args()
    select() if a.stage == "select" else test()
