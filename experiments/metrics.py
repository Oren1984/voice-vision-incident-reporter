"""Metrics used by the experiment: Macro F1, confusion matrix, bootstrap, McNemar, ECE."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

K = 5
BOOTSTRAP_N = 10_000  # protocol � 5


def macro_f1(y, p) -> float:
    """Macro F1 over K classes (zero_division=0), computed from counts; equals sklearn's value."""
    y, p = np.asarray(y), np.asarray(p)
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(K), where=denom > 0)
    return float(f1.mean())


def summary(y: np.ndarray, proba: np.ndarray) -> dict:
    pred = proba.argmax(1)
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)
    return {
        "n": int(len(y)),
        "macro_f1": macro_f1(y, pred),
        "accuracy": float(accuracy_score(y, pred)),
        "weighted_f1": float(f1_score(y, pred, labels=list(range(K)), average="weighted", zero_division=0)),
        "per_class": {"precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(), "support": sup.tolist()},
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(K))).tolist(),
        "ece_15": ece(y, proba),
    }


def ece(y: np.ndarray, proba: np.ndarray, bins: int = 15) -> float:
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def bootstrap_diffs(y, preds: dict[str, np.ndarray], pairs: dict[str, tuple], n: int | None = None, seed: int = 2026) -> dict:
    """Paired bootstrap over test examples.

    pairs: name -> (a, b) for Macro F1(a) − Macro F1(b), or (a, ("max", b, c)) for
    Macro F1(a) − max(Macro F1(b), Macro F1(c)) recomputed inside every resample (conservative).
    """
    n = n or BOOTSTRAP_N
    rng = np.random.default_rng(seed)
    N = len(y)
    out = {k: [] for k in pairs}
    for _ in range(n):
        idx = rng.integers(0, N, N)
        yy = y[idx]
        f = {m: macro_f1(yy, p[idx]) for m, p in preds.items()}
        for name, (a, b) in pairs.items():
            ref = max(f[b[1]], f[b[2]]) if isinstance(b, tuple) else f[b]
            out[name].append(f[a] - ref)
    res = {}
    for name, (a, b) in pairs.items():
        ref = max(macro_f1(y, preds[b[1]]), macro_f1(y, preds[b[2]])) if isinstance(b, tuple) else macro_f1(y, preds[b])
        d = np.array(out[name])
        res[name] = {
            "point": macro_f1(y, preds[a]) - ref,
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "share_resamples_above_0": float((d > 0).mean()),
            "resamples": n,
            "seed": seed,
        }
    return res


def mcnemar_exact(y, pa, pb) -> dict:
    a_ok, b_ok = pa == y, pb == y
    b01 = int((a_ok & ~b_ok).sum())  # a right, b wrong
    b10 = int((~a_ok & b_ok).sum())
    n = b01 + b10
    k = min(b01, b10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"a_right_b_wrong": b01, "a_wrong_b_right": b10, "p_value_two_sided": p}
