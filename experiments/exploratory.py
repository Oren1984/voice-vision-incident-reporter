"""POST-HOC, EXPLORATORY analyses (not part of the pre-registered protocol; decided after seeing RQ1 results).

Why: the primary combined model's gain over image-only rests partly on the 9-example affected_individuals class,
and dev selection gave only the combined model balanced class weights. These checks ask whether the picture changes
(a) without that class in the average and (b) when every view uses balanced class weights (C re-selected on dev).
They do not replace the pre-registered verdict. Output: results/exploratory.json
Usage: python -m experiments.exploratory
"""

from __future__ import annotations

import json

import numpy as np

from .common import RESULTS, write_json
from .metrics import bootstrap_diffs, macro_f1
from .run_rq1 import C_GRID, SELECTION, VIEWS, fit, load


def f1_subset(y, p, classes):
    from sklearn.metrics import f1_score

    return float(f1_score(y, p, labels=classes, average="macro", zero_division=0))


def main() -> None:
    sel = json.loads(SELECTION.read_text())
    data, X, y = load()
    tr, dv, te = ((data.split == s).to_numpy() for s in ("train", "dev", "test"))
    yt = y[te]
    primary = {}
    for v in VIEWS:
        s = sel["views"][v]["selected"]
        primary[v] = fit(X[v][tr], y[tr], s["C"], None if s["class_weight"] == "None" else s["class_weight"]).predict(X[v][te])
    four = [1, 2, 3, 4]  # all classes except affected_individuals (index 0)
    a = {v: f1_subset(yt, primary[v], four) for v in VIEWS}

    balanced, chosen = {}, {}
    for v in VIEWS:
        best = max(C_GRID, key=lambda C: (macro_f1(y[dv], fit(X[v][tr], y[tr], C, "balanced").predict(X[v][dv])), -C))
        chosen[v] = best
        balanced[v] = fit(X[v][tr], y[tr], best, "balanced").predict(X[v][te])
    b = {v: macro_f1(yt, balanced[v]) for v in VIEWS}
    boot = bootstrap_diffs(yt, balanced, {"combined_minus_best_unimodal": ("combined", ("max", "text", "image"))})
    out = {
        "status": "post-hoc exploratory; not pre-registered",
        "a_macro_f1_4_classes_without_affected_individuals": {**a, "combined_minus_best_unimodal": a["combined"] - max(a["text"], a["image"])},
        "b_all_views_balanced_class_weights": {"selected_C_on_dev": chosen, "test_macro_f1": b,
                                               "combined_minus_best_unimodal": boot["combined_minus_best_unimodal"]},
    }
    write_json(RESULTS / "exploratory.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
