"""Experiment pipeline: metrics, selection-before-test ordering, head export, S3 head, no pickle."""

import json

import numpy as np
import pandas as pd
import pytest

from experiments import metrics, run_rq1
from incident_reporter.labels import LABELS
from incident_reporter.ml.heads import HeadSet, LinearHead


def synthetic(n_per_split=(300, 80, 80), seed=0):
    """Features where text and image each carry part of the label signal."""
    rng = np.random.default_rng(seed)
    rows, T, I, Y = [], [], [], []
    for split, n in zip(("train", "dev", "test"), n_per_split):
        for _ in range(n):
            y = int(rng.integers(0, 5))
            t = rng.normal(0, 1, 512)
            i = rng.normal(0, 1, 512)
            t[y] += 4.0 if y < 3 else 0.0  # text separates classes 0-2
            i[10 + y] += 4.0 if y >= 2 else 0.0  # image separates classes 2-4
            T.append(t / np.linalg.norm(t)); I.append(i / np.linalg.norm(i)); Y.append(y)
            rows.append({"split": split, "event_name": "event_a" if len(rows) % 2 else "event_b", "near_dup_of_train": False})
    T, I = np.array(T, np.float32), np.array(I, np.float32)
    return pd.DataFrame(rows), {"text": T, "image": I, "combined": np.concatenate([T, I], 1)}, np.array(Y)


@pytest.fixture
def rq1(tmp_path, monkeypatch):
    data, X, y = synthetic()
    monkeypatch.setattr(run_rq1, "load", lambda: (data, X, y))
    monkeypatch.setattr(run_rq1, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run_rq1, "SELECTION", tmp_path / "results" / "rq1_selection.json")
    monkeypatch.setattr(run_rq1, "HEADS_PATH", tmp_path / "artifacts" / "heads.npz")
    monkeypatch.setattr(run_rq1, "C_GRID", (0.1, 1, 10))
    monkeypatch.setattr(metrics, "BOOTSTRAP_N", 500)
    uniform = lambda a, b: (np.full((len(a), 5), 0.2), np.full((len(b), 5), 0.2))  # noqa: E731
    monkeypatch.setattr(run_rq1, "zero_shot_probas", uniform)
    return tmp_path


def test_test_stage_refuses_before_selection(rq1):
    with pytest.raises(SystemExit, match="select"):
        run_rq1.test()


def test_pipeline_selects_on_dev_then_scores_test_once(rq1):
    run_rq1.select()
    sel = json.loads((rq1 / "results" / "rq1_selection.json").read_text())
    assert set(sel["views"]) == {"text", "image", "combined"}
    assert all("dev_macro_f1" in r for r in sel["views"]["text"]["all"])
    run_rq1.test()
    res = json.loads((rq1 / "results" / "rq1_results.json").read_text())
    r = res["results"]
    # the synthetic signal is split across modalities, so fusion must win clearly here
    assert r["combined"]["macro_f1"] > max(r["text"]["macro_f1"], r["image"]["macro_f1"])
    assert res["h1"]["verdict"].startswith("SUCCESS") and res["h1"]["ci95"][0] > 0
    assert res["test_n"] == 80 and sum(res["test_class_counts"]) == 80
    assert len(r["combined"]["confusion_matrix"]) == 5
    assert res["selection_file_sha256"]
    hs = HeadSet.load(rq1 / "artifacts" / "heads.npz")
    assert hs.kind == "trained" and hs.heads["combined"].W.shape == (5, 1024)


def test_headset_roundtrip_rejects_foreign_labels(tmp_path):
    h = LinearHead(np.zeros((5, 4), np.float32), np.zeros(5, np.float32))
    hs = HeadSet("trained", {"text": h, "image": h, "combined": h}, {})
    hs.save(tmp_path / "h.npz")
    assert np.allclose(HeadSet.load(tmp_path / "h.npz").heads["text"].proba(np.ones(4)), 0.2)
    z = dict(np.load(tmp_path / "h.npz"))
    meta = json.loads(str(z["meta"]))
    meta["labels"] = ["a", "b", "c", "d", "e"]
    z["meta"] = np.array(json.dumps(meta))
    np.savez(tmp_path / "bad.npz", **z)
    with pytest.raises(ValueError, match="labels"):
        HeadSet.load(tmp_path / "bad.npz")
    np.save(tmp_path / "obj.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):  # object arrays (pickle) are never loaded
        np.load(tmp_path / "obj.npy", allow_pickle=False)


def test_fast_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(3)
    for _ in range(20):
        y, p = rng.integers(0, 5, 100), rng.integers(0, 5, 100)
        p[:3] = 4  # include classes that may be absent from y
        assert abs(metrics.macro_f1(y, p) - f1_score(y, p, labels=range(5), average="macro", zero_division=0)) < 1e-12


def test_metrics_basics():
    y = np.array([0, 1, 2, 3, 4, 0, 1])
    assert metrics.macro_f1(y, y) == 1.0
    p = np.eye(5)[y] * 0.9 + 0.02
    s = metrics.summary(y, p / p.sum(1, keepdims=True))
    assert s["accuracy"] == 1.0 and s["n"] == 7 and s["confusion_matrix"][0][0] == 2
    m = metrics.mcnemar_exact(y, y, np.array([1, 1, 2, 3, 4, 0, 1]))
    assert m["a_right_b_wrong"] == 1 and m["a_wrong_b_right"] == 0 and m["p_value_two_sided"] == 1.0


def test_bootstrap_is_deterministic_and_paired():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 200)
    a = y.copy()
    b = np.where(rng.random(200) < 0.3, (y + 1) % 5, y)
    r1 = metrics.bootstrap_diffs(y, {"a": a, "b": b, "c": b}, {"d": ("a", ("max", "b", "c"))}, n=300, seed=7)
    r2 = metrics.bootstrap_diffs(y, {"a": a, "b": b, "c": b}, {"d": ("a", ("max", "b", "c"))}, n=300, seed=7)
    assert r1 == r2 and r1["d"]["ci95"][0] > 0


def test_s3_head_shapes_and_attention_is_not_degenerate():
    import torch

    from experiments.run_s3_diffattn import GuidedDiffAttnHead

    m = GuidedDiffAttnHead().eval()
    t, i = torch.randn(4, 512), torch.randn(4, 512)
    out = m(t, i)
    assert out.shape == (4, len(LABELS))
    # with two tokens the attention map depends on the input (unlike a length-1 sequence)
    x1 = torch.stack([m.proj_i(i), m.proj_t(t)], 1)
    x2 = torch.stack([m.proj_i(i.flip(0)), m.proj_t(t)], 1)
    assert not torch.allclose(m.diff_attn(x1), m.diff_attn(x2))
