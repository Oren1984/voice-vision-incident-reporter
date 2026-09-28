"""Secondary comparison S3: a paper-inspired fusion head over frozen CLIP embeddings (protocol § 4).

Implements the *ideas* of Munia et al. (2025) — guided cross-modal gating followed by differential
attention — as our own small head. It is not the authors' model and does not reproduce their numbers.
Unlike the authors' public code (where attention runs over a length-1 sequence, see
docs/paper-mapping.md), attention here runs over two tokens (image, text), so the softmax maps are
not constant.

Usage: python -m experiments.run_s3_diffattn   (requires `run_rq1 select` to have been run first)
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import RESULTS, load_examples, write_json
from .metrics import bootstrap_diffs, macro_f1, summary
from .run_rq1 import SELECTION, load

SEEDS = (13, 42, 7)


class GuidedDiffAttnHead(nn.Module):
    def __init__(self, dim_in: int = 512, d: int = 128, heads: int = 4, n_class: int = 5, depth: int = 1):
        super().__init__()
        self.proj_t = nn.Sequential(nn.Linear(dim_in, d), nn.BatchNorm1d(d), nn.ReLU())
        self.proj_i = nn.Sequential(nn.Linear(dim_in, d), nn.BatchNorm1d(d), nn.ReLU())
        self.gate_from_t = nn.Linear(dim_in, d)  # text decides which image features pass (guided attention)
        self.gate_from_i = nn.Linear(dim_in, d)  # image decides which text features pass
        self.modality = nn.Parameter(torch.zeros(2, d))
        self.h, self.hd = heads, d // heads // 2
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lq1, self.lk1, self.lq2, self.lk2 = (nn.Parameter(torch.randn(self.hd) * 0.1) for _ in range(4))
        self.norm = nn.GroupNorm(heads, d)
        self.out = nn.Linear(d, d)
        self.cls = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.ReLU(), nn.Dropout(0.3), nn.Linear(2 * d, n_class))

    def diff_attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q(x).view(B, T, 2 * self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, 2 * self.h, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, 2 * self.hd).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.hd), dim=-1).view(B, self.h, 2, T, T)
        lam = torch.exp((self.lq1 * self.lk1).sum()) - torch.exp((self.lq2 * self.lk2).sum()) + self.lambda_init
        attn = a[:, :, 0] - lam * a[:, :, 1]  # difference of two attention maps cancels common-mode noise
        o = (attn @ v).transpose(1, 2).reshape(B, T, D)
        o = self.norm(o.transpose(1, 2)).transpose(1, 2) * (1 - self.lambda_init)
        return self.out(o)

    def forward(self, t: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        zi = torch.sigmoid(self.gate_from_t(t)) * self.proj_i(i)
        zt = torch.sigmoid(self.gate_from_i(i)) * self.proj_t(t)
        x = torch.stack([zi, zt], 1) + self.modality
        x = x + self.diff_attn(x)
        return self.cls(x.flatten(1))


def train_one(seed, Xt, Xi, y, tr, dv, epochs=60, patience=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = GuidedDiffAttnHead()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    counts = np.bincount(y[tr], minlength=5)
    w = torch.tensor(len(y[tr]) / (5 * np.maximum(counts, 1)), dtype=torch.float32)
    T, I, Y = (torch.tensor(a) for a in (Xt, Xi, y))
    idx_tr = np.where(tr)[0]
    best, best_state, bad, hist = -1.0, None, 0, []
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        m.train()
        for b in np.array_split(rng.permutation(idx_tr), max(1, len(idx_tr) // 64)):
            opt.zero_grad()
            F.cross_entropy(m(T[b], I[b]), Y[b], weight=w).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            f = macro_f1(y[dv], m(T[dv], I[dv]).argmax(1).numpy())
        hist.append(f)
        if f > best:
            best, bad = f, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, best, len(hist)


def main() -> None:
    if not SELECTION.exists():
        raise SystemExit("run `python -m experiments.run_rq1 select` first")
    data, X, y = load()
    tr, dv, te = ((data.split == s).to_numpy() for s in ("train", "dev", "test"))
    Xt, Xi = X["text"].astype(np.float32), X["image"].astype(np.float32)
    runs, probas = [], []
    for seed in SEEDS:
        t0 = time.time()
        m, dev_f1, n_ep = train_one(seed, Xt, Xi, y, tr, dv)
        with torch.no_grad():
            p = torch.softmax(m(torch.tensor(Xt[te]), torch.tensor(Xi[te])), 1).numpy()
        probas.append(p)
        s = summary(y[te], p)
        runs.append({"seed": seed, "dev_macro_f1": dev_f1, "epochs": n_ep, "test": s, "seconds": round(time.time() - t0, 1)})
        print(f"seed {seed}: dev {dev_f1:.4f} test MacroF1 {s['macro_f1']:.4f} acc {s['accuracy']:.4f} ({n_ep} epochs)")
    f1s = np.array([r["test"]["macro_f1"] for r in runs])
    rq1 = json.loads((RESULTS / "rq1_results.json").read_text()) if (RESULTS / "rq1_results.json").exists() else None
    out = {
        "protocol": "docs/experiment-design.md v1, S3",
        "model": "GuidedDiffAttnHead (ours): guided gating + differential attention over [image, text] tokens, frozen CLIP ViT-B/32 inputs",
        "params": int(sum(p.numel() for p in GuidedDiffAttnHead().parameters())),
        "runs": runs,
        "test_macro_f1_mean": float(f1s.mean()),
        "test_macro_f1_sd": float(f1s.std(ddof=1)),
        "test_accuracy_mean": float(np.mean([r["test"]["accuracy"] for r in runs])),
    }
    if rq1:
        out["primary_combined_macro_f1"] = rq1["results"]["combined"]["macro_f1"]
        out["mean_minus_primary_combined"] = out["test_macro_f1_mean"] - out["primary_combined_macro_f1"]
    # Seed-ensemble (mean probability) vs the primary logistic-regression views, same test examples
    ens = np.mean(probas, 0).argmax(1)
    from .run_rq1 import fit  # refit the frozen primary models for paired comparison

    sel = json.loads(SELECTION.read_text())
    preds = {"s3_ensemble": ens}
    for v in ("text", "image", "combined"):
        s = sel["views"][v]["selected"]
        cw = None if s["class_weight"] == "None" else s["class_weight"]
        preds[v] = fit(X[v][tr], y[tr], s["C"], cw).predict(X[v][te])
    out["s3_seed_ensemble_macro_f1"] = macro_f1(y[te], ens)
    out["bootstrap"] = bootstrap_diffs(y[te], preds, {
        "s3_ensemble_minus_best_unimodal": ("s3_ensemble", ("max", "text", "image")),
        "s3_ensemble_minus_primary_combined": ("s3_ensemble", "combined"),
    })
    write_json(RESULTS / "s3_diffattn_results.json", out)
    print(f"S3 mean MacroF1 {out['test_macro_f1_mean']:.4f} ± {out['test_macro_f1_sd']:.4f}; ensemble {out['s3_seed_ensemble_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
