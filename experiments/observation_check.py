"""Sanity check for the app's image concept observations on CrisisMMD dev images.

The observation scores are uncalibrated design choices. This checks, on dev only, whether a concept's score
is higher for images of the related humanitarian class than for the rest (AUROC), and how often each band
fires. Aggregate output only: results/observation_check.json.
Usage: python -m experiments.observation_check
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from incident_reporter.concepts import BACKGROUND_PROMPTS, CONCEPTS, LIKELY, POSSIBLE
from incident_reporter.labels import LABELS
from incident_reporter.ml.clip_backbone import ClipBackbone
from incident_reporter.models import observation_scores

from .common import HF_CACHE, RESULTS, WORK, load_examples, write_json

# concept -> humanitarian class it should be enriched in (only where a mapping is defensible)
RELATED = {
    "structural_damage": "infrastructure_and_utility_damage",
    "debris": "infrastructure_and_utility_damage",
    "power_utility": "infrastructure_and_utility_damage",
    "flood_water": "infrastructure_and_utility_damage",
    "responders": "rescue_volunteering_or_donation_effort",
    "relief_supplies": "rescue_volunteering_or_donation_effort",
    "displaced_people": "affected_individuals",
    "injured_people": "affected_individuals",
    "non_photo": "other_relevant_information",
}


def main() -> None:
    data = load_examples()
    dv = (data.split == "dev").to_numpy()
    img = np.load(WORK / "features.npz")["image"][dv]
    labels = data.loc[dv, "label"].to_numpy()
    bb = ClipBackbone(cache_dir=str(HF_CACHE))
    prompts = [p for c in CONCEPTS for p in c.prompts] + list(BACKGROUND_PROMPTS)
    P = bb.embed_texts(prompts)
    per_image = [observation_scores(v, P) for v in img]
    out = {}
    for c in CONCEPTS:
        if not c.prompts:
            continue
        s = np.array([next(o["score"] for o in obs if o["concept"] == c.id) for obs in per_image])
        row = {"likely_share": float((s >= LIKELY).mean()), "possible_share": float(((s >= POSSIBLE) & (s < LIKELY)).mean())}
        if c.id in RELATED:
            pos = labels == RELATED[c.id]
            row.update({"related_class": RELATED[c.id], "n_related": int(pos.sum()), "auroc": float(roc_auc_score(pos, s)),
                        "mean_score_related": float(s[pos].mean()), "mean_score_other": float(s[~pos].mean())})
        out[c.id] = row
    write_json(RESULTS / "observation_check.json", {
        "note": "CrisisMMD dev split (n=%d). AUROC of each concept score for images of the related humanitarian class vs all others. "
                "A weak proxy: class labels are not concept labels." % int(dv.sum()),
        "labels": list(LABELS), "thresholds": {"likely": LIKELY, "possible": POSSIBLE}, "concepts": out})
    for cid, r in out.items():
        print(cid, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})


if __name__ == "__main__":
    main()
