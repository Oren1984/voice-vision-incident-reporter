"""Compare what the transcript says with what the image analysis suggests.

Five buckets: agreement, complementary (only one source speaks to it), contradiction, missing
information and uncertainty. The comparison never decides which source is right: a contradiction is
reported with both sides and becomes an unresolved question for the reviewer.

Important asymmetry: an image observation that is "not detected" is *not* evidence of absence — CLIP
scores are weak, scenes are partial, and photos can be taken elsewhere or earlier. So "text says X,
image does not show X" is complementary, not a contradiction. Contradictions need positive evidence on
both sides (e.g. the text denies injuries while the image likely shows injured people).
"""

from __future__ import annotations

from .concepts import BY_ID, HAZARDS
from .labels import DISPLAY, pct
from .models import ASR_HIGH_NO_SPEECH, ASR_LOW_LOGPROB, VIEW_CONFIDENT, VIEW_LOW_CONFIDENCE


def _refs(mentions: list[dict], concept: str, negated: bool | None = None) -> list[dict]:
    return [
        {"kind": "transcript", "start": m["start"], "end": m["end"], "quote": m["quote"]}
        for m in mentions
        if m["concept"] == concept and (negated is None or m["negated"] == negated)
    ]


def _img_ref(o: dict) -> dict:
    return {"kind": "image", "concept": o["concept"], "score": o["score"], "band": o["band"]}


def asr_signals(asr: dict | None) -> dict:
    if not asr or not asr.get("segments"):
        return {"mean_logprob": None, "max_no_speech": None, "low_confidence": True}
    segs = asr["segments"]
    dur = [max(0.01, s["end"] - s["start"]) for s in segs]
    lp = sum(s["avg_logprob"] * d for s, d in zip(segs, dur)) / sum(dur)
    ns = max(s["no_speech_prob"] for s in segs)
    return {"mean_logprob": round(lp, 3), "max_no_speech": round(ns, 3),
            "low_confidence": lp < ASR_LOW_LOGPROB or ns > ASR_HIGH_NO_SPEECH}


def compare(text_ev: dict | None, image: dict | None, views: dict, asr: dict | None, transcript_edited: bool) -> dict:
    agreements, complementary, contradictions, missing, uncertainty = [], [], [], [], []
    obs = {o["concept"]: o for o in (image or {}).get("observations", [])}
    mentions = (text_ev or {}).get("mentions", [])
    asserted = set((text_ev or {}).get("asserted", []))
    negated = set((text_ev or {}).get("negated", []))

    if text_ev is not None and image is not None and obs:
        for cid, o in obs.items():
            if cid == "non_photo":
                continue
            c = BY_ID[cid]
            b = o["band"]
            if cid in asserted:
                if b in ("likely", "possible"):
                    agreements.append({"concept": cid, "label": c.label, "strength": "strong" if b == "likely" else "partial",
                                       "text": f"Report and image both point to: {c.label.lower()}"
                                       + ("" if b == "likely" else " (image evidence is weak)."),
                                       "sources": _refs(mentions, cid, False) + [_img_ref(o)]})
                else:
                    complementary.append({"concept": cid, "label": c.label, "source": "transcript",
                                          "text": f"Only the report mentions {c.label.lower()}; the image analysis did not detect it "
                                                  "(not detected does not mean absent).",
                                          "sources": _refs(mentions, cid, False) + [_img_ref(o)]})
            elif cid in negated:
                if b == "likely":
                    contradictions.append({"kind": "negation", "concept": cid,
                                           "text": f"The report says there is no {c.label.lower()}, but the image analysis suggests it is likely visible.",
                                           "sources": _refs(mentions, cid, True) + [_img_ref(o)]})
                elif b == "possible":
                    uncertainty.append({"kind": "negation_weak", "text": f"The report denies {c.label.lower()}; the image may show it (weak score).",
                                        "sources": _refs(mentions, cid, True) + [_img_ref(o)]})
            else:
                if b == "likely":
                    complementary.append({"concept": cid, "label": c.label, "source": "image",
                                          "text": f"Only the image suggests {c.label.lower()}; the report does not mention it.",
                                          "sources": [_img_ref(o)]})
                elif b == "possible":
                    uncertainty.append({"kind": "image_possible", "text": f"The image may show {c.label.lower()} (weak score); the report does not mention it.",
                                        "sources": [_img_ref(o)]})
        # Hazard mismatch: the report names one hazard, the image positively suggests a different one
        for h1 in sorted(asserted & set(HAZARDS)):
            if obs.get(h1, {}).get("band") != "not_detected":
                continue
            for h2 in HAZARDS:
                if h2 != h1 and h2 not in asserted and obs.get(h2, {}).get("band") == "likely":
                    contradictions.append({"kind": "hazard", "concept": f"{h1}|{h2}",
                                           "text": f"The report describes {BY_ID[h1].label.lower()}, but the image suggests "
                                                   f"{BY_ID[h2].label.lower()} and no {BY_ID[h1].label.lower()} was detected.",
                                           "sources": _refs(mentions, h1, False) + [_img_ref(obs[h1]), _img_ref(obs[h2])]})

    # Transcript-only details the image cannot provide
    if text_ev is not None:
        for field, label in (("location", "location"), ("time", "time"), ("people_count", "people count")):
            for f in text_ev["fields"].get(field, [])[:2]:
                complementary.append({"concept": field, "label": label.capitalize(), "source": "transcript",
                                      "text": f"The report gives a {label}: “{f['quote']}”. The image cannot confirm it.",
                                      "sources": [{"kind": "transcript", "start": f["start"], "end": f["end"], "quote": f["quote"]}]})
        for q in text_ev["questions"]:
            missing.append({"field": q["field"], "text": q["text"], "sources": []})

    # Category-level agreement between the single-source views
    tv, iv = views.get("text"), views.get("image")
    if tv and iv and tv.get("available") and iv.get("available"):
        if tv["label"] == iv["label"]:
            agreements.append({"concept": "category", "label": "Category", "strength": "strong",
                               "text": f"Text-only and image-only views agree on the category: {DISPLAY[tv['label']]}.",
                               "sources": [_view_ref(tv), _view_ref(iv)]})
        elif tv["confidence"] >= VIEW_CONFIDENT and iv["confidence"] >= VIEW_CONFIDENT:
            contradictions.append({"kind": "category", "concept": "category",
                                   "text": f"The text-only view says {DISPLAY[tv['label']]} ({pct(tv['confidence'])}) while the image-only "
                                           f"view says {DISPLAY[iv['label']]} ({pct(iv['confidence'])}).",
                                   "sources": [_view_ref(tv), _view_ref(iv)]})
        else:
            uncertainty.append({"kind": "category_split", "text": f"The single-source views disagree on the category "
                                f"({DISPLAY[tv['label']]} vs {DISPLAY[iv['label']]}), and at least one is not confident.",
                                "sources": [_view_ref(tv), _view_ref(iv)]})

    for name, v in views.items():
        if v.get("available") and v["confidence"] < VIEW_LOW_CONFIDENCE:
            uncertainty.append({"kind": "low_confidence", "text": f"The {name.replace('_', ' ')} view is not confident "
                                f"({pct(v['confidence'])} for {DISPLAY[v['label']]}).", "sources": [_view_ref(v)]})
        if not v.get("available"):
            uncertainty.append({"kind": "view_unavailable", "text": f"The {name} view is unavailable: {v.get('reason')}.", "sources": []})

    # Input quality
    if image is not None:
        for flag in image.get("quality", {}).get("flags", []):
            uncertainty.append({"kind": f"image_{flag}", "text": f"Image quality: {flag.replace('_', ' ')} — observations may be unreliable.", "sources": []})
        np_obs = obs.get("non_photo")
        if np_obs and np_obs["band"] == "likely":
            uncertainty.append({"kind": "image_non_photo", "text": "The image looks like a screenshot, map or graphic rather than a photo of the scene.",
                                "sources": [_img_ref(np_obs)]})
    else:
        missing.append({"field": "image", "text": "No image was provided; nothing can be confirmed visually.", "sources": []})

    if text_ev is None:
        missing.append({"field": "audio", "text": "No usable spoken report; the incident description is missing.", "sources": []})
    else:
        sig = asr_signals(asr)
        if asr is not None and sig["low_confidence"] and not transcript_edited:
            uncertainty.append({"kind": "asr_low_confidence",
                                "text": "The speech recogniser was unsure (unclear or noisy speech). Check the transcript against the audio.",
                                "sources": [{"kind": "asr", **sig}]})
        if transcript_edited:
            uncertainty.append({"kind": "transcript_edited", "text": "The transcript was corrected by a person; analysis uses the corrected text.",
                                "sources": []})

    return {"agreements": agreements, "complementary": complementary, "contradictions": contradictions,
            "missing": missing, "uncertainty": uncertainty}


def _view_ref(v: dict) -> dict:
    return {"kind": "model", "view": v["view"], "label": v["label"], "confidence": round(v["confidence"], 3)}
