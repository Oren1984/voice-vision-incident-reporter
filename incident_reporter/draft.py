"""Build the structured incident draft from evidence and the comparison.

Each claim is a short sentence with the sources it rests on (transcript spans, image observations,
model outputs). The draft never states more than its sources: image findings are phrased as
"may be visible", the category is a *proposal*, and conflicts are listed as questions for a person.
"""

from __future__ import annotations

from .concepts import BY_ID
from .labels import DISPLAY, LABELS, pct

CRITICAL_UNCERTAINTY = {"asr_low_confidence", "image_blurry", "image_too_dark", "image_non_photo", "view_unavailable", "low_confidence"}


def _list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def build(text_ev: dict | None, image: dict | None, views: dict, comparison: dict, flags: list[str], transcript_version: str) -> dict:
    claims: list[dict] = []

    def claim(text: str, sources: list[dict], kind: str) -> None:
        claims.append({"id": f"c{len(claims) + 1}", "kind": kind, "text": text, "sources": sources})

    def tref(m: dict) -> dict:
        return {"kind": "transcript", "start": m["start"], "end": m["end"], "quote": m["quote"], "version": transcript_version}

    if text_ev is not None:
        pos = [m for m in text_ev["mentions"] if not m["negated"]]
        by_c: dict[str, list[dict]] = {}
        for m in pos:
            by_c.setdefault(m["concept"], []).append(m)
        if by_c:
            names = [BY_ID[c].label.lower() for c in by_c]
            claim(f"The spoken report mentions {_list(names)}.", [tref(ms[0]) for ms in by_c.values()], "report")
        neg = [m for m in text_ev["mentions"] if m["negated"]]
        if neg:
            claim(f"The report explicitly denies {_list(sorted({BY_ID[m['concept']].label.lower() for m in neg}))}.",
                  [tref(m) for m in neg], "report")
        for field, phrase in (("location", "gives the location as"), ("time", "gives the time as"), ("people_count", "says")):
            f = text_ev["fields"].get(field) or []
            if f:
                claim(f"The report {phrase} “{f[0]['quote']}”.", [tref(f[0])], "report")

    if image is not None:
        obs = [o for o in image.get("observations", []) if o["concept"] != "non_photo"]
        likely = [o for o in obs if o["band"] == "likely"]
        possible = [o for o in obs if o["band"] == "possible"]
        src = lambda o: {"kind": "image", "concept": o["concept"], "score": o["score"], "band": o["band"]}  # noqa: E731
        if likely:
            claim(f"Image analysis suggests {_list([o['label'].lower() for o in likely])} may be visible.", [src(o) for o in likely], "image")
        if possible:
            claim(f"Weak image signals (uncertain): {_list([o['label'].lower() for o in possible])}.", [src(o) for o in possible], "image")
        if not likely and not possible:
            claim("Image analysis found no incident-related content with confidence.", [], "image")

    for a in comparison["agreements"]:
        if a["concept"] != "category":
            claim(a["text"].rstrip(".") + ".", a["sources"], "agreement")
    for c in comparison["contradictions"]:
        claim("Conflict: " + c["text"], c["sources"], "conflict")

    # Category proposal: combined view when both inputs exist, otherwise the single available view.
    order = ("combined", "text", "image")
    basis = next((v for v in order if views.get(v, {}).get("available")), None)
    category: dict = {"proposed": None, "confidence": None, "basis": basis, "alternatives": [], "needs_decision": True, "sources": []}
    if basis:
        v = views[basis]
        alts = sorted({views[k]["label"] for k in order if views.get(k, {}).get("available")} - {v["label"]})
        conflict = any(c["kind"] in ("category", "hazard", "negation") for c in comparison["contradictions"])
        category.update({
            "proposed": v["label"], "confidence": round(v["confidence"], 3), "alternatives": alts,
            "needs_decision": conflict or v["confidence"] < 0.5 or basis != "combined",
            "sources": [{"kind": "model", "view": basis, "label": v["label"], "confidence": round(v["confidence"], 3)}],
        })
        claim(f"Proposed category: {DISPLAY[v['label']]} ({basis} view, {pct(v['confidence'])})"
              + (" — sources disagree; a reviewer must decide." if conflict else "."), category["sources"], "category")

    questions: list[dict] = []

    def question(text: str, kind: str, sources: list[dict]) -> None:
        questions.append({"id": f"q{len(questions) + 1}", "kind": kind, "text": text, "sources": sources, "resolution": None})

    for c in comparison["contradictions"]:
        question(f"Resolve conflict: {c['text']} Which account is correct, or is more information needed?", f"conflict_{c['kind']}", c["sources"])
    for m in comparison["missing"]:
        question(m["text"], f"missing_{m['field']}", m["sources"])
    for u in comparison["uncertainty"]:
        if u["kind"] in CRITICAL_UNCERTAINTY or u["kind"].startswith("image_"):
            question(u["text"], f"uncertain_{u['kind']}", u["sources"])
    if category["needs_decision"] and not any(q["kind"].startswith("conflict") for q in questions):
        question("Confirm or change the proposed category.", "category_decision", category["sources"])

    return {
        "category": category,
        "summary": " ".join(c["text"] for c in claims if c["kind"] in ("report", "image", "conflict")),
        "summary_edited": False,
        "claims": claims,
        "unresolved_questions": questions,
        "flags": sorted(set(flags)),
        "reviewer_notes": "",
        "labels": list(LABELS),
    }
