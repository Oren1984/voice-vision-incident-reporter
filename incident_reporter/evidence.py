"""Transparent, rule-based evidence extraction from the (corrected) transcript.

Every finding keeps the character span it came from, so each claim in the draft can point back to the
exact words. Rules are deliberately simple and visible; they miss paraphrases and will sometimes
mis-fire — the reviewer sees the quoted span and decides.
"""

from __future__ import annotations

import re

from .concepts import CONCEPTS, NEGATIONS

_NUM_WORDS = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|dozens|several|a few|many|hundreds"
_PEOPLE = r"(?:people|persons|residents|children|kids|adults|families|victims|workers|patients|passengers|neighbou?rs|men|women)"

FIELD_PATTERNS: dict[str, list[re.Pattern]] = {
    "location": [
        re.compile(r"\b(?:on|at|near|in|by|behind|outside|opposite|above|below|along|across|off|past)\s+(?:the\s+)?(?:\w+\s+(?:of|above|near|on)\s+)?(?:corner of\s+)?"
                   r"(?:\d+\s+)?(?:[A-Z][\w'-]+\s?){1,4}(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Lane|Drive|Way|Bridge|Park|Square|Highway|School|Station|Market|Mall|Center|Centre)\b"),
        re.compile(r"\b\d{1,5}\s+(?:[A-Z][\w'-]+\s){1,3}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Drive|Way)\b"),
        # transcripts are not always capitalised ("on station road"); require a name word other than "the"
        re.compile(r"\b(?:on|at|near|along|off)\s+(?!the\s+(?:road|street|lane|avenue|highway)\b)(?:the\s+)?(?:[a-z][\w'-]+\s){1,3}"
                   r"(?:street|road|avenue|lane|drive|boulevard)\b", re.I),
        re.compile(r"\b(?:near|at|on|outside|behind|by|inside|in front of)\s+the\s+(?:\w+\s)?(?:school|hospital|bridge|station|market|mall|park|"
                   r"church|mosque|stadium|shelter|river|harbou?r|highway|warehouse|factory|hotel|airport|community center|parking lot)\b", re.I),
        re.compile(r"\b(?:intersection|junction|corner) of [A-Z][\w'-]+(?: [A-Z][\w'-]+)* and [A-Z][\w'-]+(?: [A-Z][\w'-]+)*"),
    ],
    "time": [
        re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:a\.?m\.?|p\.?m\.?)", re.I),
        re.compile(r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)(?:\s(?:fifteen|thirty|forty[- ]five|o'clock))?"
                   r"\s(?:a\.?m\.?|p\.?m\.?|in the (?:morning|afternoon|evening))", re.I),
        re.compile(r"\bat (?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s(?:fifteen|thirty|forty[- ]five|o'clock)\b", re.I),
        re.compile(r"\b(?:about|around|at)\s+\d{1,2}(?::\d{2})\b", re.I),
        re.compile(r"\b(?:\w+\s)?(?:minutes?|hours?) ago\b|\bjust now\b|\bright now\b|\bthis (?:morning|afternoon|evening)\b|\btonight\b|\blast night\b", re.I),
    ],
    "people_count": [
        re.compile(rf"\b(?:\d+|{_NUM_WORDS})\s+(?:\w+\s)?{_PEOPLE}\b", re.I),
        re.compile(rf"\b(?:\d+|{_NUM_WORDS})\s+(?:injured|hurt|dead|missing|trapped)\b", re.I),
        re.compile(r"\b(?:nobody|no one) (?:is|was|has been|seems) (?:hurt|injured|trapped)\b", re.I),
        re.compile(r"\b(?:a|an|one)\s+(?:\w+\s)?(?:couple|family|child|baby|man|woman|person|resident|driver|neighbou?r|worker|passenger)\b", re.I),
    ],
    "ongoing": [
        re.compile(r"\b(?:still|spreading|rising|getting worse|ongoing|continues|right now|is moving)\b", re.I),
    ],
}

QUESTIONS = {
    "location": "Where exactly is the incident? No street, landmark or address was found in the report.",
    "time": "When did this happen? No time reference was found in the report.",
    "people_count": "Are people hurt, trapped or displaced, and how many? The report gives no count.",
    "hazard": "What is the hazard? The report does not name fire, flooding or another hazard.",
    "ongoing": "Is the danger still ongoing? The report does not say.",
}


def _term_regex(term: str) -> re.Pattern:
    return re.compile(r"(?<![\w-])" + r"\s+".join(map(re.escape, term.split())) + r"(?![\w-])", re.I)


_TERM_RES = [(c.id, t, _term_regex(t)) for c in CONCEPTS for t in c.terms]
_NEG_RE = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in NEGATIONS) + r")\b", re.I)
_CLAUSE_BREAK = re.compile(r"[.;!?]|\bbut\b|\bhowever\b", re.I)


def _negated(text: str, start: int) -> bool:
    """A negation word in the same clause, at most 4 words before the term."""
    window = text[max(0, start - 60) : start]
    parts = _CLAUSE_BREAK.split(window)
    clause = parts[-1] if parts else window
    words = clause.split()[-5:]
    return bool(_NEG_RE.search(" ".join(words)))


def extract(text: str) -> dict:
    """Concept mentions and incident fields, each with character spans into `text`."""
    text = text or ""
    mentions: list[dict] = []
    taken: list[tuple[int, int]] = []
    # longer terms first so "power lines" wins over "power"
    for cid, term, rx in sorted(_TERM_RES, key=lambda x: -len(x[1])):
        for m in rx.finditer(text):
            s, e = m.span()
            if any(s < te and e > ts for ts, te in taken):
                continue
            taken.append((s, e))
            mentions.append({"concept": cid, "term": term, "start": s, "end": e, "quote": text[s:e], "negated": _negated(text, s)})
    mentions.sort(key=lambda m: m["start"])

    fields: dict[str, list[dict]] = {}
    for name, pats in FIELD_PATTERNS.items():
        found = []
        for p in pats:
            for m in p.finditer(text):
                s, e = m.span()
                if not any(s < f["end"] and e > f["start"] for f in found):
                    found.append({"start": s, "end": e, "quote": text[s:e].strip()})
        fields[name] = sorted(found, key=lambda f: f["start"])

    asserted = sorted({m["concept"] for m in mentions if not m["negated"]})
    negated = sorted({m["concept"] for m in mentions if m["negated"]} - set(asserted))
    hazards_named = [c for c in asserted if c in ("fire_smoke", "flood_water", "structural_damage", "power_utility", "debris")]
    missing = [k for k in ("location", "time", "people_count") if not fields[k]]
    if not hazards_named:
        missing.append("hazard")
    if hazards_named and not fields["ongoing"]:
        missing.append("ongoing")
    return {
        "mentions": mentions,
        "asserted": asserted,
        "negated": negated,
        "fields": fields,
        "missing": missing,
        "questions": [{"field": k, "text": QUESTIONS[k]} for k in missing],
        "word_count": len(text.split()),
    }
