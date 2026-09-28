"""Text normalisation shared by the experiment, the speech evaluation and the application."""

from __future__ import annotations

import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_RT = re.compile(r"^\s*RT\s+@\w+:?\s*", re.I)
_MENTION = re.compile(r"@\w+")
_WS = re.compile(r"\s+")


def clean_post(text: str) -> str:
    """Model input cleaning (protocol § 3): drop URLs and a leading RT, anonymise mentions."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _RT.sub("", text)
    text = _URL.sub("", text)
    text = _MENTION.sub("@user", text)
    return _WS.sub(" ", text).strip()


def speakable(text: str) -> str:
    """Text as a person would read it aloud (protocol SP3): no mentions, hashtags as words, no emoji."""
    text = clean_post(text).replace("@user", "")
    text = text.replace("#", " ").replace("&amp;", " and ").replace("&", " and ")
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] in "LNPZ" or ch in "'-")
    return _WS.sub(" ", text).strip()


_PUNCT = re.compile(r"[^\w\s']|_")


def normalize_for_wer(text: str) -> list[str]:
    """WER normalisation: lower-case, punctuation removed (apostrophes kept), numbers as written."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("’", "'")
    text = _PUNCT.sub(" ", text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


def edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein distance over tokens (words or characters)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """Return (word edits, reference words). Aggregate as sum(edits)/sum(ref_words)."""
    r, h = normalize_for_wer(ref), normalize_for_wer(hyp)
    return edit_distance(r, h), len(r)


def cer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = " ".join(normalize_for_wer(ref)), " ".join(normalize_for_wer(hyp))
    return edit_distance(list(r), list(h)), len(r)
