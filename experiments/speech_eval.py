"""Speech workflow evaluation — our extension, separate from RQ1 (protocol § 7).

  python -m experiments.speech_eval scripted      # SP2: project-authored reports, synthetic voices, 3 noise levels
  python -m experiments.speech_eval librispeech   # SP1: real human read speech (LibriSpeech test-clean sample)
  python -m experiments.speech_eval sp3           # SP3: RQ1 test tweets as synthetic speech -> ASR -> frozen classifiers

Synthetic speech is produced with Kokoro-82M (Apache-2.0). It is labelled synthetic everywhere and is
never presented as a real incident report.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path

import numpy as np

from incident_reporter.evidence import extract
from incident_reporter.labels import LABELS
from incident_reporter.text_clean import cer, clean_post, normalize_for_wer, speakable, wer

from .common import CACHE, HF_CACHE, RESULTS, ROOT, WORK, environment, load_examples, write_json

KOKORO_DIR = CACHE / "kokoro"
SPEECH_CACHE = CACHE / "speech"
VOICES = ("af_heart", "am_michael", "bf_emma", "bm_george")  # 2 US + 2 UK, 2 female + 2 male
CONDITIONS = {"clean": None, "snr10": 10.0, "snr0": 0.0}
ASR_MODEL = "base.en"
ASR_REVISION = "3d3d5dee26484f91867d81cb899cfcf72b96be6c"
LIBRI_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"


# ------------------------------------------------------------------ building blocks


class TTS:
    def __init__(self) -> None:
        from kokoro_onnx import Kokoro

        self.k = Kokoro(str(KOKORO_DIR / "kokoro-v1.0.onnx"), str(KOKORO_DIR / "voices-v1.0.bin"))

    def speak(self, text: str, voice: str) -> np.ndarray:
        from scipy.signal import resample_poly

        samples, sr = self.k.create(text, voice=voice, speed=1.0, lang="en-gb" if voice.startswith("b") else "en-us")
        assert sr == 24000
        return resample_poly(samples.astype(np.float32), 2, 3).astype(np.float32)  # 24 kHz -> 16 kHz


class ASR:
    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        self.m = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8", download_root=str(HF_CACHE), cpu_threads=8)

    def __call__(self, x: np.ndarray) -> dict:
        segs, _ = self.m.transcribe(x, language="en", beam_size=5, vad_filter=False, condition_on_previous_text=False)
        segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        lp = float(np.mean([s.avg_logprob for s in segs])) if segs else None
        return {"text": text, "mean_logprob": lp}


def add_noise(x: np.ndarray, snr_db: float | None, seed: int) -> np.ndarray:
    if snr_db is None:
        return x
    rng = np.random.default_rng(seed)
    p_sig = float(np.mean(x**2)) + 1e-12
    noise = rng.normal(0, np.sqrt(p_sig / 10 ** (snr_db / 10)), x.shape).astype(np.float32)
    y = x + noise
    return (y / max(1.0, float(np.abs(y).max()))).astype(np.float32)


def save_wav(path: Path, x: np.ndarray, rate: int = 16000) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, rate, subtype="PCM_16")


def load_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32")
    assert sr == 16000
    return x


def critical_tokens(ref: str) -> list[str]:
    """Words that carry incident facts: concept terms, places, times, counts (from the reference)."""
    ev = extract(ref)
    spans = [m["quote"] for m in ev["mentions"]] + [f["quote"] for k in ("location", "time", "people_count") for f in ev["fields"][k]]
    toks = []
    for s in spans:
        toks += [t for t in normalize_for_wer(s) if t not in {"on", "at", "in", "near", "the", "of", "and", "by", "about", "around"}]
    return toks


def token_recall(ref_tokens: list[str], hyp: str) -> tuple[int, int]:
    hyp_t = normalize_for_wer(hyp)
    pool = {}
    for t in hyp_t:
        pool[t] = pool.get(t, 0) + 1
    hit = 0
    for t in ref_tokens:
        if pool.get(t, 0) > 0:
            pool[t] -= 1
            hit += 1
    return hit, len(ref_tokens)


def set_f1(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    tp = len(a & b)
    return 2 * tp / (len(a) + len(b)) if (a or b) else 1.0


def load_text_head():
    """Frozen text-only head from RQ1 (trained if available, else zero-shot) + CLIP backbone."""
    from incident_reporter.ml.clip_backbone import ClipBackbone
    from incident_reporter.ml.heads import HeadSet, zero_shot_headset

    bb = ClipBackbone(cache_dir=str(HF_CACHE))
    p = ROOT / "artifacts" / "heads" / "crisismmd_heads.npz"
    return bb, (HeadSet.load(p) if p.exists() else zero_shot_headset(bb))


# ------------------------------------------------------------------ SP2


def scripted() -> None:
    reports = json.loads((ROOT / "eval_data" / "speech" / "scripted_reports.json").read_text(encoding="utf-8"))["reports"]
    tts, asr = TTS(), ASR()
    bb, heads = load_text_head()
    ref_pred = {}
    for r in reports:
        emb = bb.embed_texts([r["text"]])
        ref_pred[r["id"]] = LABELS[int(np.argmax(heads.predict("text", emb, None)))]
    rows = []
    for ri, r in enumerate(reports):
        ref_ev = extract(r["text"])
        crit = critical_tokens(r["text"])
        for vi, voice in enumerate(VOICES):
            path = SPEECH_CACHE / "scripted" / f"{r['id']}_{voice}.wav"
            if not path.exists():
                save_wav(path, tts.speak(r["text"], voice))
            clean = load_wav(path)
            for cond, snr in CONDITIONS.items():
                x = add_noise(clean, snr, seed=1000 * ri + 10 * vi + (0 if snr is None else int(snr) + 50))
                out = asr(x)
                hyp = out["text"]
                hyp_ev = extract(hyp)
                we, wn = wer(r["text"], hyp)
                ce, cn = cer(r["text"], hyp)
                hit, tot = token_recall(crit, hyp)
                pred = LABELS[int(np.argmax(heads.predict("text", bb.embed_texts([hyp or "."]), None)))]
                rows.append({
                    "id": r["id"], "voice": voice, "condition": cond, "seconds": round(len(clean) / 16000, 2),
                    "reference": r["text"], "hypothesis": hyp, "mean_logprob": out["mean_logprob"],
                    "word_edits": we, "ref_words": wn, "char_edits": ce, "ref_chars": cn,
                    "critical_hit": hit, "critical_total": tot,
                    "concepts_ref": sorted(ref_ev["asserted"]), "concepts_hyp": sorted(hyp_ev["asserted"]),
                    "negated_ref": sorted(ref_ev["negated"]), "negated_hyp": sorted(hyp_ev["negated"]),
                    "missing_ref": sorted(ref_ev["missing"]), "missing_hyp": sorted(hyp_ev["missing"]),
                    "text_view_ref": ref_pred[r["id"]], "text_view_hyp": pred,
                })
        print(f"{r['id']} done", flush=True)
    summary = {}
    for cond in CONDITIONS:
        rs = [x for x in rows if x["condition"] == cond]
        summary[cond] = {
            "clips": len(rs),
            "wer": sum(x["word_edits"] for x in rs) / sum(x["ref_words"] for x in rs),
            "cer": sum(x["char_edits"] for x in rs) / sum(x["ref_chars"] for x in rs),
            "critical_term_recall": sum(x["critical_hit"] for x in rs) / sum(x["critical_total"] for x in rs),
            "concept_set_f1_mean": float(np.mean([set_f1(set(x["concepts_ref"]), set(x["concepts_hyp"])) for x in rs])),
            "clips_with_changed_concepts": sum(x["concepts_ref"] != x["concepts_hyp"] or x["negated_ref"] != x["negated_hyp"] for x in rs),
            "clips_with_changed_missing_fields": sum(x["missing_ref"] != x["missing_hyp"] for x in rs),
            "text_view_category_flips": sum(x["text_view_ref"] != x["text_view_hyp"] for x in rs),
            "wer_by_voice": {v: sum(x["word_edits"] for x in rs if x["voice"] == v) / sum(x["ref_words"] for x in rs if x["voice"] == v) for v in VOICES},
        }
    out = {
        "protocol": "docs/experiment-design.md v1, SP2",
        "note": "Synthetic speech (Kokoro-82M) of project-authored reports; white noise added at the stated SNR. Not real recordings.",
        "asr": {"model": f"faster-whisper {ASR_MODEL}", "revision": ASR_REVISION, "compute_type": "int8", "beam_size": 5},
        "tts": {"model": "Kokoro-82M v1.0 (kokoro-onnx)", "voices": VOICES},
        "heads": heads.kind,
        "summary": summary,
        "clips": rows,
        "environment": environment(),
    }
    write_json(RESULTS / "speech_scripted.json", out)
    for c, s in summary.items():
        print(c, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "wer_by_voice"})


# ------------------------------------------------------------------ SP1


def librispeech(n: int = 100, seed: int = 13) -> None:
    import soundfile as sf

    d = CACHE / "librispeech"
    tgz = d / "test-clean.tar.gz"
    root = d / "LibriSpeech" / "test-clean"
    if not root.exists():
        d.mkdir(parents=True, exist_ok=True)
        if not tgz.exists():
            import urllib.request

            from incident_reporter.ml.clip_backbone import maybe_use_os_truststore

            maybe_use_os_truststore()
            urllib.request.urlretrieve(LIBRI_URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(d, filter="data")
    utts = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            utts.append((uid, text, trans.parent / f"{uid}.flac"))
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(utts), size=n, replace=False))
    asr = ASR()
    rows = []
    for i in pick:
        uid, ref, path = utts[i]
        x, sr = sf.read(str(path), dtype="float32")
        assert sr == 16000
        hyp = asr(x)["text"]
        we, wn = wer(ref, hyp)
        ce, cn = cer(ref, hyp)
        rows.append({"utterance": uid, "speaker": uid.split("-")[0], "seconds": round(len(x) / sr, 2), "word_edits": we, "ref_words": wn,
                     "char_edits": ce, "ref_chars": cn, "reference": ref, "hypothesis": hyp})
    total_w = sum(r["ref_words"] for r in rows)
    out = {
        "protocol": "docs/experiment-design.md v1, SP1",
        "data": "LibriSpeech test-clean (Panayotov et al., 2015), CC BY 4.0, https://www.openslr.org/12",
        "sample": {"n": n, "seed": seed, "pool": len(utts), "speakers": len({r["speaker"] for r in rows}), "hours": round(sum(r["seconds"] for r in rows) / 3600, 3)},
        "asr": {"model": f"faster-whisper {ASR_MODEL}", "revision": ASR_REVISION},
        "wer": sum(r["word_edits"] for r in rows) / total_w,
        "cer": sum(r["char_edits"] for r in rows) / sum(r["ref_chars"] for r in rows),
        "utterances_with_zero_errors": sum(r["word_edits"] == 0 for r in rows),
        "worst_examples": sorted(rows, key=lambda r: -r["word_edits"] / max(1, r["ref_words"]))[:5],
        "per_utterance": [{k: r[k] for k in ("utterance", "seconds", "word_edits", "ref_words")} for r in rows],
    }
    write_json(RESULTS / "speech_librispeech.json", out)
    print(f"LibriSpeech sample WER {out['wer']:.4f} CER {out['cer']:.4f} ({n} utterances, {out['sample']['speakers']} speakers)")


# ------------------------------------------------------------------ SP3


def sp3() -> None:
    from .metrics import macro_f1

    data = load_examples()
    te = data[data.split == "test"].reset_index()
    feats = np.load(WORK / "features.npz")
    img = feats["image"][te["index"].to_numpy()]
    y = te["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    tts, asr = TTS(), ASR()
    bb, heads = load_text_head()
    cache = SPEECH_CACHE / "sp3"
    refs, hyps_clean, hyps_noisy, wer_c, wer_n, seconds = [], [], [], [0, 0], [0, 0], 0.0
    t0 = time.time()
    for k, row in te.iterrows():
        ref = speakable(row["tweet_text"]) or "no text"
        voice = VOICES[k % len(VOICES)]
        path = cache / f"{row['image_id']}.wav"
        if not path.exists():
            save_wav(path, tts.speak(ref, voice))
        x = load_wav(path)
        seconds += len(x) / 16000
        hc = asr(x)["text"]
        hn = asr(add_noise(x, 10.0, seed=k))["text"]
        for acc, h in ((wer_c, hc), (wer_n, hn)):
            e, n = wer(ref, h)
            acc[0] += e
            acc[1] += n
        refs.append(ref)
        hyps_clean.append(hc)
        hyps_noisy.append(hn)
        if k % 50 == 0:
            print(f"{k}/{len(te)} {time.time() - t0:.0f}s", flush=True)
    variants = {
        "original_text": [clean_post(t) for t in te["tweet_text"]],
        "speakable_reference": refs,
        "asr_clean": hyps_clean,
        "asr_snr10": hyps_noisy,
    }
    res, preds = {}, {}
    for name, texts in variants.items():
        emb = bb.embed_texts([t or "." for t in texts])
        pt = np.stack([heads.predict("text", emb[i:i + 1], None) for i in range(len(texts))]).argmax(1)
        pc = np.stack([heads.predict("combined", emb[i:i + 1], img[i:i + 1]) for i in range(len(texts))]).argmax(1)
        preds[name] = (pt, pc)
        res[name] = {"text_only_macro_f1": macro_f1(y, pt), "combined_macro_f1": macro_f1(y, pc),
                     "text_only_accuracy": float((pt == y).mean()), "combined_accuracy": float((pc == y).mean())}
    base_t, base_c = preds["original_text"]
    for name in ("speakable_reference", "asr_clean", "asr_snr10"):
        pt, pc = preds[name]
        res[name]["text_only_prediction_changes_vs_original"] = int((pt != base_t).sum())
        res[name]["combined_prediction_changes_vs_original"] = int((pc != base_c).sum())
    out = {
        "protocol": "docs/experiment-design.md v1, SP3",
        "note": "Synthetic speech (Kokoro-82M) of the 955 CrisisMMD test tweets; frozen RQ1 heads; aggregate results only (dataset confidentiality).",
        "n": int(len(te)), "audio_hours": round(seconds / 3600, 2), "heads": heads.kind,
        "wer_clean": wer_c[0] / wer_c[1], "wer_snr10": wer_n[0] / wer_n[1],
        "results": res,
        "text_only_drop_original_to_asr_clean": res["original_text"]["text_only_macro_f1"] - res["asr_clean"]["text_only_macro_f1"],
        "combined_drop_original_to_asr_clean": res["original_text"]["combined_macro_f1"] - res["asr_clean"]["combined_macro_f1"],
        "asr": {"model": f"faster-whisper {ASR_MODEL}", "revision": ASR_REVISION}, "tts_voices": VOICES,
    }
    write_json(RESULTS / "speech_sp3.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("results",)}, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("scripted", "librispeech", "sp3"))
    a = ap.parse_args()
    {"scripted": scripted, "librispeech": librispeech, "sp3": sp3}[a.part]()
