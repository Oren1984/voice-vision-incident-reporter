"""Incident workflow: intake -> transcription -> analysis -> draft -> human review -> decision.

States:
  DRAFT         processed cleanly; the reporter can correct the transcript, then submit
  NEEDS_REVIEW  waiting for a reviewer (also the fail-safe state for failures, missing input,
                conflicts and low confidence)
  APPROVED      a reviewer explicitly approved; an immutable record snapshot exists
  REJECTED      a reviewer explicitly rejected, with a reason

Only `approve()` can produce APPROVED, and it requires a human reviewer, an explicit confirmation,
the current version number, acknowledgement of open questions and (by default) a reviewer who did not
create the incident. No model output can change the status to APPROVED.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass

import numpy as np

from . import audit
from .compare import asr_signals, compare
from .config import Settings
from .db import Database, dumps, loads, now
from .draft import build
from .evidence import extract
from .labels import LABELS
from .media import MediaError, MediaStore, validate_audio, validate_image
from .models import ModelOutputError, band, image_quality, validate_observations, validate_proba, validate_transcript
from .observability import Metrics, request_id_var, timed
from .security import User
from .text_clean import edit_distance

log = logging.getLogger("vvir")

TERMINAL = ("APPROVED", "REJECTED")
FAILSAFE_FLAGS_PREFIX = ("model_failure", "missing_", "no_speech", "sources_conflict", "low_confidence", "asr_low_confidence", "image_quality")


class WorkflowError(Exception):
    """A request that is not allowed in the current state. Message is safe to show."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Upload:
    data: bytes | None


class IncidentService:
    def __init__(self, db: Database, store: MediaStore, models, settings: Settings, metrics: Metrics) -> None:
        self.db, self.store, self.models, self.settings, self.metrics = db, store, models, settings, metrics

    # ------------------------------------------------------------ access
    def get(self, user: User, incident_id: str) -> dict:
        row = self.db.one("SELECT i.*, u.username AS creator FROM incidents i JOIN users u ON u.id = i.created_by WHERE i.id = ?", (incident_id,))
        if row is None or (not user.can_review and row["created_by"] != user.id):
            raise WorkflowError("Incident not found.", 404)  # same answer for "missing" and "not yours"
        inc = dict(row)
        for k, d in (("audio_meta", None), ("image_meta", None), ("asr_meta", None), ("analysis", None), ("draft", None), ("flags", [])):
            inc[k] = loads(inc[k], d)
        if inc["decided_by"]:
            inc["decider"] = self.db.one("SELECT username FROM users WHERE id = ?", (inc["decided_by"],))["username"]
        return inc

    def list(self, user: User) -> list[dict]:
        q = "SELECT i.id, i.status, i.created_at, i.updated_at, i.flags, i.draft, u.username AS creator FROM incidents i JOIN users u ON u.id = i.created_by"
        rows = self.db.all(q + " ORDER BY i.created_at DESC") if user.can_review else self.db.all(q + " WHERE i.created_by = ? ORDER BY i.created_at DESC", (user.id,))
        out = []
        for r in rows:
            d = loads(r["draft"], {}) or {}
            out.append({**dict(r), "flags": loads(r["flags"], []), "category": (d.get("category") or {}).get("proposed"),
                        "open_questions": sum(1 for q in d.get("unresolved_questions", []) if not q.get("resolution"))})
        return out

    # ------------------------------------------------------------ intake
    def create(self, user: User, audio: bytes | None, image: bytes | None) -> str:
        rid = request_id_var.get()
        if not audio and not image:
            raise MediaError("no_input", "Upload an audio recording, an image, or both.")
        lim = self.settings.limits
        try:
            a = validate_audio(audio, lim) if audio else None
            i = validate_image(image, lim) if image else None
        except MediaError as e:
            self.metrics.inc("uploads.rejected")
            audit.record(self.db, user.username, "upload_rejected", request_id=rid, code=e.code)
            log.info("upload_rejected", extra={"code": e.code, "user": user.username})
            raise
        incident_id = "inc_" + secrets.token_hex(6)
        ts = now()
        flags = []
        if a is None:
            flags.append("missing_audio")
        if i is None:
            flags.append("missing_image")
        with self.db.transaction():
            self.db.execute(
                "INSERT INTO incidents (id, created_by, created_at, updated_at, status, audio_file, audio_sha256, audio_meta, image_file, image_sha256, image_meta, flags)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (incident_id, user.id, ts, ts, "DRAFT",
                 self.store.save(incident_id, f"audio.{a.ext}", a.data) if a else None, a.sha256 if a else None, dumps(a.meta) if a else None,
                 self.store.save(incident_id, "image.jpg", i.jpeg) if i else None, i.sha256_stored if i else None,
                 dumps({**i.meta, "sha256_original": i.sha256_original}) if i else None, dumps(flags)),
            )
            audit.record(self.db, user.username, "incident_created", incident_id, rid,
                         audio={"sha256": a.sha256, **a.meta} if a else None,
                         image={"sha256_original": i.sha256_original, "sha256_stored": i.sha256_stored, **i.meta} if i else None)
        self.metrics.inc("incidents.created")
        log.info("incident_created", extra={"incident_id": incident_id, "has_audio": a is not None, "has_image": i is not None})

        transcript = None
        if a is not None:
            transcript = self._transcribe(user, incident_id, a.samples, flags)
        self._safe_analyze(user, incident_id, transcript, i.image if i else None, flags, transcript_edited=False)
        return incident_id

    def _transcribe(self, user: User, incident_id: str, samples: np.ndarray, flags: list[str]) -> str | None:
        rid = request_id_var.get()
        try:
            with timed(self.metrics, "stage.transcribe") as t:
                out = validate_transcript(self.models.transcribe(samples))
        except Exception as e:  # model crash or invalid output -> review, never a silent result
            flags.append("model_failure:asr")
            self.metrics.inc("model_failures.asr")
            audit.record(self.db, "system", "transcription_failed", incident_id, rid, error_type=type(e).__name__)
            log.warning("transcription_failed", extra={"incident_id": incident_id, "error_type": type(e).__name__})
            return None
        text = out["text"][: self.settings.limits.transcript_max_chars]
        sig = asr_signals(out)
        if not text.strip():
            flags.append("no_speech")
        elif sig["low_confidence"]:
            flags.append("asr_low_confidence")
        meta = {"model": out.get("model"), "segments": out["segments"], "seconds": round(t.elapsed, 2), **sig}
        self.db.execute("UPDATE incidents SET transcript_original = ?, asr_meta = ?, updated_at = ? WHERE id = ?",
                        (text, dumps(meta), now(), incident_id))
        audit.record(self.db, "system", "transcribed", incident_id, rid, model=out.get("model"), chars=len(text),
                     words=len(text.split()), duration_s=round(t.elapsed, 2), low_confidence=sig["low_confidence"],
                     transcript_sha256=hashlib.sha256(text.encode()).hexdigest())
        log.info("transcribed", extra={"incident_id": incident_id, "duration_ms": round(1000 * t.elapsed), "words": len(text.split())})
        return text if text.strip() else None

    # ------------------------------------------------------------ analysis
    def _view(self, name: str, fn) -> dict:
        try:
            p = validate_proba(fn(), name)
        except Exception as e:
            self.metrics.inc(f"model_failures.view_{name}")
            return {"view": name, "available": False, "reason": f"model error ({type(e).__name__})", "failed": True}
        k = int(np.argmax(p))
        return {"view": name, "available": True, "proba": [round(float(x), 4) for x in p], "label": LABELS[k], "confidence": float(p[k])}

    def _safe_analyze(self, user: User, incident_id: str, transcript: str | None, img, flags: list[str], transcript_edited: bool) -> None:
        """Run the analysis; if anything unexpected breaks, park the incident in NEEDS_REVIEW instead of failing."""
        try:
            self._analyze(user, incident_id, transcript, img, flags, transcript_edited)
        except WorkflowError:
            raise
        except Exception as e:
            self.metrics.inc("model_failures.analysis")
            log.exception("analysis_failed", extra={"incident_id": incident_id})
            with self.db.transaction():
                cur = self.db.one("SELECT flags FROM incidents WHERE id = ?", (incident_id,))
                new_flags = sorted(set(loads(cur["flags"], []) + flags + ["model_failure:analysis"]))
                self.db.execute("UPDATE incidents SET status = 'NEEDS_REVIEW', flags = ?, version = version + 1, updated_at = ? WHERE id = ?",
                                (dumps(new_flags), now(), incident_id))
                audit.record(self.db, "system", "analysis_failed", incident_id, request_id_var.get(), error_type=type(e).__name__)

    def _analyze(self, user: User, incident_id: str, transcript: str | None, img, flags: list[str], transcript_edited: bool) -> None:
        rid = request_id_var.get()
        flags = [f for f in flags if not f.startswith(("sources_conflict", "low_confidence", "image_quality", "model_failure:clip", "model_failure:view"))]
        text_ev = extract(transcript) if transcript else None
        text_emb = image_emb = None
        image_block = None
        with timed(self.metrics, "stage.analyze") as t:
            if transcript:
                try:
                    text_emb = np.asarray(self.models.embed_text(transcript), dtype=np.float32)
                    if text_emb.shape != (1, 512) or not np.isfinite(text_emb).all():
                        raise ModelOutputError("invalid text embedding")
                except Exception as e:
                    flags.append("model_failure:clip_text")
                    text_emb = None
                    log.warning("text_embedding_failed", extra={"incident_id": incident_id, "error_type": type(e).__name__})
            if img is not None:
                image_block = {"quality": image_quality(img), "observations": []}
                try:
                    image_emb = np.asarray(self.models.embed_image(img), dtype=np.float32)
                    if image_emb.shape != (1, 512) or not np.isfinite(image_emb).all():
                        raise ModelOutputError("invalid image embedding")
                    obs = validate_observations(self.models.observations(image_emb))
                    for o in obs:
                        o["band"] = band(o["score"])  # recomputed here; never trusted from the model layer
                    image_block["observations"] = sorted(obs, key=lambda o: -o["score"])
                except Exception as e:
                    flags.append("model_failure:clip_image")
                    image_emb = None
                    image_block["error"] = type(e).__name__
                    log.warning("image_analysis_failed", extra={"incident_id": incident_id, "error_type": type(e).__name__})
                if image_block["quality"]["flags"]:
                    flags.append("image_quality")

            def unavailable(name, why):
                return {"view": name, "available": False, "reason": why}

            views = {
                "text": self._view("text", lambda: self.models.classify("text", text_emb, None)) if text_emb is not None
                else unavailable("text", "no usable transcript"),
                "image": self._view("image", lambda: self.models.classify("image", None, image_emb)) if image_emb is not None
                else unavailable("image", "no usable image"),
                "combined": self._view("combined", lambda: self.models.classify("combined", text_emb, image_emb))
                if text_emb is not None and image_emb is not None else unavailable("combined", "needs both a transcript and an image"),
            }
            for v in views.values():
                if v.get("failed"):
                    flags.append(f"model_failure:view_{v['view']}")
            if any(v.get("available") and v["confidence"] < 0.5 for v in views.values()):
                flags.append("low_confidence")
            asr = self.db.one("SELECT asr_meta FROM incidents WHERE id = ?", (incident_id,))
            asr_meta = loads(asr["asr_meta"]) if asr else None
            comparison = compare(text_ev, image_block, views, asr_meta, transcript_edited)
            if comparison["contradictions"]:
                flags.append("sources_conflict")
            version_label = "corrected" if transcript_edited else "original"
            draft = build(text_ev, image_block, views, comparison, flags, version_label)

        analysis = {
            "text_evidence": text_ev, "image": image_block, "views": views, "comparison": comparison,
            "heads": getattr(self.models, "heads_kind", "unknown"), "transcript_version": version_label,
            "duration_s": round(t.elapsed, 3),
        }
        failsafe = any(f.startswith(FAILSAFE_FLAGS_PREFIX) for f in flags)
        with self.db.transaction():
            cur = self.db.one("SELECT status, version FROM incidents WHERE id = ?", (incident_id,))
            if cur["status"] in TERMINAL:
                raise WorkflowError("This incident is closed.", 409)
            status = "NEEDS_REVIEW" if failsafe or cur["status"] == "NEEDS_REVIEW" else "DRAFT"
            self.db.execute("UPDATE incidents SET analysis = ?, draft = ?, flags = ?, status = ?, version = version + 1, updated_at = ? WHERE id = ?",
                            (dumps(analysis), dumps(draft), dumps(sorted(set(flags))), status, now(), incident_id))
            self._revision(incident_id, cur["version"] + 1, "system", "analysis", {"regenerated": True}, draft)
            audit.record(self.db, "system", "analyzed", incident_id, rid, status=status, flags=sorted(set(flags)),
                         views={k: (v.get("label"), round(v["confidence"], 3)) if v.get("available") else None for k, v in views.items()},
                         contradictions=len(comparison["contradictions"]), questions=len(draft["unresolved_questions"]),
                         duration_s=round(t.elapsed, 3), heads=analysis["heads"])
        if failsafe:
            self.metrics.inc("incidents.failsafe_review")
        log.info("analyzed", extra={"incident_id": incident_id, "status": status, "flags": sorted(set(flags)), "duration_ms": round(1000 * t.elapsed)})

    def _revision(self, incident_id: str, version: int, author: str, reason: str, changes: dict, draft: dict) -> None:
        self.db.execute("INSERT INTO draft_revisions (incident_id, version, author, created_at, reason, changes, draft) VALUES (?,?,?,?,?,?,?)",
                        (incident_id, version, author, now(), reason, dumps(changes), dumps(draft)))

    def _load_image(self, inc: dict):
        from PIL import Image

        if not inc["image_file"]:
            return None
        with Image.open(self.store.path(inc["id"], inc["image_file"])) as im:
            return im.convert("RGB")

    def _check_open(self, inc: dict, version: int | None) -> None:
        if inc["status"] in TERMINAL:
            raise WorkflowError("This incident is closed and can no longer be changed.", 409)
        if version is not None and int(version) != inc["version"]:
            raise WorkflowError("The incident changed since you opened it. Reload and review the latest version.", 409)

    # ------------------------------------------------------------ transcript correction
    def correct_transcript(self, user: User, incident_id: str, text: str, version: int) -> None:
        inc = self.get(user, incident_id)
        self._check_open(inc, version)
        if not user.can_review and inc["created_by"] != user.id:
            raise WorkflowError("Only the reporter or a reviewer can correct the transcript.", 403)
        if not inc["audio_file"]:
            raise WorkflowError("This incident has no audio recording.", 400)
        text = (text or "").strip()
        if len(text) > self.settings.limits.transcript_max_chars:
            raise WorkflowError("The transcript is too long.", 400)
        if not text:
            raise WorkflowError("The corrected transcript cannot be empty.", 400)
        before = inc["transcript_corrected"] or inc["transcript_original"] or ""
        if text == before:
            return
        orig = (inc["transcript_original"] or "").split()
        self.db.execute("UPDATE incidents SET transcript_corrected = ?, updated_at = ? WHERE id = ?", (text, now(), incident_id))
        audit.record(self.db, user.username, "transcript_corrected", incident_id, request_id_var.get(),
                     word_edits_vs_original=edit_distance(orig, text.split()), words=len(text.split()),
                     transcript_sha256=hashlib.sha256(text.encode()).hexdigest())
        self.metrics.inc("transcripts.corrected")
        flags = [f for f in inc["flags"] if f not in ("no_speech", "asr_low_confidence", "model_failure:asr")]
        self._safe_analyze(user, incident_id, text, self._load_image(inc), flags, transcript_edited=True)

    def submit(self, user: User, incident_id: str, version: int) -> None:
        inc = self.get(user, incident_id)
        self._check_open(inc, version)
        if inc["status"] != "DRAFT":
            raise WorkflowError("Only a draft can be submitted for review.", 409)
        if inc["created_by"] != user.id and not user.can_review:
            raise WorkflowError("Only the reporter can submit this draft.", 403)
        self.db.execute("UPDATE incidents SET status = 'NEEDS_REVIEW', version = version + 1, updated_at = ? WHERE id = ?", (now(), incident_id))
        audit.record(self.db, user.username, "submitted_for_review", incident_id, request_id_var.get())

    # ------------------------------------------------------------ reviewer actions
    def edit_draft(self, user: User, incident_id: str, version: int, category: str | None, summary: str | None,
                   notes: str | None, resolutions: dict[str, str]) -> None:
        if not user.can_review:
            raise WorkflowError("Only reviewers can edit the draft.", 403)
        inc = self.get(user, incident_id)
        self._check_open(inc, version)
        if inc["status"] != "NEEDS_REVIEW":
            raise WorkflowError("The draft can be edited once it is in review.", 409)
        lim = self.settings.limits.text_field_max_chars
        draft = inc["draft"]
        changes: dict = {}
        ts = now()
        if category is not None and category != (draft["category"].get("final") or draft["category"]["proposed"]):
            if category not in LABELS:
                raise WorkflowError("Unknown category.", 400)
            changes["category"] = {"from": draft["category"].get("final") or draft["category"]["proposed"], "to": category}
            draft["category"]["final"] = category
            draft["category"]["decided_by"] = user.username
        if summary is not None and summary.strip() != draft["summary"]:
            if len(summary) > lim:
                raise WorkflowError("Summary is too long.", 400)
            changes["summary"] = {"chars_from": len(draft["summary"]), "chars_to": len(summary.strip()),
                                  "word_edits": edit_distance(draft["summary"].split(), summary.split())}
            draft["summary"] = summary.strip()
            draft["summary_edited"] = True
            draft["summary_edited_by"] = user.username
        if notes is not None and notes.strip() != draft.get("reviewer_notes", ""):
            if len(notes) > lim:
                raise WorkflowError("Notes are too long.", 400)
            changes["reviewer_notes"] = {"chars_to": len(notes.strip())}
            draft["reviewer_notes"] = notes.strip()
        for q in draft["unresolved_questions"]:
            ans = (resolutions.get(q["id"]) or "").strip()
            if ans and ans != ((q.get("resolution") or {}).get("text")):
                if len(ans) > lim:
                    raise WorkflowError("Answer is too long.", 400)
                q["resolution"] = {"text": ans, "by": user.username, "at": ts}
                changes.setdefault("resolved_questions", []).append(q["id"])
        if not changes:
            return
        with self.db.transaction():
            self.db.execute("UPDATE incidents SET draft = ?, version = version + 1, updated_at = ? WHERE id = ?", (dumps(draft), ts, incident_id))
            self._revision(incident_id, inc["version"] + 1, user.username, "reviewer_edit", changes, draft)
            audit.record(self.db, user.username, "draft_edited", incident_id, request_id_var.get(), changed=sorted(changes), details=changes)
        self.metrics.inc("drafts.edited")

    def approve(self, user: User, incident_id: str, version: int, confirmed: bool, acknowledge_open: bool) -> None:
        if not user.can_review:
            raise WorkflowError("Only reviewers can approve incidents.", 403)
        inc = self.get(user, incident_id)
        self._check_open(inc, version)
        if inc["status"] != "NEEDS_REVIEW":
            raise WorkflowError("Only incidents in review can be approved.", 409)
        if self.settings.four_eyes and inc["created_by"] == user.id:
            raise WorkflowError("You created this incident; another reviewer must approve it.", 403)
        if not confirmed:
            raise WorkflowError("Tick the confirmation to approve.", 400)
        draft = inc["draft"]
        open_q = [q["id"] for q in draft["unresolved_questions"] if not q.get("resolution")]
        if open_q and not acknowledge_open:
            raise WorkflowError(f"{len(open_q)} question(s) are unresolved. Answer them or explicitly acknowledge approving with open questions.", 400)
        final_category = draft["category"].get("final") or draft["category"]["proposed"]
        if final_category is None:
            raise WorkflowError("Choose a category before approving.", 400)
        ts = now()
        record = {
            "incident_id": incident_id, "category": final_category, "summary": draft["summary"],
            "summary_edited": draft.get("summary_edited", False), "claims": draft["claims"],
            "questions": draft["unresolved_questions"], "open_questions_acknowledged": open_q,
            "reviewer_notes": draft.get("reviewer_notes", ""), "flags": inc["flags"],
            "transcript_used": inc["transcript_corrected"] or inc["transcript_original"],
            "transcript_version": "corrected" if inc["transcript_corrected"] else "original",
            "media": {"audio_sha256": inc["audio_sha256"], "image_sha256": inc["image_sha256"]},
            "approved_by": user.username, "approved_at": ts, "draft_version": inc["version"],
        }
        rec_json = dumps(record)
        sha = hashlib.sha256(rec_json.encode()).hexdigest()
        with self.db.transaction():
            n = self.db.execute("UPDATE incidents SET status = 'APPROVED', decided_by = ?, decided_at = ?, version = version + 1, updated_at = ?"
                                " WHERE id = ? AND version = ? AND status = 'NEEDS_REVIEW'", (user.id, ts, ts, incident_id, inc["version"])).rowcount
            if n != 1:
                raise WorkflowError("The incident changed while approving. Reload and try again.", 409)
            self.db.execute("INSERT INTO approved_records (incident_id, approved_by, approved_at, record, record_sha256) VALUES (?,?,?,?,?)",
                            (incident_id, user.id, ts, rec_json, sha))
            audit.record(self.db, user.username, "approved", incident_id, request_id_var.get(), record_sha256=sha, category=final_category,
                         open_questions_acknowledged=open_q, draft_version=inc["version"])
        self.metrics.inc("reviews.approved")
        log.info("approved", extra={"incident_id": incident_id, "reviewer": user.username})

    def reject(self, user: User, incident_id: str, version: int, reason: str) -> None:
        if not user.can_review:
            raise WorkflowError("Only reviewers can reject incidents.", 403)
        inc = self.get(user, incident_id)
        self._check_open(inc, version)
        if inc["status"] != "NEEDS_REVIEW":
            raise WorkflowError("Only incidents in review can be rejected.", 409)
        reason = (reason or "").strip()
        if len(reason) < 5:
            raise WorkflowError("Give a reason for rejecting (at least 5 characters).", 400)
        if len(reason) > self.settings.limits.text_field_max_chars:
            raise WorkflowError("Reason is too long.", 400)
        ts = now()
        with self.db.transaction():
            self.db.execute("UPDATE incidents SET status = 'REJECTED', decided_by = ?, decided_at = ?, decision_note = ?, version = version + 1, updated_at = ?"
                            " WHERE id = ?", (user.id, ts, reason, ts, incident_id))
            audit.record(self.db, user.username, "rejected", incident_id, request_id_var.get(), reason_chars=len(reason))
        self.metrics.inc("reviews.rejected")

    def approved_record(self, user: User, incident_id: str) -> dict | None:
        self.get(user, incident_id)  # access check
        row = self.db.one("SELECT record, record_sha256 FROM approved_records WHERE incident_id = ?", (incident_id,))
        if row is None:
            return None
        ok = hashlib.sha256(row["record"].encode()).hexdigest() == row["record_sha256"]
        return {"record": loads(row["record"]), "sha256": row["record_sha256"], "integrity_ok": ok}

    def revisions(self, user: User, incident_id: str) -> list[dict]:
        self.get(user, incident_id)
        return [{**dict(r), "changes": loads(r["changes"], {})} for r in
                self.db.all("SELECT id, version, author, created_at, reason, changes FROM draft_revisions WHERE incident_id = ? ORDER BY id", (incident_id,))]

    # ------------------------------------------------------------ ops
    def stats(self) -> dict:
        by_status = {r["status"]: r["n"] for r in self.db.all("SELECT status, COUNT(*) n FROM incidents GROUP BY status")}
        flags: dict[str, int] = {}
        for r in self.db.all("SELECT flags FROM incidents"):
            for f in loads(r["flags"], []):
                flags[f] = flags.get(f, 0) + 1
        decisions = {r["action"]: r["n"] for r in self.db.all(
            "SELECT action, COUNT(*) n FROM audit_events WHERE action IN ('approved','rejected','upload_rejected','transcription_failed','transcript_corrected','draft_edited') GROUP BY action")}
        edited_before_approval = self.db.one(
            "SELECT COUNT(DISTINCT a.incident_id) n FROM approved_records a JOIN draft_revisions r ON r.incident_id = a.incident_id AND r.reason = 'reviewer_edit'")["n"]
        return {"incidents_by_status": by_status, "flag_counts": flags, "audit_action_counts": decisions,
                "approved_after_reviewer_edit": edited_before_approval}
