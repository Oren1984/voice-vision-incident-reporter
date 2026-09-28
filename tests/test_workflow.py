"""Core workflow, approval boundary, fail-safe states, traceability and audit."""

import sqlite3

import pytest

from incident_reporter import audit
from incident_reporter.media import MediaError
from incident_reporter.workflow import WorkflowError

from .conftest import make_image, make_wav, onehot


def create(svc, user, audio=True, image=True, **img_kw):
    return svc.create(user, make_wav(3.0) if audio else None, make_image(**img_kw) if image else None)


def to_review(svc, users, iid):
    inc = svc.get(users["reporter"], iid)
    if inc["status"] == "DRAFT":
        svc.submit(users["reporter"], iid, inc["version"])
    return svc.get(users["reviewer"], iid)


def approve_ok(svc, users, iid):
    inc = to_review(svc, users, iid)
    svc.approve(users["reviewer"], iid, inc["version"], confirmed=True, acknowledge_open=True)
    return svc.get(users["reviewer"], iid)


# ---------------------------------------------------------------- happy path


def test_clean_report_is_draft_then_review_then_approved(env):
    svc, db, users, stub = env
    iid = create(svc, users["reporter"])
    inc = svc.get(users["reporter"], iid)
    assert inc["status"] == "DRAFT", inc["flags"]
    assert inc["transcript_original"] == stub.transcript
    assert set(inc["analysis"]["views"]) == {"text", "image", "combined"}
    assert all(v["available"] for v in inc["analysis"]["views"].values())
    assert inc["draft"]["category"]["proposed"] == "infrastructure_and_utility_damage"
    inc = approve_ok(svc, users, iid)
    assert inc["status"] == "APPROVED" and inc["decider"] == "vera.reviewer"
    rec = svc.approved_record(users["reviewer"], iid)
    assert rec["integrity_ok"] and rec["record"]["approved_by"] == "vera.reviewer"
    actions = [e["action"] for e in audit.events(db, iid)]
    for a in ("incident_created", "transcribed", "analyzed", "submitted_for_review", "approved"):
        assert a in actions
    assert audit.verify(db)["ok"]


def test_agreement_between_transcript_and_image(env):
    svc, _, users, _ = env
    iid = create(svc, users["reporter"])
    cmp = svc.get(users["reporter"], iid)["analysis"]["comparison"]
    assert any(a["concept"] == "fire_smoke" and a["strength"] == "strong" for a in cmp["agreements"])
    # details only the report can provide are complementary, not agreements
    assert any(c["concept"] == "location" for c in cmp["complementary"])


# ---------------------------------------------------------------- approval boundary


def test_reporter_cannot_approve_or_edit(env):
    svc, _, users, _ = env
    iid = create(svc, users["reporter"])
    inc = to_review(svc, users, iid)
    with pytest.raises(WorkflowError) as e:
        svc.approve(users["reporter"], iid, inc["version"], True, True)
    assert e.value.status == 403
    with pytest.raises(WorkflowError):
        svc.edit_draft(users["reporter"], iid, inc["version"], "not_humanitarian", None, None, {})


def test_four_eyes_creator_cannot_approve_own_incident(env):
    svc, _, users, _ = env
    iid = create(svc, users["reviewer"])
    inc = svc.get(users["reviewer"], iid)
    if inc["status"] == "DRAFT":
        svc.submit(users["reviewer"], iid, inc["version"])
    inc = svc.get(users["reviewer"], iid)
    with pytest.raises(WorkflowError, match="another reviewer"):
        svc.approve(users["reviewer"], iid, inc["version"], True, True)
    svc.approve(users["reviewer2"], iid, inc["version"], True, True)
    assert svc.get(users["reviewer2"], iid)["status"] == "APPROVED"


def test_approval_requires_explicit_confirmation_and_ack_of_open_questions(env):
    svc, _, users, stub = env
    stub.transcript = "Something happened."  # no location/time/hazard -> open questions
    iid = create(svc, users["reporter"])
    inc = to_review(svc, users, iid)
    assert any(not q["resolution"] for q in inc["draft"]["unresolved_questions"])
    with pytest.raises(WorkflowError, match="confirmation"):
        svc.approve(users["reviewer"], iid, inc["version"], confirmed=False, acknowledge_open=True)
    with pytest.raises(WorkflowError, match="unresolved"):
        svc.approve(users["reviewer"], iid, inc["version"], confirmed=True, acknowledge_open=False)
    # answering every question removes the need for the acknowledgement
    answers = {q["id"]: "Checked by phone with the caller." for q in inc["draft"]["unresolved_questions"]}
    svc.edit_draft(users["reviewer"], iid, inc["version"], None, None, None, answers)
    inc = svc.get(users["reviewer"], iid)
    svc.approve(users["reviewer"], iid, inc["version"], confirmed=True, acknowledge_open=False)
    assert svc.get(users["reviewer"], iid)["status"] == "APPROVED"


def test_draft_cannot_be_approved_before_review(env):
    svc, _, users, _ = env
    iid = create(svc, users["reporter"])
    inc = svc.get(users["reviewer"], iid)
    assert inc["status"] == "DRAFT"
    with pytest.raises(WorkflowError, match="in review"):
        svc.approve(users["reviewer"], iid, inc["version"], True, True)


def test_stale_version_is_refused(env):
    svc, _, users, _ = env
    iid = create(svc, users["reporter"])
    inc = to_review(svc, users, iid)
    svc.edit_draft(users["reviewer"], iid, inc["version"], "not_humanitarian", None, None, {})
    with pytest.raises(WorkflowError) as e:
        svc.approve(users["reviewer2"], iid, inc["version"], True, True)  # approving what they saw, not what exists
    assert e.value.status == 409


def test_approved_and_rejected_are_terminal(env):
    svc, db, users, _ = env
    iid = create(svc, users["reporter"])
    inc = approve_ok(svc, users, iid)
    for fn, args in ((svc.correct_transcript, (users["reviewer"], iid, "changed", inc["version"])),
                     (svc.edit_draft, (users["reviewer"], iid, inc["version"], "not_humanitarian", None, None, {})),
                     (svc.reject, (users["reviewer"], iid, inc["version"], "too late now")),
                     (svc.approve, (users["reviewer2"], iid, inc["version"], True, True))):
        with pytest.raises(WorkflowError) as e:
            fn(*args)
        assert e.value.status == 409
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        db.execute("UPDATE approved_records SET record = '{}' WHERE incident_id = ?", (iid,))
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        db.execute("DELETE FROM approved_records WHERE incident_id = ?", (iid,))


def test_reject_needs_reason_and_is_audited(env):
    svc, db, users, _ = env
    iid = create(svc, users["reporter"])
    inc = to_review(svc, users, iid)
    with pytest.raises(WorkflowError, match="reason"):
        svc.reject(users["reviewer"], iid, inc["version"], "")
    svc.reject(users["reviewer"], iid, inc["version"], "Duplicate of an earlier report")
    inc = svc.get(users["reviewer"], iid)
    assert inc["status"] == "REJECTED" and inc["decision_note"].startswith("Duplicate")
    assert svc.approved_record(users["reviewer"], iid) is None
    assert audit.events(db, iid)[-1]["action"] == "rejected"


def test_reviewer_edit_is_recorded_with_changes(env):
    svc, db, users, _ = env
    iid = create(svc, users["reporter"])
    inc = to_review(svc, users, iid)
    svc.edit_draft(users["reviewer"], iid, inc["version"], "affected_individuals", "Fire at Main Street, two injured.", "called caller back", {})
    inc = svc.get(users["reviewer"], iid)
    assert inc["draft"]["category"]["final"] == "affected_individuals"
    assert inc["draft"]["summary_edited"] and inc["draft"]["summary_edited_by"] == "vera.reviewer"
    rev = svc.revisions(users["reviewer"], iid)[-1]
    assert rev["author"] == "vera.reviewer" and set(rev["changes"]) == {"category", "summary", "reviewer_notes"}
    ev = [e for e in audit.events(db, iid) if e["action"] == "draft_edited"][-1]
    assert ev["actor"] == "vera.reviewer" and "Fire at Main Street" not in str(ev["details"])  # no raw text in audit
    approve_ok(svc, users, iid)
    assert svc.approved_record(users["reviewer"], iid)["record"]["category"] == "affected_individuals"


def test_reporters_only_see_their_own_incidents(env):
    svc, _, users, _ = env
    iid = create(svc, users["reporter"])
    with pytest.raises(WorkflowError) as e:
        svc.get(users["reporter2"], iid)
    assert e.value.status == 404
    assert [i["id"] for i in svc.list(users["reporter2"])] == []
    assert [i["id"] for i in svc.list(users["reviewer"])] == [iid]


# ---------------------------------------------------------------- transcript correction


def test_transcript_correction_keeps_original_and_reanalyses(env):
    svc, db, users, stub = env
    stub.transcript = "There is a fly on Main Street."  # ASR error: "fire" heard as "fly"
    iid = create(svc, users["reporter"])
    inc = svc.get(users["reporter"], iid)
    assert "fire_smoke" not in inc["analysis"]["text_evidence"]["asserted"]
    svc.correct_transcript(users["reporter"], iid, "There is a fire on Main Street.", inc["version"])
    inc = svc.get(users["reporter"], iid)
    assert inc["transcript_original"] == "There is a fly on Main Street."
    assert inc["transcript_corrected"] == "There is a fire on Main Street."
    assert "fire_smoke" in inc["analysis"]["text_evidence"]["asserted"]
    assert inc["analysis"]["transcript_version"] == "corrected"
    ev = [e for e in audit.events(db, iid) if e["action"] == "transcript_corrected"][0]
    assert ev["details"]["word_edits_vs_original"] == 1 and "fire" not in str(ev["details"])


# ---------------------------------------------------------------- fail-safe states


def test_missing_image_goes_to_review(env):
    svc, _, users, _ = env
    inc = svc.get(users["reporter"], create(svc, users["reporter"], image=False))
    assert inc["status"] == "NEEDS_REVIEW" and "missing_image" in inc["flags"]
    assert not inc["analysis"]["views"]["image"]["available"] and not inc["analysis"]["views"]["combined"]["available"]
    assert inc["draft"]["category"]["basis"] == "text" and inc["draft"]["category"]["needs_decision"]


def test_missing_audio_goes_to_review(env):
    svc, _, users, _ = env
    inc = svc.get(users["reporter"], create(svc, users["reporter"], audio=False))
    assert inc["status"] == "NEEDS_REVIEW" and "missing_audio" in inc["flags"]
    assert inc["analysis"]["views"]["image"]["available"] and not inc["analysis"]["views"]["text"]["available"]


def test_no_input_is_rejected_and_audited(env):
    svc, db, users, _ = env
    with pytest.raises(MediaError):
        svc.create(users["reporter"], None, None)
    with pytest.raises(MediaError):
        svc.create(users["reporter"], b"not audio at all", None)
    assert audit.events(db)[0]["action"] == "upload_rejected"
    assert db.one("SELECT COUNT(*) n FROM incidents")["n"] == 0


def test_no_speech_goes_to_review(env):
    svc, _, users, stub = env
    stub.transcript = ""
    inc = svc.get(users["reporter"], create(svc, users["reporter"]))
    assert inc["status"] == "NEEDS_REVIEW" and "no_speech" in inc["flags"]
    assert not inc["analysis"]["views"]["text"]["available"]


def test_unclear_speech_low_asr_confidence_goes_to_review(env):
    svc, _, users, stub = env
    stub.asr_logprob = -1.4
    inc = svc.get(users["reporter"], create(svc, users["reporter"]))
    assert inc["status"] == "NEEDS_REVIEW" and "asr_low_confidence" in inc["flags"]
    assert any(u["kind"] == "asr_low_confidence" for u in inc["analysis"]["comparison"]["uncertainty"])


def test_asr_crash_is_contained(env):
    svc, db, users, stub = env
    stub.fail.add("asr")
    inc = svc.get(users["reporter"], create(svc, users["reporter"]))
    assert inc["status"] == "NEEDS_REVIEW" and "model_failure:asr" in inc["flags"]
    assert inc["analysis"]["views"]["image"]["available"]  # the rest still ran
    assert any(e["action"] == "transcription_failed" for e in audit.events(db, inc["id"]))


@pytest.mark.parametrize("bad", [
    {"text": None, "segments": []},
    {"text": "ok"},
    {"text": "ok", "segments": [{"start": 0, "end": 1, "avg_logprob": float("nan"), "no_speech_prob": 0.1}]},
])
def test_invalid_asr_output_is_not_used(env, bad):
    svc, _, users, stub = env
    stub.raw_transcribe = bad
    inc = svc.get(users["reporter"], create(svc, users["reporter"]))
    assert inc["status"] == "NEEDS_REVIEW" and "model_failure:asr" in inc["flags"]
    assert inc["transcript_original"] is None


@pytest.mark.parametrize("fault,flag", [("classify:combined", "model_failure:view_combined"),
                                        ("shape:text", "model_failure:view_text"),
                                        ("observations", "model_failure:clip_image"),
                                        ("embed_image", "model_failure:clip_image"),
                                        ("embed_text", "model_failure:clip_text")])
def test_invalid_or_failed_model_output_goes_to_review(env, fault, flag):
    svc, _, users, stub = env
    stub.fail.add(fault)
    inc = svc.get(users["reporter"], create(svc, users["reporter"]))
    assert inc["status"] == "NEEDS_REVIEW" and flag in inc["flags"]


def test_low_confidence_goes_to_review(env):
    svc, _, users, stub = env
    stub.proba = {v: onehot("other_relevant_information", 0.35) for v in ("text", "image", "combined")}
    inc = svc.get(users["reporter"], create(svc, users["reporter"]))
    assert inc["status"] == "NEEDS_REVIEW" and "low_confidence" in inc["flags"]


def test_unclear_image_is_flagged(env):
    svc, _, users, _ = env
    inc = svc.get(users["reporter"], create(svc, users["reporter"], sharp=False, brightness=25))
    q = inc["analysis"]["image"]["quality"]["flags"]
    assert "too_dark" in q and "blurry" in q
    assert inc["status"] == "NEEDS_REVIEW" and "image_quality" in inc["flags"]


# ---------------------------------------------------------------- conflicts


def test_hazard_conflict_is_reported_not_resolved(env):
    svc, _, users, stub = env
    stub.transcript = "The river flooded Oak Road about 20 minutes ago and water is rising."
    stub.obs = {"fire_smoke": 0.92, "flood_water": 0.05}
    inc = svc.get(users["reporter"], create(svc, users["reporter"]))
    c = inc["analysis"]["comparison"]["contradictions"]
    assert any(x["kind"] == "hazard" for x in c)
    assert inc["status"] == "NEEDS_REVIEW" and "sources_conflict" in inc["flags"]
    d = inc["draft"]
    assert d["category"]["needs_decision"]
    assert any(q["kind"] == "conflict_hazard" for q in d["unresolved_questions"])
    assert "flood" in d["summary"].lower() and "fire" in d["summary"].lower()  # both sides kept


def test_negation_conflict(env):
    svc, _, users, stub = env
    stub.transcript = "Smoke over Pine Street at 9 pm, but no one is injured."
    stub.obs = {"fire_smoke": 0.8, "injured_people": 0.85}
    inc = svc.get(users["reporter"], create(svc, users["reporter"]))
    assert any(x["kind"] == "negation" and x["concept"] == "injured_people" for x in inc["analysis"]["comparison"]["contradictions"])


def test_category_conflict_between_views(env):
    svc, _, users, stub = env
    stub.proba = {"text": onehot("rescue_volunteering_or_donation_effort", 0.8), "image": onehot("infrastructure_and_utility_damage", 0.8),
                  "combined": onehot("infrastructure_and_utility_damage", 0.6)}
    inc = svc.get(users["reporter"], create(svc, users["reporter"]))
    assert any(x["kind"] == "category" for x in inc["analysis"]["comparison"]["contradictions"])
    assert set(inc["draft"]["category"]["alternatives"]) == {"rescue_volunteering_or_donation_effort"}


def test_image_not_detected_is_complementary_not_contradiction(env):
    svc, _, users, stub = env
    stub.transcript = "Two people are trapped in a car on Elm Street since 8 am, it is still dangerous."
    stub.obs = {"vehicle_damage": 0.1}
    cmp = svc.get(users["reporter"], create(svc, users["reporter"]))["analysis"]["comparison"]
    assert not cmp["contradictions"]
    assert any(c["concept"] == "injured_people" and c["source"] == "transcript" for c in cmp["complementary"])


# ---------------------------------------------------------------- traceability


def test_every_transcript_source_matches_the_transcript(env):
    svc, _, users, stub = env
    stub.transcript = "A building collapsed on King Street at 7 pm. Firefighters are there. No one is trapped."
    stub.obs = {"structural_damage": 0.8, "responders": 0.7}
    inc = svc.get(users["reporter"], create(svc, users["reporter"]))
    text = inc["transcript_original"]
    n = 0
    for c in inc["draft"]["claims"]:
        for s in c["sources"]:
            if s["kind"] == "transcript":
                assert text[s["start"]:s["end"]] == s["quote"]
                n += 1
            elif s["kind"] == "image":
                obs = {o["concept"]: o for o in inc["analysis"]["image"]["observations"]}
                assert obs[s["concept"]]["score"] == s["score"]
            elif s["kind"] == "model":
                assert inc["analysis"]["views"][s["view"]]["label"] == s["label"]
    assert n >= 3
    assert all(c["sources"] for c in inc["draft"]["claims"] if c["kind"] in ("report", "agreement", "category"))


def test_processing_never_sets_approved(env):
    svc, db, users, stub = env
    for fault in (set(), {"asr"}, {"classify:combined"}):
        stub.fail = set(fault)
        svc.create(users["reporter"], make_wav(), make_image())
    assert db.one("SELECT COUNT(*) n FROM incidents WHERE status = 'APPROVED'")["n"] == 0
    assert db.one("SELECT COUNT(*) n FROM approved_records")["n"] == 0


# ---------------------------------------------------------------- audit integrity


def test_audit_is_append_only_and_tamper_evident(env):
    svc, db, users, _ = env
    iid = create(svc, users["reporter"])
    approve_ok(svc, users, iid)
    assert audit.verify(db)["ok"]
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        db.execute("UPDATE audit_events SET actor = 'mallory' WHERE seq = 1")
    db.execute("DROP TRIGGER audit_append_only_u")  # simulate someone bypassing the trigger
    db.execute("UPDATE audit_events SET actor = 'mallory' WHERE action = 'approved'")
    res = audit.verify(db)
    assert not res["ok"] and res["broken_at_seq"] > 1


def test_audit_details_carry_no_raw_transcript(env):
    svc, db, users, stub = env
    stub.transcript = "Secret caller name Jane Q. Example reports a fire on Main Street at 10 am."
    create(svc, users["reporter"])
    assert "Jane" not in str([e["details"] for e in audit.events(db)])


def test_unexpected_analysis_error_parks_incident_in_review(env, monkeypatch):
    import incident_reporter.workflow as wf

    svc, db, users, _ = env

    def boom(*a, **k):
        raise KeyError("bug in comparison")

    monkeypatch.setattr(wf, "compare", boom)
    iid = create(svc, users["reporter"])
    inc = svc.get(users["reporter"], iid)
    assert inc["status"] == "NEEDS_REVIEW" and "model_failure:analysis" in inc["flags"]
    assert inc["transcript_original"]  # what did work is kept
    assert any(e["action"] == "analysis_failed" for e in audit.events(db, iid))
