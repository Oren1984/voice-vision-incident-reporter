"""HTTP layer: authentication, CSRF, access control, security headers, uploads, health and ops."""

import re

import pytest
from fastapi.testclient import TestClient

from incident_reporter import audit
from incident_reporter.security import create_user
from incident_reporter.web import create_app

from .conftest import PW, StubModels, make_image, make_wav


@pytest.fixture
def app(settings):
    a = create_app(settings, models=StubModels(), log_to_file=False)
    create_user(a.state.db, "rita.reporter", "reporter", PW)
    create_user(a.state.db, "vera.reviewer", "reviewer", PW)
    create_user(a.state.db, "rene.reviewer", "reviewer", PW)
    return a


def login(app, username):
    c = TestClient(app)
    r = c.post("/login", data={"username": username, "password": PW}, follow_redirects=False)
    assert r.status_code == 303, r.text
    return c


def csrf_of(client, path="/incidents/new"):
    return re.search(r'name="csrf" value="([^"]+)"', client.get(path).text).group(1)


def version_of(html):
    return re.search(r'name="version" value="(\d+)"', html).group(1)


def upload(client, audio=True, image=True):
    files = {}
    if audio:
        files["audio"] = ("report.wav", make_wav(3.0), "audio/wav")
    if image:
        files["image"] = ("scene.jpg", make_image(), "image/jpeg")
    return client.post("/incidents", data={"csrf": csrf_of(client)}, files=files, follow_redirects=False)


def test_pages_require_login(app):
    c = TestClient(app)
    for path in ("/", "/incidents/new", "/ops", "/incidents/inc_0123456789ab", "/incidents/inc_0123456789ab/media/audio", "/api/metrics"):
        r = c.get(path, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/login", path


def test_wrong_password_and_lockout(app):
    c = TestClient(app)
    for _ in range(5):
        assert c.post("/login", data={"username": "rita.reporter", "password": "nope-nope-nope"}).status_code == 401
    assert c.post("/login", data={"username": "rita.reporter", "password": PW}).status_code == 429
    assert any(e["action"] == "login_failed" for e in audit.events(app.state.db))


def test_session_cookie_flags(app):
    r = TestClient(app).post("/login", data={"username": "rita.reporter", "password": PW}, follow_redirects=False)
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie


def test_security_headers(app):
    r = TestClient(app).get("/login")
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert re.fullmatch(r"[0-9a-f]{12}", r.headers["x-request-id"])


def test_post_without_csrf_is_refused(app):
    c = login(app, "rita.reporter")
    r = c.post("/incidents", files={"audio": ("a.wav", make_wav(), "audio/wav")}, follow_redirects=False)
    assert r.status_code == 403
    assert app.state.db.one("SELECT COUNT(*) n FROM incidents")["n"] == 0


def test_full_flow_over_http(app):
    rep = login(app, "rita.reporter")
    r = upload(rep)
    assert r.status_code == 303
    iid = r.headers["location"].rsplit("/", 1)[1]
    page = rep.get(f"/incidents/{iid}").text
    assert "Text only" in page and "Image only" in page and "Combined text + image" in page
    assert "Source comparison" in page and "Unresolved questions" in page
    assert rep.get(f"/incidents/{iid}/media/audio").headers["content-type"] == "audio/wav"
    assert rep.get(f"/incidents/{iid}/media/image").headers["content-type"] == "image/jpeg"
    if "DRAFT" in page:
        rep.post(f"/incidents/{iid}/submit", data={"csrf": csrf_of(rep, f"/incidents/{iid}"), "version": version_of(page)})
    # a reporter posting to approve is refused even with a valid token
    page = rep.get(f"/incidents/{iid}").text
    r = rep.post(f"/incidents/{iid}/approve", data={"csrf": csrf_of(rep, f"/incidents/{iid}"), "version": version_of(page), "confirm": "yes", "acknowledge_open": "yes"})
    assert r.status_code == 403
    rev = login(app, "vera.reviewer")
    page = rev.get(f"/incidents/{iid}").text
    assert "Human review" in page
    r = rev.post(f"/incidents/{iid}/approve", data={"csrf": csrf_of(rev, f"/incidents/{iid}"), "version": version_of(page), "confirm": "yes", "acknowledge_open": "yes"}, follow_redirects=False)
    assert r.status_code == 303
    page = rev.get(f"/incidents/{iid}").text
    assert "APPROVED" in page and "Approved record" in page
    rec = rev.get(f"/incidents/{iid}/record.json").json()
    assert rec["integrity_ok"] and rec["record"]["approved_by"] == "vera.reviewer"


def test_upload_errors_are_shown_not_crashed(app):
    c = login(app, "rita.reporter")
    r = c.post("/incidents", data={"csrf": csrf_of(c)}, files={"audio": ("x.wav", b"not really audio", "audio/wav")})
    assert r.status_code == 400 and "Unsupported audio format" in r.text
    r = c.post("/incidents", data={"csrf": csrf_of(c)})
    assert r.status_code == 400 and "Upload an audio recording" in r.text


def test_oversized_request_is_refused_before_parsing(app):
    c = login(app, "rita.reporter")
    r = c.post("/incidents", content=b"x", headers={"content-length": str(50 * 1024 * 1024), "content-type": "multipart/form-data; boundary=x"})
    assert r.status_code == 413


@pytest.mark.parametrize("bad", ["..%2F..%2Fetc", "inc_zzzzzzzzzzzz", "inc_0123456789ab%00"])
def test_bad_incident_ids(app, bad):
    c = login(app, "vera.reviewer")
    assert c.get(f"/incidents/{bad}").status_code == 404
    assert c.get(f"/incidents/{bad}/media/image").status_code in (404, 422)


def test_media_kind_is_whitelisted(app):
    rep = login(app, "rita.reporter")
    iid = upload(rep).headers["location"].rsplit("/", 1)[1]
    assert rep.get(f"/incidents/{iid}/media/../../incidents.sqlite3").status_code == 404
    assert rep.get(f"/incidents/{iid}/media/db").status_code == 404


def test_ops_and_metrics_are_reviewer_only(app):
    rep = login(app, "rita.reporter")
    assert rep.get("/ops").status_code == 403 and rep.get("/api/metrics").status_code == 403
    rev = login(app, "vera.reviewer")
    upload(login(app, "rita.reporter"))
    m = rev.get("/api/metrics").json()
    assert m["audit_chain"]["ok"] and m["stats"]["incidents_by_status"]
    assert any(k.startswith("stage.") for k in m["metrics"]["durations"])
    assert "Audit chain" in rev.get("/ops").text


def test_healthz_is_public_and_minimal(app):
    r = TestClient(app).get("/healthz")
    assert r.status_code == 200 and r.json()["db"] is True
    assert set(r.json()) == {"status", "db", "models_loaded", "heads", "version"}


def test_logout_ends_session(app):
    c = login(app, "rita.reporter")
    c.post("/logout", data={"csrf": csrf_of(c, "/")})
    assert c.get("/", follow_redirects=False).status_code == 303


def test_scores_never_shown_as_certain():
    from incident_reporter.labels import pct

    assert [pct(x) for x in (None, 0.0, 0.005, 0.01, 0.87, 0.99, 0.995, 1.0)] == ["—", "<1%", "<1%", "1%", "87%", "99%", ">99%", ">99%"]


def test_password_reset_ends_sessions_and_old_password(app):
    from incident_reporter.security import authenticate, set_password

    c = login(app, "rita.reporter")
    assert c.get("/", follow_redirects=False).status_code == 200
    with pytest.raises(ValueError):
        set_password(app.state.db, "rita.reporter", "short")
    assert set_password(app.state.db, "rita.reporter", "a-new-local-password") is True
    assert c.get("/", follow_redirects=False).status_code == 303  # open session ended
    assert authenticate(app.state.db, "rita.reporter", PW) is None
    assert authenticate(app.state.db, "rita.reporter", "a-new-local-password") is not None
    assert set_password(app.state.db, "nobody.here", "a-new-local-password") is False
