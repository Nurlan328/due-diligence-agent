"""Asynchronous API jobs and the reviewer labelling command."""
import time

import pytest
from fastapi.testclient import TestClient

import cli
import main
import pipeline
import store
from matching import match_findings
from models import Finding


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(pipeline, "default_collectors", lambda: [])
    with TestClient(main.app) as c:  # `with` runs the lifespan: the job pool starts and stops
        yield c


def body(subject):
    return {"subject": subject.model_dump(mode="json"), "operator": "api", "deep": False}


def wait(client, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get(f"/jobs/{job_id}").json()
        if status["status"] in ("done", "failed"):
            return status
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_job_runs_in_background_and_returns_report(client, company):
    r = client.post("/jobs", json=body(company))
    assert r.status_code == 202 and r.json()["status"] == "queued"
    status = wait(client, r.json()["job_id"])
    assert status["status"] == "done" and status["error"] is None
    assert status["report"]["subject"]["name"] == company.name and status["report"]["risk_level"] == "none"


def test_failed_job_keeps_the_error(client, company, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("registry down")

    monkeypatch.setattr(main, "screen", boom)
    status = wait(client, client.post("/jobs", json=body(company)).json()["job_id"])
    assert status["status"] == "failed" and "registry down" in status["error"] and status["report"] is None


def test_unknown_job_is_404(client):
    assert client.get("/jobs/nope").status_code == 404


def test_restart_fails_jobs_left_unfinished(monkeypatch):
    store.create_job("left-over", "api", {"x": 1})
    with TestClient(main.app):
        pass
    row = store.get_job("left-over")
    assert row is not None and row["status"] == "failed" and "restarted" in row["error"]


def test_cli_label_command(company, capsys):
    f = Finding(source_type="web", source_name="s", url="https://a.kz/1", category="fraud", claim="c",
                quote="q", subject_mentioned_as="Ромашка Строй")
    match_findings(company, [f], {f.url: "текст"}, request_id="r1")
    assert cli.main(["label", f.id, "namesake", "--reviewer", "analyst"]) == 0
    (row,) = store.match_rows(labelled_only=True)
    assert row["label"] == 0 and row["labelled_by"] == "analyst"
    assert cli.main(["label", "doesnotexist", "same"]) == 1
