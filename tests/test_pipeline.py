import json

from conftest import ScriptedModel, block, message, mock_http
from fastapi.testclient import TestClient

import pipeline
from models import Category, Finding, SourceCheck, Status, Verification
from pipeline import screen


class FakeRegistry:
    name = "fake registry"

    def __init__(self, findings):
        self.findings = findings

    def check(self, subject):
        return self.findings, SourceCheck(name=self.name, status="ok", hits=len(self.findings))


def rnu_finding():
    return Finding(source_type="registry", source_name="goszakup.gov.kz РНУ", url="https://goszakup.gov.kz/x",
                   category=Category.PROCUREMENT, status=Status.LISTED, severity="high",
                   claim="Включён в РНУ", quote="включён в РНУ", identity_evidence=["поиск по БИН/ИИН"],
                   verification=Verification.REGISTRY)


def agent_record(*findings):
    return message("tool_use", block("server_tool_use", id="s", name="web_search", input={"query": "q"}),
                   block("tool_use", id="t1", name="record_findings",
                         input={"findings": list(findings), "leads": []}))


def agent_finding(url, quote, mentioned, **kw):
    return {"url": url, "title": "t", "published": "", "source_name": "news", "category": "fraud",
            "status": "charged", "severity": "high", "claim": "c", "quote": quote,
            "subject_mentioned_as": mentioned, "identity_evidence": [], **kw}


def test_end_to_end_company(company, tmp_logs):
    pages = {
        "https://news.kz/1": "<p>Суд начал рассматривать дело ТОО «Ромашка Строй» (БИН 123456789012) о хищении.</p>",
        "https://news.kz/2": "<p>ТОО Ромашка Трейд из Шымкента обвиняют в мошенничестве.</p>",
        "https://news.kz/3": "<p>Статья о другом.</p>",
    }
    model = ScriptedModel(agent_record(
        agent_finding("https://news.kz/1", "Суд начал рассматривать дело ТОО «Ромашка Строй»", "ТОО «Ромашка Строй»"),
        agent_finding("https://news.kz/2", "ТОО Ромашка Трейд из Шымкента обвиняют", "ТОО Ромашка Трейд"),
        agent_finding("https://news.kz/3", "выдуманная цитата о Ромашка Строй", "Ромашка Строй"),
    ), message("end_turn"))

    report = screen(company, operator="tester", collectors=[FakeRegistry([rnu_finding()])],
                    model_fn=model, http=mock_http(pages), max_rounds=1)

    urls = [f.url for f in report.findings]
    assert "https://goszakup.gov.kz/x" in urls and "https://news.kz/1" in urls
    assert [f.url for f in report.possible_namesakes] == ["https://news.kz/2"]
    fabricated = next(f for f in report.findings if f.url == "https://news.kz/3")
    assert fabricated.verification == Verification.QUOTE_NOT_FOUND
    assert report.findings[0].risk >= report.findings[-1].risk
    assert report.findings[-1].url == "https://news.kz/3"  # fabricated quote ranks last
    assert report.risk_level in {"medium", "high"}
    assert report.queries == ["q"]
    assert "Возможные однофамильцы" in report.markdown and "fake registry" in report.markdown

    audit = [json.loads(l) for l in (tmp_logs / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [a["event"] for a in audit] == ["screen_started", "screen_finished"]
    assert audit[1]["queries"] == ["q"] and audit[0]["purpose"] == company.purpose


def test_person_report_masks_iin(person, tmp_logs):
    report = screen(person, operator="tester", collectors=[], deep=False)
    assert report.risk_level == "none"
    assert person.identifier not in report.markdown and "********0123" in report.markdown
    audit = (tmp_logs / "audit.jsonl").read_text(encoding="utf-8")
    assert person.identifier not in audit


def test_api(company, monkeypatch):
    monkeypatch.setattr(pipeline, "default_collectors", lambda: [FakeRegistry([rnu_finding()])])
    import main

    client = TestClient(main.app)
    body = {"subject": company.model_dump(mode="json"), "operator": "api", "deep": False}
    r = client.post("/screen", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["risk_level"] == "high" and len(data["findings"]) == 1

    bad = {**body, "subject": {**body["subject"], "purpose": ""}}
    assert client.post("/screen", json=bad).status_code == 422  # purpose is mandatory
