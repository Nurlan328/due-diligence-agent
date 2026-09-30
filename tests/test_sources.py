import csv

import httpx

from models import Category, Status, Verification
from sources.goszakup import GoszakupRnuCollector
from sources.sanctions import SanctionsCollector

FIELDS = ["id", "schema", "name", "aliases", "birth_date", "countries", "addresses", "identifiers",
          "sanctions", "phones", "emails", "program_ids", "dataset", "first_seen", "last_seen", "last_change"]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    return path


def test_sanctions_person_match_with_birth_year(tmp_path, person):
    path = write_csv(tmp_path / "s.csv", [
        {"id": "p1", "schema": "Person", "name": "Akhmetov Erzhan Bolatovich",
         "aliases": "Ахметов Ержан Болатович;Ахметов Е.Б.", "birth_date": "1980-05-01",
         "countries": "kz", "sanctions": "OFAC SDN;EU", "first_seen": "2024-01-02T00:00:00"},
        # namesake: same name, different birth year -> dropped
        {"id": "p2", "schema": "Person", "name": "Ахметов Ержан Болатович", "birth_date": "1955"},
        # company with the same tokens -> wrong schema for a person
        {"id": "c1", "schema": "Company", "name": "Ахметов Ержан"},
    ])
    findings, check = SanctionsCollector(path=path).check(person)
    assert check.status == "ok" and check.hits == 1
    f = findings[0]
    assert f.url.endswith("/p1/")
    assert f.category == Category.SANCTIONS and f.status == Status.LISTED
    assert f.verification == Verification.REGISTRY
    assert any("год рождения" in e for e in f.identity_evidence)


def test_sanctions_company_by_identifier_even_if_renamed(tmp_path, company):
    path = write_csv(tmp_path / "s.csv", [
        {"id": "c9", "schema": "Company", "name": "Совсем Другое Название",
         "identifiers": "123456789012", "sanctions": "UK HMT"},
    ])
    findings, _ = SanctionsCollector(path=path).check(company)
    assert len(findings) == 1
    assert any("БИН/ИИН" in e for e in findings[0].identity_evidence)


def test_sanctions_skipped_without_export(tmp_path, company):
    findings, check = SanctionsCollector(path=tmp_path / "missing.csv").check(company)
    assert findings == [] and check.status == "skipped"


def rnu_client(status: int, payload=None) -> httpx.Client:
    def handler(request):
        assert request.headers["Authorization"] == "Bearer tok"
        assert request.url.path.endswith("/rnu/123456789012")
        return httpx.Response(status, json=payload) if payload is not None else httpx.Response(status)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_goszakup_rnu_hit(company):
    payload = {"items": [{"supplier_biin": "123456789012", "supplier_name_ru": "ТОО Ромашка Строй",
                          "customer_name_ru": "ГУ Отдел строительства", "court_decision": "2-123/2024",
                          "start_date": "2024-03-01T00:00:00", "end_date": "2026-03-01T00:00:00"}]}
    c = GoszakupRnuCollector(token="tok", base_url="https://x/v3", http=rnu_client(200, payload))
    findings, check = c.check(company)
    assert check.hits == 1
    assert findings[0].category == Category.PROCUREMENT
    assert "2-123/2024" in findings[0].quote and "2024-03-01" in findings[0].quote


def test_goszakup_not_listed(company):
    c = GoszakupRnuCollector(token="tok", base_url="https://x/v3", http=rnu_client(404))
    findings, check = c.check(company)
    assert findings == [] and check.status == "ok"


def test_goszakup_skipped_without_token(company):
    assert GoszakupRnuCollector(token="").check(company)[1].status == "skipped"


def test_goszakup_error_does_not_raise(company):
    c = GoszakupRnuCollector(token="tok", base_url="https://x/v3", http=rnu_client(500))
    findings, check = c.check(company)
    assert findings == [] and check.status == "error"
