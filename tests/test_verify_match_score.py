import json

from conftest import mock_http

from config import settings
from matching import features, match_findings, score
from models import Finding, Severity, Status, Verification
from scoring import finding_risk, overall, split_by_identity
from verify import quote_in_text, verify_findings


def web(url="https://a.kz/1", quote="Директор ТОО задержан", **kw):
    base = dict(source_type="web", source_name="a.kz", url=url, category="fraud", status="investigation",
                severity="high", claim="c", quote=quote, subject_mentioned_as="ТОО Ромашка Строй")
    return Finding(**{**base, **kw})


def test_quote_matching_tolerates_typography():
    page = "Как сообщает пресс-служба, директор ТОО «Ромашка»  задержан вчера."
    assert quote_in_text('Директор ТОО "Ромашка" задержан', page)
    assert not quote_in_text("директор ТОО Василёк оправдан", page)


def test_verify_sets_statuses():
    pages = {"https://a.kz/1": "<html><body><p>Вчера директор ТОО задержан полицией.</p></body></html>",
             "https://a.kz/2": "<html><body><p>Совсем другой текст статьи.</p></body></html>"}
    fs = [web("https://a.kz/1"), web("https://a.kz/2"), web("https://a.kz/404")]
    texts = verify_findings(fs, http=mock_http(pages))
    assert [f.verification for f in fs] == [Verification.VERIFIED, Verification.QUOTE_NOT_FOUND,
                                            Verification.UNREACHABLE]
    assert texts["https://a.kz/404"] is None


def test_identifier_in_page_confirms_identity(company):
    f = web(subject_mentioned_as="Ромашка")
    feat = features(company, f, "ТОО Ромашка (БИН 123456789012) признано банкротом")
    s, reasons = score(feat, company)
    assert s >= 0.97 and any("БИН" in r for r in reasons)


def test_person_namesake_scores_low(person):
    f = web(subject_mentioned_as="Ахметов Е.", quote="Ахметов Е. осуждён")
    s, _ = score(features(person, f, "В Караганде Ахметов Е. осуждён за кражу."), person)
    assert s < settings.MATCH_THRESHOLD


def test_context_lifts_partial_name(person):
    f = web(subject_mentioned_as="Ахметов Е.Б.")
    text = "Директор ТОО Ромашка Строй Ахметов Е.Б. (1980 г.р.) задержан в Алматы."
    s, reasons = score(features(person, f, text), person)
    assert s >= settings.MATCH_THRESHOLD
    assert "совпадает город" in reasons


def test_match_log_written(company, tmp_logs):
    f = web()
    match_findings(company, [f], {f.url: "текст"}, request_id="r1")
    row = json.loads((tmp_logs / "match.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["request_id"] == "r1" and row["label"] is None and "name_score" in row


def test_risk_weights_and_noisy_or():
    strong = web(status="convicted", severity="critical", verification="verified", match_score=1.0)
    assert finding_risk(strong) == 1.0
    acquitted = web(status="acquitted", verification="verified", match_score=1.0)
    assert finding_risk(acquitted) == 0.0
    fake = web(verification="quote_not_found", match_score=1.0)
    real = web(verification="verified", match_score=1.0)
    assert finding_risk(fake) < finding_risk(real) / 5

    a = web(category="tax", status="civil_judgment", severity="medium", verification="verified", match_score=1.0)
    b = web(category="litigation", status="civil_judgment", severity="medium", verification="verified",
            match_score=1.0)
    single, _ = overall([a])
    both, _ = overall([a, b])
    assert both > single and both < 1


def test_same_case_many_articles_counts_once():
    articles = [web(url=f"https://a.kz/{i}", verification="verified", match_score=1.0) for i in range(5)]
    assert overall(articles)[0] == overall(articles[:1])[0]


def test_levels():
    assert overall([])[1] == "none"
    sanc = web(category="sanctions", status=Status.LISTED, severity=Severity.CRITICAL,
               verification="registry", match_score=1.0)
    assert overall([sanc])[1] == "high"


def test_split_by_identity():
    hi, lo = web(match_score=0.9), web(url="https://b.kz", match_score=0.2)
    assert split_by_identity([hi, lo], threshold=0.5) == ([hi], [lo])
