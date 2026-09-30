"""The offline demo runs end to end and exercises every stage it claims to show."""
from config import settings
from demo import load_scenario, run_demo
from models import Verification


def test_demo_runs_offline_and_shows_each_stage(tmp_path):
    report, md_path = run_demo(tmp_path)

    assert md_path.exists() and (tmp_path / "sample_report.json").exists()
    text = md_path.read_text(encoding="utf-8")
    assert text.startswith("> **Демо-отчёт.**")

    verifications = {f.verification for f in report.findings}
    assert Verification.VERIFIED_ARCHIVE in verifications  # removed article, found in Wayback
    assert Verification.QUOTE_NOT_FOUND in verifications  # fabricated quote
    assert Verification.REGISTRY in verifications

    own = [x for x in report.facts if not x.about]
    affiliated = [x for x in report.facts if x.about]
    assert max(x.corroborations for x in own) == 2  # two outlets, one event
    assert {x.about for x in affiliated} == {"Примеров Ержан Болатович", "ТОО «Василёк Инвест»"}
    assert [f.url for f in report.possible_namesakes] == ["https://south-news.example/2025/bankrot"]
    assert report.rounds == 2 and len(report.queries) == 6
    assert "```mermaid" in text and report.risk_level == "high"


def test_demo_leaves_settings_and_real_logs_alone(tmp_path):
    before = (settings.MAJOR_MEDIA_EXTRA, settings.AUDIT_LOG, settings.MATCH_LOG)
    run_demo(tmp_path)
    assert (settings.MAJOR_MEDIA_EXTRA, settings.AUDIT_LOG, settings.MATCH_LOG) == before


def test_scenario_is_fictional():
    sc = load_scenario()
    urls = list(sc["pages"]) + list(sc["archive"])
    urls += [f["url"] for r in sc["registries"] for f in r["findings"]]
    assert all(".example/" in u for u in urls)  # reserved TLD, no real outlet is quoted
    assert sc["subject"]["purpose"].startswith("ДЕМО")
