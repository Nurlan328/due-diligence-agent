"""Offline demo: a recorded, fictional screening run through the real pipeline.

    python cli.py demo                # -> reports/sample_report.md / .json
    python cli.py demo --out docs     # regenerate docs/sample_report.md

No API key and no network. Everything that would go outside is replayed from
demo/scenario.json:

- the Claude agent: its recorded turns (web searches, record_findings calls with
  findings and leads) are returned in order by a stand-in model function;
- the web: page fetches and the Wayback Machine go through an httpx mock transport;
- the registries: stand-in collectors return the recorded registry entries.

Everything else is the production code: quote verification, the Wayback
fallback, identity matching, the entity graph, fact grouping, scoring and the
report. The scenario is built to show each of them once: a fact confirmed by
two outlets, a removed article verified from the archive, a fabricated quote, a
namesake company with another БИН, a registry entry, and adverse facts about
the director and the founder found by pivoting.

All companies, people, identifiers, articles and websites in the scenario are
invented; the domains use the reserved .example TLD.
"""
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx

from config import settings
from models import Finding, Report, SourceCheck, Subject

SCENARIO = Path(__file__).with_name("scenario.json")
WAYBACK_API = "https://archive.org/wayback/available"


def load_scenario(path: Path = SCENARIO) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ReplayModel:
    """Stands in for deepsearch.claude_call: returns the recorded assistant turns in order."""

    def __init__(self, turns: list[dict]):
        self.turns = [SimpleNamespace(stop_reason=t["stop_reason"],
                                      content=[SimpleNamespace(**b) for b in t["content"]]) for t in turns]
        self.calls = 0

    def __call__(self, **request):
        self.calls += 1
        if not self.turns:
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="")])
        return self.turns.pop(0)


class ReplayRegistry:
    """Stands in for a structured-source collector."""

    def __init__(self, name: str, findings: list[dict], note: str = ""):
        self.name = name
        self.findings = findings
        self.note = note

    def check(self, subject: Subject) -> tuple[list[Finding], SourceCheck]:
        found = [Finding(**f) for f in self.findings]
        return found, SourceCheck(name=self.name, status="ok", hits=len(found), note=self.note)


def replay_http(pages: dict[str, str], archive: dict[str, dict]) -> httpx.Client:
    """Recorded pages; anything else is a 404. Wayback answers for the archived URLs."""
    snapshots = {f"https://web.archive.org/web/{a['timestamp']}id_/{url}": a["html"] for url, a in archive.items()}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(WAYBACK_API):
            target = request.url.params.get("url", "")
            if target in archive:
                ts = archive[target]["timestamp"]
                return httpx.Response(200, json={"archived_snapshots": {"closest": {
                    "available": True, "status": "200", "timestamp": ts,
                    "url": f"http://web.archive.org/web/{ts}/{target}"}}})
            return httpx.Response(200, json={"archived_snapshots": {}})
        html = pages.get(url) or snapshots.get(url)
        if html is None:
            return httpx.Response(404)
        return httpx.Response(200, text=html, headers={"content-type": "text/html; charset=utf-8"})

    return httpx.Client(transport=httpx.MockTransport(handler))


@contextmanager
def _overrides(**values):
    old = {k: getattr(settings, k) for k in values}
    for k, v in values.items():
        setattr(settings, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(settings, k, v)


def run_demo(out_dir: str | Path | None = None, scenario: dict | None = None) -> tuple[Report, Path]:
    """Run the recorded scenario through pipeline.screen and write sample_report.md / .json."""
    from pipeline import screen  # late import: keeps `import demo` cheap

    sc = scenario or load_scenario()
    out = Path(out_dir or settings.REPORTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    tiers = sc.get("tiers", {})
    logs = Path(settings.DATA_DIR) / "demo"
    logs.mkdir(parents=True, exist_ok=True)
    with _overrides(
        # The demo's fictional outlets get the tiers real ones would have.
        OFFICIAL_EXTRA=tiers.get("official_extra", []),
        MAJOR_MEDIA_EXTRA=tiers.get("major_media_extra", []),
        LOW_EXTRA=tiers.get("low_extra", []),
        ARCHIVE_ENABLED=True,
        # Keep demo runs out of the real audit trail and the matcher's training log.
        AUDIT_LOG=str(logs / "audit.jsonl"),
        MATCH_LOG=str(logs / "match_features.jsonl"),
    ):
        collectors = [ReplayRegistry(r["name"], r["findings"], r.get("note", "")) for r in sc["registries"]]
        report = screen(Subject(**sc["subject"]), operator="demo", collectors=collectors,
                        model_fn=ReplayModel(sc["turns"]), http=replay_http(sc["pages"], sc.get("archive", {})),
                        max_rounds=2)
    md = out / "sample_report.md"
    banner = ("> **Демо-отчёт.** Все компании, люди, идентификаторы, публикации и сайты вымышлены. "
              "Сгенерирован командой `python cli.py demo` без API-ключа и сети: ответы агента, страницы и "
              "реестры проигрываются из `demo/scenario.json`, остальное делает настоящий пайплайн.\n\n")
    md.write_text(banner + report.markdown, encoding="utf-8", newline="\n")
    (out / "sample_report.json").write_text(report.model_dump_json(indent=2, exclude={"markdown"}),
                                            encoding="utf-8", newline="\n")
    return report, md
