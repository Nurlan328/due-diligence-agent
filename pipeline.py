"""Screening pipeline: registries -> deep search -> verification -> identity -> score -> report.

A plain sequence of stages: every stage is a function over the findings list, so
each can be tested and evaluated on its own. The multi-round loop lives inside
deep_search().
"""
import uuid
from datetime import UTC, datetime

import httpx

import audit
from deepsearch import ModelFn, deep_search
from matching import match_findings
from models import Finding, Report, SourceCheck, Subject
from report import render
from scoring import overall, split_by_identity
from sources import Collector, default_collectors
from verify import verify_findings


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen, out = set(), []
    for f in findings:
        if f.id not in seen:
            seen.add(f.id)
            out.append(f)
    return out


def screen(subject: Subject, operator: str, *, collectors: list[Collector] | None = None,
           model_fn: ModelFn | None = None, http: httpx.Client | None = None,
           deep: bool = True, max_rounds: int | None = None) -> Report:
    request_id = uuid.uuid4().hex[:12]
    audit.log_event("screen_started", request_id, subject, operator, deep_search=deep)

    findings: list[Finding] = []
    sources: list[SourceCheck] = []
    for collector in collectors if collectors is not None else default_collectors():
        found, check = collector.check(subject)
        findings += found
        sources.append(check)

    queries: list[str] = []
    rounds = 0
    if deep:
        research = deep_search(subject, model_fn=model_fn, max_rounds=max_rounds)
        findings += research.findings
        queries, rounds = research.queries, research.rounds
        sources.append(SourceCheck(
            name="Глубокий веб-поиск (Claude web search)", status="ok", hits=len(research.findings),
            note="; ".join(research.notes)[:300]))
    else:
        sources.append(SourceCheck(name="Глубокий веб-поиск (Claude web search)", status="skipped",
                                   note="отключён"))

    findings = _dedupe(findings)
    pages = verify_findings(findings, http=http)
    match_findings(subject, findings, pages, request_id=request_id)
    matched, namesakes = split_by_identity(findings)
    score, level = overall(matched)
    matched.sort(key=lambda f: -f.risk)

    report = Report(
        request_id=request_id, created_at=datetime.now(UTC), subject=subject,
        risk_score=score, risk_level=level, findings=matched,
        possible_namesakes=sorted(namesakes, key=lambda f: -f.match_score),
        sources=sources, queries=queries, rounds=rounds,
    )
    report.markdown = render(report)
    audit.log_event("screen_finished", request_id, subject, operator, risk_level=level,
                    risk_score=score, findings=len(matched), namesakes=len(namesakes),
                    queries=queries)
    return report
