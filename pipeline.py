"""Screening pipeline: registries -> deep search -> verification -> identity -> facts -> score -> report.

A plain sequence of stages: every stage is a function over the findings list, so
each can be tested and evaluated on its own. The multi-round loop and the entity
graph live inside deep_search().
"""
import uuid
from datetime import UTC, datetime

import httpx

import audit
import reliability
from deepsearch import ModelFn, deep_search
from graph import EntityGraph
from matching import match_findings
from models import Finding, Report, SourceCheck, Subject, SubjectType
from report import render
from scoring import assess, split_by_identity
from sources import Collector, default_collectors
from verify import verify_findings

# Identity of a finding about a case or an unclassified affiliate cannot be checked by
# name; it stands on the agent's link, just above the namesake threshold's default.
UNCHECKED_AFFILIATE_SCORE = 0.6


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen, out = set(), []
    for f in findings:
        if f.id not in seen:
            seen.add(f.id)
            out.append(f)
    return out


def match_all(subject: Subject, findings: list[Finding], pages: dict, graph: EntityGraph | None,
              request_id: str) -> None:
    """Findings about the subject are matched against the subject; findings about an
    affiliate against that affiliate's own name (a director's case must name the director)."""
    match_findings(subject, [f for f in findings if not f.about], pages, request_id=request_id)
    by_about: dict[str, list[Finding]] = {}
    for f in findings:
        if f.about:
            by_about.setdefault(f.about, []).append(f)
    for name, group in by_about.items():
        nid = graph.find(name) if graph else None
        kind = graph.nodes[nid].kind if nid else "other"
        if kind in ("company", "person"):
            affiliate = Subject(type=SubjectType.COMPANY if kind == "company" else SubjectType.PERSON,
                                name=name, purpose=subject.purpose)
            match_findings(affiliate, group, pages, request_id=request_id)
        else:
            for f in group:
                f.match_score = UNCHECKED_AFFILIATE_SCORE
                f.match_reasons = ["связь установлена агентом, идентичность по имени не проверялась"]


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
    graph: EntityGraph | None = None
    if deep:
        research = deep_search(subject, model_fn=model_fn, max_rounds=max_rounds)
        findings += research.findings
        queries, rounds, graph = research.queries, research.rounds, research.graph
        sources.append(SourceCheck(
            name="Глубокий веб-поиск (Claude web search)", status="ok", hits=len(research.findings),
            note="; ".join(research.notes)[:300]))
    else:
        sources.append(SourceCheck(name="Глубокий веб-поиск (Claude web search)", status="skipped",
                                   note="отключён"))

    findings = _dedupe(findings)
    for f in findings:
        reliability.annotate(f)
    pages = verify_findings(findings, http=http)
    match_all(subject, findings, pages, graph, request_id)
    matched, namesakes = split_by_identity(findings)
    score, level, facts = assess(matched)
    matched.sort(key=lambda f: -f.risk)

    graph_dict: dict = {}
    if graph is not None and len(graph.nodes) > 1:
        adverse = {nid for f in matched if f.about and (nid := graph.find(f.about))}
        if any(not f.about for f in matched):
            adverse.add(graph.subject_id)
        graph_dict = {**graph.to_dict(), "mermaid": graph.to_mermaid(adverse)}

    report = Report(
        request_id=request_id, created_at=datetime.now(UTC), subject=subject,
        risk_score=score, risk_level=level, findings=matched, facts=facts,
        possible_namesakes=sorted(namesakes, key=lambda f: -f.match_score),
        sources=sources, queries=queries, rounds=rounds, graph=graph_dict,
    )
    report.markdown = render(report)
    audit.log_event("screen_finished", request_id, subject, operator, risk_level=level,
                    risk_score=score, facts=len(facts), findings=len(matched), namesakes=len(namesakes),
                    queries=queries)
    return report
