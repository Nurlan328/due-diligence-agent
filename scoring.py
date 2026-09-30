"""Risk scoring over facts, not articles.

1. Each finding (one source reporting one event) gets an evidence weight:
   source tier × quote verification × identity score.
2. Findings about the same event are grouped into a fact: same category, same
   entity, and the same event key / case number / near-identical claim.
3. A fact's strength combines its independent sources (noisy-OR over distinct
   domains): three outlets confirming a story beat one, but one outlet
   syndicated five times is still one source.
4. fact risk = severity × status × strength × recency (× AFFILIATE_WEIGHT when
   the fact is about a director / founder / subsidiary, not the subject).
5. The overall score is a noisy-OR over facts.

Weights are explicit and meant to be tuned with the compliance team.
"""
import hashlib
import re

from rapidfuzz import fuzz

import reliability
from config import settings
from models import Fact, Finding, Severity, Status, Verification
from names import fold

SEVERITY = {Severity.LOW: 0.2, Severity.MEDIUM: 0.45, Severity.HIGH: 0.75, Severity.CRITICAL: 1.0}
STATUS = {
    Status.LISTED: 1.0, Status.CONVICTED: 1.0, Status.CIVIL_JUDGMENT: 0.8, Status.CHARGED: 0.8,
    Status.INVESTIGATION: 0.6, Status.ALLEGATION: 0.4, Status.UNKNOWN: 0.4,
    Status.CLOSED: 0.2, Status.ACQUITTED: 0.0,
}
VERIFICATION = {
    Verification.REGISTRY: 1.0, Verification.VERIFIED: 1.0, Verification.VERIFIED_ARCHIVE: 0.9,
    Verification.UNREACHABLE: 0.6, Verification.NOT_CHECKED: 0.5, Verification.QUOTE_NOT_FOUND: 0.15,
}
CONFIRMING = {Verification.REGISTRY, Verification.VERIFIED, Verification.VERIFIED_ARCHIVE}
LEVELS = [(0.7, "high"), (0.35, "medium"), (0.1, "low")]
_CASE_NO = re.compile(r"(?<!\d)\d{1,3}-\d{1,6}/\d{2,4}(?!\d)|№\s?\d{6,}")


def _annotated(f: Finding) -> Finding:
    if not f.source_tier:
        reliability.annotate(f)
    return f


def evidence(f: Finding) -> float:
    """How strongly this one source supports the fact, 0..1."""
    f = _annotated(f)
    return f.source_weight * VERIFICATION[f.verification] * f.match_score


def finding_risk(f: Finding) -> float:
    return round(SEVERITY[f.severity] * STATUS[f.status] * evidence(f) * f.recency_weight, 4)


def split_by_identity(findings: list[Finding], threshold: float | None = None) -> tuple[list[Finding], list[Finding]]:
    """(about the subject or its affiliates, possible namesakes)."""
    threshold = threshold if threshold is not None else settings.MATCH_THRESHOLD
    matched = [f for f in findings if f.match_score >= threshold]
    namesakes = [f for f in findings if f.match_score < threshold]
    return matched, namesakes


def same_event(a: Finding, b: Finding) -> bool:
    if a.category != b.category or fold(a.about) != fold(b.about):
        return False
    if a.source_type == b.source_type == "registry":
        return a.source_name == b.source_name
    if a.event_key and b.event_key and fuzz.token_set_ratio(fold(a.event_key), fold(b.event_key)) >= 85:
        return True
    cases_a = set(_CASE_NO.findall(f"{a.claim} {a.quote}"))
    if cases_a & set(_CASE_NO.findall(f"{b.claim} {b.quote}")):
        return True
    return fuzz.token_set_ratio(fold(a.claim), fold(b.claim)) >= 80


def group_facts(findings: list[Finding]) -> list[list[Finding]]:
    """Connected components of same_event (union-find)."""
    parent = list(range(len(findings)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(findings)):
        for j in range(i + 1, len(findings)):
            if find(i) != find(j) and same_event(findings[i], findings[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list[Finding]] = {}
    for i, f in enumerate(findings):
        groups.setdefault(find(i), []).append(f)
    return list(groups.values())


def build_fact(group: list[Finding]) -> Fact:
    for f in group:
        f.risk = finding_risk(f)
    # The most serious reading wins: a later conviction outranks the earlier "detained" article.
    top = max(group, key=lambda f: (SEVERITY[f.severity] * STATUS[f.status], evidence(f)))
    per_source: dict[str, float] = {}
    for f in group:
        key = reliability.source_key(f)
        per_source[key] = max(per_source.get(key, 0.0), evidence(f))
    doubt = 1.0
    for e in per_source.values():
        doubt *= 1 - e
    strength = 1 - doubt
    recency = max(f.recency_weight for f in group)
    risk = SEVERITY[top.severity] * STATUS[top.status] * strength * recency
    if top.about:
        risk *= settings.AFFILIATE_WEIGHT
    confirming = {reliability.source_key(f) for f in group if f.verification in CONFIRMING}
    fact_id = hashlib.sha1("|".join(sorted(f.id for f in group)).encode()).hexdigest()[:10]
    for f in group:
        f.fact_id = fact_id
    return Fact(id=fact_id, category=top.category, status=top.status, severity=top.severity, claim=top.claim,
                about=top.about, relation_path=top.relation_path, sources=sorted(per_source),
                corroborations=len(confirming), strength=round(strength, 3), recency_weight=recency,
                risk=round(risk, 4), finding_ids=[f.id for f in group])


def assess(findings: list[Finding]) -> tuple[float, str, list[Fact]]:
    """(overall score, level, facts sorted by risk) over identity-matched findings."""
    facts = sorted((build_fact(g) for g in group_facts(findings)), key=lambda x: -x.risk)
    clear = 1.0
    for fact in facts:
        clear *= 1 - fact.risk
    total = round(1 - clear, 3)
    level = next((name for bound, name in LEVELS if total >= bound), "none")
    return total, level, facts


def overall(findings: list[Finding]) -> tuple[float, str]:
    total, level, _ = assess(findings)
    return total, level
