"""Risk scoring: each finding contributes severity × status × verification × identity.

The overall score is a noisy-OR of the contributions: two independent medium
findings weigh more than one, but the score never exceeds 1. Weights are
explicit and meant to be tuned with the compliance team, not learned blindly.
"""
from config import settings
from models import Finding, Severity, Status, Verification

SEVERITY = {Severity.LOW: 0.2, Severity.MEDIUM: 0.45, Severity.HIGH: 0.75, Severity.CRITICAL: 1.0}
STATUS = {
    Status.LISTED: 1.0, Status.CONVICTED: 1.0, Status.CIVIL_JUDGMENT: 0.8, Status.CHARGED: 0.8,
    Status.INVESTIGATION: 0.6, Status.ALLEGATION: 0.4, Status.UNKNOWN: 0.4,
    Status.CLOSED: 0.2, Status.ACQUITTED: 0.0,
}
VERIFICATION = {
    Verification.REGISTRY: 1.0, Verification.VERIFIED: 1.0, Verification.UNREACHABLE: 0.6,
    Verification.NOT_CHECKED: 0.5, Verification.QUOTE_NOT_FOUND: 0.15,
}
LEVELS = [(0.7, "high"), (0.35, "medium"), (0.1, "low")]


def finding_risk(f: Finding) -> float:
    return round(SEVERITY[f.severity] * STATUS[f.status] * VERIFICATION[f.verification] * f.match_score, 4)


def split_by_identity(findings: list[Finding], threshold: float | None = None) -> tuple[list[Finding], list[Finding]]:
    """(about the subject, possible namesakes)."""
    threshold = threshold if threshold is not None else settings.MATCH_THRESHOLD
    matched = [f for f in findings if f.match_score >= threshold]
    namesakes = [f for f in findings if f.match_score < threshold]
    return matched, namesakes


def overall(findings: list[Finding]) -> tuple[float, str]:
    """Noisy-OR over matched findings. Several articles on one case are not independent,
    so only the strongest finding per (category, status) pair counts."""
    best: dict[tuple, float] = {}
    for f in findings:
        f.risk = finding_risk(f)
        key = (f.category, f.status)
        best[key] = max(best.get(key, 0.0), f.risk)
    clear = 1.0
    for r in best.values():
        clear *= 1 - r
    total = round(1 - clear, 3)
    level = next((name for bound, name in LEVELS if total >= bound), "none")
    return total, level
