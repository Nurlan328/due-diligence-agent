"""Identity resolution: is this source about our subject or about a namesake?

Namesakes are the main source of false positives in adverse-media screening:
"Ахметов Ержан" matches hundreds of people. Each finding gets a 0..1 score from
explicit features (name agreement, БИН/ИИН in the page, city, related names,
birth year). The scorer is a transparent heuristic on purpose: every feature
vector is logged to MATCH_LOG, and once reviewers label findings as
same / different entity, that log is the training set for a learned matcher
(logistic regression or a cross-encoder) evaluated against this baseline.
"""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from config import settings
from models import Finding, Subject, SubjectType
from names import fold, name_similarity, skeleton


@dataclass
class MatchFeatures:
    name_score: float
    name_reason: str
    id_in_text: bool
    registry_id_lookup: bool
    city_in_text: bool
    related_hits: int
    birth_year_in_text: bool
    short_company_name: bool
    llm_evidence_count: int
    page_fetched: bool


def features(subject: Subject, finding: Finding, page_text: str | None) -> MatchFeatures:
    mention = finding.subject_mentioned_as or finding.title
    name_score, name_reason = name_similarity(subject, mention) if mention else (0.0, "имя не указано")
    # Without the page, fall back to what the finding itself says.
    text = page_text or " ".join([finding.title, finding.quote, finding.claim, *finding.identity_evidence])
    digits = re.sub(r"\D", " ", text)
    folded = fold(text)
    skel = skeleton(text)

    related_hits = 0
    for rel in subject.related:
        key = skeleton(rel)
        if key and key in skel:
            related_hits += 1
    core = skeleton(subject.name)
    return MatchFeatures(
        name_score=name_score,
        name_reason=name_reason,
        id_in_text=bool(subject.identifier and re.search(rf"\b{subject.identifier}\b", digits)),
        # Registry collectors say so in identity_evidence when the hit came by identifier.
        registry_id_lookup=finding.source_type == "registry"
        and any("БИН/ИИН" in e for e in finding.identity_evidence),
        city_in_text=bool(subject.city and fold(subject.city) in folded),
        related_hits=related_hits,
        birth_year_in_text=bool(subject.birth_date and str(subject.birth_date.year) in text),
        short_company_name=subject.type == SubjectType.COMPANY and len(core.replace(" ", "")) < 5,
        llm_evidence_count=len(finding.identity_evidence),
        page_fetched=page_text is not None,
    )


def score(feat: MatchFeatures, subject: Subject) -> tuple[float, list[str]]:
    reasons = [feat.name_reason]
    if feat.registry_id_lookup:
        return 1.0, ["запись найдена по БИН/ИИН"]
    s = feat.name_score
    if feat.id_in_text:
        s = max(s, 0.97)
        reasons.append("БИН/ИИН субъекта есть в тексте источника")
    if feat.city_in_text:
        s += 0.1
        reasons.append("совпадает город")
    if feat.related_hits:
        s += min(0.15 * feat.related_hits, 0.3)
        reasons.append(f"упомянуты связанные лица/компании: {feat.related_hits}")
    if feat.birth_year_in_text and subject.type == SubjectType.PERSON:
        s += 0.1
        reasons.append("в тексте есть год рождения")
    if feat.short_company_name and not feat.id_in_text:
        s -= 0.2
        reasons.append("короткое название, высок риск совпадения с другими компаниями")
    return round(min(max(s, 0.0), 1.0), 3), reasons


def match_findings(subject: Subject, findings: list[Finding], pages: dict[str, str | None],
                   request_id: str = "", log_path: str | None = None) -> None:
    """Set match_score / match_reasons on each finding and log the feature vectors."""
    log = Path(log_path if log_path is not None else settings.MATCH_LOG)
    rows = []
    for f in findings:
        feat = features(subject, f, pages.get(f.url))
        f.match_score, f.match_reasons = score(feat, subject)
        rows.append({"request_id": request_id, "finding_id": f.id, "subject_type": subject.type,
                     "source_type": f.source_type, "verification": f.verification,
                     **asdict(feat), "score": f.match_score, "label": None})
    if rows and log_path != "":
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
