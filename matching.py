"""Identity resolution: is this source about our subject or about a namesake?

Namesakes are the main source of false positives in adverse-media screening:
"Ахметов Ержан" matches hundreds of people. Each finding gets a 0..1 score from
explicit features, in two groups:

- evidence FOR: name agreement (inflection-aware), the subject's БИН/ИИН in the
  text, the subject's city, related names, the subject's birth year or age;
- evidence AGAINST: another БИН/ИИН attached to the subject's name, and near
  the name another city, another birth year or an age that does not fit, a
  different patronymic or given name.

v1 had only the first group, and the eval (evaluate.py) showed the cost: a full
namesake from another city scored 1.0 and half of all namesakes leaked into the
risk score. The scorer stays a transparent heuristic: every feature vector is
logged to MATCH_LOG, and once reviewers label findings as same / different
entity, that log is the training set for a learned matcher evaluated against it.
"""
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import geo
from config import settings
from models import Finding, Subject, SubjectType
from names import (fold, given_name_conflict, name_similarity, patronymic_conflict, skeleton, strip_legal_form,
                   token_match)

WINDOW = 300  # chars of folded text on each side of a name occurrence
ID_LOOKBACK = 25  # tokens searched before a БИН/ИИН for the entity it belongs to

_ID = re.compile(r"(?<!\d)\d{12}(?!\d)")
# Folded text: punctuation is gone, so "г.р." is "г р" and "48-летний" is "48 летний".
_BIRTH_YEAR = [re.compile(r"(?<!\d)(19\d\d|20[0-2]\d) (?:года рождения|г р\b|гр\b|жылы туган|жылгы)"),
               re.compile(r"born(?: in)? (19\d\d|20[0-2]\d)")]
# Tokens that start another entity: a Russian-style surname or a legal form. A БИН/ИИН
# belongs to the nearest entity before it, not to everything in the window.
_SURNAME_LIKE = re.compile(r"^(?:[а-я]{3,}(?:ов|ев|ин|ын|ова|ева|ина|ына|ко|ук|юк|ян)"
                           r"|[a-z]{3,}(?:ov|ev|in|ova|eva|ina|yan))$")
_LEGAL_TOKENS = {"тоо", "ао", "жшс", "ак", "ип", "llp", "jsc", "llc", "ltd"}
_AGE = [re.compile(r"(?<!\d)(\d\d) летн"), re.compile(r"в возрасте (\d\d)"),
        re.compile(r"(?:мужчин\w*|женщин\w*|ему|ей) (\d\d) (?:лет|год)"),
        re.compile(r"(?<!\d)(\d\d) years old"), re.compile(r"aged (\d\d)"), re.compile(r"(?<!\d)(\d\d) жаста")]


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
    # Evidence against (v2). Defaults keep v1 match logs loadable.
    other_id_near: bool = False
    other_city_near: bool = False
    birth_conflict: bool = False
    patronymic_conflict: bool = False
    window_found: bool = False
    given_conflict: bool = False  # v3


def _anchor(subject: Subject) -> str:
    """The token whose occurrences mark where the text talks about the subject."""
    if subject.type == SubjectType.PERSON:
        return fold(subject.name).split()[0]  # surname
    core = fold(strip_legal_form(subject.name)).split()
    return core[0] if core else ""


def name_windows(subject: Subject, folded: str) -> list[str]:
    """Pieces of the text around each occurrence of the subject's surname / company name.

    A page often names several people and places; evidence against only counts when
    it sits next to the name, not anywhere on the page."""
    anchor = _anchor(subject)
    if not anchor:
        return []
    threshold = 92 if subject.type == SubjectType.PERSON else 85
    spans = [m.start() for m in re.finditer(r"\w+", folded) if token_match(m.group(), anchor, threshold)]
    return [folded[max(0, p - WINDOW): p + WINDOW] for p in spans]


def subject_ids(subject: Subject, folded: str) -> set[str]:
    """БИН/ИИН in the text that belong to the subject's mention.

    Each 12-digit number is attributed to the nearest entity before it: the subject's
    name, another surname, or a legal form (the start of another company). In
    "Толеубаев С.М., Касымов Е.Н. (ИИН …)" the ИИН is Касымов's, not a contradiction.
    A number with no entity before it goes to the first entity right after it
    ("ИИН … — Толеубаев С.М.")."""
    anchor = _anchor(subject)
    threshold = 92 if subject.type == SubjectType.PERSON else 85
    tokens = re.findall(r"\w+", folded)

    def owner(i: int) -> str | None:
        if anchor and token_match(tokens[i], anchor, threshold):
            return "subject"
        if tokens[i] in _LEGAL_TOKENS or _SURNAME_LIKE.match(tokens[i]):
            return "other"
        return None

    out = set()
    for i, tok in enumerate(tokens):
        if not _ID.fullmatch(tok):
            continue
        who = next((w for j in range(i - 1, max(-1, i - 1 - ID_LOOKBACK), -1) if (w := owner(j))), None)
        if who is None:
            who = next((w for j in range(i + 1, min(len(tokens), i + 6)) if (w := owner(j))), None)
        if who == "subject":
            out.add(tok)
    return out


def _birth_evidence(subject: Subject, near: str, published: str) -> tuple[bool, bool]:
    """(fits, conflicts): the subject's birth year against years and ages near the name."""
    if not subject.birth_date:
        return False, False
    year = subject.birth_date.year
    fits = conflict = False
    for pattern in _BIRTH_YEAR:
        for m in pattern.finditer(near):
            if abs(int(m.group(1)) - year) <= 1:
                fits = True
            else:
                conflict = True
    pub = re.search(r"(?:19|20)\d\d", published or "")
    now = int(pub.group()) if pub else (settings.REFERENCE_DATE or date.today()).year
    for pattern in _AGE:
        for m in pattern.finditer(near):
            implied = now - int(m.group(1))
            if abs(implied - year) <= 1:
                fits = True
            # Without a publication date the article may be old: the age can only show the
            # subject is too young to be that person, never that they are too old.
            elif pub or year > implied + 1:
                conflict = True
    return fits, conflict and not fits


def features(subject: Subject, finding: Finding, page_text: str | None) -> MatchFeatures:
    mention = finding.subject_mentioned_as or finding.title
    name_score, name_reason = name_similarity(subject, mention) if mention else (0.0, "имя не указано")
    # Without the page, fall back to what the finding itself says.
    text = page_text or " ".join([finding.title, finding.quote, finding.claim, *finding.identity_evidence])
    folded = fold(text)
    windows = name_windows(subject, folded)
    near = " | ".join(windows) if windows else folded

    id_in_text = bool(subject.identifier and subject.identifier in set(_ID.findall(folded)))
    fits_birth, birth_conflict = _birth_evidence(subject, near, finding.published)
    city_near = bool(subject.city) and geo.mentions_city(near, subject.city, folded=True)
    home = geo.canonical(subject.city)
    other_cities = geo.cities_in(near, folded=True) - ({home} if home else set())

    skel = skeleton(text)
    related_hits = sum(1 for rel in subject.related if (key := skeleton(rel)) and key in skel)
    core = skeleton(subject.name)
    return MatchFeatures(
        name_score=name_score,
        name_reason=name_reason,
        id_in_text=id_in_text,
        # Registry collectors say so in identity_evidence when the hit came by identifier.
        registry_id_lookup=finding.source_type == "registry"
        and any("БИН/ИИН" in e for e in finding.identity_evidence),
        city_in_text=city_near,
        related_hits=related_hits,
        birth_year_in_text=fits_birth,
        short_company_name=subject.type == SubjectType.COMPANY and len(core.replace(" ", "")) < 5,
        llm_evidence_count=len(finding.identity_evidence),
        page_fetched=page_text is not None,
        other_id_near=bool(subject_ids(subject, folded) - {subject.identifier}),
        other_city_near=bool(subject.city) and bool(other_cities) and not city_near,
        birth_conflict=birth_conflict,
        patronymic_conflict=subject.type == SubjectType.PERSON and patronymic_conflict(subject.name, mention),
        window_found=bool(windows),
        given_conflict=subject.type == SubjectType.PERSON and given_name_conflict(subject.name, mention),
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
        reasons.append("год рождения / возраст совпадает")
    if feat.short_company_name and not feat.id_in_text:
        s -= 0.2
        reasons.append("короткое название, высок риск совпадения с другими компаниями")

    # Evidence against. The subject's own БИН/ИИН in the text overrides all of it.
    if not feat.id_in_text:
        if feat.other_id_near:
            s = min(s, 0.1)
            reasons.append("рядом с именем другой БИН/ИИН")
        if feat.patronymic_conflict:
            s = min(s, 0.3)
            reasons.append("другое отчество")
        if feat.given_conflict:
            s = min(s, 0.3)
            reasons.append("другое имя")
        if feat.birth_conflict:
            s = min(s, 0.3)
            reasons.append("другой год рождения или возраст")
        if feat.other_city_near and not feat.related_hits:
            s = min(s, 0.45)
            reasons.append("в источнике другой город")
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
