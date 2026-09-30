"""Sanctions and PEP screening against the OpenSanctions bulk export.

The "sanctions" collection merges OFAC SDN, UN, EU, UK HMT and 100+ other lists
into one simplified CSV. It is matched locally: the subject's name never goes to
a third-party matching API. Download it with `python cli.py update-sanctions`.
"""
import csv
import io
import sys
from pathlib import Path

import httpx

from config import settings
from models import Category, Finding, Severity, SourceCheck, Status, Subject, SubjectType, Verification
from names import company_similarity, person_similarity, skeleton

PERSON_SCHEMAS = {"Person"}
COMPANY_SCHEMAS = {"Company", "Organization", "LegalEntity"}

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def default_path() -> Path:
    return Path(settings.DATA_DIR) / "opensanctions_targets.simple.csv"


def download(path: Path | None = None, url: str | None = None) -> Path:
    path = path or default_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with httpx.stream("GET", url or settings.OPENSANCTIONS_URL, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    tmp.replace(path)
    return path


def _multi(cell: str) -> list[str]:
    """Multi-valued cells are a nested CSV with ';' as the delimiter."""
    if not cell:
        return []
    return [v.strip() for v in next(csv.reader(io.StringIO(cell), delimiter=";")) if v.strip()]


class SanctionsIndex:
    """In-memory token index over the CSV: candidate lookup by name token, then fuzzy scoring."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.by_token: dict[str, list[int]] = {}
        self.by_identifier: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            for name in [row["name"], *_multi(row.get("aliases", ""))]:
                for tok in set(skeleton(name).split()):
                    if len(tok) >= 3:
                        self.by_token.setdefault(tok, []).append(i)
            for ident in _multi(row.get("identifiers", "")):
                digits = "".join(c for c in ident if c.isdigit())
                if len(digits) == 12:
                    self.by_identifier.setdefault(digits, []).append(i)

    @classmethod
    def load(cls, path: Path) -> "SanctionsIndex":
        with path.open(encoding="utf-8", newline="") as f:
            return cls(list(csv.DictReader(f)))

    def candidates(self, subject: Subject) -> set[int]:
        out: set[int] = set()
        if subject.identifier:
            out.update(self.by_identifier.get(subject.identifier, []))
        for name in [subject.name, *subject.aliases]:
            for tok in skeleton(name).split():
                out.update(self.by_token.get(tok, []))
        return out


class SanctionsCollector:
    name = "OpenSanctions (sanctions)"

    def __init__(self, path: Path | None = None, index: SanctionsIndex | None = None,
                 threshold: float | None = None):
        self.path = path or default_path()
        self._index = index
        self.threshold = (threshold if threshold is not None else settings.SANCTIONS_MATCH_THRESHOLD) / 100

    def _get_index(self) -> SanctionsIndex | None:
        if self._index is None and self.path.exists():
            self._index = SanctionsIndex.load(self.path)
        return self._index

    def check(self, subject: Subject) -> tuple[list[Finding], SourceCheck]:
        try:
            index = self._get_index()
        except (OSError, csv.Error) as e:
            return [], SourceCheck(name=self.name, status="error", note=str(e))
        if index is None:
            return [], SourceCheck(name=self.name, status="skipped",
                                   note="нет локальной выгрузки: python cli.py update-sanctions")
        findings = [f for i in index.candidates(subject) if (f := self._match(subject, index.rows[i]))]
        return findings, SourceCheck(name=self.name, status="ok", hits=len(findings),
                                     note=f"{len(index.rows)} записей в выгрузке")

    def _match(self, subject: Subject, row: dict) -> Finding | None:
        schemas = PERSON_SCHEMAS if subject.type == SubjectType.PERSON else COMPANY_SCHEMAS
        if row.get("schema") not in schemas:
            return None
        evidence: list[str] = []
        identifiers = {"".join(c for c in i if c.isdigit()) for i in _multi(row.get("identifiers", ""))}
        id_match = bool(subject.identifier and subject.identifier in identifiers)

        best_name, best_score = row["name"], 0.0
        for listed in [row["name"], *_multi(row.get("aliases", ""))]:
            for own in [subject.name, *subject.aliases]:
                if subject.type == SubjectType.PERSON:
                    score, _ = person_similarity(own, listed)
                else:
                    score = company_similarity(own, listed)
                if score > best_score:
                    best_name, best_score = listed, score

        if id_match:
            evidence.append("БИН/ИИН совпадает с идентификатором в списке")
        elif best_score < self.threshold:
            return None

        listed_dobs = _multi(row.get("birth_date", ""))
        if subject.birth_date and listed_dobs:
            years = {d[:4] for d in listed_dobs}
            if str(subject.birth_date.year) not in years:
                if not id_match:
                    return None  # same name, different person
            else:
                evidence.append(f"год рождения совпадает ({subject.birth_date.year})")
        countries = _multi(row.get("countries", ""))
        if "kz" in countries:
            evidence.append("страна в списке: KZ")
        evidence.append(f"сходство имени {best_score:.0%}")

        programs = _multi(row.get("sanctions", "")) or _multi(row.get("dataset", ""))
        return Finding(
            source_type="registry",
            source_name="OpenSanctions",
            url=f"https://www.opensanctions.org/entities/{row['id']}/",
            title=row["name"],
            published=row.get("first_seen", "")[:10],
            category=Category.SANCTIONS,
            status=Status.LISTED,
            severity=Severity.CRITICAL,
            claim=f"Совпадение с санкционным списком: {row['name']}",
            quote="; ".join(programs)[:500],
            subject_mentioned_as=best_name,
            identity_evidence=evidence,
            verification=Verification.REGISTRY,
        )
