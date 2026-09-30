"""Domain model: the subject being screened, findings about it, and the report."""
import hashlib
import re
from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class SubjectType(StrEnum):
    COMPANY = "company"  # юрлицо (БИН)
    PERSON = "person"  # физлицо (ИИН)


class Category(StrEnum):
    SANCTIONS = "sanctions"  # санкционные и террористические списки
    CORRUPTION = "corruption"  # взятки, хищение бюджетных средств, злоупотребление должностью
    FRAUD = "fraud"  # мошенничество, финансовые пирамиды, лжепредпринимательство
    MONEY_LAUNDERING = "money_laundering"  # отмывание, обналичивание
    TAX = "tax"  # налоговые долги, уклонение, бездействующий/лжеплательщик
    BANKRUPTCY = "bankruptcy"  # банкротство, реабилитация, ликвидация по суду
    LITIGATION = "litigation"  # гражданские и арбитражные споры
    ENFORCEMENT = "enforcement"  # исполнительные производства, запрет на выезд
    PROCUREMENT = "procurement"  # реестр недобросовестных участников госзакупок
    CRIMINAL = "criminal"  # прочие уголовные дела
    REGULATORY = "regulatory"  # штрафы и санкции регуляторов, отзыв лицензий
    LABOR = "labor"  # невыплата зарплаты, несчастные случаи на производстве
    ENVIRONMENTAL = "environmental"  # экологические нарушения
    OTHER = "other"


class Status(StrEnum):
    """Procedural stage. An allegation is not a conviction and the report must say which it is."""

    LISTED = "listed"  # запись в официальном реестре или санкционном списке
    CONVICTED = "convicted"  # вступивший в силу приговор
    CIVIL_JUDGMENT = "civil_judgment"  # решение суда по гражданскому/административному делу
    CHARGED = "charged"  # обвинение предъявлено, дело в суде
    INVESTIGATION = "investigation"  # досудебное расследование, обыски, задержание
    ALLEGATION = "allegation"  # публикации, жалобы, журналистские расследования
    CLOSED = "closed"  # дело прекращено, долг погашен, исключён из реестра
    ACQUITTED = "acquitted"  # оправдан
    UNKNOWN = "unknown"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Verification(StrEnum):
    REGISTRY = "registry"  # structured record from an official source, nothing to verify
    VERIFIED = "verified"  # the quote was found on the fetched page
    QUOTE_NOT_FOUND = "quote_not_found"  # page fetched, quote absent: likely hallucinated or edited
    UNREACHABLE = "unreachable"  # page could not be fetched (paywall, geo-block, removed)
    NOT_CHECKED = "not_checked"


class Subject(BaseModel):
    type: SubjectType
    name: str = Field(min_length=2)
    # БИН (company) or ИИН (person): 12 digits. For a person it never leaves the
    # machine: it is used only to confirm identity in fetched documents.
    identifier: str | None = None
    birth_date: date | None = None
    city: str | None = None
    aliases: list[str] = []
    # Known related names (director, founders, parent company) that help tell the
    # subject apart from namesakes. Official-register facts only.
    related: list[str] = []
    # Legal basis for processing: required, logged to the audit trail.
    purpose: str = Field(min_length=10)

    @field_validator("identifier")
    @classmethod
    def _twelve_digits(cls, v: str | None) -> str | None:
        if v is None or not v.strip():
            return None
        digits = re.sub(r"\D", "", v)
        if len(digits) != 12:
            raise ValueError("БИН/ИИН must have 12 digits")
        return digits


class Finding(BaseModel):
    source_type: str  # "registry" | "web"
    source_name: str  # "OpenSanctions", "goszakup РНУ", site domain
    url: str
    title: str = ""
    published: str = ""  # as found in the source, free-form date
    category: Category
    status: Status = Status.UNKNOWN
    severity: Severity = Severity.MEDIUM
    claim: str  # one-sentence summary of the adverse fact, in Russian
    quote: str = ""  # verbatim excerpt from the source that supports the claim
    subject_mentioned_as: str = ""  # how the source names the subject
    identity_evidence: list[str] = []  # why the source is believed to be about this subject
    verification: Verification = Verification.NOT_CHECKED
    match_score: float = 0.0  # 0..1: probability the source is about this subject, not a namesake
    match_reasons: list[str] = []
    risk: float = 0.0  # 0..1 contribution to the overall score

    @property
    def id(self) -> str:
        key = f"{self.url.strip().rstrip('/').lower()}|{self.category}|{self.quote[:120].lower()}"
        return hashlib.sha1(key.encode()).hexdigest()[:12]


class SourceCheck(BaseModel):
    """One source queried, with or without hits: the report lists what was checked."""

    name: str
    status: str  # "ok" | "skipped" | "error"
    hits: int = 0
    note: str = ""


class Report(BaseModel):
    request_id: str
    created_at: datetime
    subject: Subject
    risk_score: float  # 0..1
    risk_level: str  # none | low | medium | high
    findings: list[Finding]  # identity confirmed or likely, sorted by risk
    possible_namesakes: list[Finding]  # adverse, but probably another person/company
    sources: list[SourceCheck]
    queries: list[str]  # web searches the agent ran
    rounds: int
    markdown: str = ""
