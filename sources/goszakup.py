"""Реестр недобросовестных участников госзакупок (РНУ) via the goszakup.gov.kz open API.

GET {GOSZAKUP_URL}/rnu/{biin} with a Bearer token returns the registry entries
for one БИН/ИИН: customer, court decision, inclusion period, founders at the time.
Lookup is by identifier only, so it cannot hit a namesake.
"""
import httpx

from config import settings
from models import Category, Finding, Severity, SourceCheck, Status, Subject, Verification


class GoszakupRnuCollector:
    name = "goszakup.gov.kz РНУ"

    def __init__(self, token: str, base_url: str | None = None, http: httpx.Client | None = None):
        self.token = token
        self.base_url = (base_url or settings.GOSZAKUP_URL).rstrip("/")
        self.http = http

    def check(self, subject: Subject) -> tuple[list[Finding], SourceCheck]:
        if not self.token:
            return [], SourceCheck(name=self.name, status="skipped", note="GOSZAKUP_TOKEN не задан")
        if not subject.identifier:
            return [], SourceCheck(name=self.name, status="skipped", note="нужен БИН/ИИН")
        http = self.http or httpx.Client(timeout=settings.HTTP_TIMEOUT)
        try:
            r = http.get(f"{self.base_url}/rnu/{subject.identifier}",
                         headers={"Authorization": f"Bearer {self.token}"})
            if r.status_code == 404:
                return [], SourceCheck(name=self.name, status="ok", note="в реестре не найден")
            r.raise_for_status()
            payload = r.json()
        except (httpx.HTTPError, ValueError) as e:
            return [], SourceCheck(name=self.name, status="error", note=str(e)[:200])
        finally:
            if self.http is None:
                http.close()

        records = self._records(payload)
        findings = [self._finding(subject, rec) for rec in records
                    if str(rec.get("supplier_biin", subject.identifier)) == subject.identifier]
        return findings, SourceCheck(name=self.name, status="ok", hits=len(findings))

    @staticmethod
    def _records(payload) -> list[dict]:
        # The list endpoint wraps rows in {"items": [...]}; the detail endpoint may
        # return the rows, a single object or an empty body.
        if isinstance(payload, dict):
            if "items" in payload:
                return [x for x in payload["items"] or [] if isinstance(x, dict)]
            return [payload] if payload else []
        if isinstance(payload, list):
            return [x for x in payload if isinstance(x, dict)]
        return []

    def _finding(self, subject: Subject, rec: dict) -> Finding:
        start, end = str(rec.get("start_date") or "")[:10], str(rec.get("end_date") or "")[:10]
        customer = rec.get("customer_name_ru") or ""
        court = rec.get("court_decision") or ""
        parts = [f"включён в РНУ с {start or '?'} по {end or '?'}"]
        if customer:
            parts.append(f"заказчик: {customer}")
        if court:
            parts.append(f"решение суда: {court}")
        return Finding(
            source_type="registry",
            source_name=self.name,
            url=f"https://goszakup.gov.kz/ru/registry/rnu?search={subject.identifier}",
            title="Реестр недобросовестных участников государственных закупок",
            published=start,
            category=Category.PROCUREMENT,
            status=Status.LISTED,
            severity=Severity.HIGH,
            claim="Участник включён в реестр недобросовестных участников госзакупок",
            quote="; ".join(parts),
            subject_mentioned_as=rec.get("supplier_name_ru") or subject.name,
            identity_evidence=["поиск по БИН/ИИН"],
            verification=Verification.REGISTRY,
        )
