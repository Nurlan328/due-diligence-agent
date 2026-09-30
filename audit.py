"""Audit trail: who screened whom, why, and what went out to external services.

Screening a person is processing personal data. Each request is logged with the
operator, the stated purpose and the queries sent, so every check can be
justified later. A person's ИИН is masked in the log.
"""
import json
from datetime import UTC, datetime
from pathlib import Path

from config import settings
from models import Subject, SubjectType


def mask_identifier(subject: Subject) -> str | None:
    if not subject.identifier:
        return None
    if subject.type == SubjectType.COMPANY:
        return subject.identifier
    return "********" + subject.identifier[-4:]


def log_event(event: str, request_id: str, subject: Subject, operator: str, path: str | None = None,
              **extra) -> None:
    log = Path(path or settings.AUDIT_LOG)
    log.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": event,
        "request_id": request_id,
        "operator": operator,
        "purpose": subject.purpose,
        "subject_type": subject.type,
        "subject_name": subject.name,
        "identifier": mask_identifier(subject),
        **extra,
    }
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
