"""Audit trail: who screened whom, why, and what went out to external services.

Screening a person is processing personal data. Each request is logged with the
operator, the stated purpose and the queries sent, so every check can be
justified later. A person's ИИН is masked in the log. Rows go to the `audit`
table of the local store (store.py).
"""
from datetime import UTC, datetime

import store
from models import Subject, SubjectType


def mask_identifier(subject: Subject) -> str | None:
    if not subject.identifier:
        return None
    if subject.type == SubjectType.COMPANY:
        return subject.identifier
    return "********" + subject.identifier[-4:]


def log_event(event: str, request_id: str, subject: Subject, operator: str, **extra) -> None:
    store.add_audit({
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": event,
        "request_id": request_id,
        "operator": operator,
        "purpose": subject.purpose,
        "subject_type": str(subject.type),
        "subject_name": subject.name,
        "identifier": mask_identifier(subject),
        **extra,
    })
