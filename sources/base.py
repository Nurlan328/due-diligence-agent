"""Collector interface: a structured source that returns findings for a subject."""
from typing import Protocol

from models import Finding, SourceCheck, Subject


class Collector(Protocol):
    name: str

    def check(self, subject: Subject) -> tuple[list[Finding], SourceCheck]:
        """Query the source. Never raises: failures come back as SourceCheck(status="error")."""
        ...
