from config import settings
from sources.base import Collector
from sources.goszakup import GoszakupRnuCollector
from sources.sanctions import SanctionsCollector


def default_collectors() -> list[Collector]:
    return [
        SanctionsCollector(),
        GoszakupRnuCollector(token=settings.GOSZAKUP_TOKEN),
    ]
