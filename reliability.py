"""How much a source and its age are worth.

An official register outweighs a national newspaper, which outweighs a forum or a
Telegram channel. A ten-year-old story weighs less than last month's, unless it is a
current register entry. The tiers and weights are explicit on purpose: they are a
policy the compliance team owns, not something to learn from data.
"""
import re
from datetime import date
from urllib.parse import urlsplit

from config import settings
from models import Finding, Status

TIER_WEIGHT = {"registry": 1.0, "official": 1.0, "major_media": 0.9, "media": 0.7, "low": 0.4}

OFFICIAL = {"sud.kz", "adilet.zan.kz", "egov.kz", "kgd.gov.kz", "goszakup.gov.kz", "afmrk.gov.kz",
            "stat.gov.kz", "opensanctions.org", "treasury.gov", "un.org", "europa.eu", "gov.uk"}
MAJOR_MEDIA = {"tengrinews.kz", "zakon.kz", "informburo.kz", "kursiv.media", "vlast.kz", "orda.kz",
               "azattyq.org", "forbes.kz", "kapital.kz", "inbusiness.kz", "nur.kz", "inform.kz", "khabar.kz",
               "ktk.kz", "lsm.kz", "factcheck.kz", "kz.kursiv.media", "reuters.com", "occrp.org", "icij.org",
               "bloomberg.com", "ft.com", "bbc.com", "rferl.org", "interfax.kz", "total.kz", "time.kz"}
LOW = {"t.me", "telegram.me", "vk.com", "facebook.com", "instagram.com", "youtube.com", "tiktok.com",
       "livejournal.com", "pikabu.ru", "otzovik.com", "irecommend.ru", "x.com", "twitter.com"}
_SECOND_LEVEL = {"gov", "com", "org", "edu", "co", "net", "ac"}


def domain(url: str) -> str:
    """Registrable domain: news.tengrinews.kz -> tengrinews.kz, kgd.gov.kz stays kgd.gov.kz."""
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in _SECOND_LEVEL and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def source_tier(f: Finding) -> str:
    if f.source_type == "registry":
        return "registry"
    host = (urlsplit(f.url).hostname or "").lower()
    d = domain(f.url)
    # The *_EXTRA settings extend the lists without a code change (the team's own policy).
    if d in OFFICIAL | set(settings.OFFICIAL_EXTRA) or host.endswith(".gov.kz") or host.endswith(".gov"):
        return "official"
    if d in MAJOR_MEDIA | set(settings.MAJOR_MEDIA_EXTRA):
        return "major_media"
    if d in LOW | set(settings.LOW_EXTRA) or host.startswith("forum.") or "/forum" in f.url:
        return "low"
    return "media"


def source_key(f: Finding) -> str:
    """Two findings with the same key are not independent confirmations."""
    return f.source_name if f.source_type == "registry" else domain(f.url)


def published_year(published: str) -> int | None:
    m = re.search(r"(?<!\d)(19[89]\d|20\d\d)(?!\d)", published or "")
    return int(m.group(1)) if m else None


def recency_weight(f: Finding, today: date | None = None) -> float:
    """1.0 for the last two years, then halving every five years, floor 0.3.
    Unknown date: 0.8. A current register entry does not age."""
    if f.status == Status.LISTED:
        return 1.0
    year = published_year(f.published)
    if year is None:
        return 0.8
    age = (today or settings.REFERENCE_DATE or date.today()).year - year
    if age <= 2:
        return 1.0
    return round(max(0.3, 0.5 ** ((age - 2) / 5)), 3)


def annotate(f: Finding, today: date | None = None) -> None:
    f.source_tier = source_tier(f)
    f.source_weight = TIER_WEIGHT[f.source_tier]
    f.recency_weight = recency_weight(f, today)
