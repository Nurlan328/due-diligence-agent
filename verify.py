"""Faithfulness check: does the quoted text really appear on the cited page?

Every web finding carries a verbatim quote. We fetch the page ourselves, extract
the main text and look for the quote (exact after normalisation, then fuzzy for
whitespace/quote-mark/hyphenation differences). A quote missing from a page we
could read is treated as a likely hallucination and heavily discounted.

A page that is gone or no longer holds the quote is looked up in the Wayback
Machine; a match there counts as verified from the archive.

The fetched text is also handed to the identity matcher (БИН in the text, city,
related names).
"""
import re
from concurrent.futures import ThreadPoolExecutor

import httpx
from rapidfuzz import fuzz

from config import settings
from models import Finding, Verification

USER_AGENT = "Mozilla/5.0 (compatible; due-diligence-agent/0.1; compliance screening)"


def html_to_text(html: str) -> str:
    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False, include_tables=True, favor_recall=True)
        if text:
            return text
    except Exception:  # trafilatura is optional; fall back to a crude strip
        pass
    html = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", html)).strip()


def _normalise(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[«»\"“”„'`’]", "", text)
    text = re.sub(r"[‐-―−-]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_in_text(quote: str, text: str, threshold: float | None = None) -> bool:
    threshold = threshold if threshold is not None else settings.QUOTE_MATCH_THRESHOLD
    q, t = _normalise(quote), _normalise(text)
    if not q:
        return False
    if q in t:
        return True
    return fuzz.partial_ratio(q, t) >= threshold


def fetch_text(url: str, http: httpx.Client) -> str | None:
    try:
        r = http.get(url)
        if r.status_code >= 400:
            return None
        ctype = r.headers.get("content-type", "")
        if "html" in ctype or not ctype:
            return html_to_text(r.text)
        if ctype.startswith("text/"):
            return r.text
        return None  # PDF and other binaries: left unverified in the MVP
    except httpx.HTTPError:
        return None


WAYBACK_API = "https://archive.org/wayback/available"


class ArchiveUnavailable(Exception):
    """archive.org did not answer (network error or timeout)."""


def archive_snapshot(url: str, http: httpx.Client) -> str | None:
    """URL of the closest Wayback Machine snapshot, raw (no toolbar), or None if there is none.
    Raises ArchiveUnavailable when archive.org itself cannot be reached.

    The availability API answers {"archived_snapshots": {"closest": {"available": true,
    "url": ..., "timestamp": "20240102030405"}}}; inserting "id_" after the timestamp
    returns the page as archived, without the Wayback banner.
    """
    try:
        r = http.get(WAYBACK_API, params={"url": url}, timeout=settings.ARCHIVE_TIMEOUT)
    except httpx.TransportError as e:
        raise ArchiveUnavailable(str(e)) from e
    try:
        closest = (r.json().get("archived_snapshots") or {}).get("closest") or {}
    except (ValueError, AttributeError):
        return None
    ts = str(closest.get("timestamp", ""))
    if not closest.get("available") or not ts.isdigit():
        return None
    return f"https://web.archive.org/web/{ts}id_/{url}"


def _check_archive(f: Finding, http: httpx.Client, snapshots: dict[str, tuple[str, str | None] | None]) -> bool:
    """Look for the quote in the archived copy; memoised per URL. True if found."""
    if f.url not in snapshots:
        snap = archive_snapshot(f.url, http)
        snapshots[f.url] = (snap, fetch_text(snap, http)) if snap else None
    entry = snapshots[f.url]
    if entry and entry[1] and quote_in_text(f.quote, entry[1]):
        f.verification = Verification.VERIFIED_ARCHIVE
        f.archive_url = entry[0]
        return True
    return False


def verify_findings(findings: list[Finding], http: httpx.Client | None = None,
                    workers: int = 8, archive: bool | None = None) -> dict[str, str | None]:
    """Set `verification` on each web finding. Returns url -> page text (None if unreachable).

    A page that is gone, or no longer contains the quote, is looked up in the Wayback
    Machine: removed news is a common pattern in adverse media, and a quote that was
    edited out after publication is itself worth knowing."""
    archive = settings.ARCHIVE_ENABLED if archive is None else archive
    own = http is None
    http = http or httpx.Client(timeout=settings.HTTP_TIMEOUT, follow_redirects=True,
                                headers={"User-Agent": USER_AGENT})
    urls = sorted({f.url for f in findings if f.source_type == "web"})
    snapshots: dict[str, tuple[str, str | None] | None] = {}
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pages = dict(zip(urls, pool.map(lambda u: fetch_text(u, http), urls)))
        for f in findings:
            if f.source_type != "web":
                continue
            text = pages.get(f.url)
            if text is not None and quote_in_text(f.quote, text):
                f.verification = Verification.VERIFIED
                continue
            if archive:
                try:
                    if _check_archive(f, http, snapshots):
                        if text is None:
                            pages[f.url] = snapshots[f.url][1]  # the matcher reads the archived page
                        continue
                except ArchiveUnavailable:
                    # Circuit breaker: don't wait out the timeout once per finding.
                    archive = False
            f.verification = Verification.UNREACHABLE if text is None else Verification.QUOTE_NOT_FOUND
    finally:
        if own:
            http.close()
    return pages
