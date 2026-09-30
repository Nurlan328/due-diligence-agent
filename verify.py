"""Faithfulness check: does the quoted text really appear on the cited page?

Every web finding carries a verbatim quote. We fetch the page ourselves, extract
the main text and look for the quote (exact after normalisation, then fuzzy for
whitespace/quote-mark/hyphenation differences). A quote missing from a page we
could read is treated as a likely hallucination and heavily discounted.

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


def verify_findings(findings: list[Finding], http: httpx.Client | None = None,
                    workers: int = 8) -> dict[str, str | None]:
    """Set `verification` on each web finding. Returns url -> page text (None if unreachable)."""
    own = http is None
    http = http or httpx.Client(timeout=settings.HTTP_TIMEOUT, follow_redirects=True,
                                headers={"User-Agent": USER_AGENT})
    urls = sorted({f.url for f in findings if f.source_type == "web"})
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pages = dict(zip(urls, pool.map(lambda u: fetch_text(u, http), urls)))
    finally:
        if own:
            http.close()

    for f in findings:
        if f.source_type != "web":
            continue
        text = pages.get(f.url)
        if text is None:
            f.verification = Verification.UNREACHABLE
        elif quote_in_text(f.quote, text):
            f.verification = Verification.VERIFIED
        else:
            f.verification = Verification.QUOTE_NOT_FOUND
    return pages
