"""Application configuration (pydantic-settings, read from .env / environment)."""
from datetime import date

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Claude (deep search agent) ---
    ANTHROPIC_API_KEY: str = ""
    MODEL: str = "claude-opus-5-5"
    # low | medium | high | xhigh | max. Opus 5.5 defaults to medium; research needs more.
    EFFORT: str = "high"
    MAX_TOKENS: int = 32000
    # Server-side refusal fallback (fallbacks="default"): a declined request is re-run
    # on the model Anthropic recommends for that refusal category, inside the same call.
    USE_FALLBACKS: bool = True

    # --- Deep search budget ---
    # One round = one agent conversation: web searches, page fetches, recorded findings.
    # Round N+1 follows the leads (affiliated companies, case numbers) found in round N.
    MAX_ROUNDS: int = 2
    SEARCHES_PER_ROUND: int = 10
    FETCHES_PER_ROUND: int = 8
    MAX_AGENT_TURNS: int = 12  # safety cap on the tool loop inside one round

    # --- Structured sources ---
    # OpenSanctions "sanctions" collection, simplified CSV. CC BY-NC 4.0: free for
    # non-commercial use, a commercial deployment needs a data licence.
    OPENSANCTIONS_URL: str = "https://data.opensanctions.org/datasets/latest/sanctions/targets.simple.csv"
    SANCTIONS_MATCH_THRESHOLD: float = 88.0
    # goszakup.gov.kz open API token (free, issued on request). Empty -> source skipped.
    GOSZAKUP_TOKEN: str = ""
    GOSZAKUP_URL: str = "https://ows.goszakup.gov.kz/v3"

    # --- Verification ---
    HTTP_TIMEOUT: float = 20.0
    QUOTE_MATCH_THRESHOLD: float = 90.0  # rapidfuzz partial_ratio, 0..100
    # Findings below this identity score are reported as possible namesakes, not as risk.
    MATCH_THRESHOLD: float = 0.5
    # Wayback Machine fallback when a cited page is gone or no longer contains the quote.
    # archive.org is intermittently unreachable from Kazakhstani networks: failures are soft.
    ARCHIVE_ENABLED: bool = True
    ARCHIVE_TIMEOUT: float = 15.0

    # --- Source tiers (reliability.py) ---
    # Extra registrable domains per tier, on top of the built-in lists. JSON list in env:
    # MAJOR_MEDIA_EXTRA='["example-news.kz"]'
    OFFICIAL_EXTRA: list[str] = []
    MAJOR_MEDIA_EXTRA: list[str] = []
    LOW_EXTRA: list[str] = []

    # "Today" for age and recency rules. Empty = the real date; evaluate.py pins it to the date the
    # eval sets were written, so evals/RESULTS.md stays reproducible (CI checks it).
    REFERENCE_DATE: date | None = None

    # --- Scoring ---
    # A fact about a director / founder / subsidiary counts at this share of the same fact
    # about the subject itself.
    AFFILIATE_WEIGHT: float = 0.5

    # --- Entity graph (deep search pivots) ---
    MAX_DEPTH: int = 2  # hops from the subject: 1 = its directors / founders / subsidiaries
    PIVOTS_PER_ROUND: int = 5

    # --- Storage ---
    DATA_DIR: str = "data"
    REPORTS_DIR: str = "reports"
    # SQLite (store.py): the audit trail (who, when, why, which queries went out), identity-matching
    # features with reviewer labels (the training set for a learned matcher), and API jobs.
    DB_PATH: str = "data/dd.sqlite"

    # --- API ---
    JOB_WORKERS: int = 2  # screenings running at once in the background (POST /jobs)

    # --- Cost accounting (USD). Check current pricing; a refusal fallback bills at the other model's rates.
    PRICE_INPUT_PER_MTOK: float = 4.0
    PRICE_OUTPUT_PER_MTOK: float = 20.0
    PRICE_CACHE_READ_PER_MTOK: float = 0.2
    PRICE_CACHE_WRITE_PER_MTOK: float = 5.0
    PRICE_WEB_SEARCH_PER_1K: float = 10.0


settings = Settings()
