"""Application configuration (pydantic-settings, read from .env / environment)."""
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

    # --- Storage ---
    DATA_DIR: str = "data"
    REPORTS_DIR: str = "reports"
    # Every screening is logged: who, when, why, which queries went out.
    AUDIT_LOG: str = "data/audit.jsonl"
    # Identity-matching features + outcome per finding: the training set for a learned matcher.
    MATCH_LOG: str = "data/match_features.jsonl"


settings = Settings()
