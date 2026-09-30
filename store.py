"""Local SQLite store: audit trail, identity-matching features with reviewer labels, screening jobs.

Replaces the JSONL logs. SQLite in WAL mode takes concurrent writers from the API's
worker threads (and several processes on one host), rows can be queried, and a
reviewer's label is set in place on the finding it belongs to. One connection
per call keeps it thread-safe without a pool. A multi-host deployment would move
the same three tables to Postgres; nothing outside this module touches SQL.
"""
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    event        TEXT NOT NULL,
    request_id   TEXT NOT NULL,
    operator     TEXT NOT NULL,
    purpose      TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    subject_name TEXT NOT NULL,
    identifier   TEXT,               -- a person's ИИН is masked before it gets here
    extra        TEXT NOT NULL       -- JSON: queries sent out, risk level, usage...
);
CREATE INDEX IF NOT EXISTS audit_request ON audit(request_id);

CREATE TABLE IF NOT EXISTS match_features (
    request_id   TEXT NOT NULL,
    finding_id   TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    features     TEXT NOT NULL,      -- JSON of matching.MatchFeatures + context
    score        REAL NOT NULL,
    label        INTEGER,            -- 1 same entity, 0 namesake, NULL not reviewed
    labelled_by  TEXT,
    labelled_at  TEXT,
    PRIMARY KEY (request_id, finding_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    status       TEXT NOT NULL,      -- queued | running | done | failed
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    operator     TEXT NOT NULL,
    request      TEXT NOT NULL,      -- JSON of the API request
    report       TEXT,               -- JSON of models.Report when done
    error        TEXT
);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@contextmanager
def session(path: str | None = None) -> Iterator[sqlite3.Connection]:
    db = Path(path or settings.DB_PATH)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


# --- audit ---------------------------------------------------------------------------------------

def add_audit(row: dict[str, Any]) -> None:
    core = ("ts", "event", "request_id", "operator", "purpose", "subject_type", "subject_name", "identifier")
    extra = {k: v for k, v in row.items() if k not in core}
    with session() as db:
        db.execute("INSERT INTO audit (ts, event, request_id, operator, purpose, subject_type, subject_name, "
                   "identifier, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (*(row.get(k) for k in core), json.dumps(extra, ensure_ascii=False, default=str)))


def audit_events(request_id: str | None = None) -> list[dict[str, Any]]:
    sql, args = "SELECT * FROM audit", tuple[str, ...]()
    if request_id:
        sql, args = sql + " WHERE request_id = ?", (request_id,)
    with session() as db:
        rows = db.execute(sql + " ORDER BY id", args).fetchall()
    return [{**{k: r[k] for k in r.keys() if k != "extra"}, **json.loads(r["extra"])} for r in rows]


# --- identity-matching features and reviewer labels ----------------------------------------------

def add_match_rows(rows: list[dict[str, Any]]) -> None:
    with session() as db:
        db.executemany(
            "INSERT OR REPLACE INTO match_features (request_id, finding_id, subject_type, features, score) "
            "VALUES (?, ?, ?, ?, ?)",
            [(r["request_id"], r["finding_id"], r["subject_type"],
              json.dumps(r["features"], ensure_ascii=False, default=str), r["score"]) for r in rows])


def match_rows(labelled_only: bool = False, path: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM match_features" + (" WHERE label IS NOT NULL" if labelled_only else "")
    with session(path) as db:
        rows = db.execute(sql + " ORDER BY request_id, finding_id").fetchall()
    return [{**json.loads(r["features"]), "request_id": r["request_id"], "finding_id": r["finding_id"],
             "subject_type": r["subject_type"], "score": r["score"], "label": r["label"],
             "labelled_by": r["labelled_by"]} for r in rows]


def set_label(finding_id: str, label: int, reviewer: str, request_id: str | None = None) -> int:
    """Record a reviewer's decision: 1 = same entity, 0 = namesake. Returns rows updated."""
    sql = "UPDATE match_features SET label = ?, labelled_by = ?, labelled_at = ? WHERE finding_id = ?"
    args: tuple = (label, reviewer, now(), finding_id)
    if request_id:
        sql, args = sql + " AND request_id = ?", (*args, request_id)
    with session() as db:
        return db.execute(sql, args).rowcount


# --- screening jobs ------------------------------------------------------------------------------

def create_job(job_id: str, operator: str, request: dict[str, Any]) -> None:
    ts = now()
    with session() as db:
        db.execute("INSERT INTO jobs (id, status, created_at, updated_at, operator, request) VALUES (?, ?, ?, ?, ?, ?)",
                   (job_id, "queued", ts, ts, operator, json.dumps(request, ensure_ascii=False, default=str)))


def update_job(job_id: str, status: str, report: str | None = None, error: str | None = None) -> None:
    with session() as db:
        db.execute("UPDATE jobs SET status = ?, updated_at = ?, report = COALESCE(?, report), error = ? WHERE id = ?",
                   (status, now(), report, error, job_id))


def get_job(job_id: str) -> dict[str, Any] | None:
    with session() as db:
        row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def fail_unfinished_jobs(reason: str) -> int:
    """Jobs left queued/running by a previous process will never finish: mark them failed."""
    with session() as db:
        return db.execute("UPDATE jobs SET status = 'failed', error = ?, updated_at = ? "
                          "WHERE status IN ('queued', 'running')", (reason, now())).rowcount
