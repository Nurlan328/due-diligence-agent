"""HTTP API.  Run: fastapi dev main.py   (docs at /docs)

A screening takes minutes (web searches, page fetches, several agent rounds), so the
main entry point is asynchronous:

    POST /jobs          -> 202 {"job_id": ..., "status": "queued"}
    GET  /jobs/{job_id} -> {"status": "queued" | "running" | "done" | "failed", "report": ...}

Jobs run on a small thread pool inside the API process and their state lives in the
SQLite store, so a status survives the request that created it. A job still queued or
running when the process stops is marked failed on the next start; a multi-instance
deployment would move the pool to a queue (Redis / RabbitMQ) and the store to Postgres.
The blocking POST /screen stays for scripts and small checks.
"""
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

import store
from config import settings
from models import Report, Subject
from pipeline import screen

_executor: ThreadPoolExecutor | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _executor
    store.fail_unfinished_jobs("interrupted: the API process restarted")
    _executor = ThreadPoolExecutor(max_workers=settings.JOB_WORKERS, thread_name_prefix="screening")
    try:
        yield
    finally:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None


app = FastAPI(title="Due diligence agent", version="0.2.0", lifespan=lifespan)


class ScreenRequest(BaseModel):
    subject: Subject
    operator: str
    deep: bool = True
    max_rounds: int | None = None


class JobCreated(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    error: str | None = None
    report: Report | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _run_job(job_id: str, req: ScreenRequest) -> None:
    store.update_job(job_id, "running")
    try:
        report = screen(req.subject, operator=req.operator, deep=req.deep, max_rounds=req.max_rounds)
    except Exception as e:  # any failure must end up in the job row, not in a dead thread
        store.update_job(job_id, "failed", error=f"{type(e).__name__}: {e}"[:1000])
        return
    store.update_job(job_id, "done", report=report.model_dump_json())


@app.post("/jobs", response_model=JobCreated, status_code=202)
def create_job(req: ScreenRequest) -> JobCreated:
    if _executor is None:
        raise HTTPException(503, "job pool is not running")
    job_id = uuid.uuid4().hex
    store.create_job(job_id, req.operator, req.model_dump(mode="json"))
    _executor.submit(_run_job, job_id, req)
    return JobCreated(job_id=job_id, status="queued")


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    row = store.get_job(job_id)
    if row is None:
        raise HTTPException(404, "no such job")
    report = Report.model_validate_json(row["report"]) if row["report"] else None
    return JobStatus(job_id=row["id"], status=row["status"], created_at=row["created_at"],
                     updated_at=row["updated_at"], error=row["error"], report=report)


# Plain `def`: the pipeline is blocking (API calls, page fetches); FastAPI runs it in a thread pool.
@app.post("/screen", response_model=Report)
def screen_endpoint(req: ScreenRequest) -> Report:
    return screen(req.subject, operator=req.operator, deep=req.deep, max_rounds=req.max_rounds)


@app.post("/screen.md", response_class=PlainTextResponse)
def screen_markdown(req: ScreenRequest) -> str:
    return screen(req.subject, operator=req.operator, deep=req.deep, max_rounds=req.max_rounds).markdown
