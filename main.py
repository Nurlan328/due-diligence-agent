"""HTTP API.  Run: fastapi dev main.py   (docs at /docs)"""
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from models import Report, Subject
from pipeline import screen

app = FastAPI(title="Due diligence agent", version="0.1.0")


class ScreenRequest(BaseModel):
    subject: Subject
    operator: str
    deep: bool = True
    max_rounds: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Plain `def`: the pipeline is blocking (API calls, page fetches); FastAPI runs it in a thread pool.
@app.post("/screen", response_model=Report)
def screen_endpoint(req: ScreenRequest) -> Report:
    return screen(req.subject, operator=req.operator, deep=req.deep, max_rounds=req.max_rounds)


@app.post("/screen.md", response_class=PlainTextResponse)
def screen_markdown(req: ScreenRequest) -> str:
    return screen(req.subject, operator=req.operator, deep=req.deep, max_rounds=req.max_rounds).markdown
