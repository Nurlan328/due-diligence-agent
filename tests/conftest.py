"""Test fixtures.

Tests never call the Claude API (the model function is a scripted stub) and
never touch the network (page fetches go through httpx.MockTransport), so the
suite needs no keys. The store (audit, match features, jobs) is a temp SQLite file.
"""
from types import SimpleNamespace

import httpx
import pytest

from config import settings
from models import Subject


@pytest.fixture(autouse=True)
def tmp_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "dd.sqlite"))
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path / "data"))
    return tmp_path


@pytest.fixture
def company():
    return Subject(type="company", name="ТОО «Ромашка Строй»", identifier="123456789012", city="Алматы",
                   related=["Ахметов Ержан Болатович"], purpose="Проверка поставщика перед договором")


@pytest.fixture
def person():
    return Subject(type="person", name="Ахметов Ержан Болатович", identifier="800501300123",
                   birth_date="1980-05-01", city="Алматы", related=["ТОО Ромашка Строй"],
                   purpose="KYC при открытии счёта")


def block(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


def message(stop_reason, *content):
    return SimpleNamespace(stop_reason=stop_reason, content=list(content))


class ScriptedModel:
    """Stands in for claude_call: returns the scripted responses in order, records requests."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, **kwargs):
        self.requests.append(kwargs)
        # Snapshot the history length: the list is mutated after the call.
        self.requests[-1] = {**kwargs, "n_messages": len(kwargs["messages"])}
        if not self.responses:
            return message("end_turn", block("text", text="done"))
        return self.responses.pop(0)


def mock_http(pages: dict[str, str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url in pages:
            return httpx.Response(200, text=pages[url], headers={"content-type": "text/html; charset=utf-8"})
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))
