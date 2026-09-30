from conftest import ScriptedModel, block, message

from config import settings
from deepsearch import RECORD_TOOL, deep_search, run_round, subject_brief

FINDING = {
    "url": "https://tengrinews.kz/a/1", "title": "Задержан директор", "published": "2025-02-01",
    "source_name": "tengrinews.kz", "category": "fraud", "status": "investigation", "severity": "high",
    "claim": "Директор задержан по делу о мошенничестве.", "quote": "Директор ТОО задержан",
    "subject_mentioned_as": "ТОО Ромашка Строй", "identity_evidence": ["Алматы"],
}


def record(tool_id, findings=(), leads=()):
    return block("tool_use", id=tool_id, name="record_findings",
                 input={"findings": list(findings), "leads": list(leads)})


def search(query):
    return block("server_tool_use", id="s", name="web_search", input={"query": query})


def test_round_collects_findings_and_queries():
    model = ScriptedModel(
        message("tool_use", search("Ромашка Строй мошенничество"), record("t1", [FINDING])),
        message("end_turn", block("text", text="итог")),
    )
    out = run_round("prompt", model)
    assert len(out.findings) == 1 and out.findings[0].source_type == "web"
    assert out.queries == ["Ромашка Строй мошенничество"]
    # second request carries: user, assistant(tool_use), user(tool_result)
    assert model.requests[1]["n_messages"] == 3
    tool_result = model.requests[1]["messages"][2]["content"][0]
    assert tool_result["tool_use_id"] == "t1" and "is_error" not in tool_result


def test_request_shape():
    model = ScriptedModel(message("end_turn", block("text", text="ничего")))
    run_round("prompt", model)
    req = model.requests[0]
    assert req["model"] == settings.MODEL
    types = [t.get("type", t.get("name")) for t in req["tools"]]
    assert types == ["web_search_20260209", "web_fetch_20260209", "record_findings"]
    assert RECORD_TOOL["strict"] is True
    assert req["output_config"] == {"effort": settings.EFFORT}
    assert "thinking" not in req  # Opus 5.5: thinking is always on, controlled by effort


def test_invalid_tool_input_returns_error_and_agent_can_retry():
    bad = {**FINDING, "category": "gossip"}
    model = ScriptedModel(
        message("tool_use", record("t1", [bad])),
        message("tool_use", record("t2", [FINDING])),
        message("end_turn", block("text", text="ok")),
    )
    out = run_round("prompt", model)
    first_result = model.requests[1]["messages"][2]["content"][0]
    assert first_result["is_error"] is True and "INVALID_INPUT" in first_result["content"]
    assert len(out.findings) == 1


def test_pause_turn_resumes_without_extra_user_message():
    model = ScriptedModel(
        message("pause_turn", search("q1")),
        message("tool_use", record("t1", [FINDING])),
        message("end_turn"),
    )
    out = run_round("prompt", model)
    assert len(out.findings) == 1
    # after pause_turn the history ends with the paused assistant turn
    assert model.requests[1]["n_messages"] == 2
    assert model.requests[1]["messages"][1]["role"] == "assistant"


def test_refusal_stops_round():
    model = ScriptedModel(message("refusal"))
    out = run_round("prompt", model)
    assert out.findings == [] and any("refusal" in n for n in out.notes)


def test_turn_cap(monkeypatch):
    monkeypatch.setattr(settings, "MAX_AGENT_TURNS", 2)
    model = ScriptedModel(*[message("tool_use", record(f"t{i}")) for i in range(5)])
    out = run_round("prompt", model)
    assert len(model.requests) == 2 and any("лимит" in n for n in out.notes)


def test_rounds_follow_leads_and_dedupe(company):
    lead = {"name": "ТОО Василёк", "kind": "company", "reason": "общий учредитель"}
    second = {**FINDING, "url": "https://zakon.kz/b/2", "quote": "Василёк и Ромашка"}
    model = ScriptedModel(
        message("tool_use", record("t1", [FINDING], [lead])), message("end_turn"),
        message("tool_use", record("t2", [FINDING, second])), message("end_turn"),
    )
    out = deep_search(company, model_fn=model, max_rounds=3)
    assert out.rounds == 2  # no new leads after round 2
    assert [f.url for f in out.findings] == ["https://tengrinews.kz/a/1", "https://zakon.kz/b/2"]
    round2_prompt = model.requests[2]["messages"][0]["content"]
    assert "ТОО Василёк" in round2_prompt and "tengrinews.kz/a/1" in round2_prompt


def test_single_round_without_leads(company):
    model = ScriptedModel(message("tool_use", record("t1", [FINDING])), message("end_turn"))
    assert deep_search(company, model_fn=model, max_rounds=3).rounds == 1


def test_person_iin_and_birth_date_never_sent(person):
    brief = subject_brief(person)
    assert person.identifier not in brief
    assert "1980-05-01" not in brief and "1980" in brief


def test_company_bin_is_sent(company):
    assert company.identifier in subject_brief(company)


def test_usage_and_cost_are_accumulated(monkeypatch, company):
    from types import SimpleNamespace

    from deepsearch import Usage

    def usage(inp, out, searches):
        return SimpleNamespace(input_tokens=inp, output_tokens=out, cache_read_input_tokens=1000,
                               cache_creation_input_tokens=0,
                               server_tool_use=SimpleNamespace(web_search_requests=searches, web_fetch_requests=1))

    first = message("tool_use", record("t1", [FINDING]))
    first.usage = usage(10_000, 2_000, 3)
    last = message("end_turn", block("text", text="ok"))
    last.usage = usage(5_000, 500, 0)
    out = deep_search(company, model_fn=ScriptedModel(first, last), max_rounds=1)
    u = out.usage
    assert (u.requests, u.input_tokens, u.output_tokens, u.web_search_requests, u.web_fetch_requests) == \
        (2, 15_000, 2_500, 3, 2)
    for name, price in [("PRICE_INPUT_PER_MTOK", 4.0), ("PRICE_OUTPUT_PER_MTOK", 20.0),
                        ("PRICE_CACHE_READ_PER_MTOK", 0.2), ("PRICE_WEB_SEARCH_PER_1K", 10.0)]:
        monkeypatch.setattr(settings, name, price)
    # 15k*4 + 2.5k*20 + 2k*0.2 per MTok = 0.06 + 0.05 + 0.0004; 3 searches = 0.03
    assert u.cost_usd() == 0.1404
    assert Usage().to_dict()["cost_usd"] == 0
