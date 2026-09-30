"""Deep search agent: Claude researches the subject on the open web in rounds.

One round is one agent conversation. Claude runs web searches and opens pages
(Anthropic server tools `web_search` / `web_fetch`) and reports every adverse
fact through our own `record_findings` tool, with a verbatim quote, the source
URL, the category and the procedural status. The quote is later checked against
the fetched page (verify.py), so a hallucinated quote is caught, not reported.

Round N+1 starts a fresh conversation that follows the leads found in round N
(affiliated companies, directors from official records, case numbers). Rounds
stop when the budget is spent or a round finds nothing new.

Privacy: a person's ИИН and full birth date are never sent to Claude or to the
search engine. They stay local and are only used to confirm identity in fetched
pages (matching.py). The model gets the name, city, birth year and related names.
"""
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from config import settings
from models import Category, Finding, Severity, Status, Subject, SubjectType
from names import search_variants

SYSTEM_PROMPT = """\
You are an adverse-media analyst doing KYC / counterparty due diligence for a \
compliance team in Kazakhstan. Your job: find publicly reported negative \
information about ONE subject (a company or a person) and record each fact with \
its source.

What counts as adverse: sanctions and terrorist lists; corruption, bribery, \
embezzlement of budget funds; fraud, pyramid schemes, sham companies \
(лжепредпринимательство); money laundering, cash-out schemes; tax debts, tax \
evasion, "inactive" or "unreliable" taxpayer status; bankruptcy, rehabilitation, \
court-ordered liquidation; significant lawsuits; enforcement proceedings, travel \
bans for debtors; the register of unscrupulous public-procurement participants \
(РНУ); criminal cases; regulator fines and licence revocations; unpaid wages and \
workplace fatalities; environmental violations.

How to search:
- Search in Russian, Kazakh and English, with every spelling variant given. \
Combine the name with risk words (уголовное дело, мошенничество, взятка, \
банкротство, суд, долг, недобросовестный, задержан, расследование, штраф, \
санкции / fraud, bribery, sanctions, lawsuit).
- Useful Kazakhstani sources: sud.kz (court acts), kgd.gov.kz, goszakup.gov.kz, \
adilet.zan.kz, afmrk.gov.kz, egov.kz, and news media (tengrinews.kz, zakon.kz, \
informburo.kz, kursiv.media, vlast.kz, orda.kz, azattyq.org, forbes.kz, \
kapital.kz, inbusiness.kz). International: OCCRP, ICIJ, Reuters.
- Open a page with web_fetch before recording a fact from it, so the quote is \
exact.

How to record (tool record_findings, call it as often as you like):
- One finding per distinct fact per source. Several articles about the same \
case are separate findings (they corroborate each other).
- quote: copy one to three sentences VERBATIM from the page, in the original \
language. It is machine-checked against the page; a paraphrase counts as a \
fabrication. Never invent a URL.
- claim: one sentence in Russian stating the fact.
- status: listed (official register entry), convicted (final verdict), \
civil_judgment, charged (case in court), investigation (pre-trial, searches, \
detention), allegation (press reports, complaints), closed (case dropped, debt \
paid, removed from register), acquitted, unknown. Do not upgrade an allegation.
- severity: critical (sanctions, terrorism, conviction for serious crime), high \
(criminal case, fraud, corruption, РНУ, bankruptcy), medium (large lawsuits, tax \
debt, regulator fine), low (minor disputes, old closed matters).
- subject_mentioned_as: the name exactly as the source writes it.
- identity_evidence: the details that tie the source to THIS subject or set it \
apart (БИН, city, age, position, company, related names). Namesakes are common: \
record a fact if it plausibly concerns the subject and say what is uncertain; skip \
it only if the details clearly describe someone else.
- leads: affiliated companies, directors, founders and case numbers worth a \
follow-up search. Official roles only.

Do not collect private life: health, family members, religion, ethnicity, \
sexual life, home address, phone numbers, photos. Skip it even when you see it.

When the budget is spent or nothing new turns up, stop and reply with a two-line \
summary of what you covered."""

RECORD_TOOL: dict[str, Any] = {
    "name": "record_findings",
    "description": (
        "Record adverse findings about the subject, each with a verbatim quote and its source URL, "
        "plus leads for follow-up searches. Call it whenever you have confirmed facts."
    ),
    "strict": True,
    "eager_input_streaming": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["findings", "leads"],
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["url", "title", "published", "source_name", "category", "status",
                                 "severity", "claim", "quote", "subject_mentioned_as", "identity_evidence"],
                    "properties": {
                        "url": {"type": "string"},
                        "title": {"type": "string"},
                        "published": {"type": "string", "description": "Publication or decision date, empty if unknown"},
                        "source_name": {"type": "string"},
                        "category": {"type": "string", "enum": [c.value for c in Category]},
                        "status": {"type": "string", "enum": [s.value for s in Status]},
                        "severity": {"type": "string", "enum": [s.value for s in Severity]},
                        "claim": {"type": "string"},
                        "quote": {"type": "string"},
                        "subject_mentioned_as": {"type": "string"},
                        "identity_evidence": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "leads": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "kind", "reason"],
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string", "enum": ["company", "person", "case", "other"]},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    },
}


class AgentFinding(BaseModel):
    url: str
    title: str
    published: str
    source_name: str
    category: Category
    status: Status
    severity: Severity
    claim: str
    quote: str
    subject_mentioned_as: str
    identity_evidence: list[str]

    def to_finding(self) -> Finding:
        return Finding(source_type="web", **self.model_dump())


class Lead(BaseModel):
    name: str
    kind: Literal["company", "person", "case", "other"]
    reason: str


class RecordInput(BaseModel):
    findings: list[AgentFinding]
    leads: list[Lead]


@dataclass
class ResearchResult:
    findings: list[Finding] = field(default_factory=list)
    leads: list[Lead] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rounds: int = 0


# (request kwargs) -> final Message. Injected so tests run without the API.
ModelFn = Callable[..., Any]

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    # Lazy: importing this module must not require an API key.
    global _client
    if _client is None:
        _client = Anthropic(api_key=settings.ANTHROPIC_API_KEY or None)
    return _client


def claude_call(**kwargs) -> Any:
    """One Messages API call, streamed (long server-tool turns would hit HTTP timeouts)."""
    if settings.USE_FALLBACKS:
        kwargs = {**kwargs, "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    with _get_client().beta.messages.stream(**kwargs) as stream:
        return stream.get_final_message()


def subject_brief(subject: Subject) -> str:
    kind = "Компания (юрлицо)" if subject.type == SubjectType.COMPANY else "Физическое лицо"
    lines = [f"Тип: {kind}", f"Имя / название: {subject.name}",
             f"Варианты написания: {'; '.join(search_variants(subject))}"]
    if subject.type == SubjectType.COMPANY and subject.identifier:
        lines.append(f"БИН: {subject.identifier}")  # a company's БИН is public
    if subject.birth_date:
        lines.append(f"Год рождения: {subject.birth_date.year}")
    if subject.city:
        lines.append(f"Город / регион: {subject.city}")
    if subject.related:
        lines.append(f"Связанные лица и компании: {'; '.join(subject.related)}")
    return "\n".join(lines)


def _round_prompt(subject: Subject, round_no: int, known: list[Finding], leads: list[Lead]) -> str:
    parts = [f"Субъект проверки:\n{subject_brief(subject)}"]
    if round_no == 1:
        parts.append(f"Раунд 1. Проведи широкий поиск по всем категориям риска. "
                     f"Бюджет: до {settings.SEARCHES_PER_ROUND} поисков и {settings.FETCHES_PER_ROUND} открытых страниц.")
    else:
        known_lines = "\n".join(f"- {f.url} — {f.claim}" for f in known) or "- (нет)"
        lead_lines = "\n".join(f"- [{l.kind}] {l.name}: {l.reason}" for l in leads)
        parts.append(f"Раунд {round_no}. Уже записано (не дублируй):\n{known_lines}")
        parts.append(f"Проверь эти зацепки в связи с субъектом:\n{lead_lines}\n"
                     f"Бюджет: до {settings.SEARCHES_PER_ROUND} поисков и {settings.FETCHES_PER_ROUND} открытых страниц.")
    return "\n\n".join(parts)


def _request(messages: list) -> dict:
    return {
        "model": settings.MODEL,
        "max_tokens": settings.MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "tools": [
            {"type": "web_search_20260209", "name": "web_search", "max_uses": settings.SEARCHES_PER_ROUND},
            {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": settings.FETCHES_PER_ROUND},
            RECORD_TOOL,
        ],
        "output_config": {"effort": settings.EFFORT},
        "cache_control": {"type": "ephemeral"},  # the history is resent every turn
        "messages": messages,
    }


def _queries(content) -> list[str]:
    return [b.input.get("query", "") for b in content
            if getattr(b, "type", "") == "server_tool_use" and getattr(b, "name", "") == "web_search"
            and isinstance(getattr(b, "input", None), dict)]


def run_round(prompt: str, model_fn: ModelFn) -> ResearchResult:
    """One agent conversation: loop until Claude stops calling record_findings."""
    out = ResearchResult()
    messages: list = [{"role": "user", "content": prompt}]
    for _ in range(settings.MAX_AGENT_TURNS):
        response = model_fn(**_request(messages))
        out.queries += _queries(response.content)
        # Append-only history: the assistant turn goes back exactly as received.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            out.notes.append("модель отказалась продолжать раунд (refusal)")
            break
        if response.stop_reason == "pause_turn":
            continue  # server-side tool loop hit its limit; resending resumes it
        truncated = response.stop_reason == "max_tokens"

        tool_uses = [b for b in response.content if getattr(b, "type", "") == "tool_use"]
        if not tool_uses:
            if truncated:
                out.notes.append("ответ обрезан по max_tokens")
            break
        results = []
        for block in tool_uses:
            results.append(_handle_tool(block, out, truncated))
        if truncated:
            out.notes.append("ответ обрезан по max_tokens, последний вызов мог потеряться")
            break
        messages.append({"role": "user", "content": results})
    else:
        out.notes.append(f"достигнут лимит ходов агента ({settings.MAX_AGENT_TURNS})")
    return out


def _handle_tool(block, out: ResearchResult, truncated: bool) -> dict:
    if block.name != RECORD_TOOL["name"]:
        return {"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                "content": f"Unknown tool {block.name}"}
    try:
        # eager_input_streaming: the API does not validate the input, so we do.
        if not isinstance(block.input, dict):
            raise ValueError("input is not a JSON object")
        record = RecordInput.model_validate(block.input)
    except (ValidationError, ValueError) as e:
        if truncated:
            out.notes.append("вызов record_findings обрезан, находки из него потеряны")
        return {"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                "content": f"INVALID_INPUT: {str(e)[:500]}. Fix the arguments and call again."}
    out.findings += [f.to_finding() for f in record.findings]
    out.leads += record.leads
    return {"type": "tool_result", "tool_use_id": block.id,
            "content": f"Записано находок: {len(record.findings)}, зацепок: {len(record.leads)}."}


def deep_search(subject: Subject, model_fn: ModelFn | None = None,
                max_rounds: int | None = None) -> ResearchResult:
    model_fn = model_fn or claude_call
    max_rounds = max_rounds or settings.MAX_ROUNDS
    total = ResearchResult()
    seen_ids: set[str] = set()
    pursued: set[str] = set()
    pending: list[Lead] = []

    for round_no in range(1, max_rounds + 1):
        if round_no > 1 and not pending:
            break
        prompt = _round_prompt(subject, round_no, total.findings, pending)
        pursued.update(l.name.lower() for l in pending)
        result = run_round(prompt, model_fn)
        total.rounds = round_no
        total.queries += result.queries
        total.notes += [f"раунд {round_no}: {n}" for n in result.notes]

        new = [f for f in result.findings if f.id not in seen_ids]
        seen_ids.update(f.id for f in new)
        total.findings += new
        total.leads += result.leads
        pending = list({l.name.lower(): l for l in result.leads if l.name.lower() not in pursued}.values())
        if round_no > 1 and not new:
            break
    return total
