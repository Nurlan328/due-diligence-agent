"""Deep search agent: Claude researches the subject on the open web in rounds.

One round is one agent conversation. Claude runs web searches and opens pages
(Anthropic server tools `web_search` / `web_fetch`) and reports every adverse
fact through our own `record_findings` tool, with a verbatim quote, the source
URL, the category and the procedural status. The quote is later checked against
the fetched page (verify.py), so a hallucinated quote is caught, not reported.

Leads build an entity graph (graph.py): each affiliated company, director,
founder or case is a node linked to the entity it was found next to. Round N+1
starts a fresh conversation that explores the graph's frontier: the
best-supported unexplored nodes within MAX_DEPTH hops. Findings about an
affiliate are recorded with `about` set, so they are reported (and scored at a
discount) as affiliate risk, not as the subject's own. Rounds stop when the
budget is spent, the frontier is empty or a round finds nothing new.

Privacy: a person's ИИН and full birth date are never sent to Claude or to the
search engine. They stay local and are only used to confirm identity in fetched
pages (matching.py). The model gets the name, city, birth year and related names.
"""
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

from config import settings
from graph import EntityGraph, Node
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
- about: empty if the fact concerns the subject itself; otherwise the exact name \
of the affiliated entity it concerns (one you were asked to check, or a lead).
- event_key: a short label of the underlying event, 3-8 words, e.g. "уголовное \
дело о хищении топлива 2024". Use the SAME label for every article about the same \
event, so the articles can be grouped.
- identity_evidence: the details that tie the source to THIS subject or set it \
apart (БИН, city, age, position, company, related names). Namesakes are common: \
record a fact if it plausibly concerns the subject and say what is uncertain; skip \
it only if the details clearly describe someone else.
- leads: affiliated companies, directors, founders, subsidiaries and case \
numbers worth a follow-up search. Official roles only. relation: what the lead is \
to connected_to (директор, учредитель, дочерняя компания, участник дела, \
контрагент); connected_to: the entity it is linked to, empty for the subject; \
source_url: where the link is stated.

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
                                 "severity", "claim", "quote", "subject_mentioned_as", "identity_evidence",
                                 "about", "event_key"],
                    "properties": {
                        "url": {"type": "string"},
                        "title": {"type": "string"},
                        "published": {"type": "string",
                                      "description": "Publication or decision date, empty if unknown"},
                        "source_name": {"type": "string"},
                        "category": {"type": "string", "enum": [c.value for c in Category]},
                        "status": {"type": "string", "enum": [s.value for s in Status]},
                        "severity": {"type": "string", "enum": [s.value for s in Severity]},
                        "claim": {"type": "string"},
                        "quote": {"type": "string"},
                        "subject_mentioned_as": {"type": "string"},
                        "identity_evidence": {"type": "array", "items": {"type": "string"}},
                        "about": {"type": "string", "description": "Empty = the subject; else the affiliate's name"},
                        "event_key": {"type": "string"},
                    },
                },
            },
            "leads": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "kind", "relation", "connected_to", "source_url", "reason"],
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string", "enum": ["company", "person", "case", "other"]},
                        "relation": {"type": "string"},
                        "connected_to": {"type": "string", "description": "Empty = the subject"},
                        "source_url": {"type": "string"},
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
    about: str = ""
    event_key: str = ""

    def to_finding(self) -> Finding:
        return Finding(source_type="web", **self.model_dump())


class Lead(BaseModel):
    name: str
    kind: Literal["company", "person", "case", "other"]
    relation: str = ""
    connected_to: str = ""
    source_url: str = ""
    reason: str = ""


class RecordInput(BaseModel):
    findings: list[AgentFinding]
    leads: list[Lead]


@dataclass
class Usage:
    """Tokens and server-tool calls across every API request of a screening, and their cost."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    web_search_requests: int = 0
    web_fetch_requests: int = 0

    def add(self, usage: Any) -> None:
        """Add one response's `usage` (missing fields count as 0: fallbacks and stubs may omit them)."""
        self.requests += 1
        if usage is None:
            return
        for name in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            setattr(self, name, getattr(self, name) + (getattr(usage, name, 0) or 0))
        tools = getattr(usage, "server_tool_use", None)
        if tools is not None:
            self.web_search_requests += getattr(tools, "web_search_requests", 0) or 0
            self.web_fetch_requests += getattr(tools, "web_fetch_requests", 0) or 0

    def merge(self, other: "Usage") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def cost_usd(self) -> float:
        s = settings
        tokens = (self.input_tokens * s.PRICE_INPUT_PER_MTOK + self.output_tokens * s.PRICE_OUTPUT_PER_MTOK
                  + self.cache_read_input_tokens * s.PRICE_CACHE_READ_PER_MTOK
                  + self.cache_creation_input_tokens * s.PRICE_CACHE_WRITE_PER_MTOK) / 1_000_000
        return round(tokens + self.web_search_requests * s.PRICE_WEB_SEARCH_PER_1K / 1000, 4)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cost_usd": self.cost_usd()}


@dataclass
class ResearchResult:
    findings: list[Finding] = field(default_factory=list)
    leads: list[Lead] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rounds: int = 0
    graph: EntityGraph | None = None
    usage: Usage = field(default_factory=Usage)


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


def _round_prompt(subject: Subject, round_no: int, known: list[Finding], pivots: list[Node],
                  graph: EntityGraph | None = None) -> str:
    parts = [f"Субъект проверки:\n{subject_brief(subject)}"]
    budget = f"Бюджет: до {settings.SEARCHES_PER_ROUND} поисков и {settings.FETCHES_PER_ROUND} открытых страниц."
    if round_no == 1:
        parts.append(f"Раунд 1. Проведи широкий поиск по всем категориям риска. {budget}")
    else:
        known_lines = "\n".join(f"- {f.url} — {f.claim}" for f in known) or "- (нет)"
        pivot_lines = "\n".join(
            f"- [{n.kind}] {n.name} ({graph.path(n.id) if graph else ''})" for n in pivots)
        parts.append(f"Раунд {round_no}. Уже записано (не дублируй):\n{known_lines}")
        parts.append(f"Проверь негатив по этим связанным сущностям (в находках укажи about = их имя, "
                     f"если факт касается их, а не субъекта):\n{pivot_lines}\n{budget}")
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
        out.usage.add(getattr(response, "usage", None))
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
    graph = EntityGraph(subject)
    total = ResearchResult(graph=graph)
    seen_ids: set[str] = set()
    pivots: list[Node] = []

    for round_no in range(1, max_rounds + 1):
        if round_no > 1:
            pivots = graph.frontier(settings.PIVOTS_PER_ROUND, settings.MAX_DEPTH)
            if not pivots:
                break
        prompt = _round_prompt(subject, round_no, total.findings, pivots, graph)
        graph.mark_explored([graph.subject_id] + [n.id for n in pivots])
        result = run_round(prompt, model_fn)
        total.rounds = round_no
        total.usage.merge(result.usage)
        total.queries += result.queries
        total.notes += [f"раунд {round_no}: {n}" for n in result.notes]

        for lead in result.leads:
            graph.add(lead.name, lead.kind, lead.relation or "связь", lead.connected_to, lead.source_url)
        new = [f for f in result.findings if f.id not in seen_ids]
        for f in new:
            _attach_to_graph(f, graph)
        seen_ids.update(f.id for f in new)
        total.findings += new
        total.leads += result.leads
        if round_no > 1 and not new:
            break
    return total


def _attach_to_graph(f: Finding, graph: EntityGraph) -> None:
    """Resolve `about` to a graph node: the subject (cleared) or an affiliate (with its path)."""
    if not f.about.strip():
        f.about = ""
        return
    nid = graph.find(f.about)
    if nid == graph.subject_id:
        f.about = ""
        return
    if nid is None:  # an affiliate the agent did not report as a lead: link it to the subject
        nid = graph.add(f.about, "other", "упомянут в источнике", "", f.url)
    graph.nodes[nid].sources.add(f.url)
    f.about = graph.nodes[nid].name
    f.relation_path = graph.path(nid)
