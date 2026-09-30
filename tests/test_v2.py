"""Matcher v2 (morphology, evidence against), source reliability, facts, Wayback, entity graph."""
from datetime import date

import httpx
from conftest import ScriptedModel, block, message

import geo
import reliability
from deepsearch import deep_search
from graph import EntityGraph
from matching import features, score
from models import Finding, Subject, Verification
from names import patronymic_root, person_similarity, token_match
from scoring import assess, group_facts
from verify import archive_snapshot, verify_findings


def web(url="https://a.kz/1", quote="q", **kw):
    base = dict(source_type="web", source_name="src", url=url, category="fraud", status="charged",
                severity="high", claim="Директор обвиняется в хищении", quote=quote, match_score=1.0,
                verification="verified")
    return Finding(**{**base, **kw})


# --- names / geo ---------------------------------------------------------------------------------

def test_inflected_surnames_and_names():
    assert token_match("Оспановой", "Оспанова")
    assert token_match("Толеубаеву", "Толеубаев")
    assert token_match("Дархана", "Дархан")
    assert not token_match("Ахматов", "Ахметов")  # one letter off: another family


def test_patronymic_forms():
    assert patronymic_root("Болатович") == patronymic_root("Болатұлы") == patronymic_root("Болатовичу")
    assert patronymic_root("Ерлановна") == patronymic_root("Ерланқызы")
    assert person_similarity("Касымова Жанар Ерлановна", "Касымова Жанар Ерланқызы")[0] == 1.0


def test_latin_spelling_variants():
    assert person_similarity("Ахметов Ерлан Болатович", "Yerlan Akhmetov")[0] == 0.8
    assert token_match("Vasilyok", "Василёк", 85)


def test_cities_in_any_form():
    assert geo.mentions_city("задержан в Астане", "Астана")
    assert geo.mentions_city("карагандинский предприниматель", "Караганда")
    assert geo.mentions_city("Қостанай облысындағы", "Костанай")
    assert geo.mentions_city("a trader from Almaty", "Алматы")
    assert geo.cities_in("В Шымкенте и Таразе") == {"Шымкент", "Тараз"}
    assert geo.canonical("Нур-Султан") == "Астана"


# --- matcher v2 ----------------------------------------------------------------------------------

def person():
    return Subject(type="person", name="Толеубаев Самат Муратович", identifier="830709300456",
                   birth_date="1983-07-09", city="Павлодар", related=["ТОО Шиповник Логистик"],
                   purpose="KYC при открытии счёта")


def test_full_namesake_from_another_city_is_capped():
    f = web(subject_mentioned_as="Толеубаев Самат Муратович")
    s, reasons = score(features(person(), f, "В Шымкенте задержан Толеубаев Самат Муратович."), person())
    assert s < 0.5 and "в источнике другой город" in reasons


def test_other_iin_and_birth_year_and_patronymic():
    p = person()
    f = web(subject_mentioned_as="Толеубаев Самат Муратович")
    assert score(features(p, f, "Толеубаев Самат Муратович, ИИН 610101300111, должник."), p)[0] <= 0.1
    assert score(features(p, f, "Толеубаев Самат Муратович, 1961 года рождения, осуждён."), p)[0] <= 0.3
    g = web(subject_mentioned_as="Толеубаев Самат Ерланович")
    assert score(features(p, g, "Толеубаев Самат Ерланович осуждён."), p)[0] <= 0.3


def test_age_without_date_only_rules_out_too_young_subjects():
    p = person()  # 1983
    f = web(subject_mentioned_as="Самат Толеубаев")
    # "29 years old" with no date: could be an old article about the subject -> no conflict
    feat = features(p, f, "В Павлодаре Самат Толеубаев (ему 29 лет) задержан.")
    assert not feat.birth_conflict
    # dated 2025 and 27 years old -> born ~1998 -> conflict
    g = web(subject_mentioned_as="Самат Толеубаев", published="2025-03-10")
    assert features(p, g, "Самату Толеубаеву 27 лет, мужчине 27 лет").birth_conflict


def test_own_id_overrides_evidence_against():
    p = person()
    f = web(subject_mentioned_as="Толеубаев С.М.")
    text = "Толеубаев С.М., ИИН 830709300456, г. Шымкент; соучастник ИИН 790211300158."
    assert score(features(p, f, text), p)[0] >= 0.97


def test_evidence_against_only_near_the_name():
    p = person()
    f = web(subject_mentioned_as="Самат Толеубаев")
    far = "В Шымкенте прошёл форум. " + "Обсуждали экономику и логистику региона. " * 12
    s, _ = score(features(p, f, far + "В Павлодаре задержан Самат Толеубаев."), p)
    assert s >= 0.8


# --- reliability / recency / facts ---------------------------------------------------------------

def test_source_tiers():
    assert reliability.source_tier(web(url="https://office.sud.kz/x")) == "official"
    assert reliability.source_tier(web(url="https://kgd.gov.kz/ru/x")) == "official"
    assert reliability.source_tier(web(url="https://www.tengrinews.kz/x")) == "major_media"
    assert reliability.source_tier(web(url="https://t.me/channel/1")) == "low"
    assert reliability.source_tier(web(url="https://some-blog.kz/x")) == "media"
    assert reliability.domain("https://news.tengrinews.kz/a") == "tengrinews.kz"
    assert reliability.domain("https://kgd.gov.kz/a") == "kgd.gov.kz"


def test_recency_decay():
    today = date(2026, 9, 30)
    assert reliability.recency_weight(web(published="2025-01-01"), today) == 1.0
    assert reliability.recency_weight(web(published="2019"), today) == 0.5
    assert reliability.recency_weight(web(published="2001"), today) == 0.3  # floor
    assert reliability.recency_weight(web(published=""), today) == 0.8
    assert reliability.recency_weight(web(published="2001", status="listed"), today) == 1.0


def test_articles_group_into_one_fact_by_event_key():
    a = web(url="https://tengrinews.kz/1", event_key="дело о хищении топлива 2024", claim="Задержан директор")
    b = web(url="https://zakon.kz/2", event_key="Дело о хищении топлива (2024)", claim="Суд арестовал имущество")
    c = web(url="https://orda.kz/3", event_key="взятка в акимате", claim="Совсем другое дело")
    groups = group_facts([a, b, c])
    assert sorted(len(g) for g in groups) == [1, 2]


def test_case_number_links_articles():
    a = web(url="https://a.kz/1", claim="Возбуждено дело 1-234/2024", event_key="")
    b = web(url="https://b.kz/2", claim="Суд рассмотрит иск", quote="по делу 1-234/2024", event_key="")
    assert len(group_facts([a, b])) == 1


def test_corroboration_raises_fact_strength_but_not_same_domain():
    one = [web(url="https://some.kz/1", event_key="e")]
    same_domain = [web(url=f"https://some.kz/{i}", event_key="e") for i in range(3)]
    three = [web(url=f"https://site{i}.kz/1", event_key="e") for i in range(3)]
    s1, _, f1 = assess(one)
    s2, _, f2 = assess(same_domain)
    s3, _, f3 = assess(three)
    assert s1 == s2 and f2[0].corroborations == 1
    assert s3 > s1 and f3[0].corroborations == 3


def test_fabricated_quote_does_not_count_as_confirmation():
    fs = [web(url="https://a.kz/1", event_key="e"), web(url="https://b.kz/1", event_key="e",
                                                         verification="quote_not_found")]
    _, _, facts = assess(fs)
    assert facts[0].corroborations == 1 and len(facts[0].sources) == 2


def test_affiliate_fact_is_discounted():
    own = assess([web(event_key="e")])[0]
    aff = assess([web(event_key="e", about="Ахметов Е.Б.")])[0]
    assert aff < own


# --- Wayback -------------------------------------------------------------------------------------

def wayback_http(live: dict[str, tuple[int, str]], snapshot_text: str | None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith("https://archive.org/wayback/available"):
            if snapshot_text is None:
                return httpx.Response(200, json={"archived_snapshots": {}})
            return httpx.Response(200, json={"archived_snapshots": {"closest": {
                "available": True, "timestamp": "20240102030405",
                "url": "http://web.archive.org/web/20240102030405/x"}}})
        if url.startswith("https://web.archive.org/web/20240102030405id_/"):
            return httpx.Response(200, text=f"<p>{snapshot_text}</p>", headers={"content-type": "text/html"})
        if url in live:
            code, text = live[url]
            return httpx.Response(code, text=text, headers={"content-type": "text/html"})
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_archive_snapshot_url():
    http = wayback_http({}, "text")
    assert archive_snapshot("https://a.kz/1", http) == "https://web.archive.org/web/20240102030405id_/https://a.kz/1"
    assert archive_snapshot("https://a.kz/1", wayback_http({}, None)) is None


def test_removed_page_verified_from_archive():
    f = web(url="https://a.kz/removed", quote="директор задержан по делу о хищении")
    pages = verify_findings([f], http=wayback_http({}, "Вчера директор задержан по делу о хищении."), archive=True)
    assert f.verification == Verification.VERIFIED_ARCHIVE
    assert f.archive_url.startswith("https://web.archive.org/")
    assert "задержан" in pages[f.url]  # the matcher reads the archived copy


def test_edited_page_quote_only_in_archive():
    f = web(url="https://a.kz/edited", quote="директор задержан по делу о хищении")
    live = {"https://a.kz/edited": (200, "<p>Статья обновлена, текст удалён по требованию.</p>")}
    verify_findings([f], http=wayback_http(live, "директор задержан по делу о хищении"), archive=True)
    assert f.verification == Verification.VERIFIED_ARCHIVE


def test_archive_failure_is_soft():
    f = web(url="https://a.kz/gone", quote="цитата")
    verify_findings([f], http=wayback_http({}, None), archive=True)
    assert f.verification == Verification.UNREACHABLE


# --- entity graph --------------------------------------------------------------------------------

def company():
    return Subject(type="company", name="ТОО «Ромашка Строй»", identifier="123456789012",
                   purpose="Проверка поставщика перед договором")


def test_graph_links_frontier_and_path():
    g = EntityGraph(company())
    g.add("Ахметов Ержан Болатович", "person", "директор", "", "https://a.kz/1")
    g.add("ТОО Василёк", "company", "учредитель", "Ахметов Ержан Болатович", "https://b.kz/1")
    g.add("ТОО Василёк", "company", "учредитель", "Ахметов Ержан Болатович", "https://c.kz/1")
    assert g.find("ромашка строй") == g.subject_id  # the subject is not a lead
    front = g.frontier(5, max_depth=2)
    assert front[0].name == "ТОО Василёк"  # two sources beat one
    assert g.path(front[0].id) == "Субъект → директор: Ахметов Ержан Болатович → учредитель: ТОО Василёк"
    assert g.frontier(5, max_depth=1)[0].name == "Ахметов Ержан Болатович"
    assert "flowchart LR" in g.to_mermaid()


def record(tool_id, findings=(), leads=()):
    return block("tool_use", id=tool_id, name="record_findings",
                 input={"findings": list(findings), "leads": list(leads)})


def agent_finding(url, about="", event_key="e"):
    return {"url": url, "title": "t", "published": "", "source_name": "s", "category": "fraud",
            "status": "charged", "severity": "high", "claim": "c", "quote": "q", "subject_mentioned_as": "m",
            "identity_evidence": [], "about": about, "event_key": event_key}


def test_deep_search_explores_graph_and_tags_affiliate_findings():
    lead = {"name": "Ахметов Ержан Болатович", "kind": "person", "relation": "директор", "connected_to": "",
            "source_url": "https://a.kz/1", "reason": "директор субъекта"}
    model = ScriptedModel(
        message("tool_use", record("t1", [agent_finding("https://a.kz/1")], [lead])), message("end_turn"),
        message("tool_use", record("t2", [agent_finding("https://b.kz/2", about="Ахметов Е.Б.")])),
        message("end_turn"),
    )
    out = deep_search(company(), model_fn=model, max_rounds=3)
    assert out.rounds == 2
    round2 = model.requests[2]["messages"][0]["content"]
    assert "Ахметов Ержан Болатович" in round2 and "директор" in round2
    own, aff = out.findings
    assert own.about == ""
    # the source's short form resolves to the known director node
    assert aff.about == "Ахметов Ержан Болатович"
    assert aff.relation_path == "Субъект → директор: Ахметов Ержан Болатович"


def test_unknown_affiliate_becomes_a_pivot():
    model = ScriptedModel(
        message("tool_use", record("t1", [agent_finding("https://a.kz/1", about="ТОО Лютик")])), message("end_turn"),
        message("end_turn"),
    )
    out = deep_search(company(), model_fn=model, max_rounds=3)
    assert out.findings[0].relation_path == "Субъект → упомянут в источнике: ТОО Лютик"
    assert out.rounds == 2 and "ТОО Лютик" in model.requests[2]["messages"][0]["content"]


def test_short_company_name_is_not_the_company_subject():
    g = EntityGraph(company())
    assert g.find("ТОО Ромашка") is None
    g2 = EntityGraph(Subject(type="person", name="Толеубаев Самат Муратович", purpose="KYC при открытии счёта"))
    assert g2.find("Толеубаев С.М.") == g2.subject_id


def test_russian_plural():
    from report import plural
    assert [plural(n, "факт", "факта", "фактов") for n in (1, 2, 5, 11, 21, 22, 112)] == \
        ["1 факт", "2 факта", "5 фактов", "11 фактов", "21 факт", "22 факта", "112 фактов"]


def test_archive_down_trips_circuit_breaker():
    calls = {"archive": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "archive.org":
            calls["archive"] += 1
            raise httpx.ConnectError("reset", request=request)
        return httpx.Response(404)

    fs = [web(url=f"https://a.kz/{i}", quote="q") for i in range(5)]
    verify_findings(fs, http=httpx.Client(transport=httpx.MockTransport(handler)), archive=True)
    assert calls["archive"] == 1
    assert all(f.verification == Verification.UNREACHABLE for f in fs)


# --- matcher v3 ----------------------------------------------------------------------------------

def test_id_belongs_to_the_nearest_entity():
    from matching import subject_ids
    from names import fold
    p = person()
    text = "По делу проходят Толеубаев С.М., Касымов Е.Н. (ИИН 790211300158) и Садыкова Ж.А. (ИИН 880305400213)."
    assert subject_ids(p, fold(text)) == set()
    assert subject_ids(p, fold("Толеубаев С.М. (ИИН 610101300111), должник")) == {"610101300111"}
    assert subject_ids(p, fold("ИИН 830709300456 — Толеубаев С.М.")) == {"830709300456"}


def test_company_bin_attribution():
    from matching import subject_ids
    from names import fold
    c = Subject(type="company", name="ТОО «Тюльпан Энерго»", purpose="Проверка поставщика перед договором")
    text = "Подрядчик ТОО «Лютик Сервис» (БИН 180540099111) и заказчик ТОО «Тюльпан Энерго» проверяются."
    assert subject_ids(c, fold(text)) == set()
    assert subject_ids(c, fold("ТОО «Тюльпан Энерго» (БИН 200140055666), Павлодар")) == {"200140055666"}


def test_given_name_conflict():
    from names import given_name_conflict
    full = "Байжанов Ерболат Кайратович"
    assert given_name_conflict(full, "Байжанов Нурлан")
    assert given_name_conflict(full, "Nurlan Baizhanov")
    assert not given_name_conflict(full, "Байжанов Е.К.")  # initials are not a conflict
    assert not given_name_conflict(full, "Байжанов")
    assert not given_name_conflict(full, "директор Байжанов Ерболат")
    assert not given_name_conflict(full, "Сериков Нурлан")  # another surname: not this feature's job


def test_gh_romanisation():
    assert token_match("Ghalym", "Ғалым") and token_match("Galym", "Ғалым")
