"""Markdown rendering of a screening report (Russian, for the compliance reviewer)."""
from models import Category, Fact, Finding, Report, Status, SubjectType, Verification

LEVEL_RU = {"none": "не выявлено", "low": "низкий", "medium": "средний", "high": "высокий"}
CATEGORY_RU = {
    Category.SANCTIONS: "Санкции", Category.CORRUPTION: "Коррупция", Category.FRAUD: "Мошенничество",
    Category.MONEY_LAUNDERING: "Отмывание / обнал", Category.TAX: "Налоги",
    Category.BANKRUPTCY: "Банкротство", Category.LITIGATION: "Судебные споры",
    Category.ENFORCEMENT: "Исполнительные производства", Category.PROCUREMENT: "Госзакупки (РНУ)",
    Category.CRIMINAL: "Уголовные дела", Category.REGULATORY: "Регуляторы",
    Category.LABOR: "Трудовые нарушения", Category.ENVIRONMENTAL: "Экология", Category.OTHER: "Прочее",
}
STATUS_RU = {
    Status.LISTED: "в реестре", Status.CONVICTED: "приговор", Status.CIVIL_JUDGMENT: "решение суда",
    Status.CHARGED: "обвинение", Status.INVESTIGATION: "расследование", Status.ALLEGATION: "публикации",
    Status.CLOSED: "прекращено / погашено", Status.ACQUITTED: "оправдан", Status.UNKNOWN: "статус неизвестен",
}
VERIFICATION_RU = {
    Verification.REGISTRY: "официальный реестр", Verification.VERIFIED: "цитата подтверждена",
    Verification.VERIFIED_ARCHIVE: "цитата подтверждена по архивной копии",
    Verification.QUOTE_NOT_FOUND: "⚠ цитата не найдена на странице", Verification.UNREACHABLE: "страница недоступна",
    Verification.NOT_CHECKED: "не проверено",
}
TIER_RU = {"registry": "реестр", "official": "госисточник", "major_media": "крупное СМИ", "media": "СМИ",
           "low": "соцсети / форум"}


def plural(n: int, one: str, few: str, many: str) -> str:
    """Russian plural: 1 источник, 2 источника, 5 источников."""
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {one}"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f"{n} {few}"
    return f"{n} {many}"


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _source_line(f: Finding) -> str:
    parts = [f"[{_md(f.source_name or f.url)}]({f.url})"]
    if f.published:
        parts.append(f.published)
    parts.append(TIER_RU.get(f.source_tier, f.source_tier or "?"))
    parts.append(VERIFICATION_RU[f.verification])
    if f.archive_url:
        parts.append(f"[архив]({f.archive_url})")
    parts.append(f"идентификация {f.match_score:.0%}")
    line = "  - " + ", ".join(parts)
    if f.quote:
        quote = _md(f.quote)
        line += f"\n    > {quote[:400]}{'…' if len(quote) > 400 else ''}"
    return line


def _fact_block(i: int, fact: Fact, by_id: dict[str, Finding]) -> str:
    group = [by_id[x] for x in fact.finding_ids if x in by_id]
    confirmed = (plural(fact.corroborations, "независимым источником", "независимыми источниками",
                        "независимыми источниками") if fact.corroborations else "нет подтверждённых источников")
    lines = [f"### {i}. {CATEGORY_RU[fact.category]}: {_md(fact.claim)}"]
    if fact.about:
        lines.append(f"- Касается: **{_md(fact.about)}** ({_md(fact.relation_path) or 'связь с субъектом'})")
    lines += [
        f"- Статус: **{STATUS_RU[fact.status]}**, тяжесть: {fact.severity}, риск факта: {fact.risk:.2f}",
        f"- Подтверждено: {confirmed}; сила доказательств {fact.strength:.2f}, вес давности {fact.recency_weight:.2f}",
        f"- Идентификация: {'; '.join(group[0].match_reasons) if group else '—'}",
        "- Источники:",
    ]
    lines += [_source_line(f) for f in sorted(group, key=lambda f: -f.risk)]
    return "\n".join(lines)


def _finding_block(i: int, f: Finding) -> str:
    lines = [
        f"### {i}. {CATEGORY_RU[f.category]}: {_md(f.claim)}",
        f"- Статус: **{STATUS_RU[f.status]}**, тяжесть: {f.severity}",
        f"- Идентификация: {f.match_score:.0%} ({'; '.join(f.match_reasons)})",
    ]
    if f.subject_mentioned_as:
        lines.append(f"- В источнике: «{_md(f.subject_mentioned_as)}»")
    lines.append(_source_line(f))
    return "\n".join(lines)


def render(report: Report) -> str:
    s = report.subject
    kind = "Юрлицо" if s.type == SubjectType.COMPANY else "Физлицо"
    ident = ""
    if s.identifier:
        # A person's ИИН is masked in the report; a company's БИН is public.
        ident = s.identifier if s.type == SubjectType.COMPANY else f"********{s.identifier[-4:]}"
    out = [
        f"# Проверка контрагента: {s.name}",
        "",
        f"| | |\n|---|---|\n| Тип | {kind} |\n| БИН/ИИН | {ident or '—'} |\n| Цель проверки | {_md(s.purpose)} |"
        f"\n| Дата | {report.created_at:%Y-%m-%d %H:%M} UTC |\n| ID проверки | {report.request_id} |",
        "",
        f"## Итог: риск **{LEVEL_RU[report.risk_level]}** ({report.risk_score:.2f})",
        "",
    ]
    by_id = {f.id: f for f in report.findings}
    own = [x for x in report.facts if not x.about]
    affiliated = [x for x in report.facts if x.about]
    if report.facts:
        counts: dict[Category, int] = {}
        for x in own:
            counts[x.category] = counts.get(x.category, 0) + 1
        if counts:
            out.append("| Категория | Фактов |\n|---|---|")
            out += [f"| {CATEGORY_RU[c]} | {n} |" for c, n in sorted(counts.items(), key=lambda kv: -kv[1])]
            out.append("")
        if affiliated:
            out += [f"Негатив по связанным лицам и компаниям: {plural(len(affiliated), 'факт', 'факта', 'фактов')}, "
                    f"учитывается в оценке с пониженным весом.", ""]
    if own:
        out += ["## Факты о субъекте", ""]
        out += [_fact_block(i, x, by_id) + "\n" for i, x in enumerate(own, 1)]
    if affiliated:
        out += ["## Негатив по связанным лицам и компаниям", ""]
        out += [_fact_block(i, x, by_id) + "\n" for i, x in enumerate(affiliated, 1)]
    if not report.facts:
        out += ["Негативной информации, относящейся к субъекту, в проверенных источниках не найдено. "
                "Это не гарантирует её отсутствие: см. список источников и ограничения ниже.", ""]

    if report.graph.get("mermaid"):
        out += ["## Граф связей", "", "Красным отмечены сущности с негативными фактами.", "",
                "```mermaid", report.graph["mermaid"], "```", ""]

    if report.possible_namesakes:
        out += ["## Возможные однофамильцы / тёзки (требуют ручной проверки)", "",
                "Эти находки негативные, но связь с субъектом не подтверждена. В оценку риска они не входят.", ""]
        out += [_finding_block(i, f) + "\n" for i, f in enumerate(report.possible_namesakes, 1)]

    out += ["## Проверенные источники", "", "| Источник | Статус | Совпадений | Примечание |", "|---|---|---|---|"]
    out += [f"| {_md(c.name)} | {c.status} | {c.hits} | {_md(c.note)} |" for c in report.sources]
    out += ["", f"Раундов глубокого поиска: {report.rounds}, поисковых запросов: {len(report.queries)}."]
    if report.queries:
        out += ["", "<details><summary>Поисковые запросы</summary>", ""]
        out += [f"- {_md(q)}" for q in report.queries]
        out += ["", "</details>"]
    out += ["", "## Ограничения", "",
            "- Отчёт собран автоматически из открытых источников и требует проверки аналитиком.",
            "- Статус «публикации» и «расследование» означает подозрение, а не установленную вину.",
            "- Вес источника: реестр и госисточник 1.0, крупное СМИ 0.9, прочие СМИ 0.7, соцсети и форумы 0.4; "
            "старые публикации весят меньше (полураспад 5 лет).",
            "- Страницы за пейволлом, капчей или удалённые без архивной копии не проверялись; судебные акты РК "
            "обезличены, поэтому связь с субъектом часто устанавливается только косвенно.",
            ""]
    return "\n".join(out)
