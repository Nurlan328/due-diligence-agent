"""Markdown rendering of a screening report (Russian, for the compliance reviewer)."""
from models import Category, Finding, Report, Status, SubjectType, Verification

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
    Verification.QUOTE_NOT_FOUND: "⚠ цитата не найдена на странице", Verification.UNREACHABLE: "страница недоступна",
    Verification.NOT_CHECKED: "не проверено",
}


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _finding_block(i: int, f: Finding) -> str:
    lines = [
        f"### {i}. {CATEGORY_RU[f.category]}: {_md(f.claim)}",
        f"- Статус: **{STATUS_RU[f.status]}**, тяжесть: {f.severity}, вклад в риск: {f.risk:.2f}",
        f"- Источник: [{_md(f.source_name or f.url)}]({f.url})" + (f", {f.published}" if f.published else ""),
        f"- Проверка: {VERIFICATION_RU[f.verification]}",
        f"- Идентификация: {f.match_score:.0%} ({'; '.join(f.match_reasons)})",
    ]
    if f.subject_mentioned_as:
        lines.append(f"- В источнике: «{_md(f.subject_mentioned_as)}»")
    if f.quote:
        lines.append(f"\n> {_md(f.quote)}")
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
    if report.findings:
        counts: dict[Category, int] = {}
        for f in report.findings:
            counts[f.category] = counts.get(f.category, 0) + 1
        out.append("| Категория | Находок |\n|---|---|")
        out += [f"| {CATEGORY_RU[c]} | {n} |" for c, n in sorted(counts.items(), key=lambda kv: -kv[1])]
        out += ["", "## Находки", ""]
        out += [_finding_block(i, f) + "\n" for i, f in enumerate(report.findings, 1)]
    else:
        out += ["Негативной информации, относящейся к субъекту, в проверенных источниках не найдено. "
                "Это не гарантирует её отсутствие: см. список источников и ограничения ниже.", ""]

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
            "- Страницы за пейволлом, капчей или удалённые не проверялись; судебные акты РК обезличены, "
            "поэтому связь с субъектом часто устанавливается только косвенно.",
            ""]
    return "\n".join(out)
