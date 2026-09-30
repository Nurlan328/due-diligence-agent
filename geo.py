"""Kazakhstani places in free text, in any form a news article uses.

Substring matching misses most of them: «в Астане», «Караганде»,
«карагандинский предприниматель», «Қостанай облысында», «Almaty», «в столице».
Each city maps to regexes over folded text (names.fold: lowercase, Kazakh letters
folded) covering the stem (all cases), the adjective, the region and Latin
spellings.
"""
import re
from functools import cache

from names import fold

# canonical name -> alternatives (regex, applied to folded text at a word start)
CITIES: dict[str, list[str]] = {
    "Алматы": [r"алмат\w*", r"almat\w*"],
    "Астана": [r"астан\w*", r"нур султан\w*", r"столиц\w*", r"astana", r"nur sultan"],
    "Шымкент": [r"шымкент\w*", r"shymkent"],
    "Караганда": [r"караганд\w*", r"karagand\w*", r"qaragand\w*"],
    "Актобе": [r"актоб\w*", r"актюб\w*", r"aktobe", r"aqtobe"],
    "Атырау": [r"атырау\w*", r"atyrau"],
    "Павлодар": [r"павлодар\w*", r"pavlodar"],
    "Усть-Каменогорск": [r"усть каменогорск\w*", r"оскемен\w*", r"восточно казахстанск\w*",
                         r"ust kamenogorsk", r"oskemen"],
    "Костанай": [r"костана\w*", r"kostana\w*", r"qostana\w*"],
    "Тараз": [r"тараз\w*", r"жамбыл\w*", r"taraz", r"zhambyl"],
    "Туркестан": [r"туркестан\w*", r"turkestan"],
    "Кызылорда": [r"кызылорд\w*", r"kyzylorda"],
    "Актау": [r"актау\w*", r"мангиста\w*", r"мангышла\w*", r"aktau", r"mangystau"],
    "Уральск": [r"уральск\w*", r"орал\b", r"западно казахстанск\w*", r"uralsk"],
    "Петропавловск": [r"петропавловск\w*", r"северо казахстанск\w*", r"petropavl\w*"],
    "Кокшетау": [r"кокшета\w*", r"акмолинск\w*", r"kokshetau"],
    "Семей": [r"семе[йя]\w*", r"семипалатинск\w*", r"semey"],
    "Талдыкорган": [r"талдыкорган\w*", r"жетысу\w*", r"taldykorgan"],
    "Экибастуз": [r"экибастуз\w*", r"ekibastuz"],
    "Жезказган": [r"жезказган\w*", r"улытау\w*", r"zhezkazgan"],
}


@cache
def _pattern(city: str) -> re.Pattern:
    return re.compile(r"(?<!\w)(?:" + "|".join(CITIES[city]) + ")")


def canonical(city: str | None) -> str | None:
    """Map a user-supplied city to its canonical name (None if unknown)."""
    if not city:
        return None
    folded = fold(city)
    for name in CITIES:
        if _pattern(name).match(folded):
            return name
    return None


def cities_in(text: str, folded: bool = False) -> set[str]:
    """Canonical names of every known city/region mentioned in the text."""
    t = text if folded else fold(text)
    return {name for name in CITIES if _pattern(name).search(t)}


def mentions_city(text: str, city: str, folded: bool = False) -> bool:
    """True if the text mentions this city in any form; unknown cities fall back to a stem match."""
    t = text if folded else fold(text)
    name = canonical(city)
    if name:
        return bool(_pattern(name).search(t))
    stem = fold(city)
    stem = stem[:-1] if len(stem) > 4 and stem[-1] in "аеиоуыя" else stem
    return bool(stem) and re.search(r"(?<!\w)" + re.escape(stem), t) is not None
