"""Name normalisation, transliteration and fuzzy name matching.

Kazakhstani names show up in sources in several spellings: Cyrillic with Kazakh
letters (Қайрат), Cyrillic without them (Кайрат), passport Latin (Kairat,
Qairat), and with or without the patronymic. Companies appear with or without
the legal form (ТОО / ЖШС / LLP) and quotes. Everything is folded to one Latin
"skeleton" before comparison so that all of these compare equal.

Person names are expected in the Kazakhstani order: Фамилия Имя Отчество.
"""
import re
import unicodedata

from rapidfuzz import fuzz

from models import Subject, SubjectType

_KAZAKH_FOLD = str.maketrans({
    "ә": "а", "ғ": "г", "қ": "к", "ң": "н", "ө": "о", "ұ": "у", "ү": "у", "һ": "х", "і": "и", "ё": "е",
})

_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh", "з": "z", "и": "i",
    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}

# Spelling-variant collapse applied to the Latin form (both Cyrillic-derived and native Latin):
# Kh/H, Q/K, Yu/U, doubled letters (Issabayev / Isabaev) and y/i all mean the same name.
_SKELETON_RULES = [
    (r"shch", "sh"), (r"kh", "h"), (r"q", "k"), (r"yu", "u"), (r"ya", "a"), (r"iy", "i"),
    (r"y", "i"), (r"j", "zh"), (r"w", "u"), (r"x", "ks"), (r"(.)\1", r"\1"),
]

# Legal forms in Russian, Kazakh and English: stripped before comparing company names.
_LEGAL_FORMS = [
    "товарищество с ограниченной ответственностью", "акционерное общество",
    "индивидуальный предприниматель", "жауапкершілігі шектеулі серіктестігі",
    "жауапкершілігі шектеулі серіктестік", "акционерлік қоғамы", "акционерлік қоғам",
    "limited liability partnership", "joint stock company", "limited liability company",
    "тоо", "ао", "ақ", "жшс", "ип", "жк", "пк", "оо", "гкп", "ргп", "кгп", "оюл", "филиал",
    "llp", "jsc", "llc", "ltd", "inc", "plc", "gmbh",
]
_LEGAL_RE = re.compile(
    r"(?<![\wа-яёәғқңөұүһі])(" + "|".join(re.escape(f) for f in sorted(_LEGAL_FORMS, key=len, reverse=True))
    + r")(?![\wа-яёәғқңөұүһі])",
    re.IGNORECASE,
)


def fold(text: str) -> str:
    """Lowercase, drop Kazakh-specific letters and accents, keep letters/digits/spaces."""
    text = unicodedata.normalize("NFKC", text).lower().translate(_KAZAKH_FOLD)
    # Strip accents (é -> e) but keep й: in NFD it is и + a combining breve.
    text = "".join(
        c if c == "й" else "".join(d for d in unicodedata.normalize("NFD", c) if unicodedata.category(d) != "Mn")
        for c in text
    )
    text = re.sub(r"[^\w\s]", " ", text).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def transliterate(text: str) -> str:
    """Cyrillic to Latin (simplified, close to passport spelling)."""
    return "".join(_TRANSLIT.get(c, c) for c in fold(text))


def skeleton(text: str) -> str:
    """Spelling-insensitive Latin key used for all comparisons."""
    s = transliterate(text)
    for pattern, repl in _SKELETON_RULES:
        s = re.sub(pattern, repl, s)
    return s


def strip_legal_form(name: str) -> str:
    name = re.sub(r"[«»\"“”„'`]", " ", name)
    name = _LEGAL_RE.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip(" ,.-")


def _title_latin(text: str) -> str:
    return " ".join(w.capitalize() for w in transliterate(text).split())


def search_variants(subject: Subject) -> list[str]:
    """Spellings to search for: Cyrillic as given, without the legal form, Latin."""
    names = [subject.name, *subject.aliases]
    out: list[str] = []
    for name in names:
        if subject.type == SubjectType.COMPANY:
            core = strip_legal_form(name)
            out += [name, core, _title_latin(core)]
        else:
            parts = name.split()
            out.append(name)
            if len(parts) >= 2:
                surname, given = parts[0], parts[1]
                out += [f"{surname} {given}", f"{given} {surname}"]
                initials = "".join(f"{p[0].upper()}." for p in parts[1:])
                out.append(f"{surname} {initials}")
                out.append(_title_latin(f"{given} {surname}"))
    seen, uniq = set(), []
    for v in out:
        key = v.lower()
        if v and key not in seen:
            seen.add(key)
            uniq.append(v)
    return uniq


def company_similarity(a: str, b: str) -> float:
    """0..1 similarity of two company names, legal forms and spelling ignored."""
    ka, kb = skeleton(strip_legal_form(a)), skeleton(strip_legal_form(b))
    if not ka or not kb:
        return 0.0
    # "КазОйл" vs "Каз Ойл": same name, different tokenisation.
    joined = fuzz.ratio(ka.replace(" ", ""), kb.replace(" ", "")) / 100
    if joined >= 0.92:
        return joined
    # Whole-string similarity alone rates "Ромашка Строй" vs "Ромашка Трейд" at ~0.8:
    # scale it by the share of tokens that have a counterpart in the other name.
    ta, tb = ka.split(), kb.split()
    matched = sum(1 for t in ta if any(fuzz.ratio(t, u) >= 85 for u in tb))
    coverage = matched / max(len(ta), len(tb))
    return round(fuzz.token_sort_ratio(ka, kb) / 100 * coverage, 3)


def person_similarity(subject_name: str, mention: str) -> tuple[float, str]:
    """0..1 score plus the reason, for a person mention against the subject's full name.

    A bare surname matches thousands of people, so the score is driven by how much
    of the name is confirmed: surname + given name + patronymic is strong, a surname
    with an initial is weak, a surname alone is almost nothing.
    """
    subj = skeleton(subject_name).split()
    ment = skeleton(mention).split()
    if not subj or not ment:
        return 0.0, "пустое имя"

    def full_match(token: str, target: str) -> bool:
        return len(token) > 1 and fuzz.ratio(token, target) >= 85

    def initial_match(token: str, target: str) -> bool:
        return len(token) == 1 and target.startswith(token)

    surname = subj[0]
    rest = subj[1:]
    surname_ok = any(full_match(t, surname) for t in ment)
    given_full = bool(rest) and any(full_match(t, rest[0]) for t in ment)
    given_init = bool(rest) and any(initial_match(t, rest[0]) for t in ment)
    patronymic_full = len(rest) > 1 and any(full_match(t, rest[1]) for t in ment)
    patronymic_init = len(rest) > 1 and any(initial_match(t, rest[1]) for t in ment)

    if surname_ok and given_full and patronymic_full:
        return 1.0, "совпадают фамилия, имя и отчество"
    if surname_ok and given_full and patronymic_init:
        return 0.9, "совпадают фамилия, имя и инициал отчества"
    if surname_ok and given_full:
        return 0.8, "совпадают фамилия и имя"
    if surname_ok and given_init:
        # Below the default MATCH_THRESHOLD on its own: needs city / related names / БИН to count.
        return 0.45, "совпадают фамилия и инициал"
    if surname_ok:
        return 0.25, "совпадает только фамилия"
    best = max(fuzz.ratio(t, surname) for t in ment) / 100
    return round(best * 0.2, 3), "фамилия не совпадает"


def name_similarity(subject: Subject, mention: str) -> tuple[float, str]:
    """Best similarity of a mention against the subject's name and aliases."""
    best = (0.0, "имя не совпадает")
    for name in [subject.name, *subject.aliases]:
        if subject.type == SubjectType.COMPANY:
            s = company_similarity(name, mention)
            cand = (s, f"сходство названия {s:.0%}")
        else:
            cand = person_similarity(name, mention)
        if cand[0] > best[0]:
            best = cand
    return best
