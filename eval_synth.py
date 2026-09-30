"""Synthetic identity-resolution eval set: the subject vs. a source that may be about a namesake.

Each case is (subject, finding, page text, label): label 1 if the source is about
the subject, 0 if it is about another person or company with a similar name.
Cases are built from explicit trap types, so errors can be read per trap:
a full namesake from another city, a different patronymic, a surname one letter
off, the same ТОО name with another БИН, and the positives that are easy to miss
(initials, Latin spelling, Kazakh patronymic, surname + role only).

All people, companies, identifiers and events are fictional. Company names are
built from placeholder words (Ромашка, Василёк…) on purpose.

The split is by subject: every case of one subject lands in the same split, so a
learned matcher trained on dev never sees a test subject.

  python eval_synth.py                  # -> data/eval/identity_synth.jsonl
  python eval_synth.py --per-trap 40 --seed 7
"""
import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from config import settings

YEAR = 2026

MALE_GIVEN = ["Ержан", "Нурлан", "Серик", "Асхат", "Бауыржан", "Даурен", "Қайрат", "Марат", "Арман",
              "Ерлан", "Талгат", "Азамат", "Дархан", "Самат", "Ерболат", "Жандос", "Руслан", "Ғалым"]
FEMALE_GIVEN = ["Айгүл", "Гульнара", "Динара", "Жанар", "Сауле", "Асель", "Меруерт", "Алия", "Камила",
                "Салтанат", "Әсем", "Жұлдыз"]
SURNAMES = ["Ахметов", "Сериков", "Нурланов", "Искаков", "Жумабаев", "Касымов", "Омаров", "Султанов",
            "Абенов", "Байжанов", "Тулегенов", "Ермеков", "Сапаров", "Кенжебеков", "Мукашев", "Бекмуханов",
            "Оспанов", "Садыков", "Толеубаев", "Алиев", "Утебаев", "Шаймерденов", "Есенов", "Қожахметов"]
FATHERS = ["Болат", "Серик", "Марат", "Кайрат", "Нурлан", "Ерлан", "Асхат", "Жанат", "Талгат", "Бекзат",
           "Мурат", "Ербол", "Темир", "Даулет"]
# (nominative, prepositional "в ...", genitive "из ...")
CITIES = [("Алматы", "Алматы", "Алматы"), ("Астана", "Астане", "Астаны"), ("Шымкент", "Шымкенте", "Шымкента"),
          ("Караганда", "Караганде", "Караганды"), ("Актобе", "Актобе", "Актобе"), ("Атырау", "Атырау", "Атырау"),
          ("Павлодар", "Павлодаре", "Павлодара"), ("Усть-Каменогорск", "Усть-Каменогорске", "Усть-Каменогорска"),
          ("Костанай", "Костанае", "Костаная"), ("Тараз", "Таразе", "Тараза")]
COMPANY_A = ["Ромашка", "Василёк", "Лютик", "Одуванчик", "Колокольчик", "Подсолнух", "Тюльпан", "Незабудка",
             "Жасмин", "Сирень", "Шиповник", "Клевер", "Ландыш", "Пион"]
COMPANY_B = ["Строй", "Трейд", "Групп", "Инвест", "Логистик", "Агро", "Сервис", "Консалтинг", "Энерго",
             "Девелопмент", "Транс", "Мед"]
COMPANY_C = ["Плюс", "Центр", "Азия", "Холдинг", "KZ"]

# (male, female) forms
EVENTS = [("задержан по подозрению в мошенничестве", "задержана по подозрению в мошенничестве"),
          ("обвиняется в хищении бюджетных средств", "обвиняется в хищении бюджетных средств"),
          ("осуждён за получение взятки", "осуждена за получение взятки"),
          ("фигурирует в деле о лжепредпринимательстве", "фигурирует в деле о лжепредпринимательстве"),
          ("признан виновным в уклонении от уплаты налогов", "признана виновной в уклонении от уплаты налогов")]
COMPANY_EVENTS = ["подозревают в обналичивании денег", "включили в реестр недобросовестных участников госзакупок",
                  "признали банкротом по иску кредиторов", "оштрафовали за нарушение экологического законодательства",
                  "обвиняют в срыве госконтракта на строительство школы"]
FILLER = ["По данным департамента полиции, ущерб составил {n} млн тенге.", "Расследование продолжается.",
          "Вину фигурант не признаёт.", "Дело направлено в суд.",
          "Ранее проверка проводилась в отношении ТОО «{other}».", "Сумма задолженности составляет {n} млн тенге."]
ROLES = ["директор ТОО «{company}»", "учредитель ТОО «{company}»", "руководитель ТОО «{company}»"]
EN_EVENTS = ["was detained on suspicion of fraud", "is charged with embezzlement of public funds",
             "was convicted of bribery"]
CITY_EN = {"Алматы": "Almaty", "Астана": "Astana", "Шымкент": "Shymkent", "Караганда": "Karaganda",
           "Актобе": "Aktobe", "Атырау": "Atyrau", "Павлодар": "Pavlodar", "Усть-Каменогорск": "Ust-Kamenogorsk",
           "Костанай": "Kostanay", "Тараз": "Taraz"}

LATIN = {"а": "a", "б": "b", "в": "v", "г": "g", "ғ": "gh", "д": "d", "е": "e", "ё": "yo", "ж": "zh",
         "з": "z", "и": "i", "й": "y", "к": "k", "қ": "q", "л": "l", "м": "m", "н": "n", "ң": "ng", "о": "o",
         "ө": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ұ": "u", "ү": "u", "ф": "f", "х": "kh",
         "һ": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ъ": "", "ы": "y", "і": "i", "ь": "",
         "э": "e", "ю": "yu", "я": "ya", "ә": "a"}


def latin(word: str, rng: random.Random) -> str:
    """Passport-style Latin with the real-world variation: Ye- for initial Е, K or Q for Қ."""
    out = []
    for i, ch in enumerate(word.lower()):
        if i == 0 and ch == "е" and rng.random() < 0.6:
            out.append("ye")
        elif ch == "қ" and rng.random() < 0.5:
            out.append("k")
        else:
            out.append(LATIN.get(ch, ch))
    return "".join(out).capitalize()


@dataclass
class Person:
    surname: str
    given: str
    father: str
    female: bool
    city: tuple
    birth: date
    iin: str
    company: str  # a company the person is officially linked to

    @property
    def surname_g(self) -> str:
        return self.surname + "а" if self.female else self.surname

    @property
    def patronymic(self) -> str:
        return self.father + ("овна" if self.female else "ович")

    @property
    def patronymic_kz(self) -> str:
        return self.father + ("қызы" if self.female else "ұлы")

    @property
    def full(self) -> str:
        return f"{self.surname_g} {self.given} {self.patronymic}"

    @property
    def initials(self) -> str:
        return f"{self.surname_g} {self.given[0]}.{self.father[0]}."


def iin_for(birth: date, female: bool, rng: random.Random) -> str:
    century_sex = ("4" if female else "3") if birth.year < 2000 else ("6" if female else "5")
    return f"{birth:%y%m%d}{century_sex}{rng.randint(0, 99999):05d}"


def bin_for(rng: random.Random) -> str:
    return f"{rng.randint(5, 24):02d}{rng.randint(1, 12):02d}4{rng.randint(0, 9)}{rng.randint(0, 999999):06d}"


def company_name(rng: random.Random, exclude: set[str] = frozenset()) -> tuple[str, str]:
    while True:
        a, b = rng.choice(COMPANY_A), rng.choice(COMPANY_B)
        if f"{a} {b}" not in exclude:
            return a, b


def new_person(rng: random.Random) -> Person:
    female = rng.random() < 0.3
    birth = date(rng.randint(1960, 1996), rng.randint(1, 12), rng.randint(1, 28))
    a, b = company_name(rng)
    return Person(surname=rng.choice(SURNAMES), given=rng.choice(FEMALE_GIVEN if female else MALE_GIVEN),
                  father=rng.choice(FATHERS), female=female, city=rng.choice(CITIES), birth=birth,
                  iin=iin_for(birth, female, rng), company=f"{a} {b}")


def other(rng: random.Random, pool: list, current):
    return rng.choice([x for x in pool if x != current])


def filler(rng: random.Random) -> str:
    a, b = company_name(rng)
    return rng.choice(FILLER).format(n=rng.randint(12, 900), other=f"{a} {b}")


def one_letter_off(surname: str, rng: random.Random) -> str:
    vowels = "аеиоуы"
    idx = [i for i, c in enumerate(surname[1:-2], 1) if c in vowels] or [1]
    i = rng.choice(idx)
    return surname[:i] + rng.choice([v for v in vowels if v != surname[i]]) + surname[i + 1:]


def person_subject(p: Person) -> dict:
    return {"type": "person", "name": p.full, "identifier": p.iin, "birth_date": p.birth.isoformat(),
            "city": p.city[0], "related": [f"ТОО {p.company}"], "purpose": "eval: identity resolution"}


def ev(p: Person, i: int) -> str:
    return EVENTS[i][1 if p.female else 0]


# --- person traps: each returns (mention, text, label) -------------------------------------------

def p_full_city(p, rng):
    e = rng.randrange(len(EVENTS))
    return p.full, f"В {p.city[1]} {p.full} {ev(p, e)}. {filler(rng)}", 1


def p_order_given_surname(p, rng):
    m = f"{p.given} {p.surname_g}"
    role = rng.choice(ROLES).format(company=p.company)
    return m, f"{role.capitalize()} {m} {ev(p, rng.randrange(len(EVENTS)))}. {filler(rng)}", 1


def p_initials_context(p, rng):
    return p.initials, (f"В {p.city[1]} {p.initials}, {p.birth.year} года рождения, "
                        f"{ev(p, rng.randrange(len(EVENTS)))}. {filler(rng)}"), 1


def p_latin(p, rng):
    m = f"{latin(p.given, rng)} {latin(p.surname_g, rng)}"
    return m, f"{m}, a businessman from {CITY_EN[p.city[0]]}, {rng.choice(EN_EVENTS)}.", 1


def p_kazakh_patronymic(p, rng):
    m = f"{p.surname_g} {p.given} {p.patronymic_kz}"
    return m, f"В {p.city[1]} {m} {ev(p, rng.randrange(len(EVENTS)))}. {filler(rng)}", 1


def p_iin_in_text(p, rng):
    return p.initials, (f"Должник: {p.initials}, ИИН {p.iin}, г. {p.city[0]}. Взыскатель: ТОО «Кредит Финанс». "
                        f"Сумма: {rng.randint(1, 90)} млн тенге. Установлен временный запрет на выезд."), 1


def p_surname_role(p, rng):
    role = rng.choice(ROLES).format(company=p.company)
    return p.surname_g, (f"В {p.city[1]} {role} {p.surname_g} {ev(p, rng.randrange(len(EVENTS)))}. "
                         f"{filler(rng)}"), 1


def p_full_bare(p, rng):
    return p.full, f"{p.full} {ev(p, rng.randrange(len(EVENTS)))}.", 1


def n_surname_only(p, rng):
    city = other(rng, CITIES, p.city)
    return p.surname_g, f"В {city[1]} предприниматель {p.surname_g} {ev(p, rng.randrange(len(EVENTS)))}. {filler(rng)}", 0


def n_other_given(p, rng):
    given = other(rng, FEMALE_GIVEN if p.female else MALE_GIVEN, p.given)
    m = f"{p.surname_g} {given}"
    return m, f"В {p.city[1]} {m} {ev(p, rng.randrange(len(EVENTS)))}. {filler(rng)}", 0


def n_initial_collision(p, rng):
    pool = [g for g in (FEMALE_GIVEN if p.female else MALE_GIVEN) if g[0] == p.given[0] and g != p.given]
    city = other(rng, CITIES, p.city)
    if pool:
        given = rng.choice(pool)
        text = (f"В {city[1]} {p.surname_g} {given[0]}. {ev(p, rng.randrange(len(EVENTS)))}. "
                f"Как уточнили в полиции, {p.surname_g} {given} ранее не судим.")
    else:
        text = f"В {city[1]} {p.surname_g} {p.given[0]}. {ev(p, rng.randrange(len(EVENTS)))}. {filler(rng)}"
    return f"{p.surname_g} {p.given[0]}.", text, 0


def n_full_other_city_age(p, rng):
    city = other(rng, CITIES, p.city)
    year = p.birth.year + rng.choice([-1, 1]) * rng.randint(6, 20)
    return p.full, (f"В {city[1]} {p.full}, {year} года рождения, {ev(p, rng.randrange(len(EVENTS)))}. "
                    f"{filler(rng)}"), 0


def n_full_other_iin(p, rng):
    other_iin = iin_for(date(rng.randint(1955, 1999), rng.randint(1, 12), rng.randint(1, 28)), p.female, rng)
    return p.full, (f"Должник: {p.full}, ИИН {other_iin}. Взыскатель: ТОО «Кредит Финанс». "
                    f"Сумма: {rng.randint(1, 90)} млн тенге."), 0


def n_similar_surname(p, rng):
    m = f"{one_letter_off(p.surname_g, rng)} {p.given}"
    city = other(rng, CITIES, p.city)
    return m, f"В {city[1]} {m} {ev(p, rng.randrange(len(EVENTS)))}. {filler(rng)}", 0


def n_other_patronymic(p, rng):
    father = other(rng, FATHERS, p.father)
    m = f"{p.surname_g} {p.given} {father}{'овна' if p.female else 'ович'}"
    city = other(rng, CITIES, p.city)
    return m, f"В {city[1]} {m} {ev(p, rng.randrange(len(EVENTS)))}. {filler(rng)}", 0


def n_initials_same_city(p, rng):
    year = p.birth.year + rng.choice([-1, 1]) * rng.randint(6, 20)
    return p.initials, (f"В {p.city[1]} {p.initials}, {year} года рождения, {ev(p, rng.randrange(len(EVENTS)))}. "
                        f"{filler(rng)}"), 0


PERSON_TRAPS = [p_full_city, p_order_given_surname, p_initials_context, p_latin, p_kazakh_patronymic,
                p_iin_in_text, p_surname_role, p_full_bare, n_surname_only, n_other_given, n_initial_collision,
                n_full_other_city_age, n_full_other_iin, n_similar_surname, n_other_patronymic,
                n_initials_same_city]


# --- company traps -----------------------------------------------------------------------------

@dataclass
class Company:
    a: str
    b: str
    city: tuple
    bin: str
    director: Person

    @property
    def core(self) -> str:
        return f"{self.a} {self.b}"


def new_company(rng: random.Random) -> Company:
    a, b = company_name(rng)
    return Company(a=a, b=b, city=rng.choice(CITIES), bin=bin_for(rng), director=new_person(rng))


def company_subject(c: Company) -> dict:
    return {"type": "company", "name": f"ТОО «{c.core}»", "identifier": c.bin, "city": c.city[0],
            "related": [c.director.full], "purpose": "eval: identity resolution"}


def c_bin_in_text(c, rng):
    return f"ТОО «{c.core}»", f"ТОО «{c.core}» (БИН {c.bin}) {rng.choice(COMPANY_EVENTS)}. {filler(rng)}", 1


def c_latin_llp(c, rng):
    m = f"{latin(c.a, rng)} {latin(c.b, rng)} LLP"
    return m, f"{m}, a contractor based in {CITY_EN[c.city[0]]}, was blacklisted from public tenders.", 1


def c_kazakh_form(c, rng):
    m = f"«{c.core}» ЖШС"
    return m, f"{c.city[0]} қаласындағы {m} мемлекеттік сатып алудың жосықсыз қатысушылары тізіліміне енгізілді.", 1


def c_director_related(c, rng):
    m = f"ТОО «{c.core}»"
    return m, (f"{m} {rng.choice(COMPANY_EVENTS)}. Директор компании {c.director.full} от комментариев "
               f"отказался."), 1


def c_joined(c, rng):
    m = f"ТОО «{c.a}{c.b}»"
    return m, f"В {c.city[1]} {m} {rng.choice(COMPANY_EVENTS)}. {filler(rng)}", 1


def c_name_city(c, rng):
    m = f"ТОО «{c.core}»"
    return m, f"В {c.city[1]} {m} {rng.choice(COMPANY_EVENTS)}. {filler(rng)}", 1


def n_c_near_miss(c, rng):
    m = f"ТОО «{c.a} {other(rng, COMPANY_B, c.b)}»"
    city = other(rng, CITIES, c.city)
    return m, f"В {city[1]} {m} {rng.choice(COMPANY_EVENTS)}. {filler(rng)}", 0


def n_c_extra_word(c, rng):
    m = f"ТОО «{c.core} {rng.choice(COMPANY_C)}»"
    city = other(rng, CITIES, c.city)
    return m, f"В {city[1]} {m} {rng.choice(COMPANY_EVENTS)}. {filler(rng)}", 0


def n_c_other_bin(c, rng):
    m = f"ТОО «{c.core}»"
    city = other(rng, CITIES, c.city)
    return m, f"ТОО «{c.core}» (БИН {bin_for(rng)}) из {city[2]} {rng.choice(COMPANY_EVENTS)}.", 0


def n_c_other_city(c, rng):
    m = f"ТОО «{c.core}»"
    city = other(rng, CITIES, c.city)
    boss = new_person(rng)
    return m, (f"В {city[1]} {m} {rng.choice(COMPANY_EVENTS)}. Директор компании {boss.full} "
               f"от комментариев отказался."), 0


def n_c_first_word(c, rng):
    m = f"ТОО «{c.a}»"
    city = other(rng, CITIES, c.city)
    return m, f"В {city[1]} {m} {rng.choice(COMPANY_EVENTS)}. {filler(rng)}", 0


def n_c_other_first(c, rng):
    m = f"ТОО «{other(rng, COMPANY_A, c.a)} {c.b}»"
    city = other(rng, CITIES, c.city)
    return m, f"В {city[1]} {m} {rng.choice(COMPANY_EVENTS)}. {filler(rng)}", 0


COMPANY_TRAPS = [c_bin_in_text, c_latin_llp, c_kazakh_form, c_director_related, c_joined, c_name_city,
                 n_c_near_miss, n_c_extra_word, n_c_other_bin, n_c_other_city, n_c_first_word, n_c_other_first]


def _case(case_id, split, trap, label, subject, mention, text, rng) -> dict:
    # Some pages are unreachable in practice: then only the quote and the model's notes are available.
    page = None if rng.random() < 0.15 else text
    first_sentence = text.split(". ")[0].rstrip(".") + "."
    return {"id": case_id, "split": split, "trap": trap, "label": label, "subject": subject,
            "finding": {"source_type": "web", "source_name": "synthetic", "url": f"https://example.kz/{case_id}",
                        "category": "fraud", "status": "allegation", "claim": "synthetic",
                        "title": first_sentence[:120], "quote": first_sentence,
                        "subject_mentioned_as": mention, "identity_evidence": []},
            "page_text": page}


def generate(per_trap: int = 25, seed: int = 13, test_share: float = 0.5) -> list[dict]:
    rng = random.Random(seed)
    cases: list[dict] = []
    # One subject per (trap, i); the split is drawn per subject.
    for kind, traps, make, to_subject in [("p", PERSON_TRAPS, new_person, person_subject),
                                          ("c", COMPANY_TRAPS, new_company, company_subject)]:
        for trap in traps:
            for i in range(per_trap):
                entity = make(rng)
                split = "test" if rng.random() < test_share else "dev"
                mention, text, label = trap(entity, rng)
                cases.append(_case(f"{kind}-{trap.__name__}-{i:03d}", split, trap.__name__, label,
                                   to_subject(entity), mention, text, rng))
    return cases


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--per-trap", type=int, default=25)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out", default=str(Path(settings.DATA_DIR) / "eval" / "identity_synth.jsonl"))
    args = p.parse_args()
    cases = generate(args.per_trap, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    labels = Counter(c["label"] for c in cases)
    splits = Counter(c["split"] for c in cases)
    print(f"{len(cases)} cases -> {out}  (same: {labels[1]}, namesake: {labels[0]}; "
          f"dev: {splits['dev']}, test: {splits['test']})")


if __name__ == "__main__":
    main()
