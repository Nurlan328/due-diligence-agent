from models import Subject
from names import company_similarity, person_similarity, search_variants, skeleton, strip_legal_form


def test_skeleton_folds_kazakh_letters_and_latin_spellings():
    assert skeleton("Қайрат") == skeleton("Кайрат") == skeleton("Kairat") == skeleton("Qairat")
    assert skeleton("Исабаев") == skeleton("Isabaev")
    assert skeleton("Жұмабек") == skeleton("Zhumabek")


def test_strip_legal_form_ru_kz_en():
    assert strip_legal_form("ТОО «Ромашка Строй»") == "Ромашка Строй"
    assert strip_legal_form('"Romashka Stroy" LLP') == "Romashka Stroy"
    assert strip_legal_form("Ромашка Строй ЖШС") == "Ромашка Строй"
    assert strip_legal_form("Акционерное общество Каз Ойл") == "Каз Ойл"


def test_legal_form_not_stripped_inside_words():
    assert strip_legal_form("Аовица") == "Аовица"


def test_company_similarity_ignores_form_and_script():
    assert company_similarity("ТОО «Ромашка Строй»", "Romashka Stroy LLP") > 0.9
    assert company_similarity("ТОО Ромашка Строй", "ТОО Василёк Трейд") < 0.5


def test_person_similarity_levels():
    full = "Ахметов Ержан Болатович"
    assert person_similarity(full, "Ахметов Ержан Болатович")[0] == 1.0
    assert person_similarity(full, "Ержан Ахметов")[0] == 0.8
    assert person_similarity(full, "Akhmetov Yerzhan")[0] == 0.8  # passport spelling Ye- = Е-
    assert person_similarity(full, "Ахметов Е.Б.")[0] == 0.45
    assert person_similarity(full, "Ахметов")[0] == 0.25
    assert person_similarity(full, "Сериков Ержан")[0] < 0.25


def test_search_variants_person():
    s = Subject(type="person", name="Ахметов Ержан Болатович", purpose="KYC при открытии счёта")
    v = search_variants(s)
    assert "Ахметов Ержан Болатович" in v
    assert "Ержан Ахметов" in v
    assert "Ахметов Е.Б." in v
    assert "Erzhan Akhmetov" in v


def test_search_variants_company():
    s = Subject(type="company", name="ТОО «Ромашка Строй»", purpose="Проверка поставщика")
    v = search_variants(s)
    assert v[:2] == ["ТОО «Ромашка Строй»", "Ромашка Строй"]
    assert "Romashka Stroy" in v
