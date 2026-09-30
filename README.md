# Due Diligence Agent

Adverse media and counterparty screening for Kazakhstan. The input is a company
(БИН) or a person (ИИН). The output is a report of every piece of **negative
public information** about that subject: sanctions, criminal cases, fraud,
corruption, bankruptcy, tax debts, lawsuits, enforcement proceedings, the
public-procurement blacklist. Each finding comes with a source, a verbatim
quote, a procedural status and an identity-confidence score.

This is the KYC / KYB check banks, fintechs and procurement teams run before
they onboard a client or sign with a supplier. Done by hand, it means an analyst
spends an hour on search engines. The hard part is not finding pages but
**trusting them**: is the quote real, is this the same person or a namesake,
is it a conviction or only a rumour? The pipeline is built around those three
questions.

## Pipeline

```
Subject (type, name, БИН/ИИН, birth date, city, related names, purpose)
  │
  ├─ 1. Structured sources         sources/
  │     OpenSanctions (OFAC, UN, EU, UK… 100+ lists; matched locally)
  │     goszakup.gov.kz РНУ (unscrupulous procurement participants; by БИН/ИИН)
  │
  ├─ 2. Deep search agent          deepsearch.py
  │     Claude + web_search / web_fetch server tools, RU / KK / EN queries.
  │     Every fact goes through our own `record_findings` tool (strict schema):
  │     URL, verbatim quote, category, status, severity, identity evidence, leads.
  │     Round N+1 follows the leads of round N (affiliated companies, case numbers).
  │
  ├─ 3. Faithfulness check         verify.py
  │     We fetch each page ourselves and look for the quote.
  │     Quote missing → the finding is discounted ×0.15 (likely hallucination).
  │
  ├─ 4. Identity resolution        matching.py, names.py
  │     Namesake filter: name agreement (Kazakh letters, transliteration,
  │     initials, legal forms), БИН/ИИН in the page, city, related names, birth year.
  │     Below MATCH_THRESHOLD → "possible namesakes", excluded from the score.
  │
  ├─ 5. Risk score                 scoring.py
  │     severity × status × verification × identity, noisy-OR across findings
  │
  └─ 6. Report + audit trail       report.py, audit.py
        Markdown / JSON, list of sources checked (also those with no hits).
```

| module | what |
|---|---|
| `models.py` | subject, finding, category / status / severity enums, report |
| `names.py` | Kazakh-letter folding, Cyrillic→Latin, spelling skeleton, name variants, person / company similarity |
| `sources/` | structured collectors (sanctions, РНУ) behind one `Collector` interface |
| `deepsearch.py` | the Claude agent loop: server tools, `pause_turn`, refusals, tool-input validation, rounds |
| `verify.py` | page fetch, main-text extraction, exact + fuzzy quote matching |
| `matching.py` | identity features and score; every feature vector is logged for training |
| `scoring.py` | explicit risk weights, dedup of one case reported by many outlets |
| `pipeline.py` | the stages in order |
| `cli.py`, `main.py` | command line and FastAPI |

## Design decisions

**The model's claims are data, not truth.** The agent must give a verbatim
quote for every finding, and the quote is checked against the page we fetch
ourselves. A fabricated quote does not disappear silently: it stays in the
report marked "⚠ цитата не найдена", at the bottom of the list.

**Status is a first-class field.** "The director was detained" (investigation),
"the court convicted" (convicted) and "the case was dropped" (closed) are
different risks. The status is part of the score: an acquittal contributes 0.

**Namesakes are the main false-positive source.** A surname alone scores 0.25,
surname + initial 0.45 (below the threshold), surname + given name 0.8. Context
(БИН in the text, city, a related company, birth year) lifts or confirms.
Findings that stay below the threshold are shown separately for a human to
review instead of being thrown away.

**One case is not five risks.** Five articles about the same criminal case
corroborate each other, but only the strongest finding per (category, status)
counts in the noisy-OR.

**A transparent baseline before a model.** The identity matcher is a heuristic
on purpose. Every feature vector goes to `data/match_features.jsonl` with an
empty `label`. Once reviewers label findings as same / different entity, that
log is the training set for a learned matcher, evaluated against this baseline
on the same split.

## Privacy and legal basis

Screening a person is processing personal data (Закон РК «О персональных данных
и их защите»). Banks and other subjects of financial monitoring do it under the
AML law (Закон РК № 191-IV о ПОД/ФТ, due diligence on clients). The tool
enforces the following:

- `purpose` is mandatory and every screening is written to `data/audit.jsonl`:
  operator, purpose, subject, the queries sent out.
- A person's **ИИН and full birth date never leave the machine**, except the ИИН
  lookup in the official goszakup registry. Claude and the search engine get
  the name, city, birth year and related names only. The ИИН is used locally to
  confirm identity in fetched pages and is masked in reports and logs.
- The agent is told not to collect private life (health, family, religion,
  ethnicity, address, phone) even when it sees it. Leads are limited to
  official roles.
- Public sources only: no logins, captchas or paywalls are bypassed.
- The report is decision support. It labels allegations as allegations and
  says a human must review it.
- OpenSanctions data is CC BY-NC 4.0: free for non-commercial use, a commercial
  deployment needs their data licence.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env            # set ANTHROPIC_API_KEY (and GOSZAKUP_TOKEN if you have one)

python cli.py update-sanctions    # ~70 MB OpenSanctions export into data/

# registries only, no API key needed
python cli.py screen --type company --name "ТОО «Ромашка Строй»" --id 123456789012 \
    --purpose "Проверка поставщика перед заключением договора" --no-deep

# full screening with deep search
python cli.py screen --type person --name "Фамилия Имя Отчество" --birth-date 1980-05-01 \
    --city Алматы --related "ТОО Ромашка Строй" --purpose "KYC при открытии счёта"
```

Reports are written to `reports/<request_id>.md` and `.json`. HTTP API:
`fastapi dev main.py`, then `POST /screen` or `POST /screen.md` (docs at `/docs`).

Tests run offline. Claude is replaced by a scripted stub and page fetches go
through `httpx.MockTransport`:

```bash
pytest
```

## Evaluation: namesakes

`matching.py` decides whether a source is about the subject or about a namesake.
That decision is evaluated on two sets:

- **Synthetic set** (`eval_synth.py`): 700 cases from 28 explicit trap types, with
  a fixed seed. Namesakes: a full namesake from another city or with another
  birth year, the same full name with a different ИИН, a different patronymic,
  a surname one letter off, the same ТОО name with another БИН, an extra word in
  the company name. Hard positives: initials + context, Latin spelling, Kazakh
  patronymic (-ұлы / -қызы), a surname with a role at a known company, ИИН in a
  debtor list. The split into dev and test is by subject.
- **Golden set** (`evals/golden_identity.jsonl`): 20 hand-written cases with what
  templates don't produce: inflected names (Оспановой, Дархана Жумабаева), the
  city as an adjective (карагандинский), English and Kazakh articles, a short
  common surname (Ким), a page that could not be fetched.

```bash
python eval_synth.py      # data/eval/identity_synth.jsonl
python evaluate.py        # prints and writes evals/RESULTS.md
```

The threshold is tuned on dev only and reported on test. The golden set is
never used for tuning. Both sets are fictional, so they test the matcher's
logic, not its real-world accuracy. Real accuracy comes from reviewer labels in
`data/match_features.jsonl` (`python evaluate.py --match-log ...`).

**Baseline results** (test split, threshold 0.5, full tables in
[evals/RESULTS.md](evals/RESULTS.md)):

| matcher | precision | recall | namesake leak | AUC |
|---|---|---|---|---|
| name only | 0.63 | 0.79 | 0.46 | 0.68 |
| heuristic (`matching.py`) | 0.65 | 0.96 | **0.51** | 0.75 |

The context features raise recall a lot: initials + city 7% → 73%, surname +
role 0% → 80%, ИИН in text 0% → 100%. They do nothing against namesakes, and
leakage even grows. The reason is structural: **the matcher only collects
evidence for a match and never evidence against it**. A full namesake from
another city, with another birth year or another ИИН/БИН, scores 1.0. These traps
leak 100% of the time on both sets. Other weak spots found:

- inflected surnames: «Оспановой» fails the fuzzy surname check (g-08);
- the city is matched as a substring, so «в Астане», «Караганде», «карагандинский»
  are missed;
- a one-letter-off surname passes as a full match (86% similarity ≥ 85);
- `company_similarity` accepts «Колокольчик Агро KZ» at 0.94 through the
  joined-string shortcut.

Next step: negative-evidence features (a different ИИН/БИН near the name,
another city, another birth year or age, a contradicting patronymic), morphology
for surnames and cities, then a learned matcher on the same split.

## Cost

Web search is billed at $10 per 1,000 searches, plus tokens (Opus 5.5: $4 / $20
per million input / output tokens). Search results are large, and the history
is resent on every agent turn. Prompt caching is on, but tokens still dominate.
The per-screening cost has **not been measured yet**. Measure it on the first
real runs (`usage` on each response) before choosing rounds, search budget and
effort. `MAX_ROUNDS`, `SEARCHES_PER_ROUND`, `FETCHES_PER_ROUND` and `EFFORT` are
the levers.

## Roadmap

- **Identity matcher v2**: negative evidence and morphology (see Evaluation),
  measured on the existing dev/test/golden sets.
- **End-to-end evaluation set.** Real labelled subjects: positives from
  OpenSanctions (KZ) and the РНУ, negatives from random registered companies.
  Metrics: finding precision / recall, fabricated-quote rate, cost and latency
  per screening.
- **Learned identity matcher** trained on the labelled `match_features.jsonl`,
  compared with the heuristic baseline on the namesake set.
- **Adverse-media classifier** (fine-tuned rubert-tiny2) against Claude
  zero-shot on category / status, for a cheap pre-filter.
- **More Kazakhstani sources**: КГД lists (inactive taxpayers, sham companies,
  tax debtors), the АФМ list of persons linked to terrorism financing, debtor
  register (АИС ОИП), court acts from sud.kz, the state business register.
- **Review workflow**: analyst confirms or rejects each finding in a UI; the
  decisions become matcher labels (human-in-the-loop, LangGraph interrupt).
