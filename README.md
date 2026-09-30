# Due Diligence Agent

Adverse media and counterparty screening for Kazakhstan. The input is a company
(БИН) or a person (ИИН). The output is a report of every piece of **negative
public information** about that subject and its directors, founders and
subsidiaries: sanctions, criminal cases, fraud, corruption, bankruptcy, tax
debts, lawsuits, enforcement proceedings, the public-procurement blacklist. Each
fact comes with its sources, verbatim quotes, a procedural status, the number of
independent confirmations and an identity-confidence score.

This is the KYC / KYB check banks, fintechs and procurement teams run before
they onboard a client or sign with a supplier. Done by hand, it means an analyst
spends an hour on search engines. The hard part is not finding pages but
**trusting them**: is the quote real, is this the same person or a namesake,
is it a conviction or only a rumour, how many independent sources say so? The
pipeline is built around those questions.

## Pipeline

```
Subject (type, name, БИН/ИИН, birth date, city, related names, purpose)
  │
  ├─ 1. Structured sources         sources/
  │     OpenSanctions (OFAC, UN, EU, UK… 100+ lists; matched locally)
  │     goszakup.gov.kz РНУ (unscrupulous procurement participants; by БИН/ИИН)
  │
  ├─ 2. Deep search agent          deepsearch.py, graph.py
  │     Claude + web_search / web_fetch server tools, RU / KK / EN queries.
  │     Every fact goes through our own `record_findings` tool (strict schema):
  │     URL, verbatim quote, category, status, severity, identity evidence,
  │     event key, which entity it is about, leads with their relation.
  │     Leads grow an entity graph; each round explores its frontier.
  │
  ├─ 3. Faithfulness check         verify.py
  │     We fetch each page ourselves and look for the quote; a page that is gone
  │     or edited is checked in the Wayback Machine.
  │     Quote missing → the finding is discounted ×0.15 (likely hallucination).
  │
  ├─ 4. Identity resolution        matching.py, names.py, geo.py
  │     Namesake filter. Evidence for: name (inflection-aware), БИН/ИИН, city,
  │     related names, birth year. Evidence against, near the name: another
  │     БИН/ИИН, another city, another birth year or age, another patronymic.
  │     Below MATCH_THRESHOLD → "possible namesakes", excluded from the score.
  │
  ├─ 5. Facts and risk score       scoring.py, reliability.py
  │     Articles about one event → one fact; independent sources combine.
  │     severity × status × evidence strength × recency (× 0.5 for affiliates),
  │     noisy-OR across facts.
  │
  └─ 6. Report + audit trail       report.py, audit.py
        Markdown / JSON, entity graph (Mermaid), sources checked with no hits.
```

| module | what |
|---|---|
| `models.py` | subject, finding, fact, category / status / severity enums, report |
| `names.py` | Kazakh-letter folding, Cyrillic→Latin skeleton, lemmas (pymorphy3 + surname rules), patronymics, person / company similarity |
| `geo.py` | Kazakhstani cities and regions in any form: cases, adjectives, Kazakh and Latin names |
| `sources/` | structured collectors (sanctions, РНУ) behind one `Collector` interface |
| `deepsearch.py` | the Claude agent loop: server tools, `pause_turn`, refusals, tool-input validation, rounds over the graph |
| `graph.py` | entity graph: nodes, relations with source URLs, frontier, relation paths, Mermaid |
| `verify.py` | page fetch, main-text extraction, exact + fuzzy quote matching, Wayback fallback |
| `matching.py` | identity features (for and against) and score; every feature vector is logged for training |
| `reliability.py` | source tiers and recency decay |
| `scoring.py` | grouping findings into facts, corroboration, risk |
| `pipeline.py` | the stages in order |
| `eval_synth.py`, `evaluate.py`, `evals/` | namesake evaluation sets and harness |
| `cli.py`, `main.py` | command line and FastAPI |

## OSINT in this project

OSINT here is a method of working with open sources, not a library:

- **Pivoting over an entity graph.** A lead is not just a name to search next.
  It is a node (company, person, case) linked to the entity it was found next
  to, with the relation (директор, учредитель, дочерняя компания) and the
  source URL on the edge. Each round explores the frontier: unexplored nodes
  within `MAX_DEPTH` hops, the best-supported first. A fact about the director
  is reported as the director's, with its path ("Субъект → директор: …"), and
  counts at half weight.
- **Search operators (dorking).** The agent combines every spelling of the name
  with risk words (суд, мошенничество, банкротство…), in three languages, and
  targets Kazakhstani court, tax and procurement sites and news media.
- **Cross-verification.** Articles about one event are grouped into a fact by
  event key, case number or near-identical claim. Independent sources (distinct
  domains) combine; one story syndicated five times is still one source. The
  report says "confirmed by N independent sources".
- **Source reliability and recency.** Registry and government sources weigh
  1.0, major media 0.9, other media 0.7, social media and forums 0.4. Stories
  lose weight with age (half-life 5 years, floor 0.3); current register
  entries don't age.
- **Archives.** Removed or edited news is a typical adverse-media pattern. A
  quote missing from the live page is looked up in the Wayback Machine. If it is
  found there, the finding is marked verified from the archive, with a link.

What is deliberately **not** here: account and email enumeration (holehe,
maigret, sherlock, theHarvester), scraping Telegram "blacklists", bypassing
captchas. They profile a person rather than find adverse facts, conflict with
the privacy rules below, and would not survive a compliance review.

## Design decisions

**The model's claims are data, not truth.** The agent must give a verbatim
quote for every finding, and the quote is checked against the page we fetch
ourselves. A fabricated quote does not disappear silently: it stays in the
report marked "⚠ цитата не найдена", with a 0.15 weight, and it does not count
as a confirmation.

**Status is a first-class field.** "The director was detained" (investigation),
"the court convicted" (convicted) and "the case was dropped" (closed) are
different risks. The status is part of the score: an acquittal contributes 0.

**Namesakes are the main false-positive source.** A surname alone scores 0.25,
surname + initial 0.45 (below the threshold), surname + given name 0.8. Context
(БИН in the text, city, a related company, birth year) lifts the score; a
contradicting БИН/ИИН, city, age or patronymic near the name caps it. Findings
below the threshold are shown separately for a human to review instead of being
thrown away.

**A transparent baseline before a model.** The identity matcher is a heuristic
on purpose. Every feature vector goes to `data/match_features.jsonl` with an
empty `label`. Once reviewers label findings as same / different entity, that
log is the training set for a learned matcher, evaluated against this baseline.

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
  ethnicity, address, phone) even when it sees it. Graph pivots are limited to
  official roles (directors, founders, subsidiaries, cases).
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

Tests run offline. Claude is replaced by a scripted stub; page fetches and the
Wayback API go through `httpx.MockTransport`:

```bash
pytest
```

## Evaluation: namesakes

`matching.py` decides whether a source is about the subject or about a namesake.

- **Synthetic set** (`eval_synth.py`): 700 cases from 28 trap types, fixed seed,
  split into dev and test by subject. Namesakes: a full namesake from another
  city or with another birth year, the same full name with a different ИИН, a
  different patronymic, a surname one letter off, the same ТОО name with another
  БИН, an extra word in the company name. Hard positives: initials + context,
  Latin spelling, Kazakh patronymic (-ұлы / -қызы), a surname with a role at a
  known company, ИИН in a debtor list.
- **Fresh draw**: the same generator with seed 101, not looked at while writing
  the rules.
- **Golden set** (`evals/golden_identity.jsonl`): 20 hand-written cases with what
  templates don't produce: inflected names, the city as an adjective, English
  and Kazakh articles, a short common surname (Ким), an unreachable page.
- **Holdout set** (`evals/holdout_identity.jsonl`): 18 hand-written cases written
  *after* the v2 rules and never used to change them. It includes traps for the
  new features themselves: a subject arrested in another city, co-defendants'
  ИИН next to the subject's name, a dated article with an age.

```bash
python eval_synth.py      # data/eval/identity_synth.jsonl
python evaluate.py        # prints and writes evals/RESULTS.md
```

The threshold is tuned on dev only. All sets are fictional, so they test the
matcher's logic, not its real-world accuracy. Real accuracy comes from reviewer
labels in `data/match_features.jsonl` (`python evaluate.py --match-log ...`).

**Results** at threshold 0.5 (full tables, per-trap breakdown and every error in
[evals/RESULTS.md](evals/RESULTS.md)):

| matcher | set | precision | recall | namesake leak | AUC |
|---|---|---|---|---|---|
| v1 (only evidence for, no morphology) | test | 0.65 | 0.96 | 0.51 | 0.75 |
| v1 | golden | 0.59 | 0.91 | 0.78 | 0.81 |
| v2 without evidence against (ablation) | test | 0.71 | 1.00 | 0.41 | 0.79 |
| **v2** | test | 1.00 | 1.00 | **0.00** | 1.00 |
| **v2** | fresh seed 101 | 1.00 | 1.00 | **0.003** | 1.00 |
| **v2** | golden | 1.00 | 1.00 | **0.00** | 1.00 |
| **v2** | holdout | 1.00 | 0.90 | **0.00** | 0.91 |

v1 numbers are from the commit that introduced the eval. The ablation shows
where the gain comes from. Morphology (pymorphy3 lemmas, the city dictionary)
and the stricter surname bar lift recall to 1.0. The evidence-against features
bring namesake leak from 0.41 to 0. v1 only collected evidence for a match, so a
full namesake from another city scored 1.0.

**Read the perfect scores with suspicion.** The rules were written after
reading v1's errors on these trap types, and the synthetic sets come from the
same generator. The holdout set is the most honest number, and it still has a
real error. The known failures:

- **h-03, co-defendants' ИИН**: "Толеубаев С.М., Касымов Е.Н. (ИИН …)". Another
  person's ИИН sits next to the subject's name and is taken as a contradiction.
  Fix: attribute each ИИН to the nearest name instead of the whole window.
- **fresh seed, «Ghalym»**: Ғ romanised as "gh" is not folded to "g".
- **fresh seed, surname + related company + city** adds up to exactly 0.5 for a
  different given name. The additive score has no notion that "another given
  name" is itself a contradiction.

Next: fix these without touching the holdout, then a learned matcher on real
reviewer labels, compared with this baseline.

## Cost

Web search is billed at $10 per 1,000 searches, plus tokens (Opus 5.5: $4 / $20
per million input / output tokens). Search results are large, and the history
is resent on every agent turn. Prompt caching is on, but tokens still dominate.
The per-screening cost has **not been measured yet**. Measure it on the first
real runs (`usage` on each response) before choosing rounds, search budget and
effort. `MAX_ROUNDS`, `SEARCHES_PER_ROUND`, `FETCHES_PER_ROUND`,
`PIVOTS_PER_ROUND`, `MAX_DEPTH` and `EFFORT` are the levers.

## Roadmap

- **Matcher fixes** from the holdout errors above: ИИН attributed to the nearest
  name, Ғ → gh, another given name as a contradiction.
- **End-to-end evaluation set.** Real labelled subjects: positives from
  OpenSanctions (KZ) and the РНУ, negatives from random registered companies.
  Metrics: finding precision / recall, fabricated-quote rate, fact grouping
  accuracy, cost and latency per screening.
- **Learned identity matcher** trained on the labelled `match_features.jsonl`,
  compared with the heuristic baseline.
- **Adverse-media classifier** (fine-tuned rubert-tiny2) against Claude
  zero-shot on category / status, for a cheap pre-filter.
- **More Kazakhstani sources**: КГД lists (inactive taxpayers, sham companies,
  tax debtors), the АФМ list of persons linked to terrorism financing, debtor
  register (АИС ОИП), court acts from sud.kz, the state business register (to
  seed the graph with official directors and founders).
- **Review workflow**: analyst confirms or rejects each finding in a UI; the
  decisions become matcher labels (human-in-the-loop, LangGraph interrupt).
