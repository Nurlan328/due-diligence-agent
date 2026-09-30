# Due Diligence Agent

[![CI](https://github.com/Nurlan328/due-diligence-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Nurlan328/due-diligence-agent/actions/workflows/ci.yml)

**[Sample report](docs/sample_report.md)** (fictional data) · **`python cli.py demo`** runs it locally
without an API key · **[Evaluation results](evals/RESULTS.md)**

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
  └─ 6. Report + audit trail       report.py, audit.py, store.py
        Markdown / JSON, entity graph (Mermaid), sources checked with no hits,
        cost and time of the screening; audit trail and match features in SQLite.
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
| `store.py` | SQLite: audit trail, identity features with reviewer labels, API jobs |
| `eval_synth.py`, `evaluate.py`, `evals/` | namesake evaluation sets and harness |
| `cli.py`, `main.py` | command line (screen, demo, label) and FastAPI (async jobs) |

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
on purpose. Every feature vector is stored with an empty label, and each source
in the report shows its finding `id`. A reviewer records the decision with
`python cli.py label <id> same|namesake`, and `python evaluate.py --labelled`
scores the matcher on the labelled real findings. Those rows are the training
set for a learned matcher, evaluated against this baseline.

## Privacy and legal basis

Screening a person is processing personal data (Закон РК «О персональных данных
и их защите»). Banks and other subjects of financial monitoring do it under the
AML law (Закон РК № 191-IV о ПОД/ФТ, due diligence on clients). The tool
enforces the following:

- `purpose` is mandatory and every screening is written to the audit trail
  (SQLite, `data/dd.sqlite`): operator, purpose, subject, the queries sent out,
  cost.
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
pip install -r requirements.txt              # pinned lock file
pip install -r requirements-dev.txt          # pytest, ruff, mypy

# try it without an API key: a recorded fictional run through the real pipeline
python cli.py demo                # -> reports/sample_report.md

copy .env.example .env            # set ANTHROPIC_API_KEY (and GOSZAKUP_TOKEN if you have one)

python cli.py update-sanctions    # ~70 MB OpenSanctions export into data/

# registries only, no API key needed
python cli.py screen --type company --name "ТОО «Ромашка Строй»" --id 123456789012 \
    --purpose "Проверка поставщика перед заключением договора" --no-deep

# full screening with deep search
python cli.py screen --type person --name "Фамилия Имя Отчество" --birth-date 1980-05-01 \
    --city Алматы --related "ТОО Ромашка Строй" --purpose "KYC при открытии счёта"
```

Reports are written to `reports/<request_id>.md` and `.json`. Each ends with
the screening's time and cost (API requests, tokens, web searches, pages opened).

**Reviewer labels.** Each source in a report shows its finding `id`:

```bash
python cli.py label 3f2a9c1d0b7e namesake    # or: same
python evaluate.py --labelled                # the matcher on labelled real findings
```

**HTTP API** (`fastapi dev main.py`, docs at `/docs`). A screening takes minutes,
so the main entry point is asynchronous:

```bash
curl -X POST localhost:8000/jobs -H "Content-Type: application/json" -d @request.json
# -> 202 {"job_id": "…", "status": "queued"}
curl localhost:8000/jobs/<job_id>
# -> {"status": "queued | running | done | failed", "report": {…}}
```

Jobs run on a thread pool in the API process (`JOB_WORKERS`), their state is in
SQLite, and jobs left unfinished by a restart are marked failed. `POST /screen`
(blocking) stays for scripts. Scaling out means a queue (Redis / RabbitMQ) for
the pool and Postgres for the store; only `store.py` touches SQL.

**Docker:**

```bash
docker build -t due-diligence-agent .
docker run -p 8000:8000 --env-file .env -v dd-data:/app/data due-diligence-agent
```

The image installs the pinned lock file, runs as an unprivileged user, has a
health check, and keeps personal data out: `.env`, `data/` and `reports/` are
in `.dockerignore`. The volume holds the SQLite store and the sanctions export.

**Development:**

```bash
pytest            # offline: Claude is a scripted stub, pages and Wayback go through httpx.MockTransport
ruff check .
mypy
# after changing requirements.in:
uv pip compile --universal --python-version 3.12 requirements.in -o requirements.txt
```

## Demo

`python cli.py demo` replays [demo/scenario.json](demo/scenario.json), a recorded
screening of a fictional company. The agent's turns (searches, findings,
leads), the web pages, the Wayback Machine and the registries are replayed.
Everything else is the production code: quote verification, identity matching,
the entity graph, fact grouping, scoring, the report. The scenario shows each
stage once:

| in the [sample report](docs/sample_report.md) | stage |
|---|---|
| money-laundering story confirmed by 2 outlets, one fact | fact grouping, corroboration |
| wage-arrears article gone from the site, found in the archive | Wayback fallback |
| "обманула 300 дольщиков" from a blog, quote not on the page | faithfulness check, low-tier source |
| bankrupt ТОО with the same name but another БИН, in Шымкент | namesake filter (evidence against) |
| director's tax conviction, founder's cartel fine | graph pivots, affiliate facts at half weight |
| РНУ entry found by БИН | structured sources |

All names, identifiers, articles and sites in it are invented; the domains use
the reserved `.example` TLD. `python cli.py demo --out docs` regenerates the
committed sample.

## CI

[GitHub Actions](.github/workflows/ci.yml), on Python 3.12 and 3.14 from the
pinned lock files: ruff, mypy, the test suite and the demo. Then it checks that
the evaluation is **reproducible**: it regenerates the synthetic set and
`evals/RESULTS.md` and fails if the committed results differ, so metrics in this
README can't drift from the code without CI turning red. A second job builds the
Docker image, starts it and runs a background job through the API end to end.
The demo report is uploaded as a build artifact.

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
- **Holdout sets**, hand-written, each written *before* the matcher version it
  tests and never used to tune that version:
  - `evals/holdout_identity.jsonl` (18 cases) tested v2. Its errors then drove v3,
    so for v3 it is marked *used* and no longer independent.
  - `evals/holdout2_identity.jsonl` (16 cases) was written before the v3 changes.
    It includes traps for the new rules themselves: co-defendants' ИИН, a
    contractor's БИН next to the subject, another given name in the same
    company, «Ghalym», and a partner's ИИН introduced in words.

```bash
python eval_synth.py      # data/eval/identity_synth.jsonl
python evaluate.py        # prints and writes evals/RESULTS.md
```

The threshold is tuned on dev only. All sets are fictional, so they test the
matcher's logic, not its real-world accuracy. Real accuracy comes from reviewer
labels (`python cli.py label`, then `python evaluate.py --labelled`).

**Results** at threshold 0.5 (full tables, per-trap breakdown and every error in
[evals/RESULTS.md](evals/RESULTS.md)):

| matcher | set | precision | recall | namesake leak | AUC |
|---|---|---|---|---|---|
| v1 (only evidence for, no morphology) | test | 0.65 | 0.96 | 0.51 | 0.75 |
| v1 | golden | 0.59 | 0.91 | 0.78 | 0.81 |
| v2 | holdout (then unseen) | 1.00 | 0.90 | 0.00 | 0.91 |
| v3 without evidence against (ablation) | test | 0.73 | 1.00 | 0.39 | 0.79 |
| **v3** | test | 1.00 | 0.99 | **0.00** | 1.00 |
| **v3** | fresh seed 101 | 1.00 | 0.99 | **0.00** | 0.99 |
| **v3** | golden | 1.00 | 1.00 | **0.00** | 1.00 |
| **v3** | **holdout2 (unseen)** | 1.00 | 0.89 | **0.00** | 0.91 |

v1 numbers are from the commit that introduced the eval; v2 from the commit
after it. What each version added:

- **v2**: morphology (pymorphy3 lemmas, the city dictionary) and a stricter
  surname bar lift recall. Evidence against (another БИН/ИИН, city, age,
  patronymic near the name) brings namesake leak from 0.41 (ablation) to 0. v1
  only collected evidence for a match, so a full namesake from another city
  scored 1.0.
- **v3** fixes what v2's holdout and fresh draw exposed. A БИН/ИИН now belongs to
  the nearest entity before it (another surname or a legal form starts a new
  entity), so co-defendants' or a contractor's numbers no longer count against
  the subject. Another spelled-out given name is a contradiction. Ғ = gh. These
  failure types were added to the generator and fixed on dev only.

**Read the perfect scores with suspicion.** The rules were written after
reading earlier errors on these trap types, and the synthetic sets come from
the same generator. The unseen hand-written holdout is the honest number, and it
still has an error. The known failures:

- **h2-15, the owner of an ИИН named in words**: "Байжанов Е.К. и его партнёр по
  бизнесу, ИИН которого …". No surname stands between the subject and the
  number, so the ИИН is attributed to the subject.
- **fresh seed, co-defendant with the same surname**: "Султанов Б.А. (ИИН …),
  Султанов Е.А.". Attribution looks at the surname only. Fix: compare the
  initials after the surname too. Found on the test draw, so left unfixed here to
  keep that draw honest.

Next: a learned matcher on real reviewer labels, compared with this baseline.
Every new rule version needs a new holdout written before it.

## Cost

Web search is billed at $10 per 1,000 searches, plus tokens (Opus 5.5: $4 / $20
per million input / output tokens). Search results are large, and the history
is resent on every agent turn. Prompt caching is on, but tokens still dominate.
The per-screening cost has **not been measured yet**: the project has not run
against the live API. It is measured automatically once it does: every report
ends with the API requests, tokens (input, cached, output), web searches, pages
opened, the estimated cost (prices in `config.py`) and the wall-clock time, and
the same numbers go to the audit trail. Measure a handful of real screenings
before choosing rounds, search budget and effort. `MAX_ROUNDS`, `SEARCHES_PER_ROUND`, `FETCHES_PER_ROUND`,
`PIVOTS_PER_ROUND`, `MAX_DEPTH` and `EFFORT` are the levers.

## Roadmap

- **Matcher fixes** from the known failures above (initials in ИИН attribution,
  owners introduced in words), with a new holdout written first.
- **End-to-end evaluation set.** Real labelled subjects: positives from
  OpenSanctions (KZ) and the РНУ, negatives from random registered companies.
  Metrics: finding precision / recall, fabricated-quote rate, fact grouping
  accuracy, cost and latency per screening.
- **Learned identity matcher** trained on reviewer-labelled findings from the
  store, compared with the heuristic baseline.
- **Adverse-media classifier** (fine-tuned rubert-tiny2) against Claude
  zero-shot on category / status, for a cheap pre-filter.
- **More Kazakhstani sources**: КГД lists (inactive taxpayers, sham companies,
  tax debtors), the АФМ list of persons linked to terrorism financing, debtor
  register (АИС ОИП), court acts from sud.kz, the state business register (to
  seed the graph with official directors and founders).
- **Review workflow**: analyst confirms or rejects each finding in a UI; the
  decisions become matcher labels (human-in-the-loop, LangGraph interrupt).
