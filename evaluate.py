"""Identity-resolution evaluation: does the matcher tell the subject apart from namesakes?

Positive = the source is about the subject (the finding should count in the risk
score). Negative = a namesake (it must go to "possible namesakes").

  recall          share of true findings about the subject that pass the threshold
  namesake leak   share of namesake findings that pass (false-positive rate): each leak
                  puts someone else's criminal case into the subject's risk score
  AUC / AP        threshold-free ranking quality

Matchers compared: `name_only` (name similarity alone, the naive baseline),
`heuristic_for_only` (ablation: v2 without the evidence-against features) and
`heuristic` (matching.py).

The threshold is tuned on dev only. Reported on: the synthetic test split, a
second synthetic draw with another seed, the hand-written golden set and the
hand-written holdout sets: holdout (its errors drove v3, so no longer independent)
and holdout2 (written before v3, never used to change it).
All are synthetic: they check the matcher's logic, not real-world accuracy. For
that, reviewers label real findings (`python cli.py label <finding_id> same|namesake`)
and --labelled scores the matcher on them.

  python evaluate.py                      # synth (generated if missing) + golden + holdout -> evals/RESULTS.md
  python evaluate.py --no-write
  python evaluate.py --labelled [DB_PATH]  # reviewer-labelled real findings from the store
"""
import argparse
import json
from collections import defaultdict
from dataclasses import fields, replace
from datetime import date
from pathlib import Path

import store
from config import settings
from matching import MatchFeatures, features, score
from models import Finding, Subject, SubjectType
from names import name_similarity

SYNTH_PATH = Path(settings.DATA_DIR) / "eval" / "identity_synth.jsonl"
GOLDEN_PATH = Path("evals") / "golden_identity.jsonl"
HOLDOUT_PATH = Path("evals") / "holdout_identity.jsonl"
HOLDOUT2_PATH = Path("evals") / "holdout2_identity.jsonl"
FRESH_SEED = 101  # a second synthetic draw nobody looked at while writing the rules
# Ages in the cases ("48-летний") were written relative to this date.
EVAL_DATE = date(2026, 9, 30)
RESULTS_PATH = Path("evals") / "RESULTS.md"


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _objects(case: dict) -> tuple[Subject, Finding, str | None]:
    return Subject(**case["subject"]), Finding(**case["finding"]), case["page_text"]


def name_only(case: dict) -> tuple[float, list[str]]:
    subject, finding, _ = _objects(case)
    s, reason = name_similarity(subject, finding.subject_mentioned_as)
    return s, [reason]


def heuristic(case: dict) -> tuple[float, list[str]]:
    subject, finding, page = _objects(case)
    return score(features(subject, finding, page), subject)


def heuristic_for_only(case: dict) -> tuple[float, list[str]]:
    """Ablation: the v2 matcher with the evidence-against features switched off."""
    subject, finding, page = _objects(case)
    feat = replace(features(subject, finding, page), other_id_near=False, other_city_near=False,
                   birth_conflict=False, patronymic_conflict=False)
    return score(feat, subject)


MATCHERS = {"name_only": name_only, "heuristic_for_only": heuristic_for_only, "heuristic": heuristic}


# --- metrics (pure python: the set is small) -----------------------------------------------------

def confusion(labels: list[int], scores: list[float], thr: float) -> dict:
    tp = sum(1 for y, s in zip(labels, scores, strict=True) if y == 1 and s >= thr)
    fn = sum(1 for y, s in zip(labels, scores, strict=True) if y == 1 and s < thr)
    fp = sum(1 for y, s in zip(labels, scores, strict=True) if y == 0 and s >= thr)
    tn = sum(1 for y, s in zip(labels, scores, strict=True) if y == 0 and s < thr)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": len(labels), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "leak": fp / (fp + tn) if fp + tn else 0.0,
            "accuracy": (tp + tn) / len(labels) if labels else 0.0}


def auc(labels: list[int], scores: list[float]) -> float:
    """Probability that a random positive outranks a random negative (ties count half)."""
    pos = [s for y, s in zip(labels, scores, strict=True) if y == 1]
    neg = [s for y, s in zip(labels, scores, strict=True) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def average_precision(labels: list[int], scores: list[float]) -> float:
    ranked = sorted(zip(scores, labels, strict=True), key=lambda t: -t[0])
    hits, total, ap = 0, sum(labels), 0.0
    for i, (_, y) in enumerate(ranked, 1):
        if y == 1:
            hits += 1
            ap += hits / i
    return ap / total if total else float("nan")


def tune(labels: list[int], scores: list[float]) -> dict:
    """Thresholds chosen on dev: best F1, and the highest one that keeps recall >= 0.95."""
    candidates = sorted(set(scores) | {0.0, 1.01})
    best_f1 = max(candidates, key=lambda t: (confusion(labels, scores, t)["f1"], t))
    keep = [t for t in candidates if confusion(labels, scores, t)["recall"] >= 0.95]
    return {"best_f1": best_f1, "recall95": max(keep) if keep else 0.0}


# --- runs ----------------------------------------------------------------------------------------

def run(cases: list[dict]) -> dict[str, list[tuple[float, list[str]]]]:
    return {name: [fn(c) for c in cases] for name, fn in MATCHERS.items()}


def summary_row(label: str, labels, scores, thr) -> str:
    m = confusion(labels, scores, thr)
    return (f"| {label} | {m['n']} | {thr:.2f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m['leak']:.3f} | {auc(labels, scores):.3f} | {average_precision(labels, scores):.3f} |")


HEADER = ("| matcher / split | n | threshold | precision | recall | F1 | namesake leak | AUC | AP |\n"
          "|---|---|---|---|---|---|---|---|---|")


def per_trap_table(cases: list[dict], results: dict, thr: float) -> list[str]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(cases):
        groups[c["trap"]].append(i)
    lines = ["| trap | label | n | name_only correct | heuristic correct | heuristic mean score |",
             "|---|---|---|---|---|---|"]
    rows = []
    for trap, idx in groups.items():
        label = cases[idx[0]]["label"]

        def correct(name: str, idx: list[int] = idx, label: int = label) -> float:
            return sum(1 for i in idx if (results[name][i][0] >= thr) == (label == 1)) / len(idx)

        mean = sum(results["heuristic"][i][0] for i in idx) / len(idx)
        rows.append((correct("heuristic"), trap, label, len(idx), correct("name_only"), mean))
    for acc, trap, label, n, acc_name, mean in sorted(rows):
        kind = "same" if label == 1 else "namesake"
        lines.append(f"| {trap} | {kind} | {n} | {acc_name:.0%} | {acc:.0%} | {mean:.2f} |")
    return lines


def error_lines(cases: list[dict], results: dict, thr: float, limit: int = 12) -> list[str]:
    errs = []
    for c, (s, reasons) in zip(cases, results["heuristic"], strict=True):
        if (s >= thr) != (c["label"] == 1):
            errs.append((abs(s - thr), c, s, reasons))
    errs.sort(key=lambda t: -t[0])
    lines = ["| case | trap | truth | score | mention | reasons |", "|---|---|---|---|---|---|"]
    for _, c, s, reasons in errs[:limit]:
        truth = "same" if c["label"] == 1 else "namesake"
        mention = c["finding"]["subject_mentioned_as"].replace("|", "/")
        lines.append(f"| {c['id']} | {c['trap']} | {truth} | {s:.2f} | {mention} | {'; '.join(reasons)} |")
    return lines


def evaluate_sets(synth: list[dict], hand: dict[str, list[dict]], fresh: list[dict] | None = None) -> str:
    """synth: the dev/test synthetic set; hand: hand-written test sets by name; fresh: a second
    synthetic draw with another seed (reported like a test set)."""
    thr = settings.MATCH_THRESHOLD
    dev = [c for c in synth if c["split"] == "dev"]
    tests: dict[str, list[dict]] = {"test": [c for c in synth if c["split"] == "test"]}
    if fresh:
        tests[f"fresh seed {FRESH_SEED}"] = fresh
    tests.update(hand)
    res_dev = run(dev)
    results = {name: run(cases) for name, cases in tests.items()}

    def labels(cs):
        return [c["label"] for c in cs]

    def scores(res, name):
        return [s for s, _ in res[name]]

    sizes = ", ".join(f"{name}: {len(cases)}" for name, cases in tests.items())
    out = ["# Identity resolution: evaluation results", "",
           "Generated by `python evaluate.py`. Positive = the source is about the subject; negative = a "
           "namesake. *Namesake leak* = share of namesakes that pass the threshold and would enter the risk "
           "score.", "",
           f"Sets: dev {len(dev)} (threshold tuning, rule development), {sizes}. The synthetic sets come "
           "from `eval_synth.py`; golden and the holdouts are hand-written (`evals/*.jsonl`). *holdout (used)* was "
           "written after v2 and its errors drove v3, so it is no longer independent; *holdout2* was written "
           "before the v3 changes and never used to change them: it is the honest number. All are fictional: "
           "they test the matcher's logic, not real-world accuracy.", "",
           "Matchers: `name_only` (name similarity), `heuristic_for_only` (v2 with the evidence-against "
           "features switched off, an ablation), `heuristic` (matching.py, v3).", "",
           f"## At the configured threshold (MATCH_THRESHOLD = {thr})", "", HEADER]
    for name in MATCHERS:
        for set_name, cases in tests.items():
            out.append(summary_row(f"{name} / {set_name}", labels(cases), scores(results[set_name], name), thr))

    out += ["", "## Threshold tuned on dev, reported on the test sets", "", HEADER]
    for name in MATCHERS:
        t = tune(labels(dev), scores(res_dev, name))
        for tag, value in [("best F1", t["best_f1"]), ("recall ≥ 0.95", t["recall95"])]:
            for set_name, cases in tests.items():
                out.append(summary_row(f"{name} / {set_name} ({tag} on dev)", labels(cases),
                                       scores(results[set_name], name), value))

    out += ["", f"## Per trap type (test split, threshold {thr})", ""]
    out += per_trap_table(tests["test"], results["test"], thr)

    for set_name, cases in hand.items():
        res = results[set_name]
        out += ["", f"## {set_name.capitalize()} set, case by case (threshold {thr})", "",
                "| case | trap | truth | name_only | for_only | heuristic | verdict |",
                "|---|---|---|---|---|---|---|"]
        for c, (sn, _), (sf, _), (sh, _) in zip(cases, res["name_only"], res["heuristic_for_only"],
                                                res["heuristic"], strict=True):
            ok = (sh >= thr) == (c["label"] == 1)
            out.append(f"| {c['id']} | {c['trap']} | {'same' if c['label'] else 'namesake'} | {sn:.2f} "
                       f"| {sf:.2f} | {sh:.2f} | {'ok' if ok else '**error**'} |")

    for set_name, cases in tests.items():
        lines = error_lines(cases, results[set_name], thr)
        out += ["", f"## Heuristic errors ({set_name})", ""]
        out += lines if len(lines) > 2 else ["None."]
    return "\n".join(out) + "\n"


def _stub_subject(subject_type: str) -> Subject:
    """score() only reads the subject's type; logged rows carry nothing else."""
    return Subject(type=SubjectType(subject_type), name="logged subject", purpose="evaluation of logged features")


def evaluate_labelled(db_path: str | None = None) -> str:
    """Real findings labelled by reviewers: recompute the scores from the stored features, so a
    changed matcher is measured on the same findings."""
    names = {f.name for f in fields(MatchFeatures)}
    rows = store.match_rows(labelled_only=True, path=db_path)
    source = db_path or settings.DB_PATH
    if not rows:
        return f"No labelled findings in {source}: label them with `python cli.py label <finding_id> same|namesake`."
    labels = [int(r["label"]) for r in rows]
    heur = [score(MatchFeatures(**{k: r[k] for k in names if k in r}), _stub_subject(r["subject_type"]))[0]
            for r in rows]
    name = [r["name_score"] for r in rows]
    thr = settings.MATCH_THRESHOLD
    return "\n".join([f"# Reviewer-labelled findings from {source}", "", HEADER,
                      summary_row("name_only", labels, name, thr), summary_row("heuristic", labels, heur, thr)])


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Identity-resolution evaluation")
    p.add_argument("--synth", default=str(SYNTH_PATH))
    p.add_argument("--golden", default=str(GOLDEN_PATH))
    p.add_argument("--holdout", default=str(HOLDOUT_PATH))
    p.add_argument("--holdout2", default=str(HOLDOUT2_PATH))
    p.add_argument("--labelled", nargs="?", const="", metavar="DB_PATH",
                   help="evaluate on reviewer-labelled real findings in the store instead (default DB_PATH)")
    p.add_argument("--no-write", action="store_true", help=f"print only, do not update {RESULTS_PATH}")
    args = p.parse_args()
    settings.REFERENCE_DATE = settings.REFERENCE_DATE or EVAL_DATE

    if args.labelled is not None:
        print(evaluate_labelled(args.labelled or None))
        return
    import eval_synth

    synth_path = Path(args.synth)
    if not synth_path.exists():
        synth_path.parent.mkdir(parents=True, exist_ok=True)
        with synth_path.open("w", encoding="utf-8") as f:
            for c in eval_synth.generate():
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    hand = {"golden": load(Path(args.golden)), "holdout (used)": load(Path(args.holdout)),
            "holdout2": load(Path(args.holdout2))}
    report = evaluate_sets(load(synth_path), hand, fresh=eval_synth.generate(seed=FRESH_SEED))
    print(report)
    if not args.no_write:
        RESULTS_PATH.write_text(report, encoding="utf-8", newline="\n")  # same bytes on every OS (CI diffs it)


if __name__ == "__main__":
    main()
