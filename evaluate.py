"""Identity-resolution evaluation: does the matcher tell the subject apart from namesakes?

Positive = the source is about the subject (the finding should count in the risk
score). Negative = a namesake (it must go to "possible namesakes").

  recall          share of true findings about the subject that pass the threshold
  namesake leak   share of namesake findings that pass (false-positive rate): each leak
                  puts someone else's criminal case into the subject's risk score
  AUC / AP        threshold-free ranking quality

Matchers compared: `name_only` (name similarity alone, the naive baseline) and
`heuristic` (matching.py: name + БИН/ИИН in text + city + related names + birth year).

The threshold is tuned on dev only and reported on test; the golden set is a
separate hand-written test. Both are synthetic: they check the matcher's logic,
not real-world accuracy. For that, label real findings in data/match_features.jsonl
(label 1 / 0) and run with --match-log.

  python evaluate.py                      # synth (generated if missing) + golden -> evals/RESULTS.md
  python evaluate.py --no-write
  python evaluate.py --match-log data/match_features.jsonl
"""
import argparse
import json
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

from config import settings
from matching import MatchFeatures, features, score
from models import Finding, Subject
from names import name_similarity

SYNTH_PATH = Path(settings.DATA_DIR) / "eval" / "identity_synth.jsonl"
GOLDEN_PATH = Path("evals") / "golden_identity.jsonl"
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


MATCHERS = {"name_only": name_only, "heuristic": heuristic}


# --- metrics (pure python: the set is small) -----------------------------------------------------

def confusion(labels: list[int], scores: list[float], thr: float) -> dict:
    tp = sum(1 for y, s in zip(labels, scores) if y == 1 and s >= thr)
    fn = sum(1 for y, s in zip(labels, scores) if y == 1 and s < thr)
    fp = sum(1 for y, s in zip(labels, scores) if y == 0 and s >= thr)
    tn = sum(1 for y, s in zip(labels, scores) if y == 0 and s < thr)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": len(labels), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "leak": fp / (fp + tn) if fp + tn else 0.0,
            "accuracy": (tp + tn) / len(labels) if labels else 0.0}


def auc(labels: list[int], scores: list[float]) -> float:
    """Probability that a random positive outranks a random negative (ties count half)."""
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def average_precision(labels: list[int], scores: list[float]) -> float:
    ranked = sorted(zip(scores, labels), key=lambda t: -t[0])
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

        def correct(name: str) -> float:
            return sum(1 for i in idx if (results[name][i][0] >= thr) == (label == 1)) / len(idx)

        mean = sum(results["heuristic"][i][0] for i in idx) / len(idx)
        rows.append((correct("heuristic"), trap, label, len(idx), correct("name_only"), mean))
    for acc, trap, label, n, acc_name, mean in sorted(rows):
        kind = "same" if label == 1 else "namesake"
        lines.append(f"| {trap} | {kind} | {n} | {acc_name:.0%} | {acc:.0%} | {mean:.2f} |")
    return lines


def error_lines(cases: list[dict], results: dict, thr: float, limit: int = 12) -> list[str]:
    errs = []
    for c, (s, reasons) in zip(cases, results["heuristic"]):
        if (s >= thr) != (c["label"] == 1):
            errs.append((abs(s - thr), c, s, reasons))
    errs.sort(key=lambda t: -t[0])
    lines = ["| case | trap | truth | score | mention | reasons |", "|---|---|---|---|---|---|"]
    for _, c, s, reasons in errs[:limit]:
        truth = "same" if c["label"] == 1 else "namesake"
        mention = c["finding"]["subject_mentioned_as"].replace("|", "/")
        lines.append(f"| {c['id']} | {c['trap']} | {truth} | {s:.2f} | {mention} | {'; '.join(reasons)} |")
    return lines


def evaluate_sets(synth: list[dict], golden: list[dict]) -> str:
    thr = settings.MATCH_THRESHOLD
    dev = [c for c in synth if c["split"] == "dev"]
    test = [c for c in synth if c["split"] == "test"]
    res_dev, res_test, res_gold = run(dev), run(test), run(golden)

    def labels(cs):
        return [c["label"] for c in cs]

    def scores(res, name):
        return [s for s, _ in res[name]]

    out = ["# Identity resolution: evaluation results", "",
           "Generated by `python evaluate.py`. Positive = the source is about the subject; negative = a "
           "namesake. *Namesake leak* = share of namesakes that pass the threshold and would enter the risk "
           "score.", "",
           f"Synthetic set: {len(dev)} dev / {len(test)} test cases, "
           f"{sum(labels(synth))} same / {len(synth) - sum(labels(synth))} namesake "
           f"(`python eval_synth.py`, fixed seed). Golden set: {len(golden)} hand-written cases "
           "(`evals/golden_identity.jsonl`). Both are synthetic: they test the matcher's logic, not "
           "real-world accuracy.", "",
           f"## At the configured threshold (MATCH_THRESHOLD = {thr})", "", HEADER]
    for name in MATCHERS:
        out.append(summary_row(f"{name} / test", labels(test), scores(res_test, name), thr))
        out.append(summary_row(f"{name} / golden", labels(golden), scores(res_gold, name), thr))

    out += ["", "## Threshold tuned on dev, reported on test and golden", "", HEADER]
    for name in MATCHERS:
        t = tune(labels(dev), scores(res_dev, name))
        for tag, value in [("best F1", t["best_f1"]), ("recall ≥ 0.95", t["recall95"])]:
            out.append(summary_row(f"{name} / test ({tag} on dev)", labels(test), scores(res_test, name), value))
            out.append(summary_row(f"{name} / golden ({tag} on dev)", labels(golden), scores(res_gold, name),
                                   value))

    out += ["", f"## Per trap type (test split, threshold {thr})", ""]
    out += per_trap_table(test, res_test, thr)

    out += ["", f"## Golden set, case by case (threshold {thr})", "",
            "| case | trap | truth | name_only | heuristic | verdict |", "|---|---|---|---|---|---|"]
    for c, (sn, _), (sh, _) in zip(golden, res_gold["name_only"], res_gold["heuristic"]):
        ok = (sh >= thr) == (c["label"] == 1)
        out.append(f"| {c['id']} | {c['trap']} | {'same' if c['label'] else 'namesake'} | {sn:.2f} | {sh:.2f} "
                   f"| {'ok' if ok else '**error**'} |")

    out += ["", "## Largest heuristic errors (test split)", ""]
    out += error_lines(test, res_test, thr)
    out += ["", "## Largest heuristic errors (golden)", ""]
    out += error_lines(golden, res_gold, thr)
    return "\n".join(out) + "\n"


def evaluate_match_log(path: Path) -> str:
    """Real findings labelled by reviewers: recompute the scores from the logged features."""
    names = {f.name for f in fields(MatchFeatures)}
    rows = [r for r in load(path) if r.get("label") in (0, 1)]
    if not rows:
        return f"No labelled rows in {path}: set \"label\" to 1 (same entity) or 0 (namesake)."
    labels = [int(r["label"]) for r in rows]
    heur = [score(MatchFeatures(**{k: r[k] for k in names}), SimpleNamespace(type=r["subject_type"]))[0]
            for r in rows]
    name = [r["name_score"] for r in rows]
    thr = settings.MATCH_THRESHOLD
    return "\n".join([f"# Labelled findings from {path}", "", HEADER,
                      summary_row("name_only", labels, name, thr), summary_row("heuristic", labels, heur, thr)])


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Identity-resolution evaluation")
    p.add_argument("--synth", default=str(SYNTH_PATH))
    p.add_argument("--golden", default=str(GOLDEN_PATH))
    p.add_argument("--match-log", help="evaluate on reviewer-labelled rows of match_features.jsonl instead")
    p.add_argument("--no-write", action="store_true", help=f"print only, do not update {RESULTS_PATH}")
    args = p.parse_args()

    if args.match_log:
        print(evaluate_match_log(Path(args.match_log)))
        return
    synth_path = Path(args.synth)
    if not synth_path.exists():
        import eval_synth

        synth_path.parent.mkdir(parents=True, exist_ok=True)
        with synth_path.open("w", encoding="utf-8") as f:
            for c in eval_synth.generate():
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    report = evaluate_sets(load(synth_path), load(Path(args.golden)))
    print(report)
    if not args.no_write:
        RESULTS_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
