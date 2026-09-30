import json
import math
from collections import defaultdict

import eval_synth
import evaluate
from evaluate import GOLDEN_PATH, auc, average_precision, confusion, load, tune
from models import Finding, Subject


def test_confusion_and_leak():
    m = confusion([1, 1, 0, 0], [0.9, 0.2, 0.7, 0.1], thr=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["leak"] == 0.5


def test_auc_and_ap():
    assert auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auc([1, 0], [0.5, 0.5]) == 0.5  # ties count half
    assert math.isnan(auc([1, 1], [0.1, 0.2]))
    assert average_precision([1, 0, 1], [0.9, 0.8, 0.1]) == (1 + 2 / 3) / 2


def test_tune_picks_thresholds_from_dev():
    labels, scores = [1, 1, 1, 0, 0], [0.9, 0.8, 0.4, 0.6, 0.1]
    t = tune(labels, scores)
    assert t["recall95"] == 0.4  # highest threshold keeping every positive
    assert confusion(labels, scores, t["best_f1"])["f1"] >= confusion(labels, scores, 0.5)["f1"]


def test_generator_is_deterministic_and_balanced():
    a, b = eval_synth.generate(per_trap=3, seed=5), eval_synth.generate(per_trap=3, seed=5)
    assert a == b
    traps = len(eval_synth.PERSON_TRAPS) + len(eval_synth.COMPANY_TRAPS)
    assert len(a) == 3 * traps
    assert {c["label"] for c in a} == {0, 1}


def test_split_is_by_subject():
    cases = eval_synth.generate(per_trap=6, seed=1)
    splits = defaultdict(set)
    for c in cases:
        splits[json.dumps(c["subject"], sort_keys=True)].add(c["split"])
    assert all(len(s) == 1 for s in splits.values())


def test_every_case_builds_domain_objects():
    for c in eval_synth.generate(per_trap=2, seed=3) + load(GOLDEN_PATH):
        Subject(**c["subject"])
        f = Finding(**c["finding"])
        assert f.subject_mentioned_as
        assert c["label"] in (0, 1)


def test_namesake_traps_really_conflict():
    """Generated namesakes must carry the contradicting detail the trap is named after."""
    for c in eval_synth.generate(per_trap=4, seed=2):
        text = c["page_text"] or ""
        if c["trap"] == "n_full_other_iin" and c["page_text"]:
            assert c["subject"]["identifier"] not in text
        if c["trap"] == "n_c_other_bin" and c["page_text"]:
            assert "БИН" in text and c["subject"]["identifier"] not in text
        if c["trap"] == "n_full_other_city_age" and c["page_text"]:
            assert c["subject"]["birth_date"][:4] not in text


def test_report_renders(tmp_path):
    synth = eval_synth.generate(per_trap=2, seed=4)
    report = evaluate.evaluate_sets(synth, load(GOLDEN_PATH))
    assert "## Per trap type" in report and "heuristic / golden" in report


def test_match_log_eval(tmp_path):
    row = {"name_score": 0.8, "name_reason": "r", "id_in_text": False, "registry_id_lookup": False,
           "city_in_text": True, "related_hits": 0, "birth_year_in_text": False, "short_company_name": False,
           "llm_evidence_count": 0, "page_fetched": True, "subject_type": "person"}
    log = tmp_path / "m.jsonl"
    log.write_text("\n".join(json.dumps({**row, "label": lab}) for lab in (1, 0, None)), encoding="utf-8")
    out = evaluate.evaluate_match_log(log)
    assert "| heuristic | 2 |" in out
