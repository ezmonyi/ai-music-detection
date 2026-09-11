#!/usr/bin/env python3
"""Independently check final score/table consistency without importing the fitter."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yes(value):
    return str(value).lower() in {"1", "true"}


def choose(candidates):
    highest = max(float(row["selection_score"]) for row in candidates)
    tied = [row for row in candidates if highest - float(row["selection_score"]) <= 1e-12]
    return min(tied, key=lambda row: (len(row["combination"].split("+")), row["combination"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--latex-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    formal_path = args.results_root / "evaluate_expanded_formal_orchestration_audit.json"
    formal = json.loads(formal_path.read_text())
    if formal.get("status") != "formal_complete":
        raise ValueError("Final independent audit requires completed formal scoring and reporting")
    inputs = {str(formal_path.resolve()): digest(formal_path)}
    issues, details = [], {}
    combinations = {"+".join(parts) for size in range(1, 5)
                    for parts in itertools.combinations("SDRP", size)}
    quantities = ("25", "50", "100", "200", "all")
    for duration in ("10s", "30s"):
        base = args.results_root / duration
        frozen_path = base / "cv/evaluate_expanded_frozen_selection.json"
        summary_path = base / "cv/evaluate_expanded_cv_summary.csv"
        source_path = base / "locked/evaluate_expanded_locked_source_metrics.csv"
        matrix_path = args.latex_root / f"results_{duration}_j_matrix.tex"
        ba_matrix_path = args.latex_root / f"results_{duration}_ba_matrix.tex"
        for path in (frozen_path, summary_path, source_path, matrix_path, ba_matrix_path):
            inputs[str(path.resolve())] = digest(path)
        frozen = json.loads(frozen_path.read_text())
        primary = frozen["primary_feature_set"]
        table = [row for row in rows(summary_path) if row["feature_set"] == primary
                 and row["training_scope"] == "all_development"]
        keyed = {(row["combination"], row["quantity"]): row for row in table}
        if len(table) != 75 or set(keyed) != set(itertools.product(combinations, quantities)):
            issues.append(f"{duration}: primary grid is not exactly 15 by 5")
        for row in table:
            human, ai = float(row["human_holdout_roc_auc"]), float(row["generator_holdout_roc_auc"])
            expected = (human + ai) / 2
            score = float(row["selection_score"])
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (human, ai, score)):
                issues.append(f"{duration}: nonfinite/out-of-range AUC")
            elif not math.isclose(score, expected, rel_tol=0, abs_tol=1e-12):
                issues.append(f"{duration}: selection J differs from equal source-transfer AUC")
        candidates = [row for row in table if row["quantity"] == "all" and yes(row["qualifies_for_selection"])]
        leader = choose(candidates)
        single = choose([row for row in candidates if "+" not in row["combination"]])
        for key, selected in (("leader", leader), ("best_single", single)):
            if frozen[key]["combination"] != selected["combination"] or not math.isclose(
                    float(frozen[key]["selection_score"]), float(selected["selection_score"]),
                    rel_tol=0, abs_tol=1e-12):
                issues.append(f"{duration}: frozen {key} does not follow prespecified selection")
        if frozen["threshold"] != 0.5 or frozen["ridge"] != 10:
            issues.append(f"{duration}: fixed classifier parameters changed")
        if frozen["source_column"] != "source_group" or frozen["group_column"] != "group_id":
            issues.append(f"{duration}: incorrect source/group units")
        matrix = matrix_path.read_text()
        ba_matrix = ba_matrix_path.read_text()
        for combination in combinations:
            expected_cells = [f"{float(keyed[(combination, q)]['selection_score']):.3f}" for q in quantities]
            expected_line = " & ".join([combination, *expected_cells]) + r" \\"
            if expected_line not in matrix:
                issues.append(f"{duration}: LaTeX J matrix differs from AUC for {combination}")
            ba_cells = []
            for quantity in quantities:
                record = keyed[(combination, quantity)]
                human_ba = float(record["human_holdout_balanced_accuracy"])
                ai_ba = float(record["generator_holdout_balanced_accuracy"])
                if not all(math.isfinite(value) and 0 <= value <= 1 for value in (human_ba, ai_ba)):
                    issues.append(f"{duration}: nonfinite/out-of-range source-transfer balanced accuracy")
                ba_cells.append(f"{(human_ba + ai_ba) / 2:.3f}")
            if " & ".join([combination, *ba_cells]) + r" \\" not in ba_matrix:
                issues.append(f"{duration}: LaTeX balanced-accuracy matrix differs from CV for {combination}")
        source_rows = rows(source_path)
        provisional = [row for row in source_rows if row["source"] == "ai_audiox_third_party"]
        metadata_path = Path(frozen["input_files_cv"]["metadata"]["path"])
        inputs[str(metadata_path.resolve())] = digest(metadata_path)
        provisional_expected = any(row["source_group"] == "ai_audiox_third_party"
                                   for row in rows(metadata_path))
        if bool(provisional) != provisional_expected or any(int(row["known_label_rows"]) != 0 for row in provisional):
            issues.append(f"{duration}: provisional AudioX labels were treated as verified or omitted")
        for row in provisional:
            for metric in ("accuracy", "balanced_accuracy", "roc_auc", "ai_sensitivity", "human_specificity"):
                if row.get(metric, "").strip() not in {"", "nan", "NaN"}:
                    issues.append(f"{duration}: provisional AudioX has a claimed {metric}")
        details[duration] = {"primary_grid_cells": len(table), "primary_feature_set": primary,
                             "leader": frozen["leader"], "best_single": frozen["best_single"],
                             "descriptive_best_by_quantity": {
                                 quantity: {
                                     key: choose([row for row in table if row["quantity"] == quantity
                                                  and yes(row["qualifies_for_selection"])])[key]
                                     for key in ("combination", "selection_score", "human_holdout_roc_auc",
                                                 "generator_holdout_roc_auc", "human_holdout_balanced_accuracy",
                                                 "generator_holdout_balanced_accuracy")
                                 } for quantity in quantities
                             },
                             "quantity_ranking_policy": "Descriptive only; it cannot replace the prespecified all-data frozen leader.",
                             "provisional_present_in_duration_cohort": provisional_expected,
                             "provisional_source_score_rows": len(provisional)}
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "status": "passed" if not issues else "failed", "issues": sorted(set(issues)),
              "input_sha256": inputs, "durations": details,
              "scope": "Independent numerical, selection, table and provisional-label consistency; "
                       "not a replacement for physical materialization or fit/leakage audits."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
