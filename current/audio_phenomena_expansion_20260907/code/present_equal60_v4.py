#!/usr/bin/env python3
"""Export exhaustive schema-v4 exact60 tables after an independent passed audit.

This program only aggregates saved metrics.  It never fits, selects, scores,
or invents source-transfer J.  Real mode requires a current, successful,
non-synthetic audit from ``audit_equal60_v4_results.py``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


ORDER = ("S", "D", "R", "P", "F", "H", "M")
OLD_ORDER = ORDER[:4]
NEW_ORDER = ORDER[4:]
CAPS = ("25", "50", "100", "200", "all")
MEASURES = ("roc_auc", "balanced_accuracy", "ai_sensitivity", "human_specificity")
FOLD_TYPE = "ordinary_group_holdout_descriptive"
AUDITOR = Path(__file__).with_name("audit_equal60_v4_results.py")
REQUIRED_RESULTS = {
    "evaluation_plan.csv",
    "development_group_cv_metrics_by_source_pair.csv",
    "development_group_cv_pooled_fold_metrics.csv",
    "development_group_cv_matched_deltas.csv",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json_text(text, source):
    source = str(source)

    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, f"Duplicate JSON key {key}: {source}")
            output[key] = value
        return output

    def constant(value):
        raise ValueError(f"Nonfinite JSON value {value}: {source}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def strict_json(path):
    path = Path(path)
    return strict_json_text(path.read_text(), path)


def snapshot_inputs(results_dir, audit_path):
    """Capture every provenance input before semantic validation or table reads."""
    results_dir, audit_path = Path(results_dir), Path(audit_path)
    manifest_path = results_dir / "run_manifest.json"
    initial = (audit_path, manifest_path, AUDITOR)
    blobs = {}
    for path in initial:
        require(path.is_file() and not path.is_symlink(), f"Missing/non-regular provenance input: {path}")
        blobs[path.resolve()] = path.read_bytes()
    report = strict_json_text(blobs[audit_path.resolve()].decode(), audit_path)
    manifest = strict_json_text(blobs[manifest_path.resolve()].decode(), manifest_path)
    files = manifest.get("files_sha256")
    require(isinstance(files, dict), "Missing result hash manifest")
    for name in files:
        require(isinstance(name, str) and Path(name).name == name, "Unsafe result filename in manifest")
        path = results_dir / name
        require(path.is_file() and not path.is_symlink(), f"Missing/non-regular audited result: {name}")
        blobs[path.resolve()] = path.read_bytes()
    hashes = {str(path): hashlib.sha256(data).hexdigest() for path, data in blobs.items()}
    return report, manifest, hashes


def read_csv(path):
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        require(header and len(header) == len(set(header)), f"Missing/duplicate CSV header: {path}")
        rows = list(reader)
    require(all(len(row) == len(header) for row in rows), f"Malformed CSV row: {path}")
    return [dict(zip(header, row)) for row in rows], tuple(header)


def subsets(codes):
    return ["+".join(parts) for size in range(1, len(codes) + 1)
            for parts in itertools.combinations(codes, size)]


def combinations():
    return subsets(ORDER)


def comparisons():
    return [(f"incremental::{left}__plus__{right}", left, left + "+" + right)
            for left in subsets(OLD_ORDER) for right in subsets(NEW_ORDER)]


def quantity(value):
    require(value in CAPS, f"Unexpected cap: {value}")
    return value


def finite(row, name):
    try:
        value = float(row[name])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid numeric field {name}") from exc
    require(math.isfinite(value), f"Nonfinite numeric field {name}")
    return value


def integer(row, name):
    value = finite(row, name)
    require(value.is_integer() and value >= 0, f"Invalid nonnegative integer field {name}")
    return int(value)


def validate_gate(results_dir, audit_path, synthetic, report, manifest, snapshot):
    results_dir, audit_path = Path(results_dir), Path(audit_path)
    require(results_dir.is_dir(), "Results directory is missing")
    require(audit_path.is_file() and not audit_path.is_symlink(), "Audit report must be a regular file")
    manifest_path = results_dir / "run_manifest.json"
    expected_report = {
        "status": "passed", "schema_version": 4, "synthetic_test_only": synthetic,
        "candidates": 127, "caps": [25, 50, 100, 200, "all"],
        "matched_comparisons": 105, "source_transfer_J": None,
        "model_fitting_performed": False, "metrics_independently_recomputed": True,
    }
    for key, wanted in expected_report.items():
        require(report.get(key) == wanted, f"Independent audit gate mismatch: {key}")
    require(report.get("result_manifest_sha256") == snapshot[str(manifest_path.resolve())],
            "Audit does not bind the current run manifest")
    require(report.get("auditor_sha256") == snapshot[str(AUDITOR.resolve())],
            "Audit was not produced by the current independent v4 auditor")
    require(manifest.get("schema_version") == 4 and manifest.get("stage") == "dev",
            "Not a schema-v4 development result")
    require(manifest.get("source_transfer_J") is None and manifest.get("winner_selected") is False,
            "J or winner selection is forbidden")
    require(manifest.get("source_holdout_status") == "not_run_single_development_AI_source_Suno",
            "Expected single-AI-source Suno and no source-holdout run")
    wanted_auth = "synthetic_test_only" if synthetic else "frozen_verified"
    require(manifest.get("authorization", {}).get("status") == wanted_auth,
            "Result authorization mode does not match presentation mode")
    require(bool(manifest.get("contract", {}).get("synthetic_test_only")) is synthetic,
            "Contract synthetic mode mismatch")
    if synthetic:
        require(report.get("rows", 500) < 500, "Synthetic fixture must remain smaller than 500 rows")
    else:
        real_counts = {"rows": 1604, "primary_predictions": 1018540,
                       "diagnostic_predictions": 407416, "pooled_metrics": 3175,
                       "source_pair_metrics": 15875, "diagnostic_metrics": 6350,
                       "matched_delta_rows": 13125}
        for key, wanted in real_counts.items():
            require(report.get(key) == wanted, f"Real audit denominator mismatch: {key}")
    files = manifest.get("files_sha256")
    require(isinstance(files, dict) and REQUIRED_RESULTS <= set(files), "Incomplete result hash manifest")
    for name, wanted in files.items():
        path = results_dir / name
        require(snapshot[str(path.resolve())] == wanted, f"Audited result changed after audit: {name}")
    return report, manifest


def validate_plan(path):
    rows, header = read_csv(path)
    required = {"plan_type", "comparison_id", "arm", "combination", "cohort_id", "eligible_rows", "eligible_id_set_sha256"}
    require(required <= set(header), "Evaluation-plan schema is incomplete")
    expected_candidates = set(combinations())
    require({row["combination"] for row in rows} == expected_candidates, "Plan does not contain all 127 candidates")
    require(len(rows) == 232, "Plan must contain all 232 rows")
    require(len({row.get("plan_id") for row in rows}) == len(rows), "Duplicate plan row")
    require(len({row["cohort_id"] for row in rows}) == 1 and len({row["eligible_id_set_sha256"] for row in rows}) == 1,
            "Candidate cohort changed within the plan")
    require(len({row["eligible_rows"] for row in rows}) == 1, "Candidate row count changed within the plan")
    standalone_expected = ({("old_baseline_standalone", combo) for combo in subsets(OLD_ORDER)}
                           | {("new_standalone", combo) for combo in subsets(NEW_ORDER)})
    standalone_actual = {(r["plan_type"], r["combination"]) for r in rows if r["arm"] == "standalone"}
    require(standalone_actual == standalone_expected and sum(r["arm"] == "standalone" for r in rows) == 22,
            "Incomplete or altered standalone plan")
    expected = {(cid, arm, left if arm == "baseline" else added)
                for cid, left, added in comparisons() for arm in ("baseline", "added")}
    actual = {(r["comparison_id"], r["arm"], r["combination"])
              for r in rows if r["plan_type"] == "incremental_matched"}
    require(actual == expected and len(actual) == 210, "Incomplete or altered 105-comparison plan")
    return rows[0]["cohort_id"], integer(rows[0], "eligible_rows"), rows[0]["eligible_id_set_sha256"]


def common_metric_checks(row, combo, cap, cohort):
    require(row.get("combination") == combo and quantity(row.get("quantity", "")) == cap,
            "Metric candidate/cap mismatch")
    require(row.get("cohort_id") == cohort, "Metric cohort mismatch")
    require(row.get("feature_mode") == "values_plus_missing", "Only primary feature mode may be presented")
    require(row.get("fold_type") == FOLD_TYPE and row.get("heldout_source") == "__all_sources__",
            "Only ordinary descriptive group CV may be presented")
    require(row.get("ai_source", "Suno") == "Suno", "Suno must be the sole AI source")
    require(finite(row, "threshold") == 0.5, "Threshold must remain 0.5")


def aggregate_inputs(results_dir, cohort, expected_rows):
    combos, expected_combos = combinations(), set(combinations())
    pair_rows, _ = read_csv(results_dir / "development_group_cv_metrics_by_source_pair.csv")
    pooled_rows, _ = read_csv(results_dir / "development_group_cv_pooled_fold_metrics.csv")
    delta_rows, _ = read_csv(results_dir / "development_group_cv_matched_deltas.csv")

    pair_lookup = {}
    denominator_reference = {}
    for row in pair_rows:
        combo, cap = row.get("combination"), quantity(row.get("quantity", ""))
        require(combo in expected_combos, "Unknown source-pair candidate")
        common_metric_checks(row, combo, cap, cohort)
        require(row.get("ai_source") == "Suno" and row.get("human_source") not in (None, "", "Suno"),
                "Invalid Human-source/Suno pair")
        key = (combo, cap, row["fold_uid"], row["human_source"], row["ai_source"])
        require(key not in pair_lookup, "Duplicate source-pair metric cell")
        for name in MEASURES:
            finite(row, name)
        for name in ("test_human", "test_ai", "test_human_groups", "test_ai_groups"):
            require(integer(row, name) > 0, f"Empty source-pair denominator: {name}")
        require(row.get("fold_index") in {"0", "1", "2", "3", "4"}, "Unexpected source-pair fold index")
        denominator_key = (row["fold_index"], row["human_source"], row["ai_source"])
        denominator = tuple(row[name] for name in ("test_human", "test_ai", "test_human_groups", "test_ai_groups"))
        require(denominator_reference.setdefault(denominator_key, denominator) == denominator,
                "Source-pair denominator changed between candidates")
        pair_lookup[key] = row

    pair_summary = []
    expected_pair_keys = {}
    pair_aggregate_lookup = {}
    for combo in combos:
        for cap in CAPS:
            rows = [row for (c, q, *_), row in pair_lookup.items() if (c, q) == (combo, cap)]
            keys = {(r["fold_uid"], r["human_source"], r["ai_source"]) for r in rows}
            if cap not in expected_pair_keys:
                expected_pair_keys[cap] = keys
            require(keys == expected_pair_keys[cap], "Missing source-pair/fold cell; denominator shrinkage forbidden")
            require(len(rows) == 25 and len({r["fold_uid"] for r in rows}) == 5
                    and len({r["human_source"] for r in rows}) == 5
                    and {r["ai_source"] for r in rows} == {"Suno"},
                    "Every candidate/cap needs 5 folds x 5 Human-source/Suno pairs")
            output = {"cohort_id": cohort, "combination": combo, "quantity": cap,
                      "pair_fold_cells": 25, "folds": 5, "human_sources": 5, "ai_sources": 1,
                      "test_human_pair_exposures": sum(integer(r, "test_human") for r in rows),
                      "test_ai_pair_exposures": sum(integer(r, "test_ai") for r in rows)}
            for name in MEASURES:
                output["source_pair_fold_mean_" + name] = sum(finite(r, name) for r in rows) / 25
            pair_summary.append(output)
            pair_aggregate_lookup[combo, cap] = output
    require(len(pair_lookup) == 127 * 5 * 25, "Incomplete or extra 127 x 5 source-pair grid")

    pooled_lookup = {}
    pooled_denominator_reference = {}
    for row in pooled_rows:
        combo, cap = row.get("combination"), quantity(row.get("quantity", ""))
        require(combo in expected_combos, "Unknown pooled-fold candidate")
        common_metric_checks(row, combo, cap, cohort)
        key = (combo, cap, row["fold_uid"])
        require(key not in pooled_lookup, "Duplicate pooled-fold metric cell")
        for name in MEASURES:
            finite(row, name)
        for name in ("tp", "tn", "fp", "fn"):
            integer(row, name)
        require(row.get("fold_index") in {"0", "1", "2", "3", "4"}, "Unexpected pooled fold index")
        denominator = (integer(row, "tn") + integer(row, "fp"), integer(row, "tp") + integer(row, "fn"))
        require(pooled_denominator_reference.setdefault(row["fold_index"], denominator) == denominator,
                "Pooled-fold denominator changed between candidates/caps")
        pooled_lookup[key] = row

    pooled_summary, pooled_aggregate_lookup = [], {}
    for combo in combos:
        for cap in CAPS:
            rows = [row for (c, q, _), row in pooled_lookup.items() if (c, q) == (combo, cap)]
            require(len(rows) == 5 and len({r["fold_uid"] for r in rows}) == 5,
                    "Every candidate/cap needs five pooled-fold metrics")
            require({r["fold_uid"] for r in rows} == {key[0] for key in expected_pair_keys[cap]},
                    "Pooled and source-pair fold registries differ")
            human_rows = sum(integer(r, "tn") + integer(r, "fp") for r in rows)
            ai_rows = sum(integer(r, "tp") + integer(r, "fn") for r in rows)
            require(human_rows + ai_rows == expected_rows, "Pooled fold tests do not partition the complete cohort")
            output = {"cohort_id": cohort, "combination": combo, "quantity": cap,
                      "pooled_folds": 5, "pooled_test_human_rows": human_rows,
                      "pooled_test_ai_rows": ai_rows}
            for name in MEASURES:
                output["pooled_within_fold_mean_" + name] = sum(finite(r, name) for r in rows) / 5
            pooled_summary.append(output)
            pooled_aggregate_lookup[combo, cap] = output
    require(len(pooled_lookup) == 127 * 5 * 5, "Incomplete or extra 127 x 5 pooled-fold grid")

    expected_comparisons = {cid: (left, added) for cid, left, added in comparisons()}
    delta_lookup = {}
    for row in delta_rows:
        cid, cap = row.get("comparison_id"), quantity(row.get("quantity", ""))
        require(cid in expected_comparisons, "Unknown matched comparison")
        left, added = expected_comparisons[cid]
        require(row.get("baseline_combination") == left and row.get("added_combination") == added,
                "Matched-comparison arm mismatch")
        require(row.get("cohort_id") == cohort and row.get("ai_source") == "Suno",
                "Matched-comparison cohort/source mismatch")
        key = (cid, cap, row["fold_uid"], row["human_source"], row["ai_source"])
        require(key not in delta_lookup, "Duplicate matched-delta cell")
        for name in MEASURES:
            baseline = finite(row, name + "__baseline")
            addition = finite(row, name + "__added")
            delta = finite(row, "delta_" + name + "__added_minus_baseline")
            require(math.isclose(delta, addition - baseline, rel_tol=1e-10, abs_tol=1e-12),
                    "Matched delta arithmetic mismatch")
            left_metric = pair_lookup[left, cap, row["fold_uid"], row["human_source"], row["ai_source"]]
            added_metric = pair_lookup[added, cap, row["fold_uid"], row["human_source"], row["ai_source"]]
            require(math.isclose(baseline, finite(left_metric, name), rel_tol=1e-10, abs_tol=1e-12)
                    and math.isclose(addition, finite(added_metric, name), rel_tol=1e-10, abs_tol=1e-12),
                    "Matched delta does not reproduce audited arm metrics")
        delta_lookup[key] = row

    delta_summary = []
    for cid, left, added in comparisons():
        for cap in CAPS:
            rows = [row for (c, q, *_), row in delta_lookup.items() if (c, q) == (cid, cap)]
            require(len(rows) == 25 and {(r["fold_uid"], r["human_source"], r["ai_source"]) for r in rows} == expected_pair_keys[cap],
                    "Missing matched source-pair/fold delta; denominator shrinkage forbidden")
            output = {"cohort_id": cohort, "comparison_id": cid, "baseline_combination": left,
                      "added_combination": added, "quantity": cap, "pair_fold_cells": 25,
                      "folds": 5, "human_sources": 5, "ai_sources": 1}
            for name in MEASURES:
                output["source_pair_fold_mean_delta_" + name] = sum(
                    finite(r, "delta_" + name + "__added_minus_baseline") for r in rows) / 25
                output["pooled_within_fold_mean_delta_" + name] = (
                    pooled_aggregate_lookup[added, cap]["pooled_within_fold_mean_" + name]
                    - pooled_aggregate_lookup[left, cap]["pooled_within_fold_mean_" + name])
            delta_summary.append(output)
    require(len(delta_lookup) == 105 * 5 * 25, "Incomplete or extra 105 x 5 matched-delta grid")
    return pair_summary, pooled_summary, delta_summary


def write_csv(path, rows):
    require(rows, "Cannot write an empty table")
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cell(lookup, key, prefix, delta=False):
    auc = lookup[key][prefix + "roc_auc"]
    ba = lookup[key][prefix + "balanced_accuracy"]
    if delta:
        return f"{auc:+.4f} / {ba:+.4f}"
    return f"{auc:.4f} / {ba:.4f}"


def latex_escape(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def latex_row_label(value):
    if " -> " in value:
        left, right = value.split(" -> ", 1)
        return latex_escape(left) + r" $\rightarrow$ " + latex_escape(right)
    return latex_escape(value)


def table_lines(title, caption, row_names, lookup, prefix, delta=False,
                first_column="Combination", label_suffix="", display_labels=None):
    display_labels = display_labels or {}
    md = [f"## {title}", "", caption, "",
          f"| {first_column} | 25 | 50 | 100 | 200 | All |",
          "|---|---:|---:|---:|---:|---:|"]
    tex = [r"\begingroup\scriptsize", r"\setlength{\tabcolsep}{2pt}",
           r"\setlength\LTleft{0pt}\setlength\LTright{0pt}",
           r"\begin{longtable}{@{}p{0.22\linewidth}*{5}{p{0.14\linewidth}}@{}}", r"\caption{" + latex_escape(caption) + r"}\label{tab:" +
           ("equal60-v4-delta" + label_suffix if delta else "equal60-v4-" + ("pair" if "source_pair" in prefix else "pooled")) + r"}\\",
           r"\toprule " + latex_escape(first_column) + r" & 25 & 50 & 100 & 200 & All \\ \midrule\endfirsthead",
           r"\toprule " + latex_escape(first_column) + r" & 25 & 50 & 100 & 200 & All \\ \midrule\endhead",
           r"\bottomrule\endlastfoot"]
    for name in row_names:
        values = [cell(lookup, (name, cap), prefix, delta) for cap in CAPS]
        display = display_labels.get(name, name)
        md.append("| " + display + " | " + " | ".join(values) + " |")
        tex.append(latex_row_label(display) + " & " + " & ".join(values) + r" \\")
    tex.extend([r"\end{longtable}", r"\endgroup", ""])
    return md, tex


def render(pair_rows, pooled_rows, delta_rows, synthetic):
    pair = {(r["combination"], r["quantity"]): r for r in pair_rows}
    pooled = {(r["combination"], r["quantity"]): r for r in pooled_rows}
    delta = {(r["comparison_id"], r["quantity"]): r for r in delta_rows}
    prefix = "SYNTHETIC FIXTURE — NOT SCIENTIFIC RESULTS\n\n" if synthetic else ""
    md = ["# " + ("Synthetic fixture: " if synthetic else "") + "Exact60 schema-v4 exhaustive descriptive tables", "",
          prefix.rstrip(), "" if prefix else "",
          "No candidate is selected or ranked. Source-transfer J is undefined and is not reported: Suno is the sole AI source. All values are descriptive ordinary global-group CV at fixed threshold 0.5; caps are maximum training groups per source within each fold. Every candidate retains the same cohort.", ""]
    pair_caption = ("Each cell is AUC / balanced accuracy, uniformly averaged over 25 cells: five Human-source/Suno pairs in each of five folds. AI test rows are reused across the five Human pairings. This is the agreed source-pair/fold descriptive convention, not a pooled-fold metric.")
    pooled_caption = ("Each cell is AUC / balanced accuracy, uniformly averaged over five fold metrics after all Human and Suno rows were pooled within each fold. This companion table has different weighting from the source-pair/fold table.")
    delta_caption = ("Each cell is added-minus-baseline delta AUC / delta balanced accuracy, uniformly averaged over the same 25 matched source-pair/fold cells. All 105 old-subset x new-subset contrasts are shown; no top-K filtering or selection is performed.")
    pooled_delta_caption = ("Each cell is added-minus-baseline delta AUC / delta balanced accuracy computed from the companion pooled-within-fold means. It is kept separate from the agreed 25-cell source-pair/fold contrast.")
    part_md, pair_tex = table_lines("All 127 candidates: source-pair/fold mean", pair_caption, combinations(), pair, "source_pair_fold_mean_")
    md.extend(part_md + [""])
    part_md, pooled_tex = table_lines("All 127 candidates: pooled-within-fold mean", pooled_caption, combinations(), pooled, "pooled_within_fold_mean_")
    md.extend(part_md + [""])
    comparison_ids = [cid for cid, _, _ in comparisons()]
    contrast_labels = {cid: left + " -> " + added for cid, left, added in comparisons()}
    part_md, delta_tex = table_lines("All 105 matched contrasts: source-pair/fold mean deltas", delta_caption, comparison_ids, delta, "source_pair_fold_mean_delta_", True, "Contrast", "-pair", contrast_labels)
    md.extend(part_md + [""])
    part_md, pooled_delta_tex = table_lines("All 105 matched contrasts: pooled-within-fold mean deltas", pooled_delta_caption, comparison_ids, delta, "pooled_within_fold_mean_delta_", True, "Contrast", "-pooled", contrast_labels)
    md.extend(part_md + ["", "## Denominators and interpretation", "",
                         "The machine-readable CSVs report the 25 pair/fold cells, five folds, five Human sources, one AI source, pair-exposure counts, and pooled unique-class row counts for every candidate/cap. AUC is unweighted Mann–Whitney AUC with half credit for ties. Balanced accuracy is the equal mean of AI sensitivity and Human specificity at score >= 0.5. Pair/fold and pooled-fold results are descriptive views of the same group-CV predictions; neither is source-held-out evidence, unseen-generator validation, a significance test, or a generator-mechanism claim.", ""])
    tex = [r"% Exhaustive exact60 schema-v4 descriptive tables; no J and no winner selection."]
    if synthetic:
        tex.extend([r"% SYNTHETIC FIXTURE -- NOT SCIENTIFIC RESULTS",
                    r"\begin{center}\fbox{\parbox{0.92\linewidth}{\centering\textbf{SYNTHETIC FIXTURE -- NOT SCIENTIFIC RESULTS}}}\end{center}"])
    tex.extend(pair_tex + pooled_tex + delta_tex + pooled_delta_tex)
    return "\n".join(md).replace("\n\n\n", "\n\n") + "\n", "\n".join(tex) + "\n"


def export(results_dir, audit_path, output_dir, synthetic=False):
    results_dir, audit_path, output_dir = map(Path, (results_dir, audit_path, output_dir))
    require(not output_dir.exists(), "Refusing an existing output directory")
    report, manifest, before = snapshot_inputs(results_dir, audit_path)
    validate_gate(results_dir, audit_path, synthetic, report, manifest, before)
    cohort, eligible_rows, cohort_hash = validate_plan(results_dir / "evaluation_plan.csv")
    require(synthetic or eligible_rows == 1604, "Real plan cohort must contain 1,604 rows")
    pair_rows, pooled_rows, delta_rows = aggregate_inputs(results_dir, cohort, eligible_rows)
    md, tex = render(pair_rows, pooled_rows, delta_rows, synthetic)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp.", dir=output_dir.parent))
    try:
        write_csv(temporary / "all_127x5_source_pair_fold_mean.csv", pair_rows)
        write_csv(temporary / "all_127x5_pooled_fold_mean.csv", pooled_rows)
        write_csv(temporary / "all_105x5_matched_contrasts.csv", delta_rows)
        (temporary / "EQUAL60_PRESENTATION_V4_TABLES_EN.md").write_text(md)
        (temporary / "equal60_presentation_v4_tables_en.tex").write_text(tex)
        outputs = {p.name: sha(p) for p in sorted(temporary.iterdir())}
        receipt = {
            "status": "synthetic_fixture_complete" if synthetic else "complete",
            "schema_version": 4, "synthetic_fixture_only": synthetic,
            "cohort_id": cohort, "eligible_rows": eligible_rows,
            "eligible_id_set_sha256": cohort_hash,
            "candidate_cap_cells": 127 * 5, "matched_contrast_cap_cells": 105 * 5,
            "source_pair_fold_denominator_per_cell": 25, "pooled_fold_denominator_per_cell": 5,
            "source_transfer_J": None, "winner_selected": False, "classifiers_fitted": False,
            "aggregation": {
                "headline": "uniform mean over 5 Human-source/Suno pairs x 5 ordinary global-group folds",
                "companion": "uniform mean over 5 pooled-within-fold metrics",
                "matched_contrasts": "added-minus-baseline before uniform mean over the same 25 pair/fold cells",
            },
            "independent_audit": {"path": str(audit_path.resolve()), "sha256": before[str(audit_path.resolve())],
                                  "result_manifest_sha256": report["result_manifest_sha256"],
                                  "auditor_sha256": before[str(AUDITOR.resolve())]},
            "inputs": before, "presentation_code_sha256": sha(__file__), "outputs": outputs,
        }
        (temporary / "presentation_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        after = {}
        for name in before:
            path = Path(name)
            require(path.is_file() and not path.is_symlink(), f"Provenance input disappeared during export: {path}")
            after[name] = sha(path)
        require(after == before, "Audited inputs changed during presentation export")
        require(not output_dir.exists(), "Output directory appeared during export")
        os.rename(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic-fixture-only", action="store_true",
                        help="admit only a passed synthetic audit and watermark every table")
    args = parser.parse_args(argv)
    print(json.dumps(export(args.results_dir, args.audit, args.output_dir,
                            args.synthetic_fixture_only), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
