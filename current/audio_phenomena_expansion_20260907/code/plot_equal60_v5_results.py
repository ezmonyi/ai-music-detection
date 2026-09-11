#!/usr/bin/env python3
"""Create fixed scientific figures from an accepted v5 presentation package.

The plotter reads only committed presentation tables. It never reads evaluator
predictions, fits a model, changes a scalar table, or selects a candidate.
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
import platform
import re
import tempfile


FAMILIES = ("S", "D", "R", "P", "F", "H", "M")
COMBINATIONS = tuple("+".join(c) for n in range(1, 8) for c in itertools.combinations(FAMILIES, n))
CAPS = ("25", "50", "100", "200", "all")
FOLD_TYPES = ("human_source_holdout", "generator_holdout", "ordinary_group_holdout_descriptive")
FOLD_LABELS = {"human_source_holdout": "Human-source holdout",
               "generator_holdout": "Generator holdout",
               "ordinary_group_holdout_descriptive": "Ordinary group holdout"}
FIXED12 = ("S", "D", "R", "P", "F", "H", "M", "S+D+R+P", "S+D+R+P+F",
           "S+D+R+P+H", "S+D+R+P+M", "S+D+R+P+F+H+M")
ADDED = ("S+D+R+P+F", "S+D+R+P+H", "S+D+R+P+M", "S+D+R+P+F+H+M")
PRESENTATION_FILES = frozenset({"primary_pair_macro_all_arms.csv",
    "primary_pooled_fold_companion_all_arms.csv", "primary_source_endpoints_all_arms.csv",
    "diagnostic_all_cap_pair_macro_all_arms.csv", "diagnostic_all_cap_pooled_fold_companion_all_arms.csv",
    "diagnostic_all_cap_source_endpoints_all_arms.csv", "primary_fixed12_overview.csv",
    "primary_fixed_increments.csv", "training_row_group_ranges.csv",
    "EQUAL60_V5_PRESENTATION_TABLES_EN.md", "equal60_v5_presentation_tables_en.tex",
    "presentation_receipt.json"})
HEX64 = re.compile(r"[0-9a-f]{64}")
ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = ROOT / "EQUAL60_V5_FIGURES_PROTOCOL_EN.md"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def strict_json(path):
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "Duplicate JSON key: " + key)
            output[key] = value
        return output
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + token)))
    require(isinstance(value, dict), "Expected JSON object: " + str(path))
    return value


def read_csv(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
                "Missing or duplicate CSV columns: " + str(path))
        rows = list(reader)
    require(rows and all(None not in row and None not in row.values() for row in rows),
            "Empty or malformed CSV: " + str(path))
    return rows


def number(row, field):
    try:
        value = float(row[field])
    except (KeyError, ValueError) as exc:
        raise ValueError("Invalid plotted scalar: " + field) from exc
    require(math.isfinite(value), "Nonfinite plotted scalar: " + field)
    return value


def verify_commit(directory):
    commit_path = directory / "COMMIT.json"
    require(directory.is_absolute() and directory.is_dir() and directory.resolve() == directory
            and commit_path.is_file() and not commit_path.is_symlink(), "Canonical committed presentation required")
    commit = strict_json(commit_path)
    require(commit.get("status") == "committed"
            and commit.get("publication") == "exclusive directory reservation; hardlink COMMIT last",
            "Presentation COMMIT status changed")
    require(set(commit.get("files", {})) == PRESENTATION_FILES
            and {p.name for p in directory.iterdir()} == PRESENTATION_FILES | {"COMMIT.json"},
            "Presentation file inventory changed")
    hashes = {}
    for name, record in commit["files"].items():
        path = directory / name
        observed = sha(path)
        require(path.is_file() and not path.is_symlink() and path.resolve() == path
                and set(record) == {"sha256", "bytes"} and record["sha256"] == observed
                and record["bytes"] == path.stat().st_size, "Presentation file changed: " + name)
        hashes[str(path)] = observed
    hashes[str(commit_path)] = sha(commit_path)
    return commit, hashes


def validate_inputs(directory, acceptance_path, acceptance_sha, audit_path, audit_sha, synthetic):
    require(all(path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve() == path
                for path in (acceptance_path, audit_path)), "Canonical acceptance and audit files required")
    require(HEX64.fullmatch(acceptance_sha or "") and sha(acceptance_path) == acceptance_sha,
            "Explicit parent acceptance SHA-256 mismatch")
    require(HEX64.fullmatch(audit_sha or "") and sha(audit_path) == audit_sha,
            "Explicit numerical audit SHA-256 mismatch")
    commit, hashes = verify_commit(directory)
    receipt = strict_json(directory / "presentation_receipt.json")
    audit = strict_json(audit_path)
    acceptance = strict_json(acceptance_path)
    require(receipt.get("synthetic_test_only") is synthetic
            and receipt.get("status") == ("synthetic_fixture_complete" if synthetic else "complete")
            and receipt.get("independent_audit", {}).get("sha256") == audit_sha,
            "Presentation receipt does not bind the supplied audit")
    require(audit.get("status") == "passed" and audit.get("synthetic_test_only") is synthetic
            and audit.get("contract_sha256") == receipt.get("contract_sha256")
            and audit.get("aggregation_contract") == {
                "unique_oof_recording_rates_within_arm_source": True,
                "equal_global_component_rates": True, "equal_fold_component_mean": False,
                "fold_pair_macro_hierarchy": True, "pool_across_held_source_arms": False,
                "pooled_out_of_fold_auc": False}, "Supplied numerical audit scope changed")
    expected_acceptance = {"schema_version": 1, "status": "accepted_for_scientific_figure_export",
                           "reviewer": "root", "synthetic_test_only": synthetic,
                           "presentation_commit_sha256": hashes[str(directory / "COMMIT.json")],
                           "presentation_receipt_sha256": hashes[str(directory / "presentation_receipt.json")],
                           "numerical_audit_sha256": audit_sha, "figure_export_authorized": True}
    require(set(acceptance) == set(expected_acceptance) | {"reviewed_utc"}
            and all(acceptance.get(key) == value for key, value in expected_acceptance.items()),
            "Parent acceptance does not authorize this exact figure input")
    require(isinstance(acceptance.get("reviewed_utc"), str) and acceptance["reviewed_utc"].strip(),
            "Parent acceptance review time missing")
    hashes[str(acceptance_path)] = acceptance_sha
    hashes[str(audit_path)] = audit_sha
    hashes[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    hashes[str(PROTOCOL.resolve())] = sha(PROTOCOL.resolve())
    return receipt, hashes


def validate_tables(directory):
    primary = read_csv(directory / "primary_pair_macro_all_arms.csv")
    source = read_csv(directory / "primary_source_endpoints_all_arms.csv")
    increments = read_csv(directory / "primary_fixed_increments.csv")
    diagnostic = read_csv(directory / "diagnostic_all_cap_pair_macro_all_arms.csv")
    macro = [row for row in primary if row.get("summary_level") == "fold_type_macro"
             and row.get("heldout_source") == "__equal_arms__" and row.get("feature_mode") == "values_plus_missing"]
    require(len(macro) == len(FOLD_TYPES) * len(COMBINATIONS) * len(CAPS)
            and {(r["fold_type"], r["combination"], r["quantity"]) for r in macro}
            == set(itertools.product(FOLD_TYPES, COMBINATIONS, CAPS)),
            "Primary macro heatmap grid is incomplete")
    for row in macro:
        require(0 <= number(row, "roc_auc") <= 1 and 0 <= number(row, "balanced_accuracy") <= 1,
                "Primary rate outside [0,1]")
    diagnostic_macro = [row for row in diagnostic if row.get("summary_level") == "fold_type_macro"
                        and row.get("heldout_source") == "__equal_arms__" and row.get("quantity") == "all"]
    require(len(diagnostic_macro) == len(FOLD_TYPES) * len(COMBINATIONS) * 2
            and {(r["fold_type"], r["combination"], r["feature_mode"]) for r in diagnostic_macro}
            == set(itertools.product(FOLD_TYPES, COMBINATIONS, ("median_only", "missingness_only"))),
            "All-cap diagnostic macro grid is incomplete")
    for row in diagnostic_macro:
        require(0 <= number(row, "roc_auc") <= 1 and 0 <= number(row, "balanced_accuracy") <= 1,
                "Diagnostic rate outside [0,1]")
    fixed_source = [row for row in source if row.get("combination") in FIXED12
                    and row.get("feature_mode") == "values_plus_missing" and _central_source_row(row)]
    contexts = {(r["fold_type"], r["heldout_source"], r["source_group"], r["endpoint"]) for r in fixed_source}
    all_contexts = {(r["fold_type"], r["heldout_source"], r["source_group"], r["endpoint"])
                    for r in source if r.get("feature_mode") == "values_plus_missing" and _central_source_row(r)}
    require(contexts == all_contexts and contexts
            and len(fixed_source) == len(contexts) * len(FIXED12) * len(CAPS)
            and all({(r["combination"], r["quantity"]) for r in fixed_source
                     if (r["fold_type"], r["heldout_source"], r["source_group"], r["endpoint"]) == context}
                    == set(itertools.product(FIXED12, CAPS)) for context in contexts),
            "Fixed source endpoint grid is incomplete")
    for row in fixed_source:
        require(0 <= number(row, "recording_rate") <= 1 and 0 <= number(row, "equal_component_rate") <= 1,
                "Source endpoint outside [0,1]")
    fixed_increments = [row for row in increments if row.get("added_combination") in ADDED
                        and (row.get("scope") == "two_class_pair_macro" or _central_increment_row(row))]
    increment_contexts = {(r["scope"], r["summary_level"], r["fold_type"], r["heldout_source"],
                           r["source_group"], r["endpoint"]) for r in fixed_increments}
    increment_keys = [(r["scope"], r["summary_level"], r["fold_type"], r["heldout_source"],
                       r["source_group"], r["endpoint"], r["added_combination"], r["quantity"])
                      for r in fixed_increments]
    require(fixed_increments and all(r.get("baseline_combination") == "S+D+R+P" for r in fixed_increments)
            and len(increment_keys) == len(set(increment_keys))
            and all({(r["added_combination"], r["quantity"]) for r in fixed_increments
                     if (r["scope"], r["summary_level"], r["fold_type"], r["heldout_source"],
                         r["source_group"], r["endpoint"]) == context}
                    == set(itertools.product(ADDED, CAPS)) for context in increment_contexts),
            "Fixed increment grid is incomplete, duplicated, or has the wrong baseline")
    for row in fixed_increments:
        number(row, "added_minus_baseline_pp")
    return macro, fixed_source, fixed_increments, diagnostic_macro


def _central_source_row(row):
    if row["fold_type"] == "ordinary_group_holdout_descriptive":
        return True
    return row["source_group"] == row["heldout_source"]


def _central_increment_row(row):
    return row.get("scope") == "source_endpoint" and (
        row.get("fold_type") == "ordinary_group_holdout_descriptive"
        or row.get("source_group") == row.get("heldout_source"))


def plotting():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("matplotlib and numpy are required for v5 scientific figures") from exc
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11, "axes.labelsize": 9,
                                "xtick.labelsize": 7, "ytick.labelsize": 6, "figure.facecolor": "white"})
    return plt, np


def runtime_versions():
    import matplotlib
    import numpy
    return {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
            "numpy": numpy.__version__}


def watermark(fig, synthetic):
    if synthetic:
        fig.text(.5, .5, "SYNTHETIC TEST FIXTURE", ha="center", va="center", rotation=30,
                 fontsize=42, color="#b22222", alpha=.10, weight="bold")


def save_pair(fig, output_dir, stem):
    metadata = {"Creator": "plot_equal60_v5_results.py", "CreationDate": None, "ModDate": None}
    fig.savefig(output_dir / (stem + ".pdf"), bbox_inches="tight", metadata=metadata)
    fig.savefig(output_dir / (stem + ".png"), dpi=180, bbox_inches="tight",
                metadata={"Software": "plot_equal60_v5_results.py"})


def heatmap(ax, matrix, row_labels, col_labels, plt, np, title, cmap="viridis", vmin=0, vmax=1,
            show_rows=True):
    image = ax.imshow(np.asarray(matrix, dtype=float), aspect="auto", interpolation="nearest",
                      cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha="right")
    if show_rows:
        ax.set_yticks(range(len(row_labels)), row_labels)
    else:
        ax.set_yticks([])
    return image


def figure_primary_heatmaps(rows, output_dir, synthetic):
    plt, np = plotting()
    lookup = {(r["fold_type"], r["combination"], r["quantity"]): r for r in rows}
    fig, axes = plt.subplots(3, 2, figsize=(15, 34), constrained_layout=True)
    image = None
    for i, fold_type in enumerate(FOLD_TYPES):
        for j, metric in enumerate(("roc_auc", "balanced_accuracy")):
            matrix = [[number(lookup[fold_type, combo, cap], metric) for cap in CAPS] for combo in COMBINATIONS]
            image = heatmap(axes[i, j], matrix, COMBINATIONS, ("25", "50", "100", "200", "All"),
                            plt, np, FOLD_LABELS[fold_type] + " - " + ("AUC" if metric == "roc_auc" else "Balanced accuracy"),
                            show_rows=(j == 0))
    fig.colorbar(image, ax=axes, label="Rate (fixed 0 to 1)", shrink=.35)
    fig.suptitle(("Synthetic fixture: " if synthetic else "") + "Exploratory v5 primary fold-pair macro", fontsize=15)
    watermark(fig, synthetic)
    save_pair(fig, output_dir, "figure_1_primary_macro_heatmaps")
    plt.close(fig)


def short_value(value):
    mapped = {"__all_sources__": "all sources", "__equal_arms__": "equal arms",
              "__two_class__": "two-class",
              "human_specificity": "human specificity", "ai_sensitivity": "AI sensitivity",
              "roc_auc": "AUC", "balanced_accuracy": "BA",
              "two_class_roc_auc": "AUC", "two_class_balanced_accuracy": "BA",
              "equal_component_rate": "component", "recording_rate": "recording"}.get(value)
    return mapped if mapped is not None else value.strip("_").replace("_", " ")


def context_label(row, estimator=None):
    fold = {"human_source_holdout": "H", "generator_holdout": "G",
            "ordinary_group_holdout_descriptive": "O"}[row["fold_type"]]
    pieces = [fold + " | held " + short_value(row["heldout_source"])]
    if row.get("source_group"):
        pieces.append(short_value(row["source_group"]) + " " + short_value(row["endpoint"]))
    else:
        pieces.append(short_value(row["summary_level"]) + " " + short_value(row["endpoint"]))
    if estimator:
        pieces.append(short_value(estimator))
    return "\n".join(pieces)


def figure_source_and_increments(source_rows, increment_rows, output_dir, synthetic):
    plt, np = plotting()
    source_contexts = sorted({(r["fold_type"], r["heldout_source"], r["source_group"], r["endpoint"])
                              for r in source_rows}, key=lambda x: (FOLD_TYPES.index(x[0]), x[1], x[2], x[3]))
    absolute_labels, absolute_columns = [], []
    lookup = {(r["fold_type"], r["heldout_source"], r["source_group"], r["endpoint"],
               r["combination"], r["quantity"]): r for r in source_rows}
    columns = [(combo, cap) for combo in FIXED12 for cap in CAPS]
    for context in source_contexts:
        template = lookup[(*context, FIXED12[0], CAPS[0])]
        for estimator in ("recording_rate", "equal_component_rate"):
            absolute_labels.append(context_label(template, estimator))
            absolute_columns.append([number(lookup[(*context, combo, cap)], estimator) for combo, cap in columns])
    absolute_matrix = list(map(list, zip(*absolute_columns)))

    increment_contexts = sorted({(r["scope"], r["summary_level"], r["fold_type"], r["heldout_source"],
                                  r["source_group"], r["endpoint"]) for r in increment_rows},
                                key=lambda x: (x[0], FOLD_TYPES.index(x[2]), x[3], x[4], x[5]))
    inc_lookup = {(r["scope"], r["summary_level"], r["fold_type"], r["heldout_source"], r["source_group"],
                   r["endpoint"], r["added_combination"], r["quantity"]): r for r in increment_rows}
    inc_columns = [(added, cap) for added in ADDED for cap in CAPS]
    increment_columns, increment_labels = [], []
    for context in increment_contexts:
        template = inc_lookup[(*context, ADDED[0], CAPS[0])]
        increment_labels.append(context_label(template))
        increment_columns.append([number(inc_lookup[(*context, added, cap)], "added_minus_baseline_pp")
                                  for added, cap in inc_columns])
    increment_matrix = list(map(list, zip(*increment_columns)))
    limit = max(abs(value) for row in increment_matrix for value in row) or 1.0
    width = max(22, .65 * len(increment_labels))
    fig, axes = plt.subplots(2, 1, figsize=(width, 26), constrained_layout=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    image1 = heatmap(axes[0], absolute_matrix, [f"{combo} | {cap}" for combo, cap in columns],
                     absolute_labels, plt, np,
                     "Fixed 12 source endpoints", vmin=0, vmax=1)
    image2 = heatmap(axes[1], increment_matrix, [f"{added} | {cap}" for added, cap in inc_columns],
                     increment_labels, plt, np,
                     "Matched additions relative to S+D+R+P", cmap="RdBu_r", vmin=-limit, vmax=limit)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=5)
        plt.setp(ax.get_xticklabels(), rotation=62, ha="right", rotation_mode="anchor")
    fig.colorbar(image1, ax=axes[0], label="Rate (fixed 0 to 1)", shrink=.6)
    fig.colorbar(image2, ax=axes[1], label="Added minus baseline (percentage points)", shrink=.6)
    fig.suptitle(("Synthetic fixture: " if synthetic else "") + "Source endpoints and fixed matched additions", fontsize=15)
    watermark(fig, synthetic)
    save_pair(fig, output_dir, "figure_2_source_endpoints_and_increments")
    plt.close(fig)
    return limit


def figure_diagnostics(rows, output_dir, synthetic):
    plt, np = plotting()
    lookup = {(r["fold_type"], r["combination"], r["feature_mode"]): r for r in rows}
    columns = [(fold, mode, metric) for fold in FOLD_TYPES for mode in ("median_only", "missingness_only")
               for metric in ("roc_auc", "balanced_accuracy")]
    matrix = [[number(lookup[fold, combo, mode], metric) for fold, mode, metric in columns] for combo in FIXED12]
    labels = [f"{FOLD_LABELS[fold]}\n{'Median values' if mode == 'median_only' else 'Missingness only'}\n{'AUC' if metric == 'roc_auc' else 'BA'}"
              for fold, mode, metric in columns]
    fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=True)
    image = heatmap(ax, matrix, FIXED12, labels, plt, np,
                    "All-cap diagnostics shown separately from the primary model", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="Rate (fixed 0 to 1)")
    fig.suptitle(("Synthetic fixture: " if synthetic else "") + "Exploratory v5 diagnostic macro", fontsize=15)
    watermark(fig, synthetic)
    save_pair(fig, output_dir, "figure_3_all_cap_diagnostics")
    plt.close(fig)


def commit_output(directory):
    products = sorted(directory.iterdir())
    require(products and not (directory / "COMMIT.json").exists()
            and all(path.is_file() and not path.is_symlink() for path in products), "Invalid figure publication state")
    for path in products:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    marker = {"status": "committed", "publication": "exclusive directory reservation; hardlink COMMIT last",
              "files": {path.name: {"sha256": sha(path), "bytes": path.stat().st_size} for path in products}}
    fd, pending = tempfile.mkstemp(prefix=".COMMIT.", dir=directory)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(marker, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(pending, directory / "COMMIT.json")
    finally:
        Path(pending).unlink(missing_ok=True)
    return marker


def export(directory, acceptance_path, acceptance_sha, audit_path, audit_sha, output_dir, synthetic=False):
    directory, acceptance_path, audit_path, output_dir = map(Path, (directory, acceptance_path, audit_path, output_dir))
    require(output_dir.is_absolute() and output_dir.parent.is_dir() and output_dir.parent.resolve() == output_dir.parent
            and not output_dir.exists() and not output_dir.is_symlink(), "New canonical figure output directory required")
    receipt, hashes = validate_inputs(directory, acceptance_path, acceptance_sha, audit_path, audit_sha, synthetic)
    macro, source, increments, diagnostic = validate_tables(directory)
    output_dir.mkdir()
    figure_primary_heatmaps(macro, output_dir, synthetic)
    increment_limit = figure_source_and_increments(source, increments, output_dir, synthetic)
    figure_diagnostics(diagnostic, output_dir, synthetic)
    require({str(name): sha(name) for name in map(Path, hashes)} == hashes,
            "Accepted figure input changed during rendering")
    figures = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
               for path in sorted(output_dir.iterdir())}
    figure_receipt = {"schema_version": 1, "status": "synthetic_fixture_complete" if synthetic else "complete",
                      "synthetic_test_only": synthetic, "figures": 3, "formats": ["pdf", "png"],
                      "render_runtime": runtime_versions(),
                      "candidate_order": list(COMBINATIONS), "caps": list(CAPS), "fixed_candidates": list(FIXED12),
                      "rate_color_scale": [0, 1],
                      "increment_color_scale_pp": [-increment_limit, increment_limit],
                      "cap_curves_omitted_as_redundant": True,
                      "input_presentation_commit_sha256": hashes[str(directory / "COMMIT.json")],
                      "parent_acceptance_sha256": acceptance_sha, "numerical_audit_sha256": audit_sha,
                      "inputs_sha256": dict(sorted(hashes.items())), "figure_files_before_receipt": figures,
                      "scope": "Static descriptive figures from committed v5 presentation tables. No fitting, scoring, candidate selection, pooled out-of-fold AUC, or scalar-table rewrite."}
    (output_dir / "figure_receipt.json").write_text(json.dumps(figure_receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    require({str(name): sha(name) for name in map(Path, hashes)} == hashes,
            "Accepted figure input changed before COMMIT")
    commit_output(output_dir)
    return figure_receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentation-dir", type=Path, required=True)
    parser.add_argument("--parent-acceptance", type=Path, required=True)
    parser.add_argument("--parent-acceptance-sha256", required=True)
    parser.add_argument("--numerical-audit", type=Path, required=True)
    parser.add_argument("--numerical-audit-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic-test-only", action="store_true")
    args = parser.parse_args(argv)
    require(all(path.is_absolute() for path in (args.presentation_dir, args.parent_acceptance,
            args.numerical_audit, args.output_dir)), "All paths must be absolute")
    value = export(args.presentation_dir, args.parent_acceptance, args.parent_acceptance_sha256,
                   args.numerical_audit, args.numerical_audit_sha256, args.output_dir,
                   args.synthetic_test_only)
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
