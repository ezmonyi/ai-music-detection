#!/usr/bin/env python3
"""Build a concise, selection-safe Markdown report from expanded evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_expanded_reporting import family_availability_by_source, primary_quantity_ranges


def value(number: object) -> str:
    try:
        parsed = float(number)
    except (TypeError, ValueError):
        return "NA"
    return f"{parsed:.4f}" if np.isfinite(parsed) else "NA"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path, required=True)
    parser.add_argument("--locked-dir", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--duration", choices=("10s", "30s"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frozen = json.loads((args.cv_dir / "evaluate_expanded_frozen_selection.json").read_text())
    cv = pd.read_csv(args.cv_dir / "evaluate_expanded_cv_summary.csv")
    primary = frozen["primary_feature_set"]
    focus_names = [frozen["leader"]["combination"], frozen["best_single"]["combination"], "S+D+R+P"]
    focus_names = list(dict.fromkeys(focus_names))
    selected = cv[(cv.feature_set == primary) & (cv.training_scope == "all_development")
                  & (cv.quantity.astype(str) == "all") & cv.combination.isin(focus_names)]
    selected = selected.set_index("combination")
    cohort_n, quantity_ranges = primary_quantity_ranges(args.cv_dir, primary)

    lines = [
        "# Expanded four-family source-diversity evaluation",
        "",
        "## Frozen selection result",
        "",
        "The leader below was selected before any locked-set scores were read. The fixed criterion is "
        "the equal mean of human-source-held-out macro AUC and generator-held-out macro AUC. "
        "Full-band/original-S16 results and every locked all-15 result are sensitivity or secondary analyses only.",
        "",
    ]
    rows = []
    for combo in focus_names:
        if combo not in selected.index:
            continue
        row = selected.loc[combo]
        status = "leader" if combo == frozen["leader"]["combination"] else (
            "best single" if combo == frozen["best_single"]["combination"] else "all four secondary"
        )
        rows.append([
            status, combo, value(row.human_holdout_roc_auc), value(row.generator_holdout_roc_auc),
            value(row.selection_score),
            f"[{value(row.selection_score_ci_low)}, {value(row.selection_score_ci_high)}]",
            str(bool(row.qualifies_for_selection)),
        ])
    lines.extend([md_table(
        ["status", "families", "human-source AUC", "generator AUC", "fixed score", "diagnostic CI", "qualified"],
        rows,
    ), ""])

    matched_path = args.cv_dir / "evaluate_expanded_training_scope_matched.csv"
    if matched_path.exists() and matched_path.stat().st_size:
        matched = pd.read_csv(matched_path)
        matched = matched[(matched.feature_set == primary) & (matched.quantity.astype(str) == "all")
                          & matched.combination.isin(focus_names)]
        lines.extend([
            "## Prior-only versus expanded development data",
            "",
            "These deltas use only exactly matched held-out source-pair/fold recordings. They are not "
            "differences against the older, incomparable held-out experiment.",
            "",
        ])
        rows = []
        for row in matched.itertuples(index=False):
            rows.append([
                row.fold_type, row.combination, row.matched_folds, row.matched_source_pairs,
                value(row.roc_auc__prior_only), value(row.roc_auc__all_development),
                value(row.delta_roc_auc__expanded_minus_prior),
            ])
        lines.extend([md_table(
            ["fold type", "families", "matched folds", "pairs", "prior AUC", "expanded AUC", "delta"], rows
        ), ""])

    diagnostic_path = args.cv_dir / "evaluate_expanded_missingness_diagnostic_summary.csv"
    if diagnostic_path.exists():
        diagnostic = pd.read_csv(diagnostic_path)
        diagnostic = diagnostic[diagnostic.fold_type.isin(["human_source_holdout", "generator_holdout"])]
        lines.extend([
            "## Missingness diagnostics",
            "",
            "These prespecified variants cannot alter the leader. Missingness-only performance diagnoses "
            "pipeline-availability/source fingerprints; median-only performance removes missingness indicators.",
            "",
            md_table(
                ["families", "variant", "fold type", "source-macro AUC", "source-macro BA"],
                [[row.combination, row.feature_mode, row.fold_type, value(row.roc_auc_source_macro),
                  value(row.balanced_accuracy_source_macro)] for row in diagnostic.itertuples(index=False)],
            ),
            "",
        ])

    if args.locked_dir:
        locked_path = args.locked_dir / "evaluate_expanded_locked_macro_metrics.csv"
        if locked_path.exists() and locked_path.stat().st_size:
            locked = pd.read_csv(locked_path)
            locked = locked[(locked.feature_set == primary) & locked.combination.isin(focus_names)]
            lines.extend([
                "## Locked and diagnostic slices",
                "",
                "The frozen leader is unchanged. Rows marked exploratory, stress diagnostic, or secondary "
                "must not be used for post-hoc selection.",
                "",
                md_table(
                    ["slice", "families", "status", "source pairs", "macro AUC", "macro BA"],
                    [[row.slice, row.combination, row.evaluation_status, row.source_pairs,
                      value(row.roc_auc_source_macro), value(row.balanced_accuracy_source_macro)]
                     for row in locked.itertuples(index=False)],
                ),
                "",
            ])

    lines.extend([
        "## Primary S8 cohort and realized training sizes",
        "",
        f"The metadata-qualified primary common-band cohort contains {cohort_n} rows. The table records "
        "the actual fold-level ranges after held-source and group-disjoint exclusions.",
        "",
        md_table(
            ["quantity", "train rows", "groups per source"],
            [[quantity,
              f"{quantity_ranges[quantity]['train_rows_min']}–{quantity_ranges[quantity]['train_rows_max']}",
              f"{quantity_ranges[quantity]['train_groups_per_source_min']}–"
              f"{quantity_ranges[quantity]['train_groups_per_source_max']}"]
             for quantity in ("25", "50", "100", "200", "all") if quantity in quantity_ranges],
        ),
        "",
    ])

    provided = (args.metadata is not None, args.features is not None, args.duration is not None)
    if any(provided) and not all(provided):
        raise ValueError("--metadata, --features, and --duration must be supplied together")
    if all(provided):
        availability = family_availability_by_source(args.metadata, args.features, args.duration)
        lines.extend([
            "## Family observability and processing failures by source",
            "",
            "Natural unobservability is kept separate from processing failure. Missing R/P measurements "
            "remain missing and are never reported as zero. Beat-serialization incidents are an R "
            "processing failure and are also shown independently as a dependency incident; an accompanying "
            "P natural-unobservability status is retained rather than relabelled.",
            "",
            md_table(
                ["source", "role", "family", "N", "natural unobservable", "processing failure",
                 "beat dependency incident"],
                [[row.source_group, row.role, row.family, row.rows,
                  f"{100.0*row.natural_unobservable_rate:.1f}%",
                  f"{100.0*row.processing_failure_rate:.1f}%",
                  f"{100.0*row.beat_dependency_failure_rate:.1f}%"]
                 for row in availability.itertuples(index=False)],
            ),
            "",
        ])

    lines.extend([
        "## Guardrails",
        "",
        "- Locked, stress, provisional AudioX, and DiffRhythm pilot rows never enter model selection.",
        "- Common-band S8 is the only selectable spectral definition; native S16 and uncontrolled original S16 are sensitivities.",
        "- Quantities are nested deterministic creator/prompt/condition-group samples within source, with class/source/group-equal training weights.",
        "- Thirty-second S/D/R/P comparisons use the duration-qualified unpadded cohort; they are not compared with ten-second cohorts.",
        "- Provisional AudioX scores remain exploratory until labels are verified.",
        "- Final uncertainty intervals must resample group_id blocks; optional held-out-source bootstrap columns are diagnostic only.",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
