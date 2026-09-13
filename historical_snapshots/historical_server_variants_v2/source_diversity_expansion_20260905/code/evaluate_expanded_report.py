#!/usr/bin/env python3
"""Build a concise, selection-safe Markdown report from expanded evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
