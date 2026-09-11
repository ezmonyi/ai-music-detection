#!/usr/bin/env python3
"""Export English LaTeX result fragments from completed formal evaluation CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_expanded_reporting import family_availability_by_source, primary_quantity_ranges


QUANTITIES = ("25", "50", "100", "200", "all")
SINGLES = ("S", "D", "R", "P")
FAMILY_ORDER = SINGLES
SOURCE_DISPLAY = {
    "ai_audiox_third_party": "AudioX (unverified)",
    "ai_diffrhythm_pilot": "DiffRhythm pilot",
    "human_deam": "DEAM", "human_gtzan": "GTZAN",
    "human_hindustani_raag_hf": "Hindustani Raga", "human_maestro_v3": "MAESTRO",
    "human_magnatagatune": "MagnaTagATune", "human_medleydb": "MedleyDB",
    "human_moisesdb": "MoisesDB", "human_musicnet": "MusicNet", "human_urmp": "URMP",
}
ROLE_DISPLAY = {
    "development": "Development", "locked": "Locked", "pilot": "Pilot",
    "provisional": "Provisional", "stress": "Stress",
}
SLICE_DISPLAY = {
    "new_catalogue_plus_legacy_locked_ai": "New Human / legacy AI",
    "new_catalogue_plus_diff_rhythm_pilot": "New Human / DiffRhythm",
}


def source_display(value: object) -> str:
    return SOURCE_DISPLAY.get(str(value), str(value))


def role_display(value: object) -> str:
    return ROLE_DISPLAY.get(str(value), str(value).replace("_", " ").title())


def spectral_display(value: object) -> str:
    name = str(value)
    if name.startswith("common8_"):
        return "S8, common band"
    if name.startswith("native16_"):
        return r"S16, native $\geq 40$ kHz"
    if name.startswith("original_s16_uncontrolled_"):
        return "S16, all rates"
    return esc(name)


def combination_order(name: str) -> tuple[int, tuple[int, ...]]:
    parts = name.split("+")
    return len(parts), tuple(FAMILY_ORDER.index(part) for part in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", choices=("10s", "30s"), required=True)
    parser.add_argument("--cv-dir", type=Path, required=True)
    parser.add_argument("--locked-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-dir", type=Path, required=True)
    parser.add_argument("--formal-audit", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def esc(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
        "#": r"\#", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def number(value: object, digits: int = 3) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "--"
    return f"{parsed:.{digits}f}" if np.isfinite(parsed) else "--"


def boolean(value: object) -> str:
    if isinstance(value, str):
        return "yes" if value.strip().lower() == "true" else "no"
    return "yes" if bool(value) else "no"


def write_table(path: Path, columns: str, header: list[str], rows: list[list[object]],
                note: str | None = None) -> None:
    lines = [r"\begin{tabular}{" + columns + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if note:
        lines.extend([r"\par\smallskip", r"{\footnotesize " + note + "}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_longtable(path: Path, columns: str, header: list[str], rows: list[list[object]],
                    note: str) -> None:
    heading = " & ".join(header) + r" \\"
    lines = [r"\begin{longtable}{" + columns + "}", r"\toprule", heading, r"\midrule",
             r"\endfirsthead", r"\toprule", heading, r"\midrule", r"\endhead"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{longtable}", r"{\footnotesize " + note + "}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit = json.loads(args.formal_audit.read_text(encoding="utf-8"))
    if audit.get("status") not in {"formal_scoring_complete", "formal_complete"}:
        raise ValueError("LaTeX export requires a completed formal orchestration audit")
    frozen_path = args.cv_dir / "evaluate_expanded_frozen_selection.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    leader = str(frozen["leader"]["combination"])
    primary = str(frozen["primary_feature_set"])
    cv = pd.read_csv(args.cv_dir / "evaluate_expanded_cv_summary.csv", low_memory=False)
    primary_cv = cv[(cv.feature_set == primary)
                    & (cv.training_scope == "all_development")].copy()
    if primary_cv.combination.nunique() != 15 or set(primary_cv.quantity.astype(str)) != set(QUANTITIES):
        raise ValueError("Formal primary CV table is not the required 15 x 5 grid")
    primary_cv["fixed_ba"] = 0.5 * (
        primary_cv.human_holdout_balanced_accuracy + primary_cv.generator_holdout_balanced_accuracy
    )
    primary_cv["selection_j"] = primary_cv.selection_score
    if not np.allclose(primary_cv.selection_j, 0.5 * (
        primary_cv.human_holdout_roc_auc + primary_cv.generator_holdout_roc_auc
    ), rtol=0.0, atol=1e-15, equal_nan=True):
        raise ValueError("Stored selection J differs from the frozen equal-mean AUC definition")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"results_{args.duration}"
    selected_names = list(dict.fromkeys([*SINGLES, leader, "S+D+R+P"]))
    all_rows = primary_cv[primary_cv.quantity.astype(str).eq("all")].set_index("combination")
    summary_rows = []
    for name in selected_names:
        if name not in all_rows.index:
            raise ValueError(f"Required formal comparison is absent: {name}")
        row = all_rows.loc[name]
        status = "frozen leader" if name == leader else (
            "all four secondary" if name == "S+D+R+P" else "single family"
        )
        summary_rows.append([
            esc(name), status, number(row.human_holdout_roc_auc),
            number(row.generator_holdout_roc_auc), number(row.selection_j),
            boolean(row.qualifies_for_selection),
        ])
    write_table(
        args.output_dir / f"{prefix}_selection.tex", "llrrrr",
        ["Families", "Status", "Human-source AUC", "AI-source AUC", "$J$", "Qualified"],
        summary_rows,
        note=("All values use the full development quantity. The frozen criterion is "
              "$J=0.5\\,\\mathrm{AUC}_{Human}+0.5\\,\\mathrm{AUC}_{AI}$."),
    )

    coverage = pd.read_csv(args.cv_dir / "evaluate_expanded_cv_coverage.csv", low_memory=False)
    fold_manifest = pd.read_csv(args.cv_dir / "evaluate_expanded_fold_manifest.csv", low_memory=False)
    sensitivity_rows: list[dict[str, object]] = []
    feature_order = list(frozen["family_config"]["feature_sets"])
    for feature_set in feature_order:
        current = cv[(cv.feature_set == feature_set) & (cv.training_scope == "all_development")
                     & cv.quantity.astype(str).eq("all")].copy()
        current["fixed_ba"] = 0.5 * (
            current.human_holdout_balanced_accuracy + current.generator_holdout_balanced_accuracy
        )
        current["selection_j"] = current.selection_score
        s_row = current[current.combination == "S"]
        leader_row = current[current.combination == leader]
        if len(s_row) != 1 or len(leader_row) != 1:
            raise ValueError(f"Spectral sensitivity lacks S/leader results for {feature_set}")
        coverage_row = coverage[(coverage.feature_set == feature_set) & coverage.role_scope.eq("cv")]
        if len(coverage_row) != 1:
            raise ValueError(f"Spectral sensitivity lacks unique CV coverage for {feature_set}")
        train = fold_manifest[
            (fold_manifest.feature_set == feature_set)
            & (fold_manifest.training_scope == "all_development")
            & fold_manifest.quantity.astype(str).eq("all")
            & fold_manifest.status.eq("evaluated")
            & fold_manifest.fold_type.isin(["human_source_holdout", "generator_holdout"])
        ]
        if train.empty:
            raise ValueError(f"Spectral sensitivity lacks source-held-out train counts for {feature_set}")
        spec = frozen["family_config"]["feature_sets"][feature_set]
        sensitivity_rows.append({
            "feature_set": feature_set,
            "cohort": spec.get("cohort", "unspecified"),
            "selection_status": "primary selectable" if feature_set == primary else "descriptive sensitivity only",
            "cohort_n": int(coverage_row.rows_after_metadata_eligibility.iloc[0]),
            "train_n_min": int(pd.to_numeric(train.train_rows).min()),
            "train_n_max": int(pd.to_numeric(train.train_rows).max()),
            "s_only_j": float(s_row.selection_j.iloc[0]),
            "frozen_primary_leader_combination": leader,
            "frozen_primary_leader_j": float(leader_row.selection_j.iloc[0]),
        })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(args.output_dir / f"{prefix}_spectral_sensitivity.csv", index=False)
    write_table(
        args.output_dir / f"{prefix}_spectral_sensitivity.tex", "llrrrr",
        ["Spectral representation", "Status", "Cohort $N$", "Train $N$ range", "S-only $J$",
         f"{esc(leader)} $J$"],
        [[spectral_display(row.feature_set),
          "Primary" if row.feature_set == primary else "Sensitivity", int(row.cohort_n),
          f"{int(row.train_n_min)}--{int(row.train_n_max)}", number(row.s_only_j),
          number(row.frozen_primary_leader_j)] for row in sensitivity.itertuples(index=False)],
        note=("Common-band S8 is the only selectable representation. Native S16 uses a native sample-rate "
              "gate of at least 40 kHz and therefore a different cohort; original S16 over all rates is "
              "bandwidth-uncontrolled. $J$ retains the frozen equal-mean AUC definition. These rows are "
              "descriptive and are not a paired causal bandwidth comparison."),
    )

    cohort_n, ranges = primary_quantity_ranges(args.cv_dir, primary)
    combos = sorted(primary_cv.combination.unique(), key=combination_order)
    matrix_rows = []
    for combo in combos:
        current = primary_cv[primary_cv.combination == combo].copy()
        current.index = current.quantity.astype(str)
        matrix_rows.append([esc(combo), *[number(current.loc[q, "selection_j"]) for q in QUANTITIES]])
    j_matrix_csv = primary_cv.pivot(index="combination", columns="quantity", values="selection_j")
    j_matrix_csv = j_matrix_csv.reindex(index=combos, columns=list(QUANTITIES))
    j_matrix_csv.index.name = "families"
    j_matrix_csv.to_csv(args.output_dir / f"{prefix}_j_matrix.csv")
    range_note = "; ".join(
        f"{q}: rows {ranges[q]['train_rows_min']}--{ranges[q]['train_rows_max']}, "
        f"groups/source {ranges[q]['train_groups_per_source_min']}--"
        f"{ranges[q]['train_groups_per_source_max']}" for q in QUANTITIES if q in ranges
    )
    write_table(
        args.output_dir / f"{prefix}_j_matrix.tex", "lrrrrr",
        ["Families", *QUANTITIES], matrix_rows,
        note=("Entries are the frozen $J=0.5\\,\\mathrm{AUC}_{Human}+0.5\\,\\mathrm{AUC}_{AI}$. "
              f"Primary S8 cohort $N={cohort_n}$. "
              f"Actual training ranges: {range_note}."),
    )
    ba_rows = []
    for combo in combos:
        current = primary_cv[primary_cv.combination == combo].copy()
        current.index = current.quantity.astype(str)
        ba_rows.append([esc(combo), *[number(current.loc[q, "fixed_ba"]) for q in QUANTITIES]])
    write_table(
        args.output_dir / f"{prefix}_ba_matrix.tex", "lrrrrr",
        ["Families", *QUANTITIES], ba_rows,
        note=("Entries are balanced accuracy from the fixed equal mean of Human-source and AI-source "
              f"transfer evaluations. Primary S8 cohort $N={cohort_n}$. "
              f"Actual training ranges: {range_note}."),
    )

    matched = pd.read_csv(args.cv_dir / "evaluate_expanded_training_scope_matched.csv", low_memory=False)
    matched = matched[(matched.feature_set == primary) & matched.quantity.astype(str).eq("all")
                      & matched.combination.isin({leader, "S"})]
    prior_rows = []
    for combo, current in matched.groupby("combination", sort=True):
        human = current[current.fold_type == "human_source_holdout"]
        ai = current[current.fold_type == "generator_holdout"]
        if len(human) != 1 or len(ai) != 1:
            raise ValueError(f"Matched prior delta lacks Human/AI held-out rows for {combo}")
        delta_human = float(human.delta_roc_auc__expanded_minus_prior.iloc[0])
        delta_ai = float(ai.delta_roc_auc__expanded_minus_prior.iloc[0])
        prior_rows.append([
            esc(combo), int(human.matched_source_pairs.iloc[0]), int(ai.matched_source_pairs.iloc[0]),
            number(delta_human), number(delta_ai), number(0.5 * (delta_human + delta_ai)),
        ])
    write_table(
        args.output_dir / f"{prefix}_prior_delta.tex", "lrrrrr",
        ["Families", "Human pairs", "AI pairs", "$\\Delta$Human AUC", "$\\Delta$AI AUC", "$\\Delta J$"],
        prior_rows,
        note=("Deltas are expanded-development minus prior-only training on exactly matched held-out "
              "recordings; $\\Delta J$ is the equal mean of the two AUC deltas."),
    )

    locked = pd.read_csv(
        args.locked_dir / "evaluate_expanded_locked_macro_metrics.csv", low_memory=False
    )
    if locked.empty or not ((locked.feature_set == primary) & locked.combination.isin({leader, "S"})).any():
        raise ValueError("Locked primary leader/S results are absent")
    bootstrap = pd.read_csv(
        args.bootstrap_dir / "evaluate_expanded_group_bootstrap_pairs.csv", low_memory=False
    )
    if set(pd.to_numeric(bootstrap.replicates, errors="coerce").dropna().astype(int)) != {2000}:
        raise ValueError("Formal locked specificity export requires 2,000 bootstrap replicates per pair")
    locked_rows = []
    for row in bootstrap.sort_values(["slice", "human_source", "ai_source"]).itertuples(index=False):
        locked_rows.append([
            esc(SLICE_DISPLAY.get(str(row.slice), str(row.slice))),
            esc(source_display(row.human_source)), esc(source_display(row.ai_source)),
            f"{number(row.leader_human_specificity)} [{number(row.leader_human_specificity_ci_low)}, "
            f"{number(row.leader_human_specificity_ci_high)}]",
            f"{number(row.s_only_human_specificity)} [{number(row.s_only_human_specificity_ci_low)}, "
            f"{number(row.s_only_human_specificity_ci_high)}]",
            f"{number(row.delta_human_specificity__leader_minus_s)} "
            f"[{number(row.delta_human_specificity__leader_minus_s_ci_low)}, "
            f"{number(row.delta_human_specificity__leader_minus_s_ci_high)}]",
        ])
    write_longtable(
        args.output_dir / f"{prefix}_locked_specificity.tex",
        r"p{2.6cm}p{1.8cm}p{1.8cm}p{2.7cm}p{2.7cm}p{2.7cm}",
        ["Slice", "Human source", "AI source", "Leader specificity [95\\% CI]",
         "S-only specificity [95\\% CI]", "$\\Delta$ specificity [95\\% CI]"],
        locked_rows,
        note="Percentile intervals use 2,000 independent group-ID block bootstrap replicates.",
    )

    availability = family_availability_by_source(args.metadata, args.features, args.duration)
    availability.insert(2, "source_display", availability.source_group.map(source_display))
    availability.insert(4, "role_display", availability.role.map(role_display))
    availability.to_csv(args.output_dir / f"{prefix}_family_availability_by_source.csv", index=False)
    write_longtable(
        args.output_dir / f"{prefix}_family_availability_by_source.tex", "lllrrrr",
        ["Source", "Role", "Family", "$N$", "Natural unobservable", "Processing failure",
         "Beat dependency incident"],
        [[esc(row.source_display), esc(row.role_display), row.family, int(row.rows),
          f"{100.0*row.natural_unobservable_rate:.1f}\\%",
          f"{100.0*row.processing_failure_rate:.1f}\\%",
          f"{100.0*row.beat_dependency_failure_rate:.1f}\\%"]
         for row in availability.itertuples(index=False)],
        note=("Rates are derived from explicit family status fields. Natural unobservability and processing "
              "failure are reported separately; missing P or R measurements are never replaced by zero. "
              "Beat-serialization incidents count as R processing failures and are shown independently; "
              "an accompanying P natural-unobservability status is retained."),
    )
    s16 = availability[availability.family == "S16"].copy()
    s16.to_csv(args.output_dir / f"{prefix}_s16_availability_by_source.csv", index=False)
    compact_rows = []
    compact = availability[availability.family.isin(["S8", "D", "R", "P"])]
    for (source, role), current in compact.groupby(["source_group", "role"], sort=True):
        by_family = current.set_index("family")
        if set(by_family.index) != {"S8", "D", "R", "P"}:
            raise ValueError(f"Compact availability lacks a family for {source}/{role}")
        compact_rows.append([
            esc(source_display(source)), esc(role_display(role)), int(by_family.rows.iloc[0]),
            *[f"{100.0*by_family.loc[family, 'natural_unobservable_rate']:.1f}/"
              f"{100.0*by_family.loc[family, 'processing_failure_rate']:.1f}\\%"
              for family in ("S8", "D", "R", "P")],
        ])
    write_longtable(
        args.output_dir / f"{prefix}_family_availability_compact.tex", "llrrrrr",
        ["Source", "Role", "$N$", "S8 N/F", "D N/F", "R N/F", "P N/F"], compact_rows,
        note=("Each family cell is natural-unobservable/processing-failure percent (N/F). Full counts, "
              "status vocabularies, beat-dependency incident rates, and S16 results are retained in the "
              "companion CSV files; missing measurements are never converted to zero."),
    )

    manifest = {
        "duration": args.duration,
        "formal_audit": str(args.formal_audit.resolve()),
        "primary_feature_set": primary,
        "frozen_leader": leader,
        "selection_changed": False,
        "fragments": sorted(path.name for path in args.output_dir.glob(f"{prefix}_*.tex")),
        "tables_csv": sorted(path.name for path in args.output_dir.glob(f"{prefix}_*.csv")),
    }
    (args.output_dir / f"{prefix}_latex_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
