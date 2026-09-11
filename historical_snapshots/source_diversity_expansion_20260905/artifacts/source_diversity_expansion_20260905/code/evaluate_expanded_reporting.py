"""Shared reporting-only summaries for formal expanded evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


STATUS_COLUMNS = {
    "S16": "s16_feature_status", "S8": "s8_feature_status",
    "D": "d_feature_status", "R": "r_feature_status", "P": "p_feature_status",
}
PROCESSING_FAILURE_STATUSES = {
    "processing_failure", "missing_inference_output", "missing_feature_source",
}


def id_column(table: pd.DataFrame) -> str:
    candidates = [name for name in ("item_id", "track", "id", "row_id") if name in table]
    if not candidates:
        raise ValueError("Table has no item_id/track/id/row_id")
    return candidates[0]


def family_availability_by_source(
    metadata_path: Path, features_path: Path, duration: str,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path, low_memory=False)
    features = pd.read_csv(features_path, low_memory=False)
    metadata_id, feature_id = id_column(metadata), id_column(features)
    required_metadata = {"source_group", "role"}
    if not required_metadata.issubset(metadata):
        raise ValueError(f"Metadata lacks reporting fields: {sorted(required_metadata-set(metadata))}")
    missing_status = [
        column for column in (*STATUS_COLUMNS.values(), "feature_status")
        if column not in features
    ]
    if missing_status:
        raise ValueError(f"Features lack reporting status fields: {missing_status}")
    left = metadata[[metadata_id, "source_group", "role"]].copy()
    left[metadata_id] = left[metadata_id].astype(str)
    right = features[[feature_id, *STATUS_COLUMNS.values(), "feature_status"]].copy()
    right[feature_id] = right[feature_id].astype(str)
    if metadata_id != "__id":
        left = left.rename(columns={metadata_id: "__id"})
    if feature_id != "__id":
        right = right.rename(columns={feature_id: "__id"})
    merged = left.merge(right, on="__id", how="inner", validate="one_to_one")
    if len(merged) != len(features):
        raise ValueError("Not every feature row maps one-to-one to reporting metadata")
    rows = []
    for (source, role), current in merged.groupby(["source_group", "role"], sort=True):
        overall_status = current["feature_status"].astype(str).str.strip()
        beat_dependency_failure = overall_status.eq("accounted_serialization_failure")
        for family, column in STATUS_COLUMNS.items():
            status = current[column].astype(str).str.strip()
            natural = status.str.startswith("unavailable_")
            failed = status.isin(PROCESSING_FAILURE_STATUSES)
            # The Beat This serialization incident is represented in the raw
            # family columns as an R/P natural-unavailability status.  For R,
            # the failed beat serialization is the direct cause of the absent
            # rhythm measurement, so report it as processing failure and do
            # not also count it as naturally unobservable.  P remains a valid
            # natural-unobservability observation (too few sections/spans),
            # with the co-occurring beat dependency incident reported in its
            # own column rather than being silently relabelled as P failure.
            dependency = beat_dependency_failure if family in {"R", "P"} else pd.Series(
                False, index=current.index
            )
            if family == "R":
                natural = natural & ~dependency
                failed = failed | dependency
            rows.append({
                "duration": duration, "source_group": source, "role": role, "family": family,
                "rows": int(len(current)),
                "natural_unobservable_count": int(natural.sum()),
                "natural_unobservable_rate": float(natural.mean()),
                "processing_failure_count": int(failed.sum()),
                "processing_failure_rate": float(failed.mean()),
                "beat_dependency_failure_count": int(dependency.sum()),
                "beat_dependency_failure_rate": float(dependency.mean()),
                "status_counts": "|".join(
                    f"{key}:{value}" for key, value in status.value_counts().sort_index().items()
                ),
            })
    return pd.DataFrame(rows)


def primary_quantity_ranges(cv_dir: Path, primary: str) -> tuple[int, dict[str, dict[str, int]]]:
    coverage = pd.read_csv(cv_dir / "evaluate_expanded_cv_coverage.csv", low_memory=False)
    coverage_row = coverage[(coverage.feature_set == primary) & coverage.role_scope.eq("cv")]
    if len(coverage_row) != 1:
        raise ValueError("Primary CV coverage row is absent or duplicated")
    cohort_n = int(coverage_row.rows_after_metadata_eligibility.iloc[0])
    folds = pd.read_csv(cv_dir / "evaluate_expanded_fold_manifest.csv", low_memory=False)
    selected = folds[(folds.feature_set == primary) & (folds.training_scope == "all_development")
                     & folds.status.eq("evaluated")
                     & folds.fold_type.isin(["human_source_holdout", "generator_holdout"])]
    ranges: dict[str, dict[str, int]] = {}
    for quantity, current in selected.groupby(selected.quantity.astype(str)):
        ranges[quantity] = {
            "train_rows_min": int(pd.to_numeric(current.train_rows).min()),
            "train_rows_max": int(pd.to_numeric(current.train_rows).max()),
            "train_groups_per_source_min": int(pd.to_numeric(current.train_min_groups_per_source).min()),
            "train_groups_per_source_max": int(pd.to_numeric(current.train_max_groups_per_source).max()),
        }
    return cohort_n, ranges
