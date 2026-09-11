#!/usr/bin/env python3
"""Paired global-group bootstrap for the frozen 30-second F/H evaluation.

This module performs no fitting and never reads historical labels.  It filters
the frozen development source-holdout predictions to quantity ``all`` and the
four preregistered combinations::

    S+D+R, S+D+R+F, S+D+R+H, S+D+R+F+H

Each bootstrap replicate samples the *global* ``group_id`` universe with
replacement.  The resulting group multiplicity is applied unchanged to every
row carrying that group, including rows from different models, folds, sources,
or classes.  This is important because a global group can cross class/source
boundaries.  Models and the fixed 0.5 threshold remain frozen.

Metrics are weighted row-level AUC (Mann--Whitney, average credit for exact
score ties) and weighted balanced accuracy at 0.5.  Source-pair/fold metrics are
averaged within held-out source, then equally across held-out sources within
Human-source and AI-generator holdout directions.  ``J`` is the equal mean of
the two directional AUC macros.  Undefined resampled pair cells are retained as
explicit audit rows; macro means follow the evaluator's NaN-skipping behavior.

The resulting intervals are conditional cluster-bootstrap uncertainty for the
already trained models and observed source collection.  They are not population
intervals and give no guarantee for future generators or unseen source domains.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np


BASELINE = "S+D+R"
ADDED = ("S+D+R+F", "S+D+R+H", "S+D+R+F+H")
COMBINATIONS = (BASELINE,) + ADDED
FOLD_TYPES = ("human_source_holdout", "generator_holdout")
CELL_KEY_FIELDS = (
    "fold_uid", "fold_index", "fold_type", "heldout_source",
    "human_source", "ai_source", "opposite_group_fold",
)
DEFAULT_REPLICATES = 1000
DEFAULT_SEED = 20260907
THRESHOLD = 0.5
SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def finite_float(value: str, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field}: {value!r}")
    return result


def weighted_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    """Weighted binary AUC with exact-score ties receiving one-half credit."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or weights.shape != labels.shape:
        raise ValueError("labels, scores, and weights must be aligned one-dimensional arrays")
    if not np.isfinite(scores).all() or not np.isfinite(weights).all():
        raise ValueError("scores and weights must be finite")
    if (weights < 0).any() or not np.isin(labels, (0, 1)).all():
        raise ValueError("weights must be nonnegative and labels binary")
    positive_total = float(weights[labels == 1].sum())
    negative_total = float(weights[labels == 0].sum())
    if positive_total <= 0.0 or negative_total <= 0.0:
        return math.nan

    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ordered_weights = weights[order]
    numerator = 0.0
    cumulative_negative = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        block_labels = ordered_labels[start:stop]
        block_weights = ordered_weights[start:stop]
        block_positive = float(block_weights[block_labels == 1].sum())
        block_negative = float(block_weights[block_labels == 0].sum())
        numerator += block_positive * (cumulative_negative + 0.5 * block_negative)
        cumulative_negative += block_negative
        start = stop
    return numerator / (positive_total * negative_total)


def weighted_metrics(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    positive = labels == 1
    negative = labels == 0
    positive_total = float(weights[positive].sum())
    negative_total = float(weights[negative].sum())
    predicted = scores >= threshold
    sensitivity = (
        float(weights[positive & predicted].sum() / positive_total)
        if positive_total > 0.0 else math.nan
    )
    specificity = (
        float(weights[negative & ~predicted].sum() / negative_total)
        if negative_total > 0.0 else math.nan
    )
    balanced = (
        0.5 * (sensitivity + specificity)
        if math.isfinite(sensitivity) and math.isfinite(specificity) else math.nan
    )
    return {
        "roc_auc": weighted_auc(labels, scores, weights),
        "balanced_accuracy": balanced,
        "ai_sensitivity": sensitivity,
        "human_specificity": specificity,
        "human_weight": negative_total,
        "ai_weight": positive_total,
    }


def draw_group_multiplicities(
    group_count: int, replicates: int, seed: int,
) -> np.ndarray:
    """Return global-group bootstrap multiplicities, shape (replicates, groups)."""
    if group_count < 1 or replicates < 1:
        raise ValueError("group_count and replicates must be positive")
    rng = np.random.default_rng(seed)
    probability = np.full(group_count, 1.0 / group_count, dtype=float)
    return rng.multinomial(group_count, probability, size=replicates)


def _replicate_metrics(
    labels: np.ndarray, scores: np.ndarray, group_indices: np.ndarray,
    multiplicities: np.ndarray, threshold: float = THRESHOLD,
) -> dict[str, np.ndarray]:
    """Vectorized weighted metrics for one fixed source-pair cell."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    group_indices = np.asarray(group_indices, dtype=int)
    row_weights = multiplicities[:, group_indices].astype(float, copy=False)
    positive = labels == 1
    negative = labels == 0
    positive_total = row_weights[:, positive].sum(axis=1)
    negative_total = row_weights[:, negative].sum(axis=1)
    predicted = scores >= threshold
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = row_weights[:, positive & predicted].sum(axis=1) / positive_total
        specificity = row_weights[:, negative & ~predicted].sum(axis=1) / negative_total
    sensitivity[positive_total <= 0] = np.nan
    specificity[negative_total <= 0] = np.nan
    balanced = 0.5 * (sensitivity + specificity)

    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ordered_weights = row_weights[:, order]
    numerator = np.zeros(len(multiplicities), dtype=float)
    cumulative_negative = np.zeros(len(multiplicities), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        block_labels = ordered_labels[start:stop]
        block_weights = ordered_weights[:, start:stop]
        block_positive = block_weights[:, block_labels == 1].sum(axis=1)
        block_negative = block_weights[:, block_labels == 0].sum(axis=1)
        numerator += block_positive * (cumulative_negative + 0.5 * block_negative)
        cumulative_negative += block_negative
        start = stop
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = numerator / (positive_total * negative_total)
    auc[(positive_total <= 0) | (negative_total <= 0)] = np.nan
    return {
        "roc_auc": auc,
        "balanced_accuracy": balanced,
        "ai_sensitivity": sensitivity,
        "human_specificity": specificity,
        "human_weight": negative_total,
        "ai_weight": positive_total,
    }


def _quantity_all(value: str) -> bool:
    return str(value).strip().lower() == "all"


def _cell_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in CELL_KEY_FIELDS)


def _read_filtered_csv(path: Path, combinations: Iterable[str]) -> list[dict[str, str]]:
    wanted = set(combinations)
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        required = {"combination", "quantity"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV lacks {sorted(required - set(reader.fieldnames))}: {path}")
        for row in reader:
            if row["combination"] in wanted and _quantity_all(row["quantity"]):
                rows.append(row)
    if not rows:
        raise ValueError(f"No quantity=all target rows found in {path}")
    return rows


def _read_target_deltas(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        required = {"added_combination", "quantity", "comparison_id"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"Delta CSV lacks {sorted(required - set(reader.fieldnames))}: {path}")
        for row in reader:
            if row["added_combination"] in ADDED and _quantity_all(row["quantity"]):
                rows.append(row)
    if not rows:
        raise ValueError(f"No quantity=all target matched deltas found in {path}")
    return rows


def load_inputs(evaluation_dir: Path) -> dict[str, Any]:
    manifest_path = evaluation_dir / "run_manifest.json"
    prediction_path = evaluation_dir / "development_source_holdout_predictions.csv"
    metric_path = evaluation_dir / "development_source_holdout_metrics_by_source_pair.csv"
    summary_path = evaluation_dir / "development_source_holdout_summary.csv"
    delta_path = evaluation_dir / "development_source_holdout_matched_deltas.csv"
    plan_path = evaluation_dir / "evaluation_plan.csv"
    inputs = (manifest_path, prediction_path, metric_path, summary_path, delta_path, plan_path)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Evaluation is incomplete; missing: " + ", ".join(missing))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("contract", {})
    parameters = contract.get("parameters", {})
    if manifest.get("stage") != "dev":
        raise ValueError("Bootstrap accepts a completed development evaluation only")
    if manifest.get("authorization", {}).get("status") != "frozen":
        raise ValueError("Development evaluation was not authorized by a frozen receipt")
    if parameters.get("threshold_policy") != "fixed_0.5":
        raise ValueError("This frozen analysis requires threshold_policy=fixed_0.5")
    if parameters.get("model") != "ridge":
        raise ValueError("This frozen analysis requires the preregistered ridge model")
    if int(parameters.get("seed", -1)) != DEFAULT_SEED:
        raise ValueError("Evaluation seed differs from the preregistered 20260907 seed")
    if not set(COMBINATIONS).issubset(set(_plan_combinations(plan_path))):
        raise ValueError("Evaluation plan lacks one or more preregistered combinations")
    return {
        "manifest": manifest,
        "paths": inputs,
        "predictions": _read_filtered_csv(prediction_path, COMBINATIONS),
        "metrics": _read_filtered_csv(metric_path, COMBINATIONS),
        "summary": _read_filtered_csv(summary_path, COMBINATIONS),
        "deltas": _read_target_deltas(delta_path),
        "plan_path": plan_path,
    }


def _plan_combinations(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["combination"] for row in csv.DictReader(handle)]


def build_observations(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_combination: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        combination: {} for combination in COMBINATIONS
    }
    metadata_fields = (
        "row_id", "label", "source_group", "group_id", "fold_uid", "fold_index",
        "fold_type", "heldout_source", "opposite_group_fold",
    )
    for row in rows:
        combination = row["combination"]
        if combination not in by_combination:
            continue
        if row["fold_type"] not in FOLD_TYPES:
            raise ValueError(f"Unexpected fold type in source-holdout predictions: {row['fold_type']}")
        threshold = finite_float(row["threshold"], "threshold")
        if threshold != THRESHOLD:
            raise ValueError(f"Expected frozen threshold 0.5, found {threshold}")
        key = (row["fold_uid"], row["row_id"])
        if key in by_combination[combination]:
            raise ValueError(f"Duplicate prediction identity for {combination}: {key}")
        record = {field: str(row[field]) for field in metadata_fields}
        record["label"] = int(row["label"])
        record["score"] = finite_float(row["score"], "score")
        by_combination[combination][key] = record

    baseline = by_combination[BASELINE]
    if not baseline:
        raise ValueError("Baseline predictions are absent")
    baseline_keys = set(baseline)
    for combination, records in by_combination.items():
        if set(records) != baseline_keys:
            raise ValueError(f"Prediction identities differ for {combination} versus {BASELINE}")
        for key in baseline_keys:
            left, right = baseline[key], records[key]
            for field in metadata_fields:
                if left[field] != right[field]:
                    raise ValueError(f"Prediction metadata differs for {combination}, {key}, {field}")

    ordered_keys = sorted(baseline_keys)
    master = [baseline[key] for key in ordered_keys]
    groups = sorted({str(row["group_id"]) for row in master})
    group_to_index = {group: index for index, group in enumerate(groups)}
    return {
        "keys": ordered_keys,
        "master": master,
        "groups": groups,
        "group_indices": np.asarray([group_to_index[str(row["group_id"])] for row in master]),
        "scores": {
            combination: np.asarray([by_combination[combination][key]["score"] for key in ordered_keys])
            for combination in COMBINATIONS
        },
    }


def build_cells(metric_rows: list[dict[str, str]], observations: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_rows = [row for row in metric_rows if row["combination"] == BASELINE]
    if not baseline_rows:
        raise ValueError("Baseline source-pair metric rows are absent")
    master = observations["master"]
    by_fold: dict[str, list[int]] = {}
    for index, row in enumerate(master):
        by_fold.setdefault(str(row["fold_uid"]), []).append(index)
    cells: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in sorted(baseline_rows, key=_cell_key):
        key = _cell_key(row)
        if key in seen:
            raise ValueError(f"Duplicate baseline metric cell: {key}")
        seen.add(key)
        indices = np.asarray([
            index for index in by_fold.get(row["fold_uid"], [])
            if (
                (master[index]["label"] == 0 and master[index]["source_group"] == row["human_source"])
                or (master[index]["label"] == 1 and master[index]["source_group"] == row["ai_source"])
            )
        ], dtype=int)
        if not len(indices):
            raise ValueError(f"No prediction rows for metric cell: {key}")
        labels = np.asarray([master[index]["label"] for index in indices], dtype=int)
        group_count_human = len({master[index]["group_id"] for index in indices if master[index]["label"] == 0})
        group_count_ai = len({master[index]["group_id"] for index in indices if master[index]["label"] == 1})
        expected = (
            int(row["test_human"]), int(row["test_ai"]),
            int(row["test_human_groups"]), int(row["test_ai_groups"]),
        )
        actual = (
            int((labels == 0).sum()), int((labels == 1).sum()),
            group_count_human, group_count_ai,
        )
        if actual != expected:
            raise ValueError(f"Prediction/metric cell count mismatch for {key}: {actual} != {expected}")
        cells.append({"key": key, "row": row, "indices": indices})
    return cells


def compute_cell_metrics(
    observations: dict[str, Any], cells: list[dict[str, Any]], multiplicities: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    master = observations["master"]
    all_group_indices = observations["group_indices"]
    result: dict[str, dict[str, list[np.ndarray]]] = {
        combination: {"roc_auc": [], "balanced_accuracy": []} for combination in COMBINATIONS
    }
    for cell in cells:
        indices = cell["indices"]
        labels = np.asarray([master[index]["label"] for index in indices], dtype=int)
        cell_group_indices = all_group_indices[indices]
        for combination in COMBINATIONS:
            values = _replicate_metrics(
                labels, observations["scores"][combination][indices], cell_group_indices,
                multiplicities, THRESHOLD,
            )
            result[combination]["roc_auc"].append(values["roc_auc"])
            result[combination]["balanced_accuracy"].append(values["balanced_accuracy"])
    return {
        combination: {
            metric: np.column_stack(columns) for metric, columns in metrics.items()
        }
        for combination, metrics in result.items()
    }


def _mean_columns(values: np.ndarray, columns: list[int]) -> np.ndarray:
    selected = values[:, columns]
    count = np.isfinite(selected).sum(axis=1)
    total = np.nansum(selected, axis=1)
    output = np.full(len(values), np.nan, dtype=float)
    np.divide(total, count, out=output, where=count > 0)
    return output


def aggregate_macro(
    cell_values: np.ndarray, cells: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Mirror evaluator: pair/fold mean per held source, then held-source macro."""
    source_columns: dict[tuple[str, str], list[int]] = {}
    for index, cell in enumerate(cells):
        row = cell["row"]
        source_columns.setdefault((row["fold_type"], row["heldout_source"]), []).append(index)
    source_values = {
        key: _mean_columns(cell_values, columns) for key, columns in source_columns.items()
    }
    output: dict[str, np.ndarray] = {}
    for fold_type in FOLD_TYPES:
        keys = sorted(key for key in source_values if key[0] == fold_type)
        matrix = np.column_stack([source_values[key] for key in keys])
        output[fold_type] = _mean_columns(matrix, list(range(matrix.shape[1])))
        output[fold_type + "__defined_held_sources"] = np.isfinite(matrix).sum(axis=1)
        output[fold_type + "__total_held_sources"] = np.full(len(cell_values), len(keys), dtype=int)
    output["equal_mean"] = 0.5 * (
        output["human_source_holdout"] + output["generator_holdout"]
    )
    output["defined_pair_cells"] = np.isfinite(cell_values).sum(axis=1)
    output["total_pair_cells"] = np.full(len(cell_values), cell_values.shape[1], dtype=int)
    return output


def _metric_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, tuple[str, ...]], dict[str, str]]:
    lookup: dict[tuple[str, tuple[str, ...]], dict[str, str]] = {}
    for row in rows:
        key = (row["combination"], _cell_key(row))
        if key in lookup:
            raise ValueError(f"Duplicate evaluator metric row: {key}")
        lookup[key] = row
    return lookup


def validate_point_estimates(
    inputs: dict[str, Any], observations: dict[str, Any], cells: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ones = np.ones((1, len(observations["groups"])), dtype=int)
    computed = compute_cell_metrics(observations, cells, ones)
    lookup = _metric_lookup(inputs["metrics"])
    max_cell_error = 0.0
    for combination in COMBINATIONS:
        for cell_index, cell in enumerate(cells):
            for metric in ("roc_auc", "balanced_accuracy"):
                observed = float(computed[combination][metric][0, cell_index])
                expected = finite_float(lookup[(combination, cell["key"])][metric], metric)
                max_cell_error = max(max_cell_error, abs(observed - expected))
    if max_cell_error > 1e-12:
        raise ValueError(f"Point source-pair metrics fail evaluator reproduction: {max_cell_error}")

    summary_lookup = {
        (row["combination"], row["fold_type"]): row for row in inputs["summary"]
    }
    point_rows: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    max_summary_error = 0.0
    for combination in COMBINATIONS:
        aggregates[combination] = {}
        row: dict[str, Any] = {"combination": combination}
        for metric, evaluator_name in (
            ("roc_auc", "roc_auc_source_macro"),
            ("balanced_accuracy", "balanced_accuracy_source_macro"),
        ):
            macro = aggregate_macro(computed[combination][metric], cells)
            aggregates[combination][metric] = macro
            human = float(macro["human_source_holdout"][0])
            generator = float(macro["generator_holdout"][0])
            equal_mean = float(macro["equal_mean"][0])
            row[f"human_{metric}"] = human
            row[f"generator_{metric}"] = generator
            row[f"equal_mean_{metric}"] = equal_mean
            for fold_type, value in zip(FOLD_TYPES, (human, generator)):
                expected = finite_float(summary_lookup[(combination, fold_type)][evaluator_name], evaluator_name)
                max_summary_error = max(max_summary_error, abs(value - expected))
        row["max_abs_cell_error_vs_evaluator"] = max_cell_error
        row["max_abs_summary_error_vs_evaluator"] = max_summary_error
        point_rows.append(row)
    if max_summary_error > 1e-12:
        raise ValueError(f"Point macro metrics fail evaluator reproduction: {max_summary_error}")

    plan_comparisons = _comparison_ids(inputs["plan_path"])
    delta_lookup = _matched_delta_lookup(inputs["deltas"])
    point_deltas: list[dict[str, Any]] = []
    max_delta_error = 0.0
    point_by_combo = {row["combination"]: row for row in point_rows}
    for added in ADDED:
        comparison_id = plan_comparisons[added]
        for cell_index, cell in enumerate(cells):
            delta_row = delta_lookup[(comparison_id, cell["key"])]
            for metric in ("roc_auc", "balanced_accuracy"):
                observed = float(
                    computed[added][metric][0, cell_index]
                    - computed[BASELINE][metric][0, cell_index]
                )
                expected = finite_float(
                    delta_row[f"delta_{metric}__added_minus_baseline"], f"delta_{metric}"
                )
                max_delta_error = max(max_delta_error, abs(observed - expected))
        base = point_by_combo[BASELINE]
        current = point_by_combo[added]
        point_deltas.append({
            "comparison_id": comparison_id,
            "baseline_combination": BASELINE,
            "added_combination": added,
            "delta_human_auc": current["human_roc_auc"] - base["human_roc_auc"],
            "delta_generator_auc": current["generator_roc_auc"] - base["generator_roc_auc"],
            "delta_J": current["equal_mean_roc_auc"] - base["equal_mean_roc_auc"],
            "delta_human_balanced_accuracy": (
                current["human_balanced_accuracy"] - base["human_balanced_accuracy"]
            ),
            "delta_generator_balanced_accuracy": (
                current["generator_balanced_accuracy"] - base["generator_balanced_accuracy"]
            ),
            "delta_equal_mean_balanced_accuracy": (
                current["equal_mean_balanced_accuracy"] - base["equal_mean_balanced_accuracy"]
            ),
            "max_abs_cell_delta_error_vs_evaluator": max_delta_error,
        })
    if max_delta_error > 1e-12:
        raise ValueError(f"Point matched deltas fail evaluator reproduction: {max_delta_error}")
    return point_rows, point_deltas


def _comparison_ids(plan_path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    with plan_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("plan_type") == "incremental_matched"
                and row.get("arm") == "added"
                and row.get("combination") in ADDED
            ):
                output[row["combination"]] = row["comparison_id"]
    if set(output) != set(ADDED):
        raise ValueError(f"Could not resolve all target comparison IDs: {output}")
    return output


def _matched_delta_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, tuple[str, ...]], dict[str, str]]:
    output: dict[tuple[str, tuple[str, ...]], dict[str, str]] = {}
    for row in rows:
        if row.get("added_combination") not in ADDED:
            continue
        key = (row["comparison_id"], _cell_key(row))
        if key in output:
            raise ValueError(f"Duplicate evaluator matched delta row: {key}")
        output[key] = row
    return output


def percentile_interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan, math.nan, 0
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high), int(len(finite))


def bootstrap_outputs(
    observations: dict[str, Any], cells: list[dict[str, Any]], replicates: int, seed: int,
    point_deltas: list[dict[str, Any]], plan_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    multiplicities = draw_group_multiplicities(len(observations["groups"]), replicates, seed)
    computed = compute_cell_metrics(observations, cells, multiplicities)
    macro: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for combination in COMBINATIONS:
        macro[combination] = {
            metric: aggregate_macro(computed[combination][metric], cells)
            for metric in ("roc_auc", "balanced_accuracy")
        }
    comparison_ids = _comparison_ids(plan_path)
    point_lookup = {row["added_combination"]: row for row in point_deltas}
    replicate_rows: list[dict[str, Any]] = []
    undefined_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    group_universe_hash = canonical_hash(observations["groups"])

    for replicate in range(replicates):
        multiplicity = multiplicities[replicate]
        replicate_hash = hashlib.sha256(multiplicity.astype("<i8", copy=False).tobytes()).hexdigest()
        for added in ADDED:
            comparison_id = comparison_ids[added]
            row: dict[str, Any] = {
                "replicate": replicate,
                "seed": seed,
                "comparison_id": comparison_id,
                "baseline_combination": BASELINE,
                "added_combination": added,
                "global_group_count": len(observations["groups"]),
                "nonzero_resampled_groups": int((multiplicity > 0).sum()),
                "resampled_draws": int(multiplicity.sum()),
                "group_universe_sha256": group_universe_hash,
                "group_multiplicity_sha256": replicate_hash,
            }
            for metric, short in (("roc_auc", "auc"), ("balanced_accuracy", "ba")):
                base_macro = macro[BASELINE][metric]
                added_macro = macro[added][metric]
                for direction, prefix in (
                    ("human_source_holdout", "human"),
                    ("generator_holdout", "generator"),
                    ("equal_mean", "equal_mean"),
                ):
                    baseline_value = float(base_macro[direction][replicate])
                    added_value = float(added_macro[direction][replicate])
                    row[f"baseline_{prefix}_{short}"] = baseline_value
                    row[f"added_{prefix}_{short}"] = added_value
                    row[f"delta_{prefix}_{short}"] = added_value - baseline_value
                row[f"defined_pair_cells_{short}"] = int(base_macro["defined_pair_cells"][replicate])
                row[f"total_pair_cells_{short}"] = int(base_macro["total_pair_cells"][replicate])
                for fold_type, prefix in zip(FOLD_TYPES, ("human", "generator")):
                    row[f"defined_held_sources_{prefix}_{short}"] = int(
                        base_macro[fold_type + "__defined_held_sources"][replicate]
                    )
                    row[f"total_held_sources_{prefix}_{short}"] = int(
                        base_macro[fold_type + "__total_held_sources"][replicate]
                    )
            row["status"] = (
                "ok" if row["defined_pair_cells_auc"] == row["total_pair_cells_auc"]
                else "partial_undefined_pair_cells"
            )
            replicate_rows.append(row)

        # Undefined masks depend only on labels and shared group weights, so one
        # explicit baseline audit row represents all paired models.
        auc_values = computed[BASELINE]["roc_auc"][replicate]
        ba_values = computed[BASELINE]["balanced_accuracy"][replicate]
        for index, cell in enumerate(cells):
            if math.isfinite(float(auc_values[index])) and math.isfinite(float(ba_values[index])):
                continue
            indices = cell["indices"]
            labels = np.asarray([observations["master"][i]["label"] for i in indices])
            weights = multiplicity[observations["group_indices"][indices]]
            human_weight = int(weights[labels == 0].sum())
            ai_weight = int(weights[labels == 1].sum())
            undefined_rows.append({
                "replicate": replicate,
                **{field: value for field, value in zip(CELL_KEY_FIELDS, cell["key"])},
                "human_weight": human_weight,
                "ai_weight": ai_weight,
                "reason": (
                    "zero_human_and_ai_weight" if human_weight == 0 and ai_weight == 0
                    else "zero_human_weight" if human_weight == 0 else "zero_ai_weight"
                ),
                "applies_identically_to_combinations": "|".join(COMBINATIONS),
            })

    for added in ADDED:
        rows = [row for row in replicate_rows if row["added_combination"] == added]
        for column, point_column, label in (
            ("delta_human_auc", "delta_human_auc", "delta_human_auc"),
            ("delta_generator_auc", "delta_generator_auc", "delta_generator_auc"),
            ("delta_equal_mean_auc", "delta_J", "delta_J"),
            (
                "delta_human_ba", "delta_human_balanced_accuracy",
                "delta_human_balanced_accuracy",
            ),
            (
                "delta_generator_ba", "delta_generator_balanced_accuracy",
                "delta_generator_balanced_accuracy",
            ),
            (
                "delta_equal_mean_ba", "delta_equal_mean_balanced_accuracy",
                "delta_equal_mean_balanced_accuracy",
            ),
        ):
            values = np.asarray([row[column] for row in rows], dtype=float)
            low, high, valid = percentile_interval(values)
            finite = values[np.isfinite(values)]
            summary_rows.append({
                "comparison_id": comparison_ids[added],
                "baseline_combination": BASELINE,
                "added_combination": added,
                "metric": label,
                "point_estimate": point_lookup[added][point_column],
                "bootstrap_mean": float(finite.mean()) if len(finite) else math.nan,
                "bootstrap_standard_error": (
                    float(finite.std(ddof=1)) if len(finite) > 1 else math.nan
                ),
                "percentile_95_ci_low": low,
                "percentile_95_ci_high": high,
                "probability_delta_gt_zero": float((finite > 0).mean()) if len(finite) else math.nan,
                "valid_replicates": valid,
                "requested_replicates": replicates,
                "partial_undefined_pair_replicates": sum(
                    row["status"] != "ok" for row in rows
                ),
            })
    return replicate_rows, summary_rows, undefined_rows, multiplicities


def write_csv(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None,
) -> None:
    if not rows:
        if not fieldnames:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.replicates != DEFAULT_REPLICATES or args.seed != DEFAULT_SEED:
        raise ValueError("Frozen analysis requires exactly 1000 replicates and seed 20260907")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    inputs = load_inputs(args.evaluation_dir.resolve())
    observations = build_observations(inputs["predictions"])
    cells = build_cells(inputs["metrics"], observations)
    point_rows, point_deltas = validate_point_estimates(inputs, observations, cells)
    replicate_rows, summary_rows, undefined_rows, multiplicities = bootstrap_outputs(
        observations, cells, args.replicates, args.seed, point_deltas, inputs["plan_path"]
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".tmp.", dir=output_dir.parent))
    try:
        write_csv(temporary / "point_estimates.csv", point_rows)
        write_csv(temporary / "point_matched_deltas.csv", point_deltas)
        write_csv(temporary / "bootstrap_replicates.csv", replicate_rows)
        write_csv(temporary / "uncertainty_summary.csv", summary_rows)
        write_csv(
            temporary / "undefined_pair_metrics.csv", undefined_rows,
            [
                "replicate", *CELL_KEY_FIELDS, "human_weight", "ai_weight", "reason",
                "applies_identically_to_combinations",
            ],
        )
        (temporary / "LIMITATIONS_EN.md").write_text(
            "# Scope and limitations\n\n"
            "These percentile intervals use a paired nonparametric bootstrap over the "
            "observed global `group_id` units. One shared multiplicity vector is applied "
            "to all models, folds, sources, and classes. Models are not refit and the "
            "threshold remains 0.5. The intervals are therefore conditional on the fitted "
            "models, this corpus, its source composition, and these folds. They do not "
            "quantify training instability, population sampling, future-generator shift, "
            "or performance on unseen source domains. F and H remain interpretable DSP "
            "descriptors, not evidence of AI authorship by themselves.\n",
            encoding="utf-8",
        )
        input_hashes = {str(path): sha256_file(path) for path in inputs["paths"]}
        code_path = Path(__file__).resolve()
        output_hashes = {
            path.name: sha256_file(path) for path in sorted(temporary.iterdir()) if path.is_file()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "analysis": "paired_global_group_bootstrap_conditional_on_frozen_models",
            "evaluation_dir": str(args.evaluation_dir.resolve()),
            "evaluation_input_sha256": input_hashes,
            "script_sha256": sha256_file(code_path),
            "numpy_version": np.__version__,
            "output_sha256_before_manifest": output_hashes,
            "seed": args.seed,
            "replicates": args.replicates,
            "threshold": THRESHOLD,
            "baseline": BASELINE,
            "added_combinations": list(ADDED),
            "global_group_count": len(observations["groups"]),
            "group_universe_sha256": canonical_hash(observations["groups"]),
            "prediction_observations_per_combination": len(observations["master"]),
            "source_pair_fold_cells": len(cells),
            "undefined_pair_metric_rows": len(undefined_rows),
            "replicates_with_any_undefined_pair": len({row["replicate"] for row in undefined_rows}),
            "multiplicity_matrix_sha256": hashlib.sha256(
                multiplicities.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
            "point_reproduction_max_abs_error": max(
                max(row["max_abs_cell_error_vs_evaluator"], row["max_abs_summary_error_vs_evaluator"])
                for row in point_rows
            ),
            "matched_delta_reproduction_max_abs_error": max(
                row["max_abs_cell_delta_error_vs_evaluator"] for row in point_deltas
            ),
            "bootstrap_unit": "global group_id; one shared draw across models/folds/sources/classes",
            "macro_criterion": (
                "mean source-pair/fold metric within heldout_source; macro held sources "
                "within Human and AI directions; equal mean of directions"
            ),
            "auc_ties": "half credit via weighted Mann-Whitney",
            "undefined_policy": (
                "explicit pair audit; NaN-skipping held-source means matching evaluator aggregation"
            ),
            "no_model_refit": True,
            "no_historical_labels": True,
            "no_future_generator_or_population_guarantee": True,
        }
        (temporary / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
