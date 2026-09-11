#!/usr/bin/env python3
"""Preregistered, leakage-controlled evaluation of new audio phenomena.

This extension deliberately imports the tested September 5 evaluator for table
joining, global group folds, quantity sampling, weighting, ridge fitting, and
prediction.  It adds matched old-vs-new comparisons, explicit family coverage,
row-level predictions, and a hard preregistration gate.

Real data can only be scored when ``--preregistration`` names a frozen receipt
whose contract hash matches the current code, configuration, manifests, and CLI
settings.  ``--synthetic-test-only`` is reserved for generated unit-test data.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ARTIFACTS_ROOT = HERE.parents[1]
# Archive the authoritative completed local evaluator inside this new run.
# The remote historical directory can contain an older working copy; never
# silently substitute it or mutate that preserved directory.
BASE_EVALUATOR_PATH = HERE / "frozen_evaluate_expanded_20260905.py"


def _load_base_evaluator() -> Any:
    if not BASE_EVALUATOR_PATH.is_file():
        raise RuntimeError(f"Required preserved evaluator is absent: {BASE_EVALUATOR_PATH}")
    spec = importlib.util.spec_from_file_location("preserved_evaluate_expanded", BASE_EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import preserved evaluator: {BASE_EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_evaluator()

DEFAULT_SEED = 20_260_907
DEFAULT_QUANTITIES = "25,50,100,200,all"
OLD_FAMILY_ORDER = ("S", "D", "R", "P")
NEW_FAMILY_ORDER = ("F", "H", "M", "V", "B", "A", "T")
FAMILY_ORDER = OLD_FAMILY_ORDER + NEW_FAMILY_ORDER
DISALLOWED_EVALUATION_ROLES = {"pilot", "provisional", "provisional_development"}
SCHEMA_VERSION = 1


def write_process_state(output: Path, state: str, **fields: Any) -> None:
    """Atomic observation receipt; a PID must still be checked with the OS."""
    output.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "hostname": platform.node(), "state": state,
               "updated_at": datetime.now(timezone.utc).isoformat(), **fields}
    temporary = output / "process.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "process.json")
KNOWN_NON_PREDICTOR_COLUMNS = {
    "F_status", "F_quality_eligible", "F_duration_s", "F_rms_dbfs", "F_n_fft",
    "F_hop_length", "F_active_frame_count", "F_valid_phase_pair_count",
    "F_valid_group_delay_pair_count", "F_attack_frame_count", "F_sustain_frame_count",
    "F_decay_frame_count", "F_attack_eligible", "F_sustain_eligible", "F_decay_eligible",
    "H_status", "H_duration_sec", "H_input_sr_hz", "H_standardized_sr_16000",
    "H_active_coverage_fraction", "H_tonal_coverage_fraction", "H_quality_score",
    "H_eligible_block_count", "M_status", "M_duration_sec", "M_input_sr_hz",
    "M_standardized_sr_16000", "M_active_coverage_fraction", "M_tonal_coverage_fraction",
    "M_quality_score", "M_eligible_block_count", "M_evaluated_lag_count",
    "M_pattern_span_sec", "M_min_lag_sec",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preregistration-draft", "dev", "historical-descriptive"),
        required=True,
    )
    parser.add_argument("--contract-target", choices=("dev", "historical-descriptive"), default="dev")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, action="append", required=True)
    parser.add_argument("--families-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--frozen-dev-bundle", type=Path)
    parser.add_argument("--synthetic-test-only", action="store_true")
    parser.add_argument("--quantities", default=DEFAULT_QUANTITIES)
    parser.add_argument("--group-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", choices=("ridge", "logistic"), default="ridge")
    parser.add_argument(
        "--threshold-policy", choices=("train_ba", "fixed_0.5"), default="train_ba"
    )
    parser.add_argument("--development-role", default="development")
    parser.add_argument("--historical-roles", default="locked")
    parser.add_argument("--min-train-groups-per-class", type=int, default=2)
    parser.add_argument("--min-test-groups-per-class", type=int, default=1)
    parser.add_argument("--id-column")
    parser.add_argument("--label-column")
    parser.add_argument("--role-column")
    parser.add_argument("--source-column", default="source_group")
    parser.add_argument("--group-column", default="group_id")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def id_set_hash(values: Iterable[Any]) -> str:
    payload = "".join(f"{value}\n" for value in sorted({str(value) for value in values}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: pd.DataFrame | list[dict[str, Any]]) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def parse_quantities(text: str) -> list[int | str]:
    return BASE.parse_quantities(text)


def _validate_family_spec(code: str, spec: Any, *, new: bool) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"Family {code} must be an object")
    allowed_states = {"available", "planned"} if new else {"available"}
    state = str(spec.get("state", "available"))
    if state not in allowed_states:
        raise ValueError(f"Family {code} state must be one of {sorted(allowed_states)}")
    columns = spec.get("columns", [])
    if not isinstance(columns, list) or any(not isinstance(value, str) or not value for value in columns):
        raise ValueError(f"Family {code} columns must be a list of non-empty names")
    if state == "available" and not columns:
        raise ValueError(f"Available family {code} must configure at least one column")
    if state == "planned" and columns:
        raise ValueError(f"Planned family {code} must not expose columns before extraction is frozen")
    eligibility = spec.get("eligibility", {})
    if not isinstance(eligibility, dict):
        raise ValueError(f"Family {code} eligibility must be an object")
    status_columns = spec.get("status_columns", [])
    if not isinstance(status_columns, list) or any(not isinstance(value, str) for value in status_columns):
        raise ValueError(f"Family {code} status_columns must be a list")
    minimum = float(spec.get("minimum_observed_fraction", 0.0))
    if not 0.0 <= minimum <= 1.0:
        raise ValueError(f"Family {code} minimum_observed_fraction must be in [0,1]")
    phenomenon = str(spec.get("phenomenon", "")).strip()
    if new and not phenomenon:
        raise ValueError(f"New family {code} must name its intended phenomenon")
    return {
        "state": state,
        "columns": columns,
        "eligibility": eligibility,
        "status_columns": status_columns,
        "minimum_observed_fraction": minimum,
        "phenomenon": phenomenon,
    }


def load_family_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"families JSON must declare schema_version={SCHEMA_VERSION}")
    old_raw = payload.get("old_families")
    new_raw = payload.get("new_families")
    if not isinstance(old_raw, dict) or set(old_raw) != set(OLD_FAMILY_ORDER):
        raise ValueError(f"old_families must define exactly {OLD_FAMILY_ORDER}")
    if not isinstance(new_raw, dict) or set(new_raw) != set(NEW_FAMILY_ORDER):
        raise ValueError(
            "new_families must explicitly define F,H,M,V,B,A,T; future families use state=planned"
        )
    old = {code: _validate_family_spec(code, old_raw[code], new=False) for code in OLD_FAMILY_ORDER}
    new = {code: _validate_family_spec(code, new_raw[code], new=True) for code in NEW_FAMILY_ORDER}
    if not any(new[code]["state"] == "available" for code in NEW_FAMILY_ORDER):
        raise ValueError("At least one new family must be available for an extension evaluation")
    flattened = [column for code in FAMILY_ORDER for column in (old | new)[code]["columns"]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Feature columns must not overlap across phenomenon families")
    forbidden = sorted(set(flattened) & KNOWN_NON_PREDICTOR_COLUMNS)
    if forbidden:
        raise ValueError(f"Quality/status/coverage metadata cannot be predictors: {forbidden}")
    global_eligibility = payload.get("eligibility", {})
    if not isinstance(global_eligibility, dict):
        raise ValueError("Top-level eligibility must be an object")
    status_columns = payload.get("status_columns", [])
    if not isinstance(status_columns, list) or any(not isinstance(value, str) for value in status_columns):
        raise ValueError("Top-level status_columns must be a list")
    return {
        "schema_version": SCHEMA_VERSION,
        "cohort": str(payload.get("cohort", "configured cohort")),
        "eligibility": global_eligibility,
        "status_columns": status_columns,
        "old_families": old,
        "new_families": new,
    }


def family_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {**config["old_families"], **config["new_families"]}


def active_new_families(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(code for code in NEW_FAMILY_ORDER if config["new_families"][code]["state"] == "available")


def nonempty_combinations(codes: Iterable[str]) -> list[tuple[str, ...]]:
    ordered = [code for code in FAMILY_ORDER if code in set(codes)]
    return [combo for size in range(1, len(ordered) + 1) for combo in itertools.combinations(ordered, size)]


def combo_name(codes: Iterable[str]) -> str:
    selected = set(codes)
    return "+".join(code for code in FAMILY_ORDER if code in selected)


def build_evaluation_plan(config: dict[str, Any]) -> pd.DataFrame:
    old_combos = nonempty_combinations(OLD_FAMILY_ORDER)
    new_combos = nonempty_combinations(active_new_families(config))
    rows: list[dict[str, Any]] = []
    for old in old_combos:
        name = combo_name(old)
        rows.append({
            "plan_id": f"baseline::{name}", "plan_type": "old_baseline_standalone",
            "comparison_id": "", "arm": "standalone", "combination": name,
            "family_codes": json.dumps(list(old)), "eligibility_family_codes": json.dumps(list(old)),
        })
    for new in new_combos:
        name = combo_name(new)
        rows.append({
            "plan_id": f"new::{name}", "plan_type": "new_standalone",
            "comparison_id": "", "arm": "standalone", "combination": name,
            "family_codes": json.dumps(list(new)), "eligibility_family_codes": json.dumps(list(new)),
        })
    for old in old_combos:
        old_name = combo_name(old)
        for new in new_combos:
            new_name = combo_name(new)
            comparison_id = f"incremental::{old_name}__plus__{new_name}"
            eligibility = tuple(code for code in FAMILY_ORDER if code in set(old + new))
            for arm, candidate in (("baseline", old), ("added", eligibility)):
                rows.append({
                    "plan_id": f"{comparison_id}::{arm}", "plan_type": "incremental_matched",
                    "comparison_id": comparison_id, "arm": arm,
                    "combination": combo_name(candidate),
                    "family_codes": json.dumps(list(candidate)),
                    "eligibility_family_codes": json.dumps(list(eligibility)),
                })
    return pd.DataFrame(rows)


def load_table(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
    if args.source_column != "source_group" or args.group_column != "group_id":
        raise ValueError("This extension requires source_group and globally applied group_id")
    namespace = Namespace(
        metadata=args.metadata,
        features=args.features,
        id_column=args.id_column,
        label_column=args.label_column,
        role_column=args.role_column,
        source_column=args.source_column,
        group_column=args.group_column,
    )
    table, resolved = BASE.load_table(namespace)
    labelled = table[np.isfinite(table["__label"])].copy()
    source_label_counts = labelled.groupby("__source")["__label"].nunique()
    mixed = source_label_counts[source_label_counts > 1]
    if len(mixed):
        raise ValueError(f"source_group must not cross class labels: {mixed.index.tolist()[:10]}")
    return table, resolved


def validate_feature_columns(table: pd.DataFrame, config: dict[str, Any]) -> None:
    specs = family_specs(config)
    columns = [column for code in FAMILY_ORDER for column in specs[code]["columns"]]
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"Configured feature columns are absent: {missing}")
    status_columns = config["status_columns"] + [
        column for code in FAMILY_ORDER for column in specs[code]["status_columns"]
    ]
    absent_status = sorted(set(status_columns) - set(table.columns))
    if absent_status:
        raise ValueError(f"Configured status columns are absent: {absent_status}")
    for column in columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")


def eligibility_contract(config: dict[str, Any], codes: Iterable[str]) -> dict[str, Any]:
    specs = family_specs(config)
    ordered = [code for code in FAMILY_ORDER if code in set(codes)]
    return {
        "global": config["eligibility"],
        "families": [{"code": code, "eligibility": specs[code]["eligibility"]} for code in ordered],
        "missingness_policy": "never_complete_case_filter; training-fold imputation plus indicators",
    }


def apply_contract(table: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    selected = BASE.apply_filters(table, contract["global"])
    for item in contract["families"]:
        selected = BASE.apply_filters(selected, item["eligibility"])
    return selected.copy()


def _check_development_cohort(table: pd.DataFrame, group_folds: int) -> None:
    counts = table["__label"].value_counts()
    if not {0.0, 1.0}.issubset(set(counts.index)):
        raise ValueError("Development cohort must contain both verified classes")
    for label in (0.0, 1.0):
        current = table[table["__label"] == label]
        if current["__source"].nunique() < 2:
            raise ValueError("Development source holdout needs at least two source_group values per class")
        if current["__group"].nunique() < group_folds:
            raise ValueError(
                f"Class {int(label)} has fewer global group_id units than --group-folds={group_folds}"
            )


def materialize_plan(
    plan: pd.DataFrame, table: pd.DataFrame, config: dict[str, Any], role: str, group_folds: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    role_table = table[table["__role"] == role].copy()
    role_table = role_table[np.isfinite(role_table["__label"])].copy()
    cohorts: dict[str, pd.DataFrame] = {}
    registry: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for raw_codes, current in plan.groupby("eligibility_family_codes", sort=False):
        codes = json.loads(raw_codes)
        contract = eligibility_contract(config, codes)
        eligible = apply_contract(role_table, contract)
        _check_development_cohort(eligible, group_folds)
        digest = id_set_hash(eligible["__id"])
        cohort_id = f"dev-{digest[:16]}"
        cohorts.setdefault(cohort_id, eligible)
        contract_hash = canonical_hash(contract)
        registry.setdefault(cohort_id, {
            "cohort_id": cohort_id,
            "eligible_id_set_sha256": digest,
            "rows": int(len(eligible)),
            "human_rows": int((eligible["__label"] == 0).sum()),
            "ai_rows": int((eligible["__label"] == 1).sum()),
            "human_sources": int(eligible.loc[eligible["__label"] == 0, "__source"].nunique()),
            "ai_sources": int(eligible.loc[eligible["__label"] == 1, "__source"].nunique()),
            "human_groups": int(eligible.loc[eligible["__label"] == 0, "__group"].nunique()),
            "ai_groups": int(eligible.loc[eligible["__label"] == 1, "__group"].nunique()),
            "eligibility_contract_hashes": set(),
        })["eligibility_contract_hashes"].add(contract_hash)
        for index in current.index:
            record = current.loc[index].to_dict()
            record.update({
                "cohort_id": cohort_id,
                "eligibility_contract_sha256": contract_hash,
                "eligible_rows": int(len(eligible)),
                "eligible_id_set_sha256": digest,
            })
            record["candidate_key"] = candidate_key(cohort_id, record["combination"])
            records.append(record)
    registry_rows = []
    for value in registry.values():
        value = dict(value)
        value["eligibility_contract_hashes"] = json.dumps(sorted(value["eligibility_contract_hashes"]))
        registry_rows.append(value)
    return pd.DataFrame(records), cohorts, pd.DataFrame(registry_rows)


def candidate_key(cohort_id: str, combination: str) -> str:
    return hashlib.sha256(f"{cohort_id}|{combination}".encode("utf-8")).hexdigest()[:20]


def columns_for_combination(config: dict[str, Any], combination: str) -> list[str]:
    specs = family_specs(config)
    return [column for code in combination.split("+") for column in specs[code]["columns"]]


def fit_logistic(
    table: pd.DataFrame, columns: list[str], feature_mode: str = "values_plus_missing",
) -> dict[str, Any]:
    # Reuse the preserved fitter's training-only medians, scaling and weighting;
    # only the penalized objective changes from ridge least squares to logistic.
    model = BASE.fit_model(table, columns, feature_mode=feature_mode)
    x = table[columns].to_numpy(float)
    missing = ~np.isfinite(x)
    medians = np.asarray(model["medians"], dtype=float)
    imputed = np.where(missing, medians, x)
    if feature_mode == "values_plus_missing":
        matrix = np.concatenate((imputed, missing.astype(float)), axis=1)
    elif feature_mode == "median_only":
        matrix = imputed
    elif feature_mode == "missingness_only":
        matrix = missing.astype(float)
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    design = np.column_stack((np.ones(len(matrix)), (matrix - mean) / scale))
    y = table["__label"].to_numpy(float)
    weights = BASE.sample_weights(table)
    coefficients = np.zeros(design.shape[1], dtype=float)
    penalty = np.full(design.shape[1], float(BASE.RIDGE))
    penalty[0] = 0.0
    for _ in range(100):
        linear = np.clip(design @ coefficients, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        gradient = design.T @ (weights * (probability - y)) + penalty * coefficients
        curvature = weights * probability * (1.0 - probability)
        hessian = design.T @ (design * curvature[:, None]) + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    model["coefficients_with_intercept"] = coefficients.tolist()
    model["model_type"] = "weighted_logistic_l2"
    model["prediction_link"] = "sigmoid"
    return model


def fit_candidate(
    table: pd.DataFrame, columns: list[str], model_type: str,
    feature_mode: str = "values_plus_missing",
) -> dict[str, Any]:
    if model_type == "ridge":
        model = BASE.fit_model(table, columns, feature_mode=feature_mode)
        model["model_type"] = "weighted_ridge_linear_probability"
        model["prediction_link"] = "identity"
        return model
    return fit_logistic(table, columns, feature_mode=feature_mode)


def predict_candidate(table: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    linear = BASE.predict(table, model)
    if model.get("prediction_link") == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
    return linear


def choose_train_threshold(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Threshold training needs both classes")
    candidates = np.unique(scores)
    candidates = np.concatenate((candidates, [float(np.nextafter(scores.max(), math.inf))]))
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        prediction = scores >= threshold
        positive = labels == 1
        negative = ~positive
        sensitivity = float(weights[prediction & positive].sum() / weights[positive].sum())
        specificity = float(weights[(~prediction) & negative].sum() / weights[negative].sum())
        balanced = 0.5 * (sensitivity + specificity)
        key = (balanced, -abs(float(threshold) - 0.5), -float(threshold))
        if best is None or key > best:
            best = key
            chosen = float(threshold)
    return chosen


def threshold_for_training(table: pd.DataFrame, model: dict[str, Any], policy: str) -> float:
    if policy == "fixed_0.5":
        return 0.5
    scores = predict_candidate(table, model)
    return choose_train_threshold(table["__label"].to_numpy(int), scores, BASE.sample_weights(table))


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prediction = scores >= threshold
    tp = int(((prediction == 1) & (labels == 1)).sum())
    tn = int(((prediction == 0) & (labels == 0)).sum())
    fp = int(((prediction == 1) & (labels == 0)).sum())
    fn = int(((prediction == 0) & (labels == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    balanced = (
        0.5 * (sensitivity + specificity)
        if np.isfinite(sensitivity) and np.isfinite(specificity) else math.nan
    )
    return {
        "roc_auc": BASE.auc(labels, scores), "balanced_accuracy": balanced,
        "ai_sensitivity": sensitivity, "human_specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def pair_metrics(
    test: pd.DataFrame, scores: np.ndarray, fold: dict[str, Any], threshold: float,
    minimum_groups: int,
) -> list[dict[str, Any]]:
    scored = test[["__id", "__label", "__source", "__group"]].copy()
    scored["score"] = scores
    fold_type = fold["fold_type"]
    output: list[dict[str, Any]] = []
    if fold_type == "ordinary_group_holdout_descriptive":
        human_sources = sorted(scored.loc[scored["__label"] == 0, "__source"].unique())
        ai_sources = sorted(scored.loc[scored["__label"] == 1, "__source"].unique())
        pairs = ((human, ai) for human in human_sources for ai in ai_sources)
    else:
        held_source = fold["heldout_source"]
        held_label = 0 if fold_type == "human_source_holdout" else 1
        opposites = sorted(scored.loc[scored["__label"] != held_label, "__source"].unique())
        pairs = (
            (held_source, opposite) if held_label == 0 else (opposite, held_source)
            for opposite in opposites
        )
    for human_source, ai_source in pairs:
        pair = scored[
            ((scored["__label"] == 0) & (scored["__source"] == human_source))
            | ((scored["__label"] == 1) & (scored["__source"] == ai_source))
        ]
        if pair["__label"].nunique() != 2:
            continue
        group_counts = pair.groupby("__label")["__group"].nunique()
        if any(int(group_counts.get(label, 0)) < minimum_groups for label in (0.0, 1.0)):
            continue
        output.append({
            "fold_type": fold_type,
            "heldout_source": fold.get("heldout_source", "__all_sources__"),
            "human_source": human_source, "ai_source": ai_source,
            "opposite_group_fold": fold["opposite_group_fold"],
            "test_human": int((pair["__label"] == 0).sum()),
            "test_ai": int((pair["__label"] == 1).sum()),
            "test_human_groups": int(pair.loc[pair["__label"] == 0, "__group"].nunique()),
            "test_ai_groups": int(pair.loc[pair["__label"] == 1, "__group"].nunique()),
            "threshold": threshold,
            **metrics_at_threshold(pair["__label"].to_numpy(int), pair["score"].to_numpy(float), threshold),
        })
    return output


def valid_train(table: pd.DataFrame, minimum_groups: int) -> tuple[bool, str]:
    if table["__label"].nunique() != 2:
        return False, "training fold lacks one class"
    counts = table.groupby("__label")["__group"].nunique()
    if any(int(counts.get(label, 0)) < minimum_groups for label in (0.0, 1.0)):
        return False, f"training fold has fewer than {minimum_groups} group_id units in a class"
    return True, ""


def coverage_rows(
    cohorts: dict[str, pd.DataFrame], config: dict[str, Any], plan: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = family_specs(config)
    family_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for cohort_id, table in cohorts.items():
        for source, current in table.groupby("__source", sort=True):
            for code in FAMILY_ORDER:
                spec = specs[code]
                columns = spec["columns"]
                observed = (
                    np.isfinite(current[columns].to_numpy(float)).any(axis=1)
                    if columns else np.zeros(len(current), dtype=bool)
                )
                complete = (
                    np.isfinite(current[columns].to_numpy(float)).all(axis=1)
                    if columns else np.zeros(len(current), dtype=bool)
                )
                family_rows.append({
                    "cohort_id": cohort_id, "source_group": source, "family": code,
                    "state": spec["state"], "phenomenon": spec["phenomenon"],
                    "eligible_rows": int(len(current)), "any_observed_rows": int(observed.sum()),
                    "complete_rows": int(complete.sum()),
                    "any_observed_fraction": float(observed.mean()) if len(current) else math.nan,
                    "complete_fraction": float(complete.mean()) if len(current) else math.nan,
                    "minimum_observed_fraction": spec["minimum_observed_fraction"],
                    "coverage_qualified": bool(
                        spec["state"] == "available"
                        and (float(observed.mean()) if len(current) else 0.0)
                        >= spec["minimum_observed_fraction"]
                    ),
                })
        candidates = sorted(set(plan.loc[plan["cohort_id"] == cohort_id, "combination"]))
        for combination in candidates:
            codes = combination.split("+")
            columns = columns_for_combination(config, combination)
            finite = np.isfinite(table[columns].to_numpy(float))
            candidate_rows.append({
                "cohort_id": cohort_id, "candidate_key": candidate_key(cohort_id, combination),
                "combination": combination, "eligible_rows": int(len(table)),
                "feature_count": len(columns),
                "rows_with_any_observed_feature": int(finite.any(axis=1).sum()),
                "rows_complete_all_selected_features": int(finite.all(axis=1).sum()),
                "any_observed_fraction": float(finite.any(axis=1).mean()),
                "complete_case_fraction_diagnostic_only": float(finite.all(axis=1).mean()),
                "missingness_policy": "no complete-case deletion; train-fold median plus indicators",
                "coverage_qualified": bool(all(
                    specs[code]["state"] == "available"
                    and float(np.isfinite(table[specs[code]["columns"]].to_numpy(float)).any(axis=1).mean())
                    >= specs[code]["minimum_observed_fraction"]
                    for code in codes
                )),
            })
    return pd.DataFrame(family_rows), pd.DataFrame(candidate_rows)


def source_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    keys = ["cohort_id", "candidate_key", "combination", "quantity", "fold_type"]
    source_level = (
        metrics.groupby(keys + ["heldout_source"], dropna=False)
        .agg(
            roc_auc=("roc_auc", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
            ai_sensitivity=("ai_sensitivity", "mean"),
            human_specificity=("human_specificity", "mean"), folds=("fold_index", "nunique"),
        ).reset_index()
    )
    return (
        source_level.groupby(keys, dropna=False)
        .agg(
            roc_auc_source_macro=("roc_auc", "mean"),
            balanced_accuracy_source_macro=("balanced_accuracy", "mean"),
            ai_sensitivity_source_macro=("ai_sensitivity", "mean"),
            human_specificity_source_macro=("human_specificity", "mean"),
            heldout_sources=("heldout_source", "nunique"),
        ).reset_index()
    )


def matched_metric_deltas(metrics: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Build fold/source-pair deltas only after exact-cohort matching is verified."""
    if metrics.empty:
        return pd.DataFrame()
    incremental = plan[plan["plan_type"] == "incremental_matched"].copy()
    baseline = incremental[incremental["arm"] == "baseline"]
    added = incremental[incremental["arm"] == "added"]
    plan_pairs = baseline.merge(
        added, on=["comparison_id", "cohort_id", "eligibility_contract_sha256",
                   "eligible_id_set_sha256"], suffixes=("__baseline", "__added"),
        validate="one_to_one",
    )
    metric_keys = [
        "cohort_id", "quantity", "fold_uid", "fold_index", "fold_type",
        "heldout_source", "human_source", "ai_source", "opposite_group_fold",
        "test_human", "test_ai", "test_human_groups", "test_ai_groups",
    ]
    measures = ["roc_auc", "balanced_accuracy", "ai_sensitivity", "human_specificity"]
    output: list[pd.DataFrame] = []
    for _, pair in plan_pairs.iterrows():
        left = metrics[metrics["candidate_key"] == pair["candidate_key__baseline"]]
        right = metrics[metrics["candidate_key"] == pair["candidate_key__added"]]
        matched = left[metric_keys + measures].merge(
            right[metric_keys + measures], on=metric_keys, suffixes=("__baseline", "__added"),
            validate="one_to_one",
        )
        matched.insert(0, "comparison_id", pair["comparison_id"])
        matched.insert(1, "baseline_combination", pair["combination__baseline"])
        matched.insert(2, "added_combination", pair["combination__added"])
        matched.insert(3, "eligible_id_set_sha256", pair["eligible_id_set_sha256"])
        for measure in measures:
            matched[f"delta_{measure}__added_minus_baseline"] = (
                matched[f"{measure}__added"] - matched[f"{measure}__baseline"]
            )
        output.append(matched)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def build_contract(args: argparse.Namespace, target: str) -> dict[str, Any]:
    quantities = parse_quantities(args.quantities)
    contract: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authorized_stage": target,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "preserved_base_evaluator_sha256": sha256_file(BASE_EVALUATOR_PATH),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "thread_settings": {key: os.environ.get(key) for key in
                                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
        "families_json": {"path": str(args.families_json.resolve()), "sha256": sha256_file(args.families_json)},
        "metadata": {"path": str(args.metadata.resolve()), "sha256": sha256_file(args.metadata)},
        "features": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.features
        ],
        "source_contract": {"source_column": args.source_column, "global_group_column": args.group_column},
        "parameters": {
            "seed": args.seed, "quantities": quantities, "group_folds": args.group_folds,
            "model": args.model, "threshold_policy": args.threshold_policy,
            "development_role": args.development_role,
            "historical_roles": sorted(parse_roles(args.historical_roles)),
            "min_train_groups_per_class": args.min_train_groups_per_class,
            "min_test_groups_per_class": args.min_test_groups_per_class,
        },
    }
    if target == "historical-descriptive":
        if args.frozen_dev_bundle is None:
            raise ValueError("historical-descriptive contract requires --frozen-dev-bundle")
        contract["frozen_dev_bundle"] = {
            "path": str(args.frozen_dev_bundle.resolve()),
            "sha256": sha256_file(args.frozen_dev_bundle),
        }
    return contract


def verify_authorization(args: argparse.Namespace, contract: dict[str, Any]) -> dict[str, Any]:
    if args.synthetic_test_only:
        return {"status": "synthetic_test_only", "contract_sha256": canonical_hash(contract)}
    if args.preregistration is None:
        raise ValueError(
            "Real-corpus scoring is disabled until --preregistration names a frozen matching receipt"
        )
    receipt = json.loads(args.preregistration.read_text(encoding="utf-8"))
    expected = canonical_hash(contract)
    if receipt.get("status") != "frozen":
        raise ValueError("Preregistration receipt status must be frozen")
    if receipt.get("contract_sha256") != expected:
        raise ValueError("Preregistration contract hash does not match current code/config/manifest/settings")
    if receipt.get("authorized_stage") != contract["authorized_stage"]:
        raise ValueError("Preregistration receipt authorizes a different stage")
    return {
        "status": "frozen_verified", "contract_sha256": expected,
        "receipt_path": str(args.preregistration.resolve()),
        "receipt_sha256": sha256_file(args.preregistration),
    }


def parse_roles(text: str) -> set[str]:
    roles = {value.strip() for value in text.split(",") if value.strip()}
    if not roles:
        raise ValueError("At least one role is required")
    return roles


def write_preregistration_draft(args: argparse.Namespace, config: dict[str, Any]) -> None:
    del config  # validation happened before hashing the exact file bytes
    contract = build_contract(args, args.contract_target)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "draft",
        "authorized_stage": args.contract_target,
        "contract_sha256": canonical_hash(contract),
        "contract": contract,
        "freeze_instruction": (
            "Independent root review must change status to frozen without changing contract or hash. "
            "A draft does not authorize scoring."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "preregistration_draft.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "preregistration-draft", "path": str(path),
                      "contract_sha256": payload["contract_sha256"]}, indent=2))


def _prediction_rows(
    test: pd.DataFrame, scores: np.ndarray, threshold: float, common: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (_, row), score in zip(test.iterrows(), scores):
        rows.append({
            **common,
            "row_id": row["__id"], "label": int(row["__label"]),
            "source_group": row["__source"], "group_id": row["__group"],
            "score": float(score), "threshold": threshold,
            "predicted_label": int(float(score) >= threshold),
        })
    return rows


def run_dev(
    args: argparse.Namespace, table: pd.DataFrame, config: dict[str, Any],
    contract: dict[str, Any], authorization: dict[str, Any], resolved: dict[str, str],
) -> None:
    if args.development_role in DISALLOWED_EVALUATION_ROLES or args.development_role == "locked":
        raise ValueError("Development role must not be locked, pilot, or provisional")
    quantities = parse_quantities(args.quantities)
    plan = build_evaluation_plan(config)
    plan, cohorts, cohort_registry = materialize_plan(
        plan, table, config, args.development_role, args.group_folds
    )
    family_coverage, candidate_coverage = coverage_rows(cohorts, config, plan)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "evaluation_plan.csv", plan)
    write_csv(args.output_dir / "cohort_registry.csv", cohort_registry)
    write_csv(args.output_dir / "family_coverage_by_source.csv", family_coverage)
    write_csv(args.output_dir / "candidate_coverage.csv", candidate_coverage)

    source_predictions: list[dict[str, Any]] = []
    group_predictions: list[dict[str, Any]] = []
    source_metrics: list[dict[str, Any]] = []
    group_metrics: list[dict[str, Any]] = []
    diagnostic_source_predictions: list[dict[str, Any]] = []
    diagnostic_group_predictions: list[dict[str, Any]] = []
    diagnostic_source_metrics: list[dict[str, Any]] = []
    diagnostic_group_metrics: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    model_bundle: dict[str, Any] = {}

    for cohort_id, cohort in cohorts.items():
        candidates = sorted(
            set(plan.loc[plan["cohort_id"] == cohort_id, "combination"]),
            key=lambda name: (len(name.split("+")), [FAMILY_ORDER.index(code) for code in name.split("+")]),
        )
        folds = BASE.make_folds(cohort, args.group_folds, args.seed)
        for fold_index, fold in enumerate(folds):
            train_groups = set(fold["train"]["__group"].astype(str))
            test_groups = set(fold["test"]["__group"].astype(str))
            if train_groups & test_groups:
                raise AssertionError("Preserved fold builder allowed global group_id leakage")
            for quantity in quantities:
                progress = {"cohort_id": cohort_id, "fold_index": fold_index,
                            "fold_count": len(folds), "quantity": quantity,
                            "candidates": len(candidates), "fold_type": fold["fold_type"]}
                write_process_state(args.output_dir, "running", **progress)
                print(json.dumps({"event": "fold_quantity_start", **progress}), flush=True)
                train = BASE.deterministic_quantity(fold["train"], quantity, args.seed)
                assert train is not None
                valid, reason = valid_train(train, args.min_train_groups_per_class)
                fold_uid = canonical_hash({
                    "cohort_id": cohort_id, "fold_index": fold_index,
                    "fold_type": fold["fold_type"], "quantity": quantity,
                    "seed": args.seed, "train_ids": sorted(train["__id"].astype(str)),
                    "test_ids": sorted(fold["test"]["__id"].astype(str)),
                })[:24]
                fold_records.append({
                    "fold_uid": fold_uid, "cohort_id": cohort_id, "fold_index": fold_index,
                    "fold_type": fold["fold_type"], "heldout_source": fold["heldout_source"],
                    "opposite_group_fold": fold["opposite_group_fold"], "quantity": quantity,
                    "train_id_set_sha256": id_set_hash(train["__id"]),
                    "test_id_set_sha256": id_set_hash(fold["test"]["__id"]),
                    "train_ids": sorted(train["__id"].astype(str).tolist()),
                    "test_ids": sorted(fold["test"]["__id"].astype(str).tolist()),
                    "train_group_ids": sorted(train["__group"].astype(str).unique().tolist()),
                    "test_group_ids": sorted(fold["test"]["__group"].astype(str).unique().tolist()),
                })
                if not valid:
                    skipped.append({
                        "cohort_id": cohort_id, "fold_index": fold_index,
                        "fold_type": fold["fold_type"], "quantity": quantity, "reason": reason,
                    })
                    continue
                for combination in candidates:
                    columns = columns_for_combination(config, combination)
                    model = fit_candidate(train, columns, args.model)
                    threshold = threshold_for_training(train, model, args.threshold_policy)
                    scores = predict_candidate(fold["test"], model)
                    model_sha = canonical_hash({"model": model, "threshold": threshold})
                    common = {
                        "fold_uid": fold_uid, "cohort_id": cohort_id,
                        "candidate_key": candidate_key(cohort_id, combination),
                        "combination": combination, "quantity": quantity,
                        "fold_index": fold_index, "fold_type": fold["fold_type"],
                        "heldout_source": fold["heldout_source"],
                        "opposite_group_fold": fold["opposite_group_fold"],
                        "model_sha256": model_sha,
                    }
                    predictions = _prediction_rows(fold["test"], scores, threshold, common)
                    metric_rows = pair_metrics(
                        fold["test"], scores, fold, threshold, args.min_test_groups_per_class
                    )
                    enriched_metrics = [{**common, **row} for row in metric_rows]
                    if fold["fold_type"] == "ordinary_group_holdout_descriptive":
                        group_predictions.extend(predictions)
                        group_metrics.extend(enriched_metrics)
                    else:
                        source_predictions.extend(predictions)
                        source_metrics.extend(enriched_metrics)

                    # Missingness sensitivities run only at the all-groups quantity and
                    # are permanently excluded from candidate selection/report ranking.
                    if quantity == "all":
                        for feature_mode in ("median_only", "missingness_only"):
                            diagnostic_model = fit_candidate(
                                train, columns, args.model, feature_mode=feature_mode
                            )
                            diagnostic_threshold = threshold_for_training(
                                train, diagnostic_model, args.threshold_policy
                            )
                            diagnostic_scores = predict_candidate(fold["test"], diagnostic_model)
                            diagnostic_common = {
                                **common,
                                "feature_mode": feature_mode,
                                "diagnostic_status": "sensitivity_only_never_select",
                                "model_sha256": canonical_hash({
                                    "model": diagnostic_model,
                                    "threshold": diagnostic_threshold,
                                }),
                            }
                            diagnostic_predictions = _prediction_rows(
                                fold["test"], diagnostic_scores, diagnostic_threshold,
                                diagnostic_common,
                            )
                            diagnostic_metric_rows = pair_metrics(
                                fold["test"], diagnostic_scores, fold, diagnostic_threshold,
                                args.min_test_groups_per_class,
                            )
                            diagnostic_enriched = [
                                {**diagnostic_common, **row} for row in diagnostic_metric_rows
                            ]
                            if fold["fold_type"] == "ordinary_group_holdout_descriptive":
                                diagnostic_group_predictions.extend(diagnostic_predictions)
                                diagnostic_group_metrics.extend(diagnostic_enriched)
                            else:
                                diagnostic_source_predictions.extend(diagnostic_predictions)
                                diagnostic_source_metrics.extend(diagnostic_enriched)

        # Frozen all-development models are retained for descriptive historical scoring.
        for combination in candidates:
            columns = columns_for_combination(config, combination)
            final_model = fit_candidate(cohort, columns, args.model)
            threshold = threshold_for_training(cohort, final_model, args.threshold_policy)
            key = candidate_key(cohort_id, combination)
            training_config = {
                "cohort_id": cohort_id, "combination": combination, "columns": columns,
                "training_id_set_sha256": id_set_hash(cohort["__id"]),
                "model": args.model, "threshold_policy": args.threshold_policy,
                "seed": args.seed,
            }
            model_bundle[key] = {
                "candidate_key": key, "cohort_id": cohort_id, "combination": combination,
                "training_config": training_config,
                "training_config_sha256": canonical_hash(training_config),
                "threshold": threshold, "model": final_model,
                "model_sha256": canonical_hash({"model": final_model, "threshold": threshold}),
            }

    source_metrics_frame = pd.DataFrame(source_metrics)
    group_metrics_frame = pd.DataFrame(group_metrics)
    source_deltas = matched_metric_deltas(source_metrics_frame, plan)
    group_deltas = matched_metric_deltas(group_metrics_frame, plan)
    write_csv(args.output_dir / "development_source_holdout_predictions.csv", source_predictions)
    write_csv(args.output_dir / "development_source_holdout_metrics_by_source_pair.csv", source_metrics_frame)
    write_csv(args.output_dir / "development_source_holdout_summary.csv", source_summary(source_metrics_frame))
    write_csv(args.output_dir / "development_source_holdout_matched_deltas.csv", source_deltas)
    write_csv(args.output_dir / "development_group_cv_predictions.csv", group_predictions)
    write_csv(args.output_dir / "development_group_cv_metrics_by_source_pair.csv", group_metrics_frame)
    write_csv(args.output_dir / "development_group_cv_summary.csv", source_summary(group_metrics_frame))
    write_csv(args.output_dir / "development_group_cv_matched_deltas.csv", group_deltas)
    write_csv(
        args.output_dir / "development_missingness_source_holdout_predictions.csv",
        diagnostic_source_predictions,
    )
    write_csv(
        args.output_dir / "development_missingness_source_holdout_metrics.csv",
        diagnostic_source_metrics,
    )
    write_csv(
        args.output_dir / "development_missingness_group_cv_predictions.csv",
        diagnostic_group_predictions,
    )
    write_csv(
        args.output_dir / "development_missingness_group_cv_metrics.csv",
        diagnostic_group_metrics,
    )
    write_csv(args.output_dir / "skipped_folds.csv", skipped)
    with (args.output_dir / "fold_registry.jsonl").open("w", encoding="utf-8") as handle:
        for record in fold_records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_development_models_for_historical_description",
        "selection": None,
        "selection_note": "No winner is selected automatically; historical labels must never select.",
        "contract_sha256": canonical_hash(contract),
        "authorization": authorization,
        "evaluator_sha256": contract["evaluator_sha256"],
        "preserved_base_evaluator_sha256": contract["preserved_base_evaluator_sha256"],
        "families_json_sha256": contract["families_json"]["sha256"],
        "metadata_sha256": contract["metadata"]["sha256"],
        "feature_hashes": contract["features"],
        "source_contract": contract["source_contract"],
        "parameters": contract["parameters"],
        "resolved_columns": resolved,
        "evaluation_plan": plan.to_dict(orient="records"),
        "models": model_bundle,
    }
    bundle_path = args.output_dir / "frozen_dev_models.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION, "stage": "dev",
        "interpretation": "development_only_source_holdout_and_group_cv; not historical test",
        "authorization": authorization, "contract": contract,
        "active_new_families": list(active_new_families(config)),
        "planned_new_families": [
            code for code in NEW_FAMILY_ORDER if config["new_families"][code]["state"] == "planned"
        ],
        "plan_rows": int(len(plan)), "unique_candidates": int(plan["candidate_key"].nunique()),
        "cohorts": int(len(cohorts)), "frozen_dev_bundle": str(bundle_path),
        "frozen_dev_bundle_sha256": sha256_file(bundle_path),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "stage": "dev", "status": authorization["status"],
        "unique_candidates": manifest["unique_candidates"], "cohorts": len(cohorts),
        "frozen_dev_bundle": str(bundle_path),
    }, indent=2))


def _historical_source_rows(
    scored: pd.DataFrame, threshold: float, common: dict[str, Any], minimum_groups: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_source: list[dict[str, Any]] = []
    by_pair: list[dict[str, Any]] = []
    for source, current in scored.groupby("__source", sort=True):
        metric = metrics_at_threshold(
            current["__label"].to_numpy(int), current["score"].to_numpy(float), threshold
        )
        by_source.append({
            **common, "source_group": source, "rows": int(len(current)),
            "groups": int(current["__group"].nunique()), **metric,
        })
    humans = sorted(scored.loc[scored["__label"] == 0, "__source"].unique())
    ais = sorted(scored.loc[scored["__label"] == 1, "__source"].unique())
    for human in humans:
        for ai in ais:
            pair = scored[
                ((scored["__label"] == 0) & (scored["__source"] == human))
                | ((scored["__label"] == 1) & (scored["__source"] == ai))
            ]
            groups = pair.groupby("__label")["__group"].nunique()
            if any(int(groups.get(label, 0)) < minimum_groups for label in (0.0, 1.0)):
                continue
            by_pair.append({
                **common, "human_source": human, "ai_source": ai,
                "human_rows": int((pair["__label"] == 0).sum()),
                "ai_rows": int((pair["__label"] == 1).sum()),
                **metrics_at_threshold(
                    pair["__label"].to_numpy(int), pair["score"].to_numpy(float), threshold
                ),
            })
    return by_source, by_pair


def historical_matched_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    current = metrics[
        (metrics["plan_type"] == "incremental_matched") & metrics["comparison_id"].astype(bool)
    ].copy()
    keys = ["comparison_id", "human_source", "ai_source", "historical_id_set_sha256"]
    measures = ["roc_auc", "balanced_accuracy", "ai_sensitivity", "human_specificity"]
    baseline = current[current["arm"] == "baseline"][
        keys + ["combination"] + measures
    ].rename(columns={"combination": "baseline_combination"})
    added = current[current["arm"] == "added"][
        keys + ["combination"] + measures
    ].rename(columns={"combination": "added_combination"})
    matched = baseline.merge(added, on=keys, suffixes=("__baseline", "__added"),
                             validate="one_to_one")
    for measure in measures:
        matched[f"delta_{measure}__added_minus_baseline"] = (
            matched[f"{measure}__added"] - matched[f"{measure}__baseline"]
        )
    return matched


def run_historical(
    args: argparse.Namespace, table: pd.DataFrame, config: dict[str, Any],
    contract: dict[str, Any], authorization: dict[str, Any],
) -> None:
    if args.frozen_dev_bundle is None:
        raise ValueError("historical-descriptive requires --frozen-dev-bundle")
    roles = parse_roles(args.historical_roles)
    forbidden = roles & DISALLOWED_EVALUATION_ROLES
    if forbidden:
        raise ValueError(f"Pilot/provisional roles cannot enter historical scoring: {sorted(forbidden)}")
    bundle = json.loads(args.frozen_dev_bundle.read_text(encoding="utf-8"))
    if bundle.get("status") != "frozen_development_models_for_historical_description":
        raise ValueError("Frozen development bundle has the wrong status")
    if bundle.get("selection") is not None:
        raise ValueError("Historical scoring refuses a bundle carrying historical label selection")
    if bundle.get("evaluator_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("Evaluator changed after development models were frozen")
    if bundle.get("families_json_sha256") != sha256_file(args.families_json):
        raise ValueError("Family config changed after development models were frozen")
    if bundle.get("metadata_sha256") != sha256_file(args.metadata):
        raise ValueError("Metadata/role manifest changed after development models were frozen")
    expected_feature_hashes = bundle.get("feature_hashes")
    current_feature_hashes = [
        {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in args.features
    ]
    if expected_feature_hashes != current_feature_hashes:
        raise ValueError("Feature files changed after development models were frozen")

    historical = table[table["__role"].isin(roles)].copy()
    labelled = historical[np.isfinite(historical["__label"])].copy()
    excluded_unverified = int(len(historical) - len(labelled))
    predictions: list[dict[str, Any]] = []
    by_source: list[dict[str, Any]] = []
    by_pair: list[dict[str, Any]] = []
    plan = pd.DataFrame(bundle["evaluation_plan"])
    for _, plan_row in plan.iterrows():
        codes = json.loads(plan_row["eligibility_family_codes"])
        eligible = apply_contract(labelled, eligibility_contract(config, codes))
        model_record = bundle["models"][plan_row["candidate_key"]]
        model = model_record["model"]
        threshold = float(model_record["threshold"])
        scores = predict_candidate(eligible, model)
        common = {
            "plan_id": plan_row["plan_id"], "plan_type": plan_row["plan_type"],
            "comparison_id": plan_row["comparison_id"], "arm": plan_row["arm"],
            "candidate_key": plan_row["candidate_key"], "combination": plan_row["combination"],
            "development_cohort_id": plan_row["cohort_id"],
            "evaluation_status": "historical_descriptive_only_never_select",
            "model_sha256": model_record["model_sha256"], "threshold": threshold,
            "historical_id_set_sha256": id_set_hash(eligible["__id"]),
        }
        scored = eligible[["__id", "__label", "__source", "__group"]].copy()
        scored["score"] = scores
        for (_, row), score in zip(eligible.iterrows(), scores):
            predictions.append({
                **common, "row_id": row["__id"], "label": int(row["__label"]),
                "source_group": row["__source"], "group_id": row["__group"],
                "score": float(score), "predicted_label": int(float(score) >= threshold),
            })
        source_rows, pair_rows = _historical_source_rows(
            scored, threshold, common, args.min_test_groups_per_class
        )
        by_source.extend(source_rows)
        by_pair.extend(pair_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "historical_descriptive_predictions.csv", predictions)
    write_csv(args.output_dir / "historical_descriptive_metrics_by_source.csv", by_source)
    write_csv(args.output_dir / "historical_descriptive_metrics_by_source_pair.csv", by_pair)
    write_csv(
        args.output_dir / "historical_descriptive_matched_deltas.csv",
        historical_matched_deltas(pd.DataFrame(by_pair)),
    )
    write_csv(args.output_dir / "historical_evaluation_plan.csv", plan)
    manifest = {
        "schema_version": SCHEMA_VERSION, "stage": "historical-descriptive",
        "evaluation_status": "historical_descriptive_only_never_select",
        "selection_performed": False,
        "roles": sorted(roles), "rows_before_unverified_exclusion": int(len(historical)),
        "verified_label_rows": int(len(labelled)), "excluded_unverified_rows": excluded_unverified,
        "pilot_and_provisional_roles_forbidden": True,
        "authorization": authorization, "contract": contract,
        "frozen_dev_bundle": str(args.frozen_dev_bundle.resolve()),
        "frozen_dev_bundle_sha256": sha256_file(args.frozen_dev_bundle),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "stage": "historical-descriptive", "status": authorization["status"],
        "verified_rows": len(labelled), "excluded_unverified_rows": excluded_unverified,
        "selection_performed": False,
    }, indent=2))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_family_config(args.families_json)
    if args.stage == "preregistration-draft":
        write_preregistration_draft(args, config)
        return
    target = args.stage
    contract = build_contract(args, target)
    authorization = verify_authorization(args, contract)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output_dir / "run_manifest.json").exists():
            raise RuntimeError("Completed evaluation exists; use a new output directory")
        write_process_state(args.output_dir, "running", stage=args.stage,
                            contract_sha256=canonical_hash(contract))
        try:
            table, resolved = load_table(args)
            validate_feature_columns(table, config)
            if args.stage == "dev":
                run_dev(args, table, config, contract, authorization, resolved)
            else:
                run_historical(args, table, config, contract, authorization)
        except BaseException as error:
            write_process_state(args.output_dir, "failed", stage=args.stage, error=repr(error))
            raise
        write_process_state(args.output_dir, "finished", stage=args.stage,
                            contract_sha256=canonical_hash(contract))


if __name__ == "__main__":
    main()
