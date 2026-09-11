#!/usr/bin/env python3
"""Leakage-controlled four-family evaluation for the expanded music cohort.

The program deliberately has two stages. ``cv`` performs source-held-out model
selection and freezes every candidate model. ``locked`` only loads those frozen
models and scores named evaluation slices; it never refits or changes the leader.

The family JSON is explicit.  Its preferred form is::

  {
    "primary_feature_set": "common8_30s",
    "feature_sets": {
      "common8_30s": {
        "selection_eligible": true,
        "cohort": "duration-qualified 30 s",
        "eligibility": {"min": {"native_duration_sec": 30}},
        "families": {"S": ["s8__..."], "D": ["d__..."],
                     "R": ["r__..."], "P": ["p__..."]}
      },
      "native16_30s": {
        "selection_eligible": false,
        "cohort": "native-band sensitivity; SR >= 40 kHz",
        "eligibility": {"min": {"native_duration_sec": 30,
                                  "native_sample_rate_hz": 40000}},
        "families": {"S": ["s16__..."], "D": ["d__..."],
                     "R": ["r__..."], "P": ["p__..."]}
      }
    }
  }

A simple top-level ``{"S": [...], "D": [...], "R": [...], "P": [...]}``
mapping is also accepted as one primary feature set.  All combinations within a
feature set use the same metadata-qualified row cohort. Missing values are imputed
from each training fold only and accompanied by missingness indicators; coverage
is reported, and a wholly unobserved family cannot qualify a combination as best.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_SEED = 20_260_905
RIDGE = 10.0
THRESHOLD = 0.5
SELECTION_TIE_TOLERANCE = 1e-12
FAMILY_ORDER = ("S", "D", "R", "P")
DEFAULT_CV_ROLES = ("development",)
EVALUATION_ROLES = (
    "locked",
    "stress",
    "provisional",
    "pilot",
    "locked_catalogue_test",
    "stress_test_only",
    "provisional_development",
    "locked_pilot_only",
)
ALIASES = {
    "id": ("row_id", "track", "id", "item_id"),
    "label": ("label", "class_label", "is_ai", "class_name"),
    "role": ("role", "split"),
    "source": ("source_id", "source_group", "generator", "source"),
    "group": ("group_id", "condition_id"),
}
ELIGIBILITY_ALIASES = {
    "native_duration_sec": ("native_duration_sec", "native_duration_s", "duration_sec", "duration_s"),
    "native_sample_rate_hz": ("native_sample_rate_hz", "native_sr", "sample_rate_hz", "sample_rate"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cv", "locked"), required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, action="append", required=True)
    parser.add_argument("--families-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-selection", type=Path,
                        help="Required for locked; produced by the cv stage")
    parser.add_argument("--evaluation-slices-json", type=Path,
                        help="Optional mapping from slice name to role list")
    parser.add_argument("--training-scopes-json", type=Path,
                        help="Optional CV training-scope filters; must include all_development")
    parser.add_argument("--prior-heldout", type=Path, action="append", default=[],
                        help="Prior immutable result to register by path/hash (never read for selection)")
    parser.add_argument("--cv-roles", default=",".join(DEFAULT_CV_ROLES))
    parser.add_argument("--quantities", default="25,50,100,200,all")
    parser.add_argument("--opposite-class-folds", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=0,
                        help="Optional held-out-source macro bootstrap (paper CIs should instead resample group_id)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--id-column")
    parser.add_argument("--label-column")
    parser.add_argument("--role-column")
    parser.add_argument("--source-column")
    parser.add_argument("--group-column")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_column(table: pd.DataFrame, explicit: str | None, kind: str) -> str:
    if explicit:
        if explicit not in table:
            raise ValueError(f"Configured {kind} column is absent: {explicit}")
        return explicit
    candidates = [name for name in ALIASES[kind] if name in table]
    if not candidates:
        raise ValueError(f"No {kind} column found; accepted aliases={ALIASES[kind]}")
    return candidates[0]


def normalize_label(value: Any) -> float:
    if pd.isna(value) or str(value).strip().lower() in {"", "unknown", "unverified", "na", "nan"}:
        return math.nan
    text = str(value).strip().lower()
    if text in {"0", "0.0", "human", "real"}:
        return 0.0
    if text in {"1", "1.0", "ai", "generated", "synthetic"}:
        return 1.0
    raise ValueError(f"Unsupported label value: {value!r}")


def load_table(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str]]:
    metadata = pd.read_csv(args.metadata, low_memory=False)
    resolved = {
        "id": resolve_column(metadata, args.id_column, "id"),
        "label": resolve_column(metadata, args.label_column, "label"),
        "role": resolve_column(metadata, args.role_column, "role"),
        "source": resolve_column(metadata, args.source_column, "source"),
        "group": resolve_column(metadata, args.group_column, "group"),
    }
    metadata = metadata.rename(columns={value: f"__{key}" for key, value in resolved.items()})
    if metadata.__id.isna().any() or metadata.__id.astype(str).duplicated().any():
        raise ValueError("Metadata IDs must be non-empty and unique")
    metadata["__id"] = metadata.__id.astype(str)
    for name in ("__role", "__source", "__group"):
        if metadata[name].isna().any() or (metadata[name].astype(str).str.strip() == "").any():
            raise ValueError(f"{name[2:]} values must be non-empty")
        metadata[name] = metadata[name].astype(str)
    metadata["__label"] = metadata.__label.map(normalize_label)
    metadata["__claimed_label"] = metadata["__label"]
    provisional = metadata.__role.isin({"provisional", "provisional_development"})
    metadata.loc[provisional, "__label"] = math.nan
    feature_master: pd.DataFrame | None = None
    for path in args.features:
        current = pd.read_csv(path, low_memory=False)
        feature_id = resolved["id"] if resolved["id"] in current else resolve_column(current, None, "id")
        current = current.rename(columns={feature_id: "__id"})
        current["__id"] = current.__id.astype(str)
        if current.__id.duplicated().any():
            raise ValueError(f"Feature IDs are not unique in {path}")
        current = current.set_index("__id")
        if feature_master is None:
            feature_master = current
            continue
        union_index = feature_master.index.union(current.index)
        feature_master = feature_master.reindex(union_index)
        current = current.reindex(union_index)
        for column in current:
            if column not in feature_master:
                feature_master[column] = current[column]
                continue
            left, right = feature_master[column], current[column]
            overlap = left.notna() & right.notna()
            if overlap.any():
                left_values, right_values = left[overlap], right[overlap]
                try:
                    conflict = ~np.isclose(
                        pd.to_numeric(left_values), pd.to_numeric(right_values),
                        equal_nan=True, rtol=1e-9, atol=1e-12,
                    )
                except (TypeError, ValueError):
                    conflict = left_values.astype(str).to_numpy() != right_values.astype(str).to_numpy()
                if bool(np.asarray(conflict).any()):
                    raise ValueError(f"Conflicting duplicate feature values for column {column} in {path}")
            feature_master[column] = left.combine_first(right)
    assert feature_master is not None
    feature_master["__has_any_feature_row"] = True
    # Extractor provenance repeats some authoritative metadata. Keep it for
    # audit under an explicit namespace, while verifying overlapping values so
    # pandas never creates ambiguous ``_x``/``_y`` eligibility columns.
    metadata_by_id = metadata.set_index("__id")
    common_ids = metadata_by_id.index.intersection(feature_master.index)
    semantic_checks = {
        resolved["label"]: "__label", resolved["source"]: "__source", resolved["group"]: "__group",
    }
    for audit_column, authoritative_column in semantic_checks.items():
        if audit_column not in feature_master:
            continue
        left = metadata_by_id.loc[common_ids, authoritative_column]
        right = feature_master.loc[common_ids, audit_column]
        present = left.notna() & right.notna()
        if audit_column in ALIASES["label"]:
            right_normalized = right[present].map(normalize_label)
            conflict = ~np.isclose(left[present].astype(float), right_normalized.astype(float), equal_nan=True)
        else:
            conflict = left[present].astype(str).to_numpy() != right[present].astype(str).to_numpy()
        if bool(np.asarray(conflict).any()):
            raise ValueError(f"Extractor {audit_column} conflicts with authoritative metadata")
    for audit_column in ("label", "class_label", "is_ai", "class_name",
                         "source_id", "source_group", "generator", "source",
                         "group_id", "condition_id"):
        if audit_column in feature_master:
            feature_master = feature_master.rename(columns={audit_column: f"extractor__{audit_column}"})
    overlap_columns = sorted((set(feature_master.columns) & set(metadata.columns)) - {"__id"})
    for column in overlap_columns:
        common_ids = metadata_by_id.index.intersection(feature_master.index)
        left = metadata_by_id.loc[common_ids, column]
        right = feature_master.loc[common_ids, column]
        present = left.notna() & right.notna()
        if present.any():
            left_values, right_values = left[present], right[present]
            try:
                conflict = ~np.isclose(
                    pd.to_numeric(left_values), pd.to_numeric(right_values),
                    equal_nan=True, rtol=1e-9, atol=1e-12,
                )
            except (TypeError, ValueError):
                conflict = left_values.astype(str).to_numpy() != right_values.astype(str).to_numpy()
            if bool(np.asarray(conflict).any()):
                raise ValueError(f"Extractor audit field conflicts with metadata: {column}")
        feature_master = feature_master.rename(columns={column: f"extractor__{column}"})
    table = metadata.merge(feature_master.reset_index(), on="__id", how="left", validate="one_to_one")
    if "available" in table:
        available = table["available"].astype(str).str.strip().str.lower().isin(
            {"1", "1.0", "true", "yes", "available"}
        )
        table = table[available].copy()
    missing_feature_row = table["__has_any_feature_row"].isna()
    if missing_feature_row.any():
        status_columns = [name for name in ("feature_status", "extraction_status", "failure_reason", "feature_error")
                          if name in table]
        success_tokens = {"ok", "success", "complete", "completed", "1", "1.0", "true", "available"}

        def explicit_failure(row: pd.Series) -> bool:
            values = [str(row[name]).strip().lower() for name in status_columns
                      if pd.notna(row[name]) and str(row[name]).strip()]
            return any(value not in success_tokens for value in values)

        explained = table.loc[missing_feature_row].apply(explicit_failure, axis=1)
        if not bool(explained.all()):
            examples = table.loc[missing_feature_row].loc[~explained, "__id"].head(10).tolist()
            raise ValueError(f"Available rows have no feature record or explicit failure: {examples}")
    table["__has_any_feature_row"] = table["__has_any_feature_row"].eq(True)
    return table, resolved


def load_family_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload).issuperset(FAMILY_ORDER) and all(isinstance(payload[name], list) for name in FAMILY_ORDER):
        payload = {
            "primary_feature_set": "primary",
            "feature_sets": {
                "primary": {"selection_eligible": True, "cohort": "configured cohort",
                            "eligibility": {}, "families": payload}
            },
        }
    if "feature_sets" not in payload or not isinstance(payload["feature_sets"], dict):
        raise ValueError("families JSON must contain feature_sets or be a simple S/D/R/P mapping")
    primary = payload.get("primary_feature_set")
    if primary not in payload["feature_sets"]:
        raise ValueError("primary_feature_set must name one configured feature set")
    for set_name, spec in payload["feature_sets"].items():
        families = spec.get("families", {})
        unknown = sorted(set(families) - set(FAMILY_ORDER))
        if unknown:
            raise ValueError(f"Unknown families in {set_name}: {unknown}")
        if not families or any(not isinstance(value, list) or not value for value in families.values()):
            raise ValueError(f"Every configured family in {set_name} must have a non-empty list")
        flattened = sum((families[name] for name in families), [])
        if len(flattened) != len(set(flattened)):
            raise ValueError(f"Feature columns overlap across families in {set_name}")
        if bool(spec.get("selection_eligible", False)) and set(families) != set(FAMILY_ORDER):
            raise ValueError(f"Selection feature set {set_name} must define all S/D/R/P families")
    eligible = [name for name, spec in payload["feature_sets"].items()
                if bool(spec.get("selection_eligible", False))]
    if eligible != [primary]:
        raise ValueError("Exactly primary_feature_set must have selection_eligible=true")
    return payload


def combinations(families: dict[str, list[str]]) -> dict[str, list[str]]:
    names = [name for name in FAMILY_ORDER if name in families]
    output: dict[str, list[str]] = {}
    for size in range(1, len(names) + 1):
        for selected in itertools.combinations(names, size):
            output["+".join(selected)] = sum((families[name] for name in selected), [])
    return output


def apply_filters(table: pd.DataFrame, eligibility: dict[str, Any]) -> pd.DataFrame:
    selected = table.copy()

    any_clauses = eligibility.get("any", [])
    if any_clauses:
        if not isinstance(any_clauses, list) or any(not isinstance(item, dict) for item in any_clauses):
            raise ValueError("Eligibility any must be a list of filter objects")
        indices: set[Any] = set()
        for clause in any_clauses:
            indices.update(apply_filters(selected, clause).index)
        selected = selected.loc[selected.index.isin(indices)].copy()

    def actual_column(column: str) -> str:
        if column in selected:
            return column
        internal = {
            "label": "__label", "role": "__role", "source_id": "__source",
            "source": "__source", "group_id": "__group",
        }.get(column)
        if internal in selected:
            return str(internal)
        alternatives = ELIGIBILITY_ALIASES.get(column, ())
        found = [name for name in alternatives if name in selected]
        if found:
            return found[0]
        raise ValueError(f"Eligibility column absent: {column}")

    for column, minimum in eligibility.get("min", {}).items():
        actual = actual_column(column)
        selected = selected[pd.to_numeric(selected[actual], errors="coerce") >= float(minimum)]
    for column, maximum in eligibility.get("max", {}).items():
        actual = actual_column(column)
        selected = selected[pd.to_numeric(selected[actual], errors="coerce") <= float(maximum)]
    for column, expected in eligibility.get("equals", {}).items():
        actual = actual_column(column)
        allowed = expected if isinstance(expected, list) else [expected]
        selected = selected[selected[actual].isin(allowed)]
    for column, allowed in eligibility.get("in", {}).items():
        actual = actual_column(column)
        allowed_values = allowed if isinstance(allowed, list) else [allowed]
        selected = selected[selected[actual].isin(allowed_values)]
    return selected


def load_training_scopes(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {"all_development": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "all_development" not in payload:
        raise ValueError("Training scopes must be an object containing all_development")
    if any(not isinstance(spec, dict) for spec in payload.values()):
        raise ValueError("Every training scope must be a filter object")
    return payload


def apply_eligibility(table: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    families = spec["families"]
    union = list(dict.fromkeys(sum((families[name] for name in FAMILY_ORDER if name in families), [])))
    missing = [name for name in union if name not in table]
    if missing:
        raise ValueError(f"Missing configured feature columns: {missing}")
    eligibility = spec.get("eligibility", {})
    selected = apply_filters(table, eligibility)
    numeric = selected[union].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(float)).all(axis=1)
    configured_status = list(spec.get("status_columns", []))
    absent_status = [name for name in configured_status if name not in selected]
    if absent_status:
        raise ValueError(f"Configured status columns are absent: {absent_status}")
    if configured_status:
        has_status = selected[configured_status].apply(
            lambda row: any(pd.notna(value) and bool(str(value).strip()) for value in row), axis=1
        )
        if not bool(has_status.all()):
            examples = selected.loc[~has_status, "__id"].astype(str).head(10).tolist()
            raise ValueError(f"Available evaluated rows lack feature status: {examples}")
    for column in union:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    known = selected[np.isfinite(selected.__label)]
    coverage = {
        "rows_before_eligibility": int(len(table)),
        "rows_after_metadata_eligibility": int(len(selected)),
        "rows_complete_all_families": int(finite.sum()),
        "complete_all_families_fraction": float(finite.mean()) if len(finite) else math.nan,
        "union_feature_count": len(union),
        "cohort": spec.get("cohort", "unspecified"),
        "eligibility": eligibility,
        "eligible_source_counts": json.dumps(
            source_key(known).value_counts().sort_index().to_dict(), sort_keys=True
        ) if len(known) else "",
    }
    return selected, coverage


def parse_quantities(text: str) -> list[int | str]:
    output: list[int | str] = []
    for value in text.split(","):
        value = value.strip().lower()
        if value == "all":
            output.append("all")
        else:
            amount = int(value)
            if amount <= 0:
                raise ValueError("Quantities must be positive")
            output.append(amount)
    if "all" not in output:
        raise ValueError("Quantities must include all for frozen selection")
    return output


def source_key(table: pd.DataFrame) -> pd.Series:
    return table.__label.astype(int).astype(str) + ":" + table.__source.astype(str)


def sample_weights(table: pd.DataFrame) -> np.ndarray:
    labels = table.__label.to_numpy(int)
    sources = source_key(table).to_numpy(str)
    weights = np.zeros(len(table), dtype=float)
    for label in (0, 1):
        label_indices = np.flatnonzero(labels == label)
        current_sources = sorted(set(sources[label_indices]))
        if not current_sources:
            raise ValueError("Training fold must contain both classes")
        for source in current_sources:
            indices = label_indices[sources[label_indices] == source]
            source_groups = table.iloc[indices].__group.astype(str).to_numpy()
            groups = sorted(set(source_groups))
            for group in groups:
                group_indices = indices[source_groups == group]
                weights[group_indices] = 0.5 / (
                    len(current_sources) * len(groups) * len(group_indices)
                )
    return weights * (len(table) / weights.sum())


def fit_model(table: pd.DataFrame, columns: list[str], feature_mode: str = "values_plus_missing") -> dict[str, Any]:
    x = table[columns].to_numpy(float)
    missing = ~np.isfinite(x)
    medians = np.asarray([
        float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else 0.0
        for column in x.T
    ])
    imputed = np.where(missing, medians, x)
    if feature_mode == "values_plus_missing":
        x = np.concatenate((imputed, missing.astype(float)), axis=1)
    elif feature_mode == "median_only":
        x = imputed
    elif feature_mode == "missingness_only":
        x = missing.astype(float)
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")
    y = table.__label.to_numpy(float)
    weights = sample_weights(table)
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    root_weights = np.sqrt(weights)
    weighted_design = design * root_weights[:, None]
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ (y * root_weights),
    )
    return {
        "columns": columns,
        "medians": medians.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients_with_intercept": coefficients.tolist(),
        "ridge": RIDGE,
        "threshold": THRESHOLD,
        "feature_mode": feature_mode,
        "training_rows": int(len(table)),
        "training_source_counts": source_key(table).value_counts().sort_index().to_dict(),
        "weighting": "classes equal; sources equal within class; groups equal within source; samples equal within group",
        "missing_value_policy": "training-only median imputation plus missingness indicators",
        "observed_fraction_by_column": {
            column: float((~missing[:, index]).mean()) for index, column in enumerate(columns)
        },
    }


def prepare_training(table: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    """Prepare column-independent training transforms once for many subsets.

    Medians, weighted means, and weighted variances are separable by column, so
    selecting a family subset from this union produces the same design matrix as
    calling :func:`fit_model` on that subset directly.
    """
    if len(columns) != len(set(columns)):
        raise ValueError("Prepared union columns must be unique")
    raw = table[columns].to_numpy(float)
    missing = ~np.isfinite(raw)
    medians = np.asarray([
        float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else 0.0
        for column in raw.T
    ])
    imputed = np.where(missing, medians, raw)
    matrix = np.concatenate((imputed, missing.astype(float)), axis=1)
    weights = sample_weights(table)
    mean = np.average(matrix, axis=0, weights=weights)
    variance = np.average((matrix - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    return {
        "columns": columns,
        "column_index": {column: index for index, column in enumerate(columns)},
        "missing": missing,
        "medians": medians,
        "z": (matrix - mean) / scale,
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "y": table.__label.to_numpy(float),
        "training_rows": int(len(table)),
        "training_source_counts": source_key(table).value_counts().sort_index().to_dict(),
    }


def fit_prepared(prepared: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    union_count = len(prepared["columns"])
    value_indices = [prepared["column_index"][column] for column in columns]
    indices = np.asarray([*value_indices, *[union_count + index for index in value_indices]], dtype=int)
    z = prepared["z"][:, indices]
    weights = prepared["weights"]
    design = np.column_stack((np.ones(len(z)), z))
    root_weights = np.sqrt(weights)
    weighted_design = design * root_weights[:, None]
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ (prepared["y"] * root_weights),
    )
    missing = prepared["missing"][:, value_indices]
    return {
        "columns": columns,
        "medians": prepared["medians"][value_indices].tolist(),
        "mean": prepared["mean"][indices].tolist(),
        "scale": prepared["scale"][indices].tolist(),
        "coefficients_with_intercept": coefficients.tolist(),
        "ridge": RIDGE,
        "threshold": THRESHOLD,
        "feature_mode": "values_plus_missing",
        "training_rows": prepared["training_rows"],
        "training_source_counts": prepared["training_source_counts"],
        "weighting": "classes equal; sources equal within class; groups equal within source; samples equal within group",
        "missing_value_policy": "training-only median imputation plus missingness indicators",
        "observed_fraction_by_column": {
            column: float((~missing[:, index]).mean()) for index, column in enumerate(columns)
        },
    }


def predict(table: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    x = table[list(model["columns"])].to_numpy(float)
    missing = ~np.isfinite(x)
    medians = np.asarray(model["medians"], dtype=float)
    imputed = np.where(missing, medians, x)
    feature_mode = model.get("feature_mode", "values_plus_missing")
    if feature_mode == "values_plus_missing":
        x = np.concatenate((imputed, missing.astype(float)), axis=1)
    elif feature_mode == "median_only":
        x = imputed
    elif feature_mode == "missingness_only":
        x = missing.astype(float)
    else:
        raise ValueError(f"Unknown frozen feature mode: {feature_mode}")
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coefficients = np.asarray(model["coefficients_with_intercept"], dtype=float)
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    return design @ coefficients


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    if not n0 or not n1:
        return math.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    prediction = scores >= THRESHOLD
    tp = int(((prediction == 1) & (labels == 1)).sum())
    tn = int(((prediction == 0) & (labels == 0)).sum())
    fp = int(((prediction == 1) & (labels == 0)).sum())
    fn = int(((prediction == 0) & (labels == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    return {
        "roc_auc": auc(labels, scores),
        "balanced_accuracy": (0.5 * (sensitivity + specificity)
                              if np.isfinite(sensitivity) and np.isfinite(specificity) else math.nan),
        "ai_sensitivity": sensitivity,
        "human_specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def hash_fold(value: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def make_folds(table: pd.DataFrame, opposite_folds: int, seed: int) -> list[dict[str, Any]]:
    if opposite_folds < 2:
        raise ValueError("opposite-class-folds must be at least 2")
    humans = sorted(table.loc[table.__label == 0, "__source"].unique())
    generators = sorted(table.loc[table.__label == 1, "__source"].unique())
    if len(humans) < 2 or len(generators) < 2:
        raise ValueError("Source-held-out CV needs at least two human and two AI sources")
    folds: list[dict[str, Any]] = []
    for held_label, held_sources, opposite_label, fold_type in (
        (0, humans, 1, "human_source_holdout"),
        (1, generators, 0, "generator_holdout"),
    ):
        opposite = table[table.__label == opposite_label]
        for held_source in held_sources:
            for inner in range(opposite_folds):
                group_fold = table.__group.map(
                    lambda value: hash_fold(str(value), opposite_folds, seed)
                )
                held_mask = (
                    (table.__label == held_label) & (table.__source == held_source)
                    & (group_fold == inner)
                )
                opposite_mask = (table.__label == opposite_label) & table.__group.map(
                    lambda value: hash_fold(str(value), opposite_folds, seed) == inner
                )
                test_mask = held_mask | opposite_mask
                test = table[test_mask].copy()
                test_groups = set(test.__group)
                train = table[
                    ~((table.__label == held_label) & (table.__source == held_source))
                    & (group_fold != inner)
                    & ~table.__group.isin(test_groups)
                ].copy()
                if train.__label.nunique() != 2 or test.__label.nunique() != 2:
                    continue
                folds.append({
                    "fold_type": fold_type,
                    "heldout_source": held_source,
                    "opposite_group_fold": inner,
                    "train": train,
                    "test": test,
                })
    # Ordinary group-disjoint folds are descriptive only. They exercise every
    # source while holding creator/prompt/condition groups out globally, but do
    # not enter the predeclared source-transfer selection criterion.
    group_fold = table.__group.map(lambda value: hash_fold(str(value), opposite_folds, seed))
    for inner in range(opposite_folds):
        test = table[group_fold == inner].copy()
        train = table[group_fold != inner].copy()
        if train.__label.nunique() == 2 and test.__label.nunique() == 2:
            folds.append({
                "fold_type": "ordinary_group_holdout_descriptive",
                "heldout_source": "__all_sources__",
                "opposite_group_fold": inner,
                "train": train,
                "test": test,
            })
    if not folds:
        raise ValueError("No valid grouped source-held-out folds could be constructed")
    return folds


def make_generator_family_folds(
    table: pd.DataFrame, family_column: str, opposite_folds: int, seed: int,
) -> list[dict[str, Any]]:
    if family_column not in table:
        return []
    ai = table[table.__label == 1]
    families = sorted(str(value) for value in ai[family_column].dropna().unique()
                      if str(value).strip())
    group_fold = table.__group.map(lambda value: hash_fold(str(value), opposite_folds, seed))
    output: list[dict[str, Any]] = []
    for family in families:
        family_mask = (table.__label == 1) & (table[family_column].astype(str) == family)
        for inner in range(opposite_folds):
            test = table[
                (group_fold == inner)
                & (family_mask | (table.__label == 0))
            ].copy()
            train = table[(group_fold != inner) & ~family_mask].copy()
            if train.__label.nunique() != 2 or test.__label.nunique() != 2:
                continue
            output.append({
                "heldout_generator_family": family,
                "opposite_group_fold": inner,
                "train": train,
                "test": test,
            })
    return output


def deterministic_quantity(table: pd.DataFrame, quantity: int | str, seed: int) -> pd.DataFrame | None:
    if quantity == "all":
        return table.copy()
    selected_parts: list[pd.DataFrame] = []
    keyed = table.assign(__source_key=source_key(table))
    for source, current in keyed.groupby("__source_key", sort=True):
        groups = sorted(current.__group.unique(), key=lambda group: hashlib.sha256(
            f"{seed}|{group}".encode()).hexdigest())
        chosen = set(groups[:min(int(quantity), len(groups))])
        selected_parts.append(current[current.__group.isin(chosen)].drop(columns="__source_key"))
    return pd.concat(selected_parts, ignore_index=True)


def per_source_group_range(table: pd.DataFrame) -> tuple[int, int]:
    keyed = table.assign(__source_key=source_key(table))
    counts = keyed.groupby("__source_key").__group.nunique()
    return int(counts.min()), int(counts.max())


def paired_rows(test: pd.DataFrame, scores: np.ndarray, fold: dict[str, Any]) -> list[dict[str, Any]]:
    scored = test[["__id", "__label", "__source"]].copy()
    scored["score"] = scores
    fold_type = fold["fold_type"]
    if fold_type == "ordinary_group_holdout_descriptive":
        output = []
        human_sources = sorted(scored.loc[scored.__label == 0, "__source"].unique())
        ai_sources = sorted(scored.loc[scored.__label == 1, "__source"].unique())
        for human_source in human_sources:
            for ai_source in ai_sources:
                pair = scored[((scored.__label == 0) & (scored.__source == human_source))
                              | ((scored.__label == 1) & (scored.__source == ai_source))]
                if pair.__label.nunique() != 2:
                    continue
                output.append({
                    "fold_type": fold_type,
                    "heldout_source": human_source,
                    "opposite_source": ai_source,
                    "opposite_group_fold": fold["opposite_group_fold"],
                    "test_human": int((pair.__label == 0).sum()),
                    "test_ai": int((pair.__label == 1).sum()),
                    **metrics(pair.__label.to_numpy(int), pair.score.to_numpy(float)),
                })
        return output
    held_source = fold["heldout_source"]
    held_label = 0 if fold_type == "human_source_holdout" else 1
    held = scored[(scored.__label == held_label) & (scored.__source == held_source)]
    opposite_sources = sorted(scored.loc[scored.__label != held_label, "__source"].unique())
    output = []
    for opposite_source in opposite_sources:
        opposite = scored[(scored.__label != held_label) & (scored.__source == opposite_source)]
        pair = pd.concat((held, opposite), ignore_index=True)
        if pair.__label.nunique() != 2:
            continue
        output.append({
            "fold_type": fold_type,
            "heldout_source": held_source,
            "opposite_source": opposite_source,
            "opposite_group_fold": fold["opposite_group_fold"],
            "test_human": int((pair.__label == 0).sum()),
            "test_ai": int((pair.__label == 1).sum()),
            **metrics(pair.__label.to_numpy(int), pair.score.to_numpy(float)),
        })
    return output


def write_csv(path: Path, rows: pd.DataFrame | list[dict[str, Any]]) -> None:
    table = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    table.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def aggregate_cv(raw: pd.DataFrame, bootstrap_replicates: int, seed: int) -> pd.DataFrame:
    keys = ["feature_set", "selection_eligible", "training_scope", "combination",
            "family_count", "quantity", "qualifies_for_selection"]
    source_level = (
        raw.groupby(keys + ["fold_type", "heldout_source"], dropna=False)
        .agg(roc_auc=("roc_auc", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
             ai_sensitivity=("ai_sensitivity", "mean"),
             human_specificity=("human_specificity", "mean"), folds=("roc_auc", "size"))
        .reset_index()
    )
    macro = (
        source_level.groupby(keys + ["fold_type"], dropna=False)
        .agg(roc_auc=("roc_auc", "mean"), balanced_accuracy=("balanced_accuracy", "mean"),
             ai_sensitivity=("ai_sensitivity", "mean"),
             human_specificity=("human_specificity", "mean"),
             heldout_sources=("heldout_source", "nunique"))
        .reset_index()
    )
    human = macro[macro.fold_type == "human_source_holdout"].drop(columns="fold_type").rename(
        columns={name: f"human_holdout_{name}" for name in
                 ("roc_auc", "balanced_accuracy", "ai_sensitivity", "human_specificity", "heldout_sources")})
    generator = macro[macro.fold_type == "generator_holdout"].drop(columns="fold_type").rename(
        columns={name: f"generator_holdout_{name}" for name in
                 ("roc_auc", "balanced_accuracy", "ai_sensitivity", "human_specificity", "heldout_sources")})
    summary = human.merge(generator, on=keys, how="outer", validate="one_to_one")
    summary["selection_score"] = 0.5 * (
        summary.human_holdout_roc_auc + summary.generator_holdout_roc_auc
    )
    summary["selection_criterion"] = "0.5*human-source-heldout macro AUC + 0.5*generator-heldout macro AUC"
    bootstrap_rows: list[dict[str, Any]] = []
    for values, current in source_level.groupby(keys, dropna=False):
        human_values = current.loc[current.fold_type == "human_source_holdout", "roc_auc"].dropna().to_numpy(float)
        generator_values = current.loc[current.fold_type == "generator_holdout", "roc_auc"].dropna().to_numpy(float)
        row = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        row.update({
            "human_holdout_roc_auc_ci_low": math.nan,
            "human_holdout_roc_auc_ci_high": math.nan,
            "generator_holdout_roc_auc_ci_low": math.nan,
            "generator_holdout_roc_auc_ci_high": math.nan,
            "selection_score_ci_low": math.nan,
            "selection_score_ci_high": math.nan,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_unit": "disabled; final confidence intervals must resample group_id blocks",
        })
        if len(human_values) and len(generator_values) and bootstrap_replicates > 0:
            identity = "|".join(map(str, values if isinstance(values, tuple) else (values,)))
            local_seed = seed ^ int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
            rng = np.random.default_rng(local_seed)
            human_boot = rng.choice(human_values, (bootstrap_replicates, len(human_values)), replace=True).mean(axis=1)
            generator_boot = rng.choice(
                generator_values, (bootstrap_replicates, len(generator_values)), replace=True
            ).mean(axis=1)
            selection_boot = 0.5 * (human_boot + generator_boot)
            row.update({
                "human_holdout_roc_auc_ci_low": float(np.quantile(human_boot, 0.025)),
                "human_holdout_roc_auc_ci_high": float(np.quantile(human_boot, 0.975)),
                "generator_holdout_roc_auc_ci_low": float(np.quantile(generator_boot, 0.025)),
                "generator_holdout_roc_auc_ci_high": float(np.quantile(generator_boot, 0.975)),
                "selection_score_ci_low": float(np.quantile(selection_boot, 0.025)),
                "selection_score_ci_high": float(np.quantile(selection_boot, 0.975)),
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_unit": "heldout source macro (diagnostic only; not final group-block CI)",
            })
        bootstrap_rows.append(row)
    summary = summary.merge(pd.DataFrame(bootstrap_rows), on=keys, how="left", validate="one_to_one")
    return summary


def matched_training_scope_comparison(raw: pd.DataFrame) -> pd.DataFrame:
    if not {"prior_only", "all_development"}.issubset(set(raw.training_scope)):
        return pd.DataFrame()
    keys = [
        "feature_set", "combination", "family_count", "quantity", "fold_index", "fold_type",
        "heldout_source", "opposite_source", "opposite_group_fold", "test_human", "test_ai",
    ]
    metrics_to_compare = ["roc_auc", "balanced_accuracy", "ai_sensitivity", "human_specificity"]
    prior = raw[raw.training_scope == "prior_only"][keys + metrics_to_compare].copy()
    expanded = raw[raw.training_scope == "all_development"][keys + metrics_to_compare].copy()
    matched = prior.merge(expanded, on=keys, suffixes=("__prior_only", "__all_development"),
                          validate="one_to_one")
    for metric in metrics_to_compare:
        matched[f"delta_{metric}__expanded_minus_prior"] = (
            matched[f"{metric}__all_development"] - matched[f"{metric}__prior_only"]
        )
    group_keys = ["feature_set", "combination", "family_count", "quantity", "fold_type"]
    aggregations: dict[str, tuple[str, str]] = {
        "matched_source_pairs": ("roc_auc__prior_only", "size"),
        "matched_folds": ("fold_index", "nunique"),
    }
    for metric in metrics_to_compare:
        aggregations[f"{metric}__prior_only"] = (f"{metric}__prior_only", "mean")
        aggregations[f"{metric}__all_development"] = (f"{metric}__all_development", "mean")
        aggregations[f"delta_{metric}__expanded_minus_prior"] = (
            f"delta_{metric}__expanded_minus_prior", "mean"
        )
    return matched.groupby(group_keys, dropna=False).agg(**aggregations).reset_index()


def choose_with_tolerance(candidates: pd.DataFrame) -> pd.Series:
    if candidates.empty or candidates.selection_score.notna().sum() == 0:
        raise ValueError("No finite selection candidates")
    maximum = float(candidates.selection_score.max())
    tied = candidates[
        candidates.selection_score >= maximum - SELECTION_TIE_TOLERANCE
    ].copy()
    tied = tied.sort_values(["family_count", "combination"], ascending=[True, True])
    return tied.iloc[0]


def missingness_diagnostics(
    eligible: pd.DataFrame,
    spec: dict[str, Any],
    combo_names: list[str],
    opposite_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combos = combinations(spec["families"])
    folds = make_folds(eligible, opposite_folds, seed)
    rows: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        train = fold["train"]
        for combo_name in combo_names:
            columns = combos[combo_name]
            for mode in ("values_plus_missing", "median_only", "missingness_only"):
                model = fit_model(train, columns, feature_mode=mode)
                scores = predict(fold["test"], model)
                for row in paired_rows(fold["test"], scores, fold):
                    rows.append({
                        "combination": combo_name,
                        "feature_mode": mode,
                        "fold_index": fold_index,
                        "diagnostic_status": "prespecified_sensitivity_only_never_select",
                        **row,
                    })
    raw = pd.DataFrame(rows)
    source_level = (
        raw.groupby(["combination", "feature_mode", "fold_type", "heldout_source"], dropna=False)
        .agg(roc_auc=("roc_auc", "mean"), balanced_accuracy=("balanced_accuracy", "mean"))
        .reset_index()
    )
    summary = (
        source_level.groupby(["combination", "feature_mode", "fold_type"], dropna=False)
        .agg(heldout_sources=("heldout_source", "nunique"), roc_auc_source_macro=("roc_auc", "mean"),
             balanced_accuracy_source_macro=("balanced_accuracy", "mean"))
        .reset_index()
    )
    summary["diagnostic_status"] = "prespecified_sensitivity_only_never_select"
    return raw, summary


def generator_family_diagnostics(
    eligible: pd.DataFrame,
    spec: dict[str, Any],
    combo_names: list[str],
    opposite_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = make_generator_family_folds(
        eligible, "generator_family", opposite_folds, seed
    )
    if not folds:
        return pd.DataFrame(), pd.DataFrame()
    combos = combinations(spec["families"])
    union_columns = list(dict.fromkeys(sum(
        (spec["families"][name] for name in FAMILY_ORDER if name in spec["families"]), []
    )))
    rows: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        prepared = prepare_training(fold["train"], union_columns)
        for combo_name in combo_names:
            model = fit_prepared(prepared, combos[combo_name])
            test = fold["test"]
            scored = test[["__label", "__source"]].copy()
            scored["score"] = predict(test, model)
            humans = sorted(scored.loc[scored.__label == 0, "__source"].unique())
            ais = sorted(scored.loc[scored.__label == 1, "__source"].unique())
            for human_source in humans:
                for ai_source in ais:
                    pair = scored[
                        ((scored.__label == 0) & (scored.__source == human_source))
                        | ((scored.__label == 1) & (scored.__source == ai_source))
                    ]
                    rows.append({
                        "heldout_generator_family": fold["heldout_generator_family"],
                        "group_fold": fold["opposite_group_fold"],
                        "fold_index": fold_index,
                        "combination": combo_name,
                        "human_source": human_source,
                        "ai_source_variant": ai_source,
                        "train_rows": int(len(fold["train"])),
                        "train_human_sources": int(fold["train"].loc[
                            fold["train"].__label == 0, "__source"
                        ].nunique()),
                        "train_ai_sources": int(fold["train"].loc[
                            fold["train"].__label == 1, "__source"
                        ].nunique()),
                        "test_human": int((pair.__label == 0).sum()),
                        "test_ai": int((pair.__label == 1).sum()),
                        "diagnostic_status": "post_selection_architecture_family_holdout_never_select",
                        **metrics(pair.__label.to_numpy(int), pair.score.to_numpy(float)),
                    })
    raw = pd.DataFrame(rows)
    pair_level = (
        raw.groupby(["heldout_generator_family", "combination", "human_source", "ai_source_variant"],
                    dropna=False)
        .agg(folds=("roc_auc", "size"), roc_auc=("roc_auc", "mean"),
             balanced_accuracy=("balanced_accuracy", "mean"),
             ai_sensitivity=("ai_sensitivity", "mean"),
             human_specificity=("human_specificity", "mean"))
        .reset_index()
    )
    summary = (
        pair_level.groupby(["heldout_generator_family", "combination"], dropna=False)
        .agg(source_pairs=("roc_auc", "size"), roc_auc_pair_source_macro=("roc_auc", "mean"),
             balanced_accuracy_pair_source_macro=("balanced_accuracy", "mean"),
             ai_sensitivity_pair_source_macro=("ai_sensitivity", "mean"),
             human_specificity_pair_source_macro=("human_specificity", "mean"))
        .reset_index()
    )
    summary["diagnostic_status"] = "post_selection_architecture_family_holdout_never_select"
    return raw, summary


def cv_stage(args: argparse.Namespace, table: pd.DataFrame, config: dict[str, Any],
             resolved: dict[str, str]) -> None:
    quantities = parse_quantities(args.quantities)
    training_scopes = load_training_scopes(args.training_scopes_json)
    cv_roles = {value.strip() for value in args.cv_roles.split(",") if value.strip()}
    forbidden = set(EVALUATION_ROLES)
    overlap = cv_roles & forbidden
    if overlap:
        raise ValueError(f"Evaluation-only roles cannot enter CV: {sorted(overlap)}")
    known = table[np.isfinite(table.__label)].copy()
    cv_all = known[known.__role.isin(cv_roles)].copy()
    if cv_all.empty:
        raise ValueError(f"No rows found for CV roles {sorted(cv_roles)}")

    raw_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    frozen_models: dict[str, dict[str, Any]] = {}
    eligible_sets: dict[str, pd.DataFrame] = {}
    for set_name, spec in config["feature_sets"].items():
        eligible, coverage = apply_eligibility(cv_all, spec)
        eligible_sets[set_name] = eligible
        coverage_rows.append({"feature_set": set_name, "role_scope": "cv", **coverage})
        combos = combinations(spec["families"])
        union_columns = list(dict.fromkeys(sum(
            (spec["families"][name] for name in FAMILY_ORDER if name in spec["families"]), []
        )))
        family_informative = {}
        for family, columns in spec["families"].items():
            family_informative[family] = all(
                np.isfinite(eligible.loc[eligible.__label == label, columns].to_numpy(float)).any()
                for label in (0, 1)
            )
        folds = make_folds(eligible, args.opposite_class_folds, args.seed)
        for fold_index, fold in enumerate(folds):
            for scope_name, scope_filter in training_scopes.items():
                scoped_fold_train = apply_filters(fold["train"], scope_filter)
                if scoped_fold_train.__label.nunique() != 2:
                    fold_rows.append({
                        "feature_set": set_name, "training_scope": scope_name,
                        "fold_index": fold_index, "fold_type": fold["fold_type"],
                        "heldout_source": fold["heldout_source"],
                        "opposite_group_fold": fold["opposite_group_fold"],
                        "quantity": "not_evaluable", "status": "insufficient_two_classes_after_scope_filter",
                        "train_rows": int(len(scoped_fold_train)),
                        "train_human_sources": int(scoped_fold_train.loc[scoped_fold_train.__label == 0, "__source"].nunique()),
                        "train_ai_sources": int(scoped_fold_train.loc[scoped_fold_train.__label == 1, "__source"].nunique()),
                        "test_rows": int(len(fold["test"])),
                    })
                    continue
                for quantity in quantities:
                    train = deterministic_quantity(scoped_fold_train, quantity, args.seed)
                    assert train is not None
                    min_groups, max_groups = per_source_group_range(train)
                    prepared = prepare_training(train, union_columns)
                    fold_rows.append({
                        "feature_set": set_name,
                        "training_scope": scope_name,
                        "fold_index": fold_index,
                        "fold_type": fold["fold_type"],
                        "heldout_source": fold["heldout_source"],
                        "opposite_group_fold": fold["opposite_group_fold"],
                        "quantity": str(quantity),
                        "status": "evaluated",
                        "train_rows": int(len(train)),
                        "train_human_sources": int(train.loc[train.__label == 0, "__source"].nunique()),
                        "train_ai_sources": int(train.loc[train.__label == 1, "__source"].nunique()),
                        "train_min_groups_per_source": min_groups,
                        "train_max_groups_per_source": max_groups,
                        "test_rows": int(len(fold["test"])),
                        "test_human_sources": int(fold["test"].loc[fold["test"].__label == 0, "__source"].nunique()),
                        "test_ai_sources": int(fold["test"].loc[fold["test"].__label == 1, "__source"].nunique()),
                    })
                    for combo_name, columns in combos.items():
                        included = combo_name.split("+")
                        qualifies = all(family_informative[name] for name in included)
                        model = fit_prepared(prepared, columns)
                        scores = predict(fold["test"], model)
                        for row in paired_rows(fold["test"], scores, fold):
                            raw_rows.append({
                                "feature_set": set_name,
                                "selection_eligible": bool(spec.get("selection_eligible", False)),
                                "training_scope": scope_name,
                                "cohort": spec.get("cohort", "unspecified"),
                                "combination": combo_name,
                                "family_count": combo_name.count("+") + 1,
                                "qualifies_for_selection": qualifies,
                                "quantity": str(quantity),
                                "fold_index": fold_index,
                                "train_rows": int(len(train)),
                                "train_sources": int(source_key(train).nunique()),
                                "train_min_groups_per_source": min_groups,
                                "train_max_groups_per_source": max_groups,
                                **row,
                            })
        full_train = deterministic_quantity(eligible, "all", args.seed)
        assert full_train is not None
        full_prepared = prepare_training(full_train, union_columns)
        frozen_models[set_name] = {
            combo_name: fit_prepared(full_prepared, columns) for combo_name, columns in combos.items()
        }

    raw = pd.DataFrame(raw_rows)
    if raw.empty:
        raise ValueError("No CV results were produced")
    summary = aggregate_cv(raw, args.bootstrap_replicates, args.seed)
    primary = config["primary_feature_set"]
    candidates = summary[
        (summary.feature_set == primary) & (summary.training_scope == "all_development")
        & (summary.quantity == "all")
        & summary.selection_score.notna()
    ].copy()
    if len(candidates) != 15:
        raise ValueError(f"Primary all-data comparison must contain all 15 combinations; got {len(candidates)}")
    qualified_candidates = candidates[candidates.qualifies_for_selection].copy()
    if qualified_candidates.empty:
        raise ValueError("No primary combination has observed features in both classes")
    candidates = candidates.sort_values(
        ["selection_score", "family_count", "combination"], ascending=[False, True, True]
    )
    leader = choose_with_tolerance(qualified_candidates)
    singles = qualified_candidates[qualified_candidates.family_count == 1]
    best_single = choose_with_tolerance(singles)
    all_four = candidates[candidates.combination == "S+D+R+P"]
    if len(all_four) != 1:
        raise ValueError("Primary feature set did not produce the S+D+R+P comparison")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "evaluate_expanded_cv_raw.csv", raw)
    write_csv(args.output_dir / "evaluate_expanded_cv_summary.csv", summary)
    write_csv(args.output_dir / "evaluate_expanded_cv_coverage.csv", coverage_rows)
    write_csv(args.output_dir / "evaluate_expanded_fold_manifest.csv", fold_rows)
    matched_scopes = matched_training_scope_comparison(raw)
    write_csv(args.output_dir / "evaluate_expanded_training_scope_matched.csv", matched_scopes)
    ordinary = raw[raw.fold_type == "ordinary_group_holdout_descriptive"]
    if len(ordinary):
        ordinary_summary = (
            ordinary.groupby(["feature_set", "training_scope", "combination", "family_count", "quantity"],
                             dropna=False)
            .agg(source_pairs=("roc_auc", "size"), roc_auc_source_pair_macro=("roc_auc", "mean"),
                 balanced_accuracy_source_pair_macro=("balanced_accuracy", "mean"))
            .reset_index()
        )
    else:
        ordinary_summary = pd.DataFrame()
    write_csv(args.output_dir / "evaluate_expanded_group_cv_descriptive.csv", ordinary_summary)
    focus_names = {str(leader.combination), str(best_single.combination), "S+D+R+P"}
    focus = summary[(summary.feature_set == primary)
                    & (summary.training_scope == "all_development")
                    & summary.combination.isin(focus_names)].copy()
    focus["focus_status"] = focus.combination.map(
        lambda value: "cv_leader" if value == leader.combination else
        ("best_single" if value == best_single.combination else "all_four_secondary"))
    write_csv(args.output_dir / "evaluate_expanded_quantity_focus.csv", focus)
    diagnostic_combos = list(dict.fromkeys([str(leader.combination), "S"]))
    diagnostic_raw, diagnostic_summary = missingness_diagnostics(
        eligible_sets[primary], config["feature_sets"][primary], diagnostic_combos,
        args.opposite_class_folds, args.seed,
    )
    write_csv(args.output_dir / "evaluate_expanded_missingness_diagnostic_raw.csv", diagnostic_raw)
    write_csv(args.output_dir / "evaluate_expanded_missingness_diagnostic_summary.csv", diagnostic_summary)
    family_raw, family_summary = generator_family_diagnostics(
        eligible_sets[primary], config["feature_sets"][primary], diagnostic_combos,
        args.opposite_class_folds, args.seed,
    )
    write_csv(args.output_dir / "evaluate_expanded_generator_family_holdout_raw.csv", family_raw)
    write_csv(args.output_dir / "evaluate_expanded_generator_family_holdout_summary.csv", family_summary)

    frozen = {
        "schema_version": 1,
        "created_by_stage": "cv_before_locked_evaluation",
        "seed": args.seed,
        "ridge": RIDGE,
        "threshold": THRESHOLD,
        "selection_tie_tolerance": SELECTION_TIE_TOLERANCE,
        "cv_roles": sorted(cv_roles),
        "training_scopes": training_scopes,
        "source_column": resolved["source"],
        "group_column": resolved["group"],
        "primary_feature_set": primary,
        "family_config_sha256": canonical_hash(config),
        "family_config": config,
        "selection_criterion": "fixed equal mean of human-source-heldout macro AUC and generator-heldout macro AUC",
        "tie_break": (
            f"scores within {SELECTION_TIE_TOLERANCE:g} are tied; then fewer families; then combination name"
        ),
        "selection_quantity": "all",
        "leader": {
            "feature_set": primary,
            "combination": str(leader.combination),
            "selection_score": float(leader.selection_score),
            "human_holdout_roc_auc": float(leader.human_holdout_roc_auc),
            "generator_holdout_roc_auc": float(leader.generator_holdout_roc_auc),
        },
        "best_single": {
            "feature_set": primary,
            "combination": str(best_single.combination),
            "selection_score": float(best_single.selection_score),
        },
        "all_four": {
            "feature_set": primary,
            "combination": "S+D+R+P",
            "selection_score": float(all_four.iloc[0].selection_score),
            "qualifies_for_selection": bool(all_four.iloc[0].qualifies_for_selection),
            "status": "reported comparison; never selected from locked results",
        },
        "models": frozen_models,
        "input_files_cv": {
            "metadata": {"path": str(args.metadata), "sha256": sha256_file(args.metadata)},
            "features": [{"path": str(path), "sha256": sha256_file(path)} for path in args.features],
            "families_json": {"path": str(args.families_json), "sha256": sha256_file(args.families_json)},
        },
        "prior_heldout_results_preserved": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.prior_heldout
        ],
        "guardrails": [
            "locked/stress/provisional/pilot roles excluded from model selection",
            "all combinations compared on the same metadata-qualified row cohort per feature set",
            "training-only median imputation and missingness indicators",
            "common-band primary only; native/full-band sets are sensitivity-only",
            "locked all-15 results cannot change the frozen leader",
            "generator-family holdout is a post-selection leader-vs-S diagnostic only",
        ],
    }
    frozen_path = args.output_dir / "evaluate_expanded_frozen_selection.json"
    frozen_path.write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    run_summary = {
        "stage": "cv",
        "cv_rows_before_feature_eligibility": int(len(cv_all)),
        "cv_class_counts": cv_all.__label.value_counts().sort_index().to_dict(),
        "cv_source_counts": source_key(cv_all).value_counts().sort_index().to_dict(),
        "leader": frozen["leader"],
        "best_single": frozen["best_single"],
        "all_four": frozen["all_four"],
        "frozen_selection": str(frozen_path),
    }
    (args.output_dir / "evaluate_expanded_cv_run_summary.json").write_text(
        json.dumps(run_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(run_summary, indent=2, allow_nan=False))


def default_slices(table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if "original_role" in table:
        return {
            "legacy_locked_400": {
                "roles": ["locked"],
                "filters": {"equals": {"original_role": "locked_test"}},
                "evaluation_status": "previously_used_test_reference",
            },
            "new_catalogue_human": {
                "roles": ["locked"],
                "filters": {"equals": {"original_role": "locked_catalogue_test"}},
                "evaluation_status": "new_locked_human_specificity",
            },
            "new_catalogue_plus_legacy_locked_ai": {
                "roles": ["locked"],
                "filters": {"any": [
                    {"equals": {"original_role": "locked_catalogue_test"}},
                    {"equals": {"original_role": "locked_test", "label": 1}},
                ]},
                "evaluation_status": "new_human_vs_previously_used_ai_diagnostic",
            },
            "new_catalogue_plus_diff_rhythm_pilot": {
                "roles": ["locked", "pilot"],
                "filters": {"any": [
                    {"equals": {"original_role": "locked_catalogue_test"}},
                    {"equals": {"role": "pilot"}},
                ]},
                "evaluation_status": "exploratory_pilot",
            },
            "stress_human": {"roles": ["stress"], "filters": {},
                             "evaluation_status": "stress_diagnostic"},
            "audiox_unverified_scores": {"roles": ["provisional"], "filters": {},
                                         "evaluation_status": "unverified_score_distribution_only"},
        }
    roles = sorted(set(table.__role) & set(EVALUATION_ROLES))
    return {role: {"roles": [role], "filters": {}} for role in roles}


def normalize_slices(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("evaluation slices JSON must be an object")
    output: dict[str, dict[str, Any]] = {}
    for name, value in payload.items():
        if isinstance(value, list):
            output[name] = {"roles": value, "filters": {}}
        elif isinstance(value, dict) and isinstance(value.get("roles"), list):
            output[name] = {"roles": value["roles"], "filters": value.get("filters", {}),
                            "note": value.get("note", ""),
                            "evaluation_status": value.get("evaluation_status")}
        else:
            raise ValueError(f"Slice {name} must be a role list or an object with roles")
    return output


def score_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "score_mean": float(np.mean(values)),
        "score_sd": float(np.std(values)),
        "score_q05": float(np.quantile(values, 0.05)),
        "score_median": float(np.median(values)),
        "score_q95": float(np.quantile(values, 0.95)),
    }


def evaluation_status(roles: list[str], frozen_leader: bool) -> str:
    role_set = set(roles)
    if role_set & {"provisional", "pilot", "provisional_development", "locked_pilot_only"}:
        return "exploratory_only"
    if role_set & {"stress", "stress_test_only"}:
        return "stress_diagnostic"
    return "frozen_primary" if frozen_leader else "secondary_no_selection"


def locked_stage(args: argparse.Namespace, table: pd.DataFrame, config: dict[str, Any]) -> None:
    if args.frozen_selection is None:
        raise ValueError("--frozen-selection is required for locked stage")
    frozen = json.loads(args.frozen_selection.read_text(encoding="utf-8"))
    if frozen.get("created_by_stage") != "cv_before_locked_evaluation":
        raise ValueError("Frozen selection was not produced by the CV stage")
    if frozen.get("family_config_sha256") != canonical_hash(config):
        raise ValueError("Family config differs from the pre-locked frozen selection")
    if float(frozen.get("ridge")) != RIDGE or float(frozen.get("threshold")) != THRESHOLD:
        raise ValueError("Frozen model protocol does not match ridge/threshold constants")
    if args.evaluation_slices_json:
        slices = normalize_slices(json.loads(args.evaluation_slices_json.read_text(encoding="utf-8")))
    else:
        slices = default_slices(table)
    cv_roles = set(frozen["cv_roles"])
    for name, slice_spec in slices.items():
        roles = slice_spec["roles"]
        if set(roles) & cv_roles:
            raise ValueError(f"Locked slice {name} includes CV roles")

    detailed_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    score_rows: list[pd.DataFrame] = []
    leader = frozen["leader"]
    for set_name, spec in config["feature_sets"].items():
        models = frozen["models"][set_name]
        for slice_name, slice_spec in slices.items():
            roles = slice_spec["roles"]
            current = table[table.__role.isin(roles)].copy()
            current = apply_filters(current, slice_spec.get("filters", {}))
            eligible, coverage = apply_eligibility(current, spec)
            coverage_rows.append({"feature_set": set_name, "slice": slice_name,
                                  "roles": "|".join(roles), **coverage})
            for combo_name in combinations(spec["families"]):
                model = models[combo_name]
                scores = predict(eligible, model)
                scored = eligible[["__id", "__label", "__claimed_label", "__role", "__source", "__group"]].copy()
                scored["score"] = scores
                scored["feature_set"] = set_name
                scored["slice"] = slice_name
                scored["combination"] = combo_name
                scored["frozen_leader"] = (
                    set_name == leader["feature_set"] and combo_name == leader["combination"]
                )
                score_rows.append(scored)
                for source, group in scored.groupby("__source", sort=True):
                    known = group[np.isfinite(group.__label)]
                    row = {
                        "feature_set": set_name,
                        "selection_eligible": bool(spec.get("selection_eligible", False)),
                        "slice": slice_name,
                        "roles": "|".join(roles),
                        "combination": combo_name,
                        "frozen_leader": bool(scored.frozen_leader.iloc[0]) if len(scored) else False,
                        "source": source,
                        "rows": int(len(group)),
                        "known_label_rows": int(len(known)),
                        "evaluation_status": evaluation_status(
                            roles, bool(scored.frozen_leader.iloc[0]) if len(scored) else False
                        ) if not slice_spec.get("evaluation_status") else slice_spec["evaluation_status"],
                        **score_summary(group.score.to_numpy(float)),
                    }
                    if len(known):
                        row.update(metrics(known.__label.to_numpy(int), known.score.to_numpy(float)))
                    detailed_rows.append(row)
                known = scored[np.isfinite(scored.__label)].copy()
                humans = sorted(known.loc[known.__label == 0, "__source"].unique())
                ais = sorted(known.loc[known.__label == 1, "__source"].unique())
                for human_source in humans:
                    for ai_source in ais:
                        pair = known[((known.__label == 0) & (known.__source == human_source))
                                     | ((known.__label == 1) & (known.__source == ai_source))]
                        pair_rows.append({
                            "feature_set": set_name,
                            "selection_eligible": bool(spec.get("selection_eligible", False)),
                            "slice": slice_name,
                            "combination": combo_name,
                            "frozen_leader": bool(scored.frozen_leader.iloc[0]) if len(scored) else False,
                            "human_source": human_source,
                            "ai_source": ai_source,
                            "test_human": int((pair.__label == 0).sum()),
                            "test_ai": int((pair.__label == 1).sum()),
                            "evaluation_status": evaluation_status(
                                roles, bool(scored.frozen_leader.iloc[0]) if len(scored) else False
                            ) if not slice_spec.get("evaluation_status") else slice_spec["evaluation_status"],
                            **metrics(pair.__label.to_numpy(int), pair.score.to_numpy(float)),
                        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details = pd.DataFrame(detailed_rows)
    pairs = pd.DataFrame(pair_rows)
    scores = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    write_csv(args.output_dir / "evaluate_expanded_locked_source_metrics.csv", details)
    write_csv(args.output_dir / "evaluate_expanded_locked_pair_metrics.csv", pairs)
    write_csv(args.output_dir / "evaluate_expanded_locked_scores.csv", scores)
    write_csv(args.output_dir / "evaluate_expanded_locked_coverage.csv", coverage_rows)
    if not pairs.empty:
        macro = (
            pairs.groupby(["feature_set", "selection_eligible", "slice", "combination",
                           "frozen_leader", "evaluation_status"], dropna=False)
            .agg(source_pairs=("roc_auc", "size"), roc_auc_source_macro=("roc_auc", "mean"),
                 balanced_accuracy_source_macro=("balanced_accuracy", "mean"),
                 ai_sensitivity_source_macro=("ai_sensitivity", "mean"),
                 human_specificity_source_macro=("human_specificity", "mean"))
            .reset_index()
        )
    else:
        macro = pd.DataFrame(columns=["feature_set", "slice", "combination", "frozen_leader"])
    write_csv(args.output_dir / "evaluate_expanded_locked_macro_metrics.csv", macro)
    run_summary = {
        "stage": "locked_score_only",
        "frozen_leader_unchanged": leader,
        "slices": slices,
        "source_metric_rows": int(len(details)),
        "source_pair_metric_rows": int(len(pairs)),
        "interpretation": (
            "Every model, including all-15 secondary comparisons, was frozen before locked scoring. "
            "Provisional slices are exploratory and cannot alter the leader."
        ),
    }
    (args.output_dir / "evaluate_expanded_locked_run_summary.json").write_text(
        json.dumps(run_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(run_summary, indent=2, allow_nan=False))


def main() -> None:
    args = parse_args()
    table, resolved = load_table(args)
    config = load_family_config(args.families_json)
    if args.stage == "cv":
        cv_stage(args, table, config, resolved)
    else:
        locked_stage(args, table, config)


if __name__ == "__main__":
    main()
