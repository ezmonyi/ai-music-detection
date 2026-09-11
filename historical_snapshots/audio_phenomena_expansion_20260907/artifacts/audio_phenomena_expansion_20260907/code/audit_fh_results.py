#!/usr/bin/env python3
"""Independent, read-only audit of frozen 30-second S/D/R/P/F/H dev results.

The auditor never fits a model.  It verifies the frozen contract and inputs,
reconstructs cohorts and coverage, checks fold/role/group invariants, streams
saved predictions, and recomputes every saved source-pair point metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


EXPECTED_FAMILIES = ("S", "D", "R", "P", "F", "H")
OLD_FAMILIES = ("S", "D", "R", "P")
NEW_FAMILIES = ("F", "H")
EXPECTED_PLANNED = ("M", "V", "B", "A", "T")
EXPECTED_QUANTITIES = ("25", "50", "100", "200", "all")
EXPECTED_PARAMETERS = {
    "seed": 20260907,
    "quantities": [25, 50, 100, 200, "all"],
    "group_folds": 5,
    "model": "ridge",
    "threshold_policy": "fixed_0.5",
    "development_role": "development",
    "historical_roles": ["locked"],
    "min_train_groups_per_class": 5,
    "min_test_groups_per_class": 2,
}
DISALLOWED_ROLES = {"locked", "pilot", "provisional", "provisional_development"}
METRIC_NAMES = ("roc_auc", "balanced_accuracy", "ai_sensitivity", "human_specificity")


class AuditFailure(RuntimeError):
    pass


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.counts: dict[str, Any] = {}

    def check(self, name: str, condition: bool, detail: str = "", *, fatal: bool = True) -> None:
        passed = bool(condition)
        self.checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            message = f"{name}: {detail or 'check failed'}"
            self.errors.append(message)
            if fatal:
                raise AuditFailure(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def report(self) -> dict[str, Any]:
        return {
            "status": "passed" if not self.errors else "failed",
            "checks_passed": sum(check["passed"] for check in self.checks),
            "checks_total": len(self.checks),
            "errors": self.errors,
            "warnings": self.warnings,
            "counts": self.counts,
            "checks": self.checks,
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def id_set_hash(values: Iterable[Any]) -> str:
    payload = "".join(f"{value}\n" for value in sorted({str(value) for value in values}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_key(cohort_id: str, combination: str) -> str:
    return hashlib.sha256(f"{cohort_id}|{combination}".encode("utf-8")).hexdigest()[:20]


def nonempty_combinations(codes: Sequence[str]) -> set[str]:
    return {
        "+".join(combo)
        for size in range(1, len(codes) + 1)
        for combo in itertools.combinations(codes, size)
    }


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def as_float(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(value)


def close(left: Any, right: Any, tolerance: float) -> bool:
    a, b = as_float(left), as_float(right)
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def normalize_label(value: Any) -> float:
    if pd.isna(value) or str(value).strip().lower() in {"", "unknown", "unverified", "na", "nan"}:
        return float("nan")
    text = str(value).strip().lower()
    if text in {"0", "0.0", "human", "real"}:
        return 0.0
    if text in {"1", "1.0", "ai", "generated", "synthetic"}:
        return 1.0
    raise ValueError(f"unsupported label value {value!r}")


def resolve_path(recorded: str, override: Path | None) -> Path:
    path = override if override is not None else Path(recorded)
    return path.resolve()


def verify_hashed_path(
    audit: Audit, name: str, record: Mapping[str, Any], override: Path | None
) -> Path:
    path = resolve_path(str(record["path"]), override)
    audit.check(f"{name}_exists", path.is_file(), str(path))
    actual = file_sha256(path)
    audit.check(f"{name}_sha256", actual == record["sha256"], f"{actual} != {record['sha256']}")
    return path


def load_contract_and_bundle(args: argparse.Namespace, audit: Audit) -> dict[str, Any]:
    results = args.results_dir.resolve()
    manifest_path = results / "run_manifest.json"
    if not manifest_path.is_file():
        print(json.dumps({"status": "not_ready", "reason": "run_manifest.json absent", "results_dir": str(results)}, indent=2))
        raise SystemExit(3)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit.check("manifest_stage_dev", manifest.get("stage") == "dev", repr(manifest.get("stage")))
    audit.check(
        "manifest_interpretation_dev_only",
        manifest.get("interpretation") == "development_only_source_holdout_and_group_cv; not historical test",
        repr(manifest.get("interpretation")),
    )
    audit.check("authorization_frozen_verified", manifest.get("authorization", {}).get("status") == "frozen_verified")
    contract = manifest["contract"]
    authorization = manifest["authorization"]
    contract_hash = canonical_hash(contract)
    audit.check("authorization_contract_hash", contract_hash == authorization.get("contract_sha256"))
    audit.check("source_column_contract", contract.get("source_contract") == {
        "source_column": "source_group", "global_group_column": "group_id"
    })
    audit.check("frozen_parameters_exact", contract.get("parameters") == EXPECTED_PARAMETERS, repr(contract.get("parameters")))
    runtime = contract.get("runtime", {})
    audit.check("runtime_python_frozen", runtime.get("python") == "3.11.15", repr(runtime))
    audit.check("runtime_numpy_frozen", runtime.get("numpy") == "1.26.4", repr(runtime))
    audit.check("runtime_pandas_frozen", runtime.get("pandas") == "3.0.5", repr(runtime))
    audit.check(
        "runtime_thread_limits_frozen",
        runtime.get("thread_settings") == {
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"
        },
        repr(runtime.get("thread_settings")),
    )

    evaluator_path = resolve_path("", args.evaluator_path) if args.evaluator_path else Path(
        args.results_dir.parents[1] / "code" / "evaluate_new_phenomena.py"
    )
    if not evaluator_path.is_file():
        evaluator_path = Path(contract.get("families_json", {}).get("path", "")).parents[2] / "code" / "evaluate_new_phenomena.py"
    audit.check("frozen_evaluator_exists", evaluator_path.is_file(), str(evaluator_path))
    audit.check("frozen_evaluator_hash", file_sha256(evaluator_path) == contract["evaluator_sha256"])

    base_path = args.base_evaluator_path
    if base_path is None:
        base_path = evaluator_path.parent / "frozen_evaluate_expanded_20260905.py"
    base_path = base_path.resolve()
    audit.check("preserved_base_evaluator_exists", base_path.is_file(), str(base_path))
    audit.check("preserved_base_evaluator_hash", file_sha256(base_path) == contract["preserved_base_evaluator_sha256"])

    families_path = verify_hashed_path(audit, "families_json", contract["families_json"], args.families_json)
    metadata_path = verify_hashed_path(audit, "metadata", contract["metadata"], args.metadata)
    feature_overrides = args.feature_path or []
    audit.check(
        "feature_override_count",
        not feature_overrides or len(feature_overrides) == len(contract["features"]),
        f"overrides={len(feature_overrides)} expected={len(contract['features'])}",
    )
    feature_paths = [
        verify_hashed_path(
            audit,
            f"feature_{index}",
            record,
            feature_overrides[index] if feature_overrides else None,
        )
        for index, record in enumerate(contract["features"])
    ]

    receipt_path = resolve_path(authorization["receipt_path"], args.preregistration)
    audit.check("preregistration_exists", receipt_path.is_file(), str(receipt_path))
    audit.check("preregistration_file_hash", file_sha256(receipt_path) == authorization["receipt_sha256"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    audit.check("preregistration_status_frozen", receipt.get("status") == "frozen")
    audit.check("preregistration_stage_dev", receipt.get("authorized_stage") == "dev")
    audit.check("preregistration_contract_exact", receipt.get("contract") == contract)
    audit.check("preregistration_contract_hash", receipt.get("contract_sha256") == contract_hash)

    bundle_path = resolve_path(manifest["frozen_dev_bundle"], args.frozen_dev_bundle)
    audit.check("frozen_bundle_exists", bundle_path.is_file(), str(bundle_path))
    audit.check("frozen_bundle_file_hash", file_sha256(bundle_path) == manifest["frozen_dev_bundle_sha256"])
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    audit.check("bundle_status", bundle.get("status") == "frozen_development_models_for_historical_description")
    audit.check("bundle_has_no_selection", bundle.get("selection") is None)
    audit.check("bundle_contract_hash", bundle.get("contract_sha256") == contract_hash)
    audit.check("bundle_evaluator_hash", bundle.get("evaluator_sha256") == contract["evaluator_sha256"])
    audit.check("bundle_base_hash", bundle.get("preserved_base_evaluator_sha256") == contract["preserved_base_evaluator_sha256"])
    audit.check("bundle_input_hashes", bundle.get("families_json_sha256") == contract["families_json"]["sha256"]
                and bundle.get("metadata_sha256") == contract["metadata"]["sha256"]
                and bundle.get("feature_hashes") == contract["features"])
    audit.check("active_new_families_FH_only", manifest.get("active_new_families") == ["F", "H"])
    audit.check("planned_families_explicit", manifest.get("planned_new_families") == list(EXPECTED_PLANNED))
    return {
        "manifest": manifest,
        "contract": contract,
        "bundle": bundle,
        "bundle_path": bundle_path,
        "families_path": families_path,
        "metadata_path": metadata_path,
        "feature_paths": feature_paths,
        "evaluator_path": evaluator_path,
        "base_path": base_path,
    }


def apply_filters(frame: pd.DataFrame, contract: Mapping[str, Any]) -> pd.DataFrame:
    selected = frame.copy()
    aliases = {
        "native_duration_sec": ("native_duration_sec", "native_duration_s", "duration_sec", "duration_s"),
        "native_sample_rate_hz": ("native_sample_rate_hz", "native_sr", "sample_rate_hz", "sample_rate"),
    }

    def actual(column: str) -> str:
        if column in selected:
            return column
        matches = [candidate for candidate in aliases.get(column, ()) if candidate in selected]
        if matches:
            return matches[0]
        raise KeyError(column)

    any_clauses = contract.get("any", [])
    if any_clauses:
        indices: set[Any] = set()
        for clause in any_clauses:
            indices.update(apply_filters(selected, clause).index)
        selected = selected.loc[selected.index.isin(indices)].copy()
    for column, value in contract.get("min", {}).items():
        selected = selected[pd.to_numeric(selected[actual(column)], errors="coerce") >= float(value)]
    for column, value in contract.get("max", {}).items():
        selected = selected[pd.to_numeric(selected[actual(column)], errors="coerce") <= float(value)]
    for column, value in contract.get("equals", {}).items():
        allowed = value if isinstance(value, list) else [value]
        selected = selected[selected[actual(column)].isin(allowed)]
    for column, value in contract.get("in", {}).items():
        allowed = value if isinstance(value, list) else [value]
        selected = selected[selected[actual(column)].isin(allowed)]
    return selected


def load_inputs(context: Mapping[str, Any], audit: Audit) -> tuple[pd.DataFrame, dict[str, Any]]:
    bundle = context["bundle"]
    resolved = bundle["resolved_columns"]
    metadata = pd.read_csv(context["metadata_path"], low_memory=False)
    audit.check("metadata_ids_unique", metadata[resolved["id"]].notna().all()
                and not metadata[resolved["id"]].astype(str).duplicated().any())
    metadata = metadata.rename(columns={
        resolved["id"]: "__id", resolved["label"]: "__label", resolved["role"]: "__role",
        resolved["source"]: "__source", resolved["group"]: "__group",
    })
    metadata["__id"] = metadata["__id"].astype(str)
    metadata["__role"] = metadata["__role"].astype(str)
    metadata["__source"] = metadata["__source"].astype(str)
    metadata["__group"] = metadata["__group"].astype(str)
    metadata["__label"] = metadata["__label"].map(normalize_label)
    metadata.loc[metadata["__role"].isin({"provisional", "provisional_development"}), "__label"] = np.nan

    features: pd.DataFrame | None = None
    for path in context["feature_paths"]:
        current = pd.read_csv(path, low_memory=False)
        candidates = [resolved["id"], "id", "item_id", "track_id", "clip_id", "uid"]
        feature_id = next((column for column in candidates if column in current.columns), None)
        audit.check(f"feature_id_found_{path.name}", feature_id is not None)
        assert feature_id is not None
        current = current.rename(columns={feature_id: "__id"})
        current["__id"] = current["__id"].astype(str)
        audit.check(f"feature_ids_unique_{path.name}", not current["__id"].duplicated().any())
        current = current.set_index("__id")
        if features is None:
            features = current
        else:
            overlap = (set(features.columns) & set(current.columns))
            audit.check(f"feature_columns_nonoverlap_{path.name}", not overlap, repr(sorted(overlap)))
            features = features.join(current, how="outer")
    assert features is not None
    table = metadata.merge(features.reset_index(), on="__id", how="left", validate="one_to_one")
    if "available" in table:
        table = table[table["available"].map(as_bool)].copy()
    config = json.loads(context["families_path"].read_text(encoding="utf-8"))
    specs = {**config["old_families"], **config["new_families"]}
    configured_columns = [column for family in specs.values() for column in family["columns"]]
    audit.check("configured_feature_columns_present", set(configured_columns).issubset(table.columns))
    status_columns = set(config.get("status_columns", [])) | {
        column for family in specs.values() for column in family.get("status_columns", [])
    }
    audit.check("status_columns_not_predictors", not (status_columns & set(configured_columns)), repr(status_columns & set(configured_columns)))
    for column in configured_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    audit.check(
        "source_groups_do_not_cross_labels",
        bool((table[np.isfinite(table["__label"])].groupby("__source")["__label"].nunique() <= 1).all()),
    )
    return table, config


def audit_plan_and_coverage(
    results_dir: Path, table: pd.DataFrame, config: Mapping[str, Any], context: Mapping[str, Any], audit: Audit,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    plan = pd.read_csv(results_dir / "evaluation_plan.csv", dtype=str, keep_default_na=False)
    registry = pd.read_csv(results_dir / "cohort_registry.csv", dtype=str, keep_default_na=False)
    audit.check("one_development_cohort", len(registry) == 1, f"rows={len(registry)}")
    audit.check("plan_rows_108", len(plan) == 108, f"rows={len(plan)}")
    audit.check("manifest_plan_rows", context["manifest"].get("plan_rows") == len(plan))
    expected_combinations = nonempty_combinations(EXPECTED_FAMILIES)
    audit.check("unique_combinations_63", set(plan["combination"]) == expected_combinations,
                f"observed={plan['combination'].nunique()}")
    audit.check("unique_candidates_63", plan["candidate_key"].nunique() == 63,
                f"observed={plan['candidate_key'].nunique()}")
    audit.check("manifest_unique_candidates_63", context["manifest"].get("unique_candidates") == 63)
    audit.check("standalone_old_count_15", int((plan["plan_type"] == "old_baseline_standalone").sum()) == 15)
    audit.check("standalone_new_count_3", int((plan["plan_type"] == "new_standalone").sum()) == 3)
    incremental = plan[plan["plan_type"] == "incremental_matched"]
    audit.check("incremental_rows_90", len(incremental) == 90)
    audit.check("incremental_comparisons_45", incremental["comparison_id"].nunique() == 45)
    audit.check("incremental_two_arms_each", bool((incremental.groupby("comparison_id").size() == 2).all()))

    all_specs = {**config["old_families"], **config["new_families"]}
    audit.check("available_family_set_exact", {
        family for family, spec in all_specs.items() if spec["state"] == "available"
    } == set(EXPECTED_FAMILIES))
    audit.check("planned_family_set_exact", {
        family for family, spec in all_specs.items() if spec["state"] == "planned"
    } == set(EXPECTED_PLANNED))
    for row in plan.to_dict(orient="records"):
        combination = row["combination"]
        audit.check(
            f"candidate_key_{row['plan_id']}",
            row["candidate_key"] == candidate_key(row["cohort_id"], combination),
            fatal=True,
        )
    for comparison, arms in incremental.groupby("comparison_id"):
        baseline = arms[arms["arm"] == "baseline"].iloc[0]
        added = arms[arms["arm"] == "added"].iloc[0]
        audit.check(
            f"matched_plan_contract_{comparison}",
            baseline["cohort_id"] == added["cohort_id"]
            and baseline["eligible_id_set_sha256"] == added["eligible_id_set_sha256"]
            and baseline["eligibility_contract_sha256"] == added["eligibility_contract_sha256"],
        )

    base = table[(table["__role"] == EXPECTED_PARAMETERS["development_role"])
                 & np.isfinite(table["__label"])].copy()
    cohort_tables: dict[str, pd.DataFrame] = {}
    for eligibility_codes, rows in plan.groupby("eligibility_family_codes", sort=False):
        codes = json.loads(eligibility_codes)
        selected = apply_filters(base, config.get("eligibility", {}))
        for code in codes:
            selected = apply_filters(selected, all_specs[code].get("eligibility", {}))
        digest = id_set_hash(selected["__id"])
        for row in rows.to_dict(orient="records"):
            audit.check(f"eligible_rows_{row['plan_id']}", int(row["eligible_rows"]) == len(selected))
            audit.check(f"eligible_hash_{row['plan_id']}", row["eligible_id_set_sha256"] == digest)
            cohort_tables.setdefault(row["cohort_id"], selected)
    cohort = next(iter(cohort_tables.values()))
    cohort_id = next(iter(cohort_tables))
    registry_row = registry.iloc[0]
    audit.check("cohort_rows_3317", len(cohort) == 3317, f"rows={len(cohort)}")
    audit.check("cohort_registry_hash", registry_row["eligible_id_set_sha256"] == id_set_hash(cohort["__id"]))
    audit.check("cohort_registry_counts", int(registry_row["rows"]) == len(cohort)
                and int(registry_row["human_rows"]) == int((cohort["__label"] == 0).sum())
                and int(registry_row["ai_rows"]) == int((cohort["__label"] == 1).sum())
                and int(registry_row["human_sources"]) == cohort.loc[cohort["__label"] == 0, "__source"].nunique()
                and int(registry_row["ai_sources"]) == cohort.loc[cohort["__label"] == 1, "__source"].nunique())

    family_coverage = pd.read_csv(results_dir / "family_coverage_by_source.csv", dtype=str, keep_default_na=False)
    audit.check("family_coverage_rows", len(family_coverage) == cohort["__source"].nunique() * len(all_specs))
    family_lookup = {(row["cohort_id"], row["source_group"], row["family"]): row
                     for row in family_coverage.to_dict(orient="records")}
    for source, current in cohort.groupby("__source"):
        for family, spec in all_specs.items():
            row = family_lookup[(cohort_id, source, family)]
            columns = spec["columns"]
            finite = np.isfinite(current[columns].to_numpy(float)) if columns else np.zeros((len(current), 0), bool)
            any_observed = finite.any(axis=1) if columns else np.zeros(len(current), bool)
            complete = finite.all(axis=1) if columns else np.zeros(len(current), bool)
            expected_qualified = spec["state"] == "available" and float(any_observed.mean()) >= float(spec["minimum_observed_fraction"])
            audit.check(f"family_coverage_{source}_{family}",
                        int(row["eligible_rows"]) == len(current)
                        and int(row["any_observed_rows"]) == int(any_observed.sum())
                        and int(row["complete_rows"]) == int(complete.sum())
                        and close(row["any_observed_fraction"], any_observed.mean(), 1e-12)
                        and close(row["complete_fraction"], complete.mean(), 1e-12)
                        and as_bool(row["coverage_qualified"]) == expected_qualified)

    candidate_coverage = pd.read_csv(results_dir / "candidate_coverage.csv", dtype=str, keep_default_na=False)
    audit.check("candidate_coverage_rows_63", len(candidate_coverage) == 63)
    for row in candidate_coverage.to_dict(orient="records"):
        codes = row["combination"].split("+")
        columns = [column for code in codes for column in all_specs[code]["columns"]]
        finite = np.isfinite(cohort[columns].to_numpy(float))
        any_observed, complete = finite.any(axis=1), finite.all(axis=1)
        qualified = all(
            all_specs[code]["state"] == "available"
            and float(np.isfinite(cohort[all_specs[code]["columns"]].to_numpy(float)).any(axis=1).mean())
            >= float(all_specs[code]["minimum_observed_fraction"])
            for code in codes
        )
        audit.check(f"candidate_coverage_{row['combination']}",
                    int(row["eligible_rows"]) == len(cohort)
                    and int(row["feature_count"]) == len(columns)
                    and int(row["rows_with_any_observed_feature"]) == int(any_observed.sum())
                    and int(row["rows_complete_all_selected_features"]) == int(complete.sum())
                    and close(row["any_observed_fraction"], any_observed.mean(), 1e-12)
                    and close(row["complete_case_fraction_diagnostic_only"], complete.mean(), 1e-12)
                    and as_bool(row["coverage_qualified"]) == qualified)
    audit.counts.update({
        "cohort_rows": len(cohort),
        "human_rows": int((cohort["__label"] == 0).sum()),
        "ai_rows": int((cohort["__label"] == 1).sum()),
        "human_sources": int(cohort.loc[cohort["__label"] == 0, "__source"].nunique()),
        "ai_sources": int(cohort.loc[cohort["__label"] == 1, "__source"].nunique()),
        "plan_rows": len(plan), "unique_candidates": plan["candidate_key"].nunique(),
    })
    return plan, registry, cohort_tables


def load_fold_registry(
    results_dir: Path, metadata_table: pd.DataFrame, plan: pd.DataFrame, audit: Audit,
) -> tuple[dict[str, dict[str, Any]], set[tuple[str, str]]]:
    records = [json.loads(line) for line in (results_dir / "fold_registry.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    audit.check("fold_registry_rows_275", len(records) == 275, f"rows={len(records)}")
    by_uid: dict[str, dict[str, Any]] = {}
    metadata_lookup = metadata_table.set_index("__id")[["__role", "__label", "__source", "__group"]].to_dict(orient="index")
    fold_type_counts: dict[str, set[tuple[int, str]]] = defaultdict(set)
    for record in records:
        uid = record["fold_uid"]
        audit.check(f"fold_uid_unique_{uid}", uid not in by_uid)
        by_uid[uid] = record
        quantity = str(record["quantity"])
        audit.check(f"fold_quantity_{uid}", quantity in EXPECTED_QUANTITIES)
        train_ids, test_ids = list(map(str, record["train_ids"])), list(map(str, record["test_ids"]))
        train_groups, test_groups = set(map(str, record["train_group_ids"])), set(map(str, record["test_group_ids"]))
        audit.check(f"train_test_ids_disjoint_{uid}", not (set(train_ids) & set(test_ids)))
        audit.check(f"train_test_groups_disjoint_{uid}", not (train_groups & test_groups))
        audit.check(f"train_hash_{uid}", id_set_hash(train_ids) == record["train_id_set_sha256"])
        audit.check(f"test_hash_{uid}", id_set_hash(test_ids) == record["test_id_set_sha256"])
        all_ids = train_ids + test_ids
        audit.check(f"fold_ids_known_{uid}", all(item in metadata_lookup for item in all_ids))
        roles = {metadata_lookup[item]["__role"] for item in all_ids}
        audit.check(f"development_role_only_{uid}", roles == {"development"}, repr(roles))
        audit.check(f"no_disallowed_roles_{uid}", not (roles & DISALLOWED_ROLES), repr(roles))
        audit.check(f"train_groups_match_ids_{uid}", train_groups == {metadata_lookup[item]["__group"] for item in train_ids})
        audit.check(f"test_groups_match_ids_{uid}", test_groups == {metadata_lookup[item]["__group"] for item in test_ids})
        if quantity != "all":
            cap = int(quantity)
            per_source_groups: dict[str, set[str]] = defaultdict(set)
            for item in train_ids:
                per_source_groups[metadata_lookup[item]["__source"]].add(metadata_lookup[item]["__group"])
            audit.check(f"quantity_cap_{uid}", all(len(groups) <= cap for groups in per_source_groups.values()))
        fold_type_counts[record["fold_type"]].add((int(record["fold_index"]), quantity))
    audit.check("fold_quantities_exact", {str(record["quantity"]) for record in records} == set(EXPECTED_QUANTITIES))
    audit.check("human_source_fold_cells_150", len(fold_type_counts["human_source_holdout"]) == 150)
    audit.check("generator_fold_cells_100", len(fold_type_counts["generator_holdout"]) == 100)
    audit.check("group_fold_cells_25", len(fold_type_counts["ordinary_group_holdout_descriptive"]) == 25)

    skipped: set[tuple[str, str]] = set()
    skipped_path = results_dir / "skipped_folds.csv"
    if skipped_path.stat().st_size > 1:
        with skipped_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                matching = [record for record in records if record["cohort_id"] == row["cohort_id"]
                            and str(record["fold_index"]) == str(row["fold_index"])
                            and record["fold_type"] == row["fold_type"]
                            and str(record["quantity"]) == str(row["quantity"])]
                audit.check("skipped_fold_resolves_unique", len(matching) == 1)
                skipped.add((matching[0]["fold_uid"], str(row["quantity"])))
    audit.counts["fold_registry_rows"] = len(records)
    audit.counts["skipped_fold_quantity_cells"] = len(skipped)
    return by_uid, skipped


def independent_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n0, n1 = int(np.sum(labels == 0)), int(np.sum(labels == 1))
    if not n0 or not n1:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and scores[order[end]] == scores[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * ((index + 1) + end)
        index = end
    return float((np.sum(ranks[labels == 1]) - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def independent_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "roc_auc": independent_auc(labels, scores),
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "ai_sensitivity": sensitivity,
        "human_specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def metric_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row["fold_uid"]), str(row["candidate_key"]),
        str(row.get("feature_mode", "main")), str(row["human_source"]), str(row["ai_source"]),
    )


def load_metrics(path: Path, mode_default: str, audit: Audit) -> dict[tuple[str, ...], dict[str, str]]:
    output: dict[tuple[str, ...], dict[str, str]] = {}
    if path.stat().st_size <= 1:
        return output
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if "feature_mode" not in row:
                row["feature_mode"] = mode_default
            key = metric_key(row)
            if key in output:
                raise AuditFailure(f"duplicate metric key in {path.name}: {key}")
            output[key] = row
    audit.check(f"metric_keys_unique_{path.name}", True, f"rows={len(output)}")
    return output


def prediction_cell_key(row: Mapping[str, Any], diagnostic: bool) -> tuple[str, ...]:
    return (
        str(row["fold_uid"]), str(row["candidate_key"]),
        str(row.get("feature_mode", "main")) if diagnostic else "main",
    )


def process_prediction_file(
    path: Path,
    metrics: dict[tuple[str, ...], dict[str, str]],
    fold_by_uid: Mapping[str, Mapping[str, Any]],
    plan_candidate: Mapping[str, Mapping[str, str]],
    metadata_lookup: Mapping[str, Mapping[str, Any]],
    model_lookup: Mapping[str, Mapping[str, Any]],
    audit: Audit,
    tolerance: float,
    diagnostic: bool,
    macro_accumulator: dict[Any, Any] | None,
) -> tuple[set[tuple[str, ...]], dict[tuple[str, ...], str], int, int]:
    seen_cells: set[tuple[str, ...]] = set()
    test_hashes: dict[tuple[str, ...], str] = {}
    seen_metric_keys: set[tuple[str, ...]] = set()
    prediction_rows = 0
    point_metrics = 0

    def ensure(condition: bool, message: str) -> None:
        if not condition:
            raise AuditFailure(f"{path.name}: {message}")

    def flush(rows: list[dict[str, str]]) -> None:
        nonlocal point_metrics
        if not rows:
            return
        first = rows[0]
        cell = prediction_cell_key(first, diagnostic)
        ensure(cell not in seen_cells, f"prediction cell is non-contiguous or duplicated: {cell}")
        seen_cells.add(cell)
        uid, candidate, mode = cell
        ensure(uid in fold_by_uid, f"unknown fold_uid {uid}")
        ensure(candidate in plan_candidate, f"unknown candidate_key {candidate}")
        fold = fold_by_uid[uid]
        plan_row = plan_candidate[candidate]
        for field in (
            "fold_uid", "cohort_id", "candidate_key", "combination", "quantity",
            "fold_index", "fold_type", "heldout_source", "opposite_group_fold",
        ):
            ensure(len({str(row[field]) for row in rows}) == 1, f"{field} not constant in {cell}")
        ensure(first["combination"] == plan_row["combination"], f"combination mismatch in {cell}")
        ensure(first["cohort_id"] == fold["cohort_id"] == plan_row["cohort_id"], f"cohort mismatch in {cell}")
        ensure(first["fold_type"] == fold["fold_type"], f"fold type mismatch in {cell}")
        ensure(str(first["quantity"]) == str(fold["quantity"]), f"quantity mismatch in {cell}")
        row_ids = [str(row["row_id"]) for row in rows]
        ensure(len(row_ids) == len(set(row_ids)), f"duplicate prediction row IDs in {cell}")
        test_hash = id_set_hash(row_ids)
        ensure(test_hash == fold["test_id_set_sha256"], f"test ID set mismatch in {cell}")
        test_hashes[cell] = test_hash
        thresholds = {float(row["threshold"]) for row in rows}
        model_hashes = {row["model_sha256"] for row in rows}
        ensure(len(thresholds) == 1, f"threshold not constant in {cell}")
        ensure(len(model_hashes) == 1, f"model hash not constant in {cell}")
        if not diagnostic:
            fold_model_hash = next(iter(model_hashes))
            ensure(len(fold_model_hash) == 64 and all(character in "0123456789abcdef" for character in fold_model_hash),
                   f"invalid fold-model hash in {cell}")
            ensure(thresholds == {0.5}, f"prediction threshold differs from preregistered fixed 0.5 in {cell}")
        else:
            ensure(str(first["quantity"]) == "all", f"diagnostic is not all-quantity in {cell}")
            ensure(mode in {"median_only", "missingness_only"}, f"unknown diagnostic mode in {cell}")
            ensure(all(row["diagnostic_status"] == "sensitivity_only_never_select" for row in rows), f"diagnostic status mismatch in {cell}")
        threshold = next(iter(thresholds))
        by_label_source: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            item = str(row["row_id"])
            ensure(item in metadata_lookup, f"prediction ID absent from metadata: {item}")
            truth = metadata_lookup[item]
            label = int(row["label"])
            score = float(row["score"])
            ensure(truth["__role"] == "development", f"non-development prediction ID {item}")
            ensure(label == int(truth["__label"])
                   and row["source_group"] == truth["__source"]
                   and row["group_id"] == truth["__group"], f"prediction metadata mismatch for {item}")
            ensure(int(row["predicted_label"]) == int(score >= threshold), f"predicted-label rule mismatch for {item}")
            by_label_source[(label, row["source_group"])].append(row)
        humans = sorted(source for label, source in by_label_source if label == 0)
        ais = sorted(source for label, source in by_label_source if label == 1)
        fold_type = first["fold_type"]
        heldout = first["heldout_source"]
        if fold_type == "ordinary_group_holdout_descriptive":
            pairs = list(itertools.product(humans, ais))
        elif fold_type == "human_source_holdout":
            pairs = [(heldout, ai) for ai in ais]
        elif fold_type == "generator_holdout":
            pairs = [(human, heldout) for human in humans]
        else:
            raise AuditFailure(f"unknown fold type {fold_type}")
        minimum_groups = EXPECTED_PARAMETERS["min_test_groups_per_class"]
        for human, ai in pairs:
            human_rows = by_label_source.get((0, human), [])
            ai_rows = by_label_source.get((1, ai), [])
            human_groups = {row["group_id"] for row in human_rows}
            ai_groups = {row["group_id"] for row in ai_rows}
            if len(human_groups) < minimum_groups or len(ai_groups) < minimum_groups:
                continue
            pair_rows = human_rows + ai_rows
            labels = np.asarray([int(row["label"]) for row in pair_rows], dtype=int)
            scores = np.asarray([float(row["score"]) for row in pair_rows], dtype=float)
            computed = independent_metrics(labels, scores, threshold)
            key = (uid, candidate, mode, human, ai)
            ensure(key in metrics, f"saved metric absent for {key}")
            stored = metrics[key]
            seen_metric_keys.add(key)
            ensure(stored["model_sha256"] in model_hashes, f"metric model hash mismatch {key}")
            ensure(close(stored["threshold"], threshold, tolerance), f"metric threshold mismatch {key}")
            ensure(stored["fold_type"] == fold_type and stored["heldout_source"] == heldout,
                   f"metric fold identity mismatch {key}")
            for name in METRIC_NAMES:
                ensure(close(stored[name], computed[name], tolerance),
                       f"point metric mismatch {name} {key}: stored={stored[name]} computed={computed[name]}")
            for name in ("tp", "tn", "fp", "fn"):
                ensure(int(stored[name]) == computed[name], f"confusion count mismatch {name} {key}")
            ensure(int(stored["test_human"]) == len(human_rows)
                   and int(stored["test_ai"]) == len(ai_rows)
                   and int(stored["test_human_groups"]) == len(human_groups)
                   and int(stored["test_ai_groups"]) == len(ai_groups), f"metric denominator mismatch {key}")
            point_metrics += 1
            if macro_accumulator is not None:
                summary_key = (
                    first["cohort_id"], candidate, first["combination"], str(first["quantity"]), fold_type
                )
                source_key = summary_key + (heldout,)
                bucket = macro_accumulator.setdefault(source_key, {
                    "folds": set(), **{name: [] for name in METRIC_NAMES}
                })
                bucket["folds"].add(int(first["fold_index"]))
                for name in METRIC_NAMES:
                    bucket[name].append(computed[name])

    current: list[dict[str, str]] = []
    current_key: tuple[str, ...] | None = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prediction_rows += 1
            key = prediction_cell_key(row, diagnostic)
            if current_key is None:
                current_key = key
            if key != current_key:
                flush(current)
                current = []
                current_key = key
            current.append(row)
    flush(current)
    audit.check(f"prediction_cells_valid_{path.name}", True, f"cells={len(seen_cells)} rows={prediction_rows}")
    audit.check(f"all_metrics_recomputed_{path.name}", seen_metric_keys == set(metrics),
                f"seen={len(seen_metric_keys)} stored={len(metrics)}")
    return seen_cells, test_hashes, prediction_rows, point_metrics


def verify_source_summary(
    path: Path, accumulator: Mapping[Any, Any], audit: Audit, tolerance: float
) -> int:
    source_means: dict[tuple[str, ...], list[dict[str, float]]] = defaultdict(list)
    for source_key, bucket in accumulator.items():
        summary_key = source_key[:-1]
        source_means[summary_key].append({
            name: float(np.mean(bucket[name])) for name in METRIC_NAMES
        })
    expected: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, values in source_means.items():
        expected[key] = {
            f"{name}_source_macro": float(np.mean([value[name] for value in values]))
            for name in METRIC_NAMES
        }
        expected[key]["heldout_sources"] = len(values)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    audit.check(f"summary_row_count_{path.name}", len(rows) == len(expected), f"saved={len(rows)} expected={len(expected)}")
    for row in rows:
        key = (row["cohort_id"], row["candidate_key"], row["combination"], str(row["quantity"]), row["fold_type"])
        audit.check(f"summary_key_{path.name}_{key}", key in expected)
        computed = expected[key]
        for name in METRIC_NAMES:
            field = f"{name}_source_macro"
            audit.check(f"summary_metric_{path.name}_{field}_{key}", close(row[field], computed[field], tolerance))
        audit.check(f"summary_sources_{path.name}_{key}", int(row["heldout_sources"]) == computed["heldout_sources"])
    return len(rows)


def verify_models(plan: pd.DataFrame, config: Mapping[str, Any], bundle: Mapping[str, Any], registry: pd.DataFrame, audit: Audit) -> dict[str, Mapping[str, Any]]:
    models = bundle["models"]
    unique_plan = plan.drop_duplicates("candidate_key").set_index("candidate_key")
    audit.check("frozen_models_63", len(models) == 63, f"models={len(models)}")
    audit.check("frozen_model_keys_match_plan", set(models) == set(unique_plan.index))
    specs = {**config["old_families"], **config["new_families"]}
    cohort_hash = dict(zip(registry["cohort_id"], registry["eligible_id_set_sha256"]))
    for key, record in models.items():
        row = unique_plan.loc[key]
        expected_columns = [column for code in row["combination"].split("+") for column in specs[code]["columns"]]
        training = record["training_config"]
        audit.check(f"model_combination_{key}", record["combination"] == row["combination"])
        audit.check(f"model_columns_{key}", training["columns"] == expected_columns)
        audit.check(f"model_training_hash_{key}", canonical_hash(training) == record["training_config_sha256"])
        audit.check(f"model_training_ids_{key}", training["training_id_set_sha256"] == cohort_hash[row["cohort_id"]])
        audit.check(f"model_type_ridge_{key}", record["model"].get("model_type") == "weighted_ridge_linear_probability")
        audit.check(f"model_identity_link_{key}", record["model"].get("prediction_link") == "identity")
        audit.check(f"model_threshold_fixed_{key}", float(record["threshold"]) == 0.5)
        audit.check(f"model_hash_{key}", canonical_hash({"model": record["model"], "threshold": record["threshold"]}) == record["model_sha256"])
    return models


def verify_matched_arm_hashes(
    plan: pd.DataFrame, fold_by_uid: Mapping[str, Mapping[str, Any]], test_hashes: Mapping[tuple[str, str], str], audit: Audit,
) -> None:
    incremental = plan[plan["plan_type"] == "incremental_matched"]
    comparisons = {
        comparison: (
            arms.loc[arms["arm"] == "baseline", "candidate_key"].iloc[0],
            arms.loc[arms["arm"] == "added", "candidate_key"].iloc[0],
        )
        for comparison, arms in incremental.groupby("comparison_id")
    }
    checks = 0
    for uid in fold_by_uid:
        for baseline, added in comparisons.values():
            left, right = test_hashes.get((uid, baseline, "main")), test_hashes.get((uid, added, "main"))
            if left is None and right is None:
                continue
            if left is None or left != right:
                raise AuditFailure(f"matched arm test-ID mismatch: fold={uid} baseline={baseline} added={added}")
            checks += 1
    audit.check("matched_arm_test_ids_all_equal", True, f"checks={checks}")
    audit.counts["matched_arm_fold_quantity_id_checks"] = checks


def verify_matched_deltas(
    path: Path, metrics: Mapping[tuple[str, ...], Mapping[str, str]], plan: pd.DataFrame, audit: Audit, tolerance: float
) -> int:
    pair_map = {}
    incremental = plan[plan["plan_type"] == "incremental_matched"]
    for comparison, arms in incremental.groupby("comparison_id"):
        pair_map[comparison] = {
            "baseline": arms[arms["arm"] == "baseline"].iloc[0],
            "added": arms[arms["arm"] == "added"].iloc[0],
        }
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    def ensure(condition: bool, message: str) -> None:
        if not condition:
            raise AuditFailure(f"{path.name}: {message}")
    for row in rows:
        pair = pair_map[row["comparison_id"]]
        baseline_key = (row["fold_uid"], pair["baseline"]["candidate_key"], "main", row["human_source"], row["ai_source"])
        added_key = (row["fold_uid"], pair["added"]["candidate_key"], "main", row["human_source"], row["ai_source"])
        ensure(baseline_key in metrics and added_key in metrics, f"metric key absent for {row['comparison_id']} {row['fold_uid']}")
        baseline, added = metrics[baseline_key], metrics[added_key]
        for name in METRIC_NAMES:
            ensure(close(row[f"{name}__baseline"], baseline[name], tolerance), f"baseline mismatch {name}")
            ensure(close(row[f"{name}__added"], added[name], tolerance), f"added mismatch {name}")
            computed = float(added[name]) - float(baseline[name])
            ensure(close(row[f"delta_{name}__added_minus_baseline"], computed, tolerance), f"delta mismatch {name}")
    audit.check(f"matched_deltas_valid_{path.name}", True, f"rows={len(rows)}")
    return len(rows)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    audit = Audit()
    try:
        context = load_contract_and_bundle(args, audit)
        results_dir = args.results_dir.resolve()
        table, config = load_inputs(context, audit)
        plan, registry, _ = audit_plan_and_coverage(results_dir, table, config, context, audit)
        fold_by_uid, skipped = load_fold_registry(results_dir, table, plan, audit)
        models = verify_models(plan, config, context["bundle"], registry, audit)
        unique_plan = plan.drop_duplicates("candidate_key").set_index("candidate_key").to_dict(orient="index")
        metadata_lookup = table.set_index("__id")[["__role", "__label", "__source", "__group"]].to_dict(orient="index")

        source_metrics = load_metrics(results_dir / "development_source_holdout_metrics_by_source_pair.csv", "main", audit)
        group_metrics = load_metrics(results_dir / "development_group_cv_metrics_by_source_pair.csv", "main", audit)
        source_macro: dict[Any, Any] = {}
        group_macro: dict[Any, Any] = {}
        source_cells, source_hashes, source_rows, source_points = process_prediction_file(
            results_dir / "development_source_holdout_predictions.csv", source_metrics,
            fold_by_uid, unique_plan, metadata_lookup, models, audit, args.metric_tolerance,
            False, source_macro,
        )
        group_cells, group_hashes, group_rows, group_points = process_prediction_file(
            results_dir / "development_group_cv_predictions.csv", group_metrics,
            fold_by_uid, unique_plan, metadata_lookup, models, audit, args.metric_tolerance,
            False, group_macro,
        )
        main_cells = source_cells | group_cells
        main_hashes = {**source_hashes, **group_hashes}
        candidate_sets = {
            cohort: set(rows["candidate_key"])
            for cohort, rows in plan.drop_duplicates("candidate_key").groupby("cohort_id")
        }
        expected_main = {
            (uid, candidate, "main")
            for uid, fold in fold_by_uid.items()
            if (uid, str(fold["quantity"])) not in skipped
            for candidate in candidate_sets[fold["cohort_id"]]
        }
        audit.check("main_prediction_grid_complete", main_cells == expected_main,
                    f"observed={len(main_cells)} expected={len(expected_main)}")
        audit.check("main_candidate_quantity_grid_63x5", len({(unique_plan[cell[1]]["combination"], str(fold_by_uid[cell[0]]["quantity"])) for cell in main_cells}) == 315)
        verify_matched_arm_hashes(plan, fold_by_uid, main_hashes, audit)
        source_summary_rows = verify_source_summary(
            results_dir / "development_source_holdout_summary.csv", source_macro, audit, args.metric_tolerance
        )
        group_summary_rows = verify_source_summary(
            results_dir / "development_group_cv_summary.csv", group_macro, audit, args.metric_tolerance
        )
        source_delta_rows = verify_matched_deltas(
            results_dir / "development_source_holdout_matched_deltas.csv", source_metrics, plan, audit, args.metric_tolerance
        )
        group_delta_rows = verify_matched_deltas(
            results_dir / "development_group_cv_matched_deltas.csv", group_metrics, plan, audit, args.metric_tolerance
        )

        diagnostic_source_metrics = load_metrics(results_dir / "development_missingness_source_holdout_metrics.csv", "", audit)
        diagnostic_group_metrics = load_metrics(results_dir / "development_missingness_group_cv_metrics.csv", "", audit)
        ds_cells, ds_hashes, ds_rows, ds_points = process_prediction_file(
            results_dir / "development_missingness_source_holdout_predictions.csv",
            diagnostic_source_metrics, fold_by_uid, unique_plan, metadata_lookup, models,
            audit, args.metric_tolerance, True, None,
        )
        dg_cells, dg_hashes, dg_rows, dg_points = process_prediction_file(
            results_dir / "development_missingness_group_cv_predictions.csv",
            diagnostic_group_metrics, fold_by_uid, unique_plan, metadata_lookup, models,
            audit, args.metric_tolerance, True, None,
        )
        diagnostic_cells = ds_cells | dg_cells
        expected_diagnostic = {
            (uid, candidate, mode)
            for uid, fold in fold_by_uid.items()
            if str(fold["quantity"]) == "all" and (uid, "all") not in skipped
            for candidate in candidate_sets[fold["cohort_id"]]
            for mode in ("median_only", "missingness_only")
        }
        audit.check("diagnostic_grid_complete_all_only", diagnostic_cells == expected_diagnostic,
                    f"observed={len(diagnostic_cells)} expected={len(expected_diagnostic)}")
        for (uid, candidate, mode), digest in {**ds_hashes, **dg_hashes}.items():
            audit.check(f"diagnostic_test_ids_match_main_{uid}_{candidate}_{mode}", digest == main_hashes[(uid, candidate, "main")])

        audit.counts.update({
            "main_prediction_cells": len(main_cells),
            "main_prediction_rows": source_rows + group_rows,
            "main_point_metrics_recomputed": source_points + group_points,
            "source_prediction_rows": source_rows,
            "group_prediction_rows": group_rows,
            "source_point_metrics": source_points,
            "group_point_metrics": group_points,
            "source_summary_rows_recomputed": source_summary_rows,
            "group_summary_rows_recomputed": group_summary_rows,
            "source_matched_delta_rows_checked": source_delta_rows,
            "group_matched_delta_rows_checked": group_delta_rows,
            "diagnostic_prediction_cells": len(diagnostic_cells),
            "diagnostic_prediction_rows": ds_rows + dg_rows,
            "diagnostic_point_metrics_recomputed": ds_points + dg_points,
        })
        audit.warn("Source-held-out development results are development evidence, not a historical test.")
        audit.warn("Ordinary group CV is descriptive and must remain separate from source-held-out transfer evidence.")
        audit.warn("There are only six human and four generator source groups; source-macro uncertainty must use source-level resampling.")
        audit.warn("The 315 candidate-by-quantity cells and 45 incremental comparisons are highly dependent; point estimates are not multiplicity-adjusted discoveries.")
        audit.warn("F and H are direct original-mix DSP representations; classifier association does not establish scalar measurement or causal validity.")
        audit.warn("Missingness-only and median-only outputs are sensitivity diagnostics and must never select a candidate.")
        audit.warn("V failed its external microstructure gate and T whole-clip evidence is weak; neither is present in this F/H run.")
        audit.warn("Synthetic tests are sensitivity evidence only and cannot admit a phenomenon family without external natural-label validation.")
    except (AuditFailure, KeyError, ValueError, OSError, json.JSONDecodeError) as error:
        if not audit.errors or str(error) not in audit.errors[-1]:
            audit.errors.append(f"{type(error).__name__}: {error}")
    return audit.report()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--evaluator-path", type=Path)
    parser.add_argument("--base-evaluator-path", type=Path)
    parser.add_argument("--families-json", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--feature-path", action="append", type=Path)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--frozen-dev-bundle", type=Path)
    parser.add_argument("--metric-tolerance", type=float, default=1e-10)
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_audit(args)
    if args.compact:
        report = {key: value for key, value in report.items() if key != "checks"}
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
