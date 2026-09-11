#!/usr/bin/env python3
"""Independent, read-only audit of the frozen schema-v2 10-second SDRFH run.

The auditor refuses to inspect partial outputs: ``run_manifest.json`` must exist
before any plan, prediction, metric, or model file is opened.  It never fits a
model.  Frozen parameters come from the hash-verified preregistration contract;
candidate and fold expectations are derived from the configured family states
and reconstructed eligible source/group cohort rather than from 30-second
constants.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import audit_fh_results as core


FAMILY_ORDER = ("S", "D", "R", "P", "F", "H", "M", "V", "B", "A", "T")
OLD_FAMILY_ORDER = ("S", "D", "R", "P")
NEW_FAMILY_ORDER = ("F", "H", "M", "V", "B", "A", "T")
EXPECTED_ACTIVE_OLD = ("S", "D", "R")
EXPECTED_PLANNED_OLD = ("P",)
EXPECTED_ACTIVE_NEW = ("F", "H")
EXPECTED_PLANNED_NEW = ("M", "V", "B", "A", "T")
EXPECTED_SCHEMA_VERSION = 2
DISALLOWED_ROLES = {"locked", "pilot", "provisional", "provisional_development", "stress"}
METRIC_NAMES = core.METRIC_NAMES


Audit = core.Audit
AuditFailure = core.AuditFailure
canonical_hash = core.canonical_hash
candidate_key = core.candidate_key
file_sha256 = core.file_sha256
id_set_hash = core.id_set_hash
apply_filters = core.apply_filters


def ordered_combinations(codes: Sequence[str]) -> set[str]:
    selected = set(codes)
    ordered = [code for code in FAMILY_ORDER if code in selected]
    return {
        "+".join(combo)
        for size in range(1, len(ordered) + 1)
        for combo in itertools.combinations(ordered, size)
    }


def active_and_planned(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    old = config["old_families"]
    new = config["new_families"]
    return {
        "active_old": tuple(code for code in OLD_FAMILY_ORDER if old[code]["state"] == "available"),
        "planned_old": tuple(code for code in OLD_FAMILY_ORDER if old[code]["state"] == "planned"),
        "active_new": tuple(code for code in NEW_FAMILY_ORDER if new[code]["state"] == "available"),
        "planned_new": tuple(code for code in NEW_FAMILY_ORDER if new[code]["state"] == "planned"),
    }


def lattice_expectations(config: Mapping[str, Any]) -> dict[str, Any]:
    states = active_and_planned(config)
    old = ordered_combinations(states["active_old"])
    new = ordered_combinations(states["active_new"])
    all_candidates = ordered_combinations(states["active_old"] + states["active_new"])
    comparisons = len(old) * len(new)
    return {
        **states,
        "old_combinations": old,
        "new_combinations": new,
        "all_candidates": all_candidates,
        "incremental_comparisons": comparisons,
        "plan_rows": len(old) + len(new) + 2 * comparisons,
    }


def expected_fold_counts(
    human_sources: int, ai_sources: int, group_folds: int, quantities: Sequence[Any]
) -> dict[str, int]:
    quantity_count = len(quantities)
    return {
        "human_source_holdout": human_sources * group_folds * quantity_count,
        "generator_holdout": ai_sources * group_folds * quantity_count,
        "ordinary_group_holdout_descriptive": group_folds * quantity_count,
        "per_quantity": (human_sources + ai_sources + 1) * group_folds,
        "total": (human_sources + ai_sources + 1) * group_folds * quantity_count,
    }


def _artifact_root(results_dir: Path) -> Path:
    return results_dir.resolve().parents[1]


def _first_existing(candidates: Iterable[Path]) -> Path:
    paths = [Path(path).resolve() for path in candidates]
    return next((path for path in paths if path.is_file()), paths[0])


def _matching_hash(candidates: Iterable[Path], expected: str) -> Path:
    paths = [Path(path).resolve() for path in candidates]
    for path in paths:
        if path.is_file() and file_sha256(path) == expected:
            return path
    return _first_existing(paths)


def resolve_record_path(
    results_dir: Path, record: Mapping[str, Any], override: Path | None
) -> Path:
    if override is not None:
        return override.resolve()
    recorded = Path(str(record["path"]))
    root = _artifact_root(results_dir)
    return _first_existing((
        recorded,
        root / "evaluation_inputs" / "sdrfh_10s_v2" / recorded.name,
        results_dir / recorded.name,
    ))


def verify_record(
    audit: Audit, name: str, results_dir: Path, record: Mapping[str, Any], override: Path | None
) -> Path:
    path = resolve_record_path(results_dir, record, override)
    audit.check(f"{name}_exists", path.is_file(), str(path))
    actual = file_sha256(path)
    audit.check(f"{name}_sha256", actual == record["sha256"], f"{actual} != {record['sha256']}")
    return path


def load_contract_and_bundle(args: argparse.Namespace, audit: Audit) -> dict[str, Any]:
    results = args.results_dir.resolve()
    manifest_path = results / "run_manifest.json"
    if not manifest_path.is_file():
        print(json.dumps({
            "status": "not_ready",
            "reason": "run_manifest.json absent; partial outputs were not inspected",
            "results_dir": str(results),
        }, indent=2))
        raise SystemExit(3)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit.check("manifest_schema_v2", manifest.get("schema_version") == EXPECTED_SCHEMA_VERSION)
    audit.check("manifest_stage_dev", manifest.get("stage") == "dev", repr(manifest.get("stage")))
    audit.check(
        "manifest_interpretation_dev_only",
        manifest.get("interpretation")
        == "development_only_source_holdout_and_group_cv; not historical test",
    )
    authorization = manifest["authorization"]
    contract = manifest["contract"]
    audit.check("contract_schema_v2", contract.get("schema_version") == EXPECTED_SCHEMA_VERSION)
    audit.check("contract_stage_dev", contract.get("authorized_stage") == "dev")
    audit.check("authorization_frozen_verified", authorization.get("status") == "frozen_verified")
    contract_hash = canonical_hash(contract)
    audit.check("authorization_contract_hash", authorization.get("contract_sha256") == contract_hash)
    audit.check(
        "source_group_global_group_contract",
        contract.get("source_contract")
        == {"source_column": "source_group", "global_group_column": "group_id"},
    )

    parameters = contract.get("parameters", {})
    expected_parameter_keys = {
        "seed", "quantities", "group_folds", "model", "threshold_policy",
        "development_role", "historical_roles", "min_train_groups_per_class",
        "min_test_groups_per_class",
    }
    audit.check("frozen_parameter_keys", set(parameters) == expected_parameter_keys, repr(parameters))
    audit.check("frozen_ridge_model", parameters.get("model") == "ridge")
    audit.check("frozen_fixed_threshold", parameters.get("threshold_policy") == "fixed_0.5")
    audit.check("frozen_development_role", parameters.get("development_role") == "development")
    audit.check("historical_roles_excluded", not ({str(x) for x in parameters.get("historical_roles", [])} & {"development"}))
    audit.check("group_folds_valid", int(parameters.get("group_folds", 0)) >= 2)
    quantities = parameters.get("quantities", [])
    audit.check("quantities_nonempty_unique", bool(quantities) and len({str(x) for x in quantities}) == len(quantities))
    audit.check("minimum_group_checks_positive", int(parameters.get("min_train_groups_per_class", 0)) > 0
                and int(parameters.get("min_test_groups_per_class", 0)) > 0)
    runtime = contract.get("runtime", {})
    audit.check("runtime_contract_complete", set(runtime) == {"python", "numpy", "pandas", "thread_settings"})
    audit.check(
        "runtime_thread_limits_frozen",
        runtime.get("thread_settings")
        == {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        repr(runtime.get("thread_settings")),
    )

    root = _artifact_root(results)
    evaluator_path = (
        args.evaluator_path.resolve()
        if args.evaluator_path
        else _matching_hash(
            (root / "code" / "evaluate_new_phenomena_v2.py",
             root / "code" / "evaluate_new_phenomena.py"),
            contract["evaluator_sha256"],
        )
    )
    audit.check("frozen_evaluator_exists", evaluator_path.is_file(), str(evaluator_path))
    audit.check("frozen_evaluator_hash", file_sha256(evaluator_path) == contract["evaluator_sha256"])
    base_path = (
        args.base_evaluator_path.resolve()
        if args.base_evaluator_path
        else _matching_hash(
            (evaluator_path.parent / "frozen_evaluate_expanded_20260905.py",
             root / "code" / "frozen_evaluate_expanded_20260905.py"),
            contract["preserved_base_evaluator_sha256"],
        )
    )
    audit.check("preserved_base_exists", base_path.is_file(), str(base_path))
    audit.check("preserved_base_hash", file_sha256(base_path) == contract["preserved_base_evaluator_sha256"])

    families_path = verify_record(audit, "families_json", results, contract["families_json"], args.families_json)
    metadata_path = verify_record(audit, "metadata", results, contract["metadata"], args.metadata)
    feature_overrides = args.feature_path or []
    audit.check(
        "feature_override_count",
        not feature_overrides or len(feature_overrides) == len(contract["features"]),
        f"overrides={len(feature_overrides)} expected={len(contract['features'])}",
    )
    feature_paths = [
        verify_record(
            audit, f"feature_{index}", results, record,
            feature_overrides[index] if feature_overrides else None,
        )
        for index, record in enumerate(contract["features"])
    ]

    receipt_recorded = Path(str(authorization["receipt_path"]))
    receipt_path = (
        args.preregistration.resolve()
        if args.preregistration
        else _first_existing((
            receipt_recorded,
            root / "preregistration" / "sdrfh_10s_v2" / receipt_recorded.name,
        ))
    )
    audit.check("preregistration_exists", receipt_path.is_file(), str(receipt_path))
    audit.check("preregistration_file_hash", file_sha256(receipt_path) == authorization["receipt_sha256"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    audit.check("preregistration_schema_v2", receipt.get("schema_version") == EXPECTED_SCHEMA_VERSION)
    audit.check("preregistration_status_frozen", receipt.get("status") == "frozen")
    audit.check("preregistration_stage_dev", receipt.get("authorized_stage") == "dev")
    audit.check("preregistration_contract_exact", receipt.get("contract") == contract)
    audit.check("preregistration_contract_hash", receipt.get("contract_sha256") == contract_hash)

    bundle_recorded = Path(str(manifest["frozen_dev_bundle"]))
    bundle_path = (
        args.frozen_dev_bundle.resolve()
        if args.frozen_dev_bundle
        else _first_existing((bundle_recorded, results / bundle_recorded.name))
    )
    audit.check("frozen_bundle_exists", bundle_path.is_file(), str(bundle_path))
    audit.check("frozen_bundle_file_hash", file_sha256(bundle_path) == manifest["frozen_dev_bundle_sha256"])
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    audit.check("bundle_schema_v2", bundle.get("schema_version") == EXPECTED_SCHEMA_VERSION)
    audit.check("bundle_status", bundle.get("status") == "frozen_development_models_for_historical_description")
    audit.check("bundle_has_no_selection", bundle.get("selection") is None)
    audit.check("bundle_contract_hash", bundle.get("contract_sha256") == contract_hash)
    audit.check("bundle_evaluator_hash", bundle.get("evaluator_sha256") == contract["evaluator_sha256"])
    audit.check("bundle_base_hash", bundle.get("preserved_base_evaluator_sha256") == contract["preserved_base_evaluator_sha256"])
    audit.check("bundle_source_contract", bundle.get("source_contract") == contract["source_contract"])
    audit.check("bundle_parameters", bundle.get("parameters") == parameters)
    audit.check("bundle_input_hashes", bundle.get("families_json_sha256") == contract["families_json"]["sha256"]
                and bundle.get("metadata_sha256") == contract["metadata"]["sha256"]
                and bundle.get("feature_hashes") == contract["features"])

    return {
        "manifest": manifest,
        "contract": contract,
        "parameters": parameters,
        "bundle": bundle,
        "bundle_path": bundle_path,
        "families_path": families_path,
        "metadata_path": metadata_path,
        "feature_paths": feature_paths,
        "evaluator_path": evaluator_path,
        "base_path": base_path,
    }


def load_inputs(context: Mapping[str, Any], audit: Audit) -> tuple[pd.DataFrame, dict[str, Any]]:
    return core.load_inputs(context, audit)


def _canonical_codes(text: str) -> tuple[str, ...]:
    codes = tuple(json.loads(text))
    if any(code not in FAMILY_ORDER for code in codes):
        raise AuditFailure(f"unknown family code in {text}")
    expected = tuple(code for code in FAMILY_ORDER if code in set(codes))
    if codes != expected or len(codes) != len(set(codes)):
        raise AuditFailure(f"family codes are not unique canonical order: {text}")
    return codes


def audit_plan_and_coverage(
    results_dir: Path,
    table: pd.DataFrame,
    config: Mapping[str, Any],
    context: Mapping[str, Any],
    audit: Audit,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    plan = pd.read_csv(results_dir / "evaluation_plan.csv", dtype=str, keep_default_na=False)
    registry = pd.read_csv(results_dir / "cohort_registry.csv", dtype=str, keep_default_na=False)
    lattice = lattice_expectations(config)
    audit.check("active_old_sdr", lattice["active_old"] == EXPECTED_ACTIVE_OLD, repr(lattice["active_old"]))
    audit.check("planned_old_p", lattice["planned_old"] == EXPECTED_PLANNED_OLD, repr(lattice["planned_old"]))
    audit.check("active_new_fh", lattice["active_new"] == EXPECTED_ACTIVE_NEW, repr(lattice["active_new"]))
    audit.check("planned_new_exact", lattice["planned_new"] == EXPECTED_PLANNED_NEW, repr(lattice["planned_new"]))
    audit.check("manifest_active_old", context["manifest"].get("active_old_families") == list(lattice["active_old"]))
    audit.check("manifest_unavailable_old", context["manifest"].get("unavailable_old_families") == list(lattice["planned_old"]))
    audit.check("manifest_active_new", context["manifest"].get("active_new_families") == list(lattice["active_new"]))
    audit.check("manifest_planned_new", context["manifest"].get("planned_new_families") == list(lattice["planned_new"]))
    audit.check("plan_ids_unique", plan["plan_id"].nunique() == len(plan))
    audit.check("plan_rows_derived", len(plan) == lattice["plan_rows"], f"observed={len(plan)} expected={lattice['plan_rows']}")
    audit.check("manifest_plan_rows", context["manifest"].get("plan_rows") == len(plan))
    audit.check("candidate_combinations_derived", set(plan["combination"]) == lattice["all_candidates"])
    audit.check("candidate_keys_derived", plan["candidate_key"].nunique() == len(lattice["all_candidates"]))
    audit.check("manifest_unique_candidates", context["manifest"].get("unique_candidates") == len(lattice["all_candidates"]))
    audit.check("standalone_old_rows", int((plan["plan_type"] == "old_baseline_standalone").sum()) == len(lattice["old_combinations"]))
    audit.check("standalone_new_rows", int((plan["plan_type"] == "new_standalone").sum()) == len(lattice["new_combinations"]))
    incremental = plan[plan["plan_type"] == "incremental_matched"]
    audit.check("incremental_rows", len(incremental) == 2 * lattice["incremental_comparisons"])
    audit.check("incremental_comparisons", incremental["comparison_id"].nunique() == lattice["incremental_comparisons"])
    audit.check("incremental_two_arms", bool((incremental.groupby("comparison_id").size() == 2).all()))
    audit.check("incremental_arm_names", set(incremental["arm"]) == {"baseline", "added"})

    for row in plan.to_dict(orient="records"):
        codes = _canonical_codes(row["family_codes"])
        eligibility_codes = _canonical_codes(row["eligibility_family_codes"])
        audit.check(f"plan_combination_{row['plan_id']}", row["combination"] == "+".join(codes))
        audit.check(f"candidate_key_{row['plan_id']}", row["candidate_key"] == candidate_key(row["cohort_id"], row["combination"]))
        if row["plan_type"] != "incremental_matched":
            audit.check(f"standalone_eligibility_{row['plan_id']}", codes == eligibility_codes)
    for comparison, arms in incremental.groupby("comparison_id"):
        baseline = arms[arms["arm"] == "baseline"].iloc[0]
        added = arms[arms["arm"] == "added"].iloc[0]
        baseline_codes = _canonical_codes(baseline["family_codes"])
        added_codes = _canonical_codes(added["family_codes"])
        eligibility_codes = _canonical_codes(added["eligibility_family_codes"])
        added_new = tuple(code for code in added_codes if code in lattice["active_new"])
        audit.check(f"incremental_old_arm_{comparison}", set(baseline_codes).issubset(lattice["active_old"]))
        audit.check(f"incremental_new_arm_{comparison}", bool(added_new) and set(added_new).issubset(lattice["active_new"]))
        audit.check(f"incremental_union_{comparison}", set(added_codes) == set(baseline_codes) | set(added_new))
        audit.check(f"incremental_eligibility_union_{comparison}", eligibility_codes == added_codes)
        audit.check(f"incremental_matched_contract_{comparison}",
                    baseline["cohort_id"] == added["cohort_id"]
                    and baseline["eligible_id_set_sha256"] == added["eligible_id_set_sha256"]
                    and baseline["eligibility_contract_sha256"] == added["eligibility_contract_sha256"]
                    and baseline["eligibility_family_codes"] == added["eligibility_family_codes"])

    bundle_plan = pd.DataFrame(context["bundle"]["evaluation_plan"])
    audit.check("bundle_plan_columns", set(bundle_plan.columns) == set(plan.columns))
    normalized_bundle = bundle_plan[plan.columns].fillna("").astype(str).sort_values("plan_id").to_dict(orient="records")
    normalized_csv = plan.fillna("").astype(str).sort_values("plan_id").to_dict(orient="records")
    audit.check("bundle_plan_matches_csv", normalized_bundle == normalized_csv)

    specs = {**config["old_families"], **config["new_families"]}
    base = table[(table["__role"] == context["parameters"]["development_role"])
                 & np.isfinite(table["__label"])].copy()
    cohort_tables: dict[str, pd.DataFrame] = {}
    for eligibility_text, rows in plan.groupby("eligibility_family_codes", sort=False):
        codes = _canonical_codes(eligibility_text)
        selected = apply_filters(base, config.get("eligibility", {}))
        for code in codes:
            selected = apply_filters(selected, specs[code].get("eligibility", {}))
        digest = id_set_hash(selected["__id"])
        expected_cohort_id = f"dev-{digest[:16]}"
        for row in rows.to_dict(orient="records"):
            audit.check(f"eligible_rows_{row['plan_id']}", int(row["eligible_rows"]) == len(selected))
            audit.check(f"eligible_hash_{row['plan_id']}", row["eligible_id_set_sha256"] == digest)
            audit.check(f"cohort_id_{row['plan_id']}", row["cohort_id"] == expected_cohort_id)
        if expected_cohort_id in cohort_tables:
            audit.check(f"cohort_identity_reused_{expected_cohort_id}",
                        id_set_hash(cohort_tables[expected_cohort_id]["__id"]) == digest)
        else:
            cohort_tables[expected_cohort_id] = selected

    audit.check("manifest_cohort_count", context["manifest"].get("cohorts") == len(cohort_tables))
    audit.check("registry_cohort_count", len(registry) == len(cohort_tables))
    registry_by_id = registry.set_index("cohort_id")
    audit.check("registry_cohort_ids", set(registry_by_id.index) == set(cohort_tables))
    for cohort_id, cohort in cohort_tables.items():
        row = registry_by_id.loc[cohort_id]
        human = cohort[cohort["__label"] == 0]
        ai = cohort[cohort["__label"] == 1]
        audit.check(f"registry_hash_{cohort_id}", row["eligible_id_set_sha256"] == id_set_hash(cohort["__id"]))
        audit.check(f"registry_rows_{cohort_id}", int(row["rows"]) == len(cohort))
        audit.check(f"registry_class_rows_{cohort_id}", int(row["human_rows"]) == len(human) and int(row["ai_rows"]) == len(ai))
        audit.check(f"registry_source_counts_{cohort_id}", int(row["human_sources"]) == human["__source"].nunique()
                    and int(row["ai_sources"]) == ai["__source"].nunique())
        audit.check(f"registry_group_counts_{cohort_id}", int(row["human_groups"]) == human["__group"].nunique()
                    and int(row["ai_groups"]) == ai["__group"].nunique())

    family_coverage = pd.read_csv(results_dir / "family_coverage_by_source.csv", dtype=str, keep_default_na=False)
    expected_family_rows = sum(cohort["__source"].nunique() * len(specs) for cohort in cohort_tables.values())
    audit.check("family_coverage_rows_derived", len(family_coverage) == expected_family_rows)
    family_lookup = {(row["cohort_id"], row["source_group"], row["family"]): row
                     for row in family_coverage.to_dict(orient="records")}
    audit.check("family_coverage_keys_unique", len(family_lookup) == len(family_coverage))
    for cohort_id, cohort in cohort_tables.items():
        for source, current in cohort.groupby("__source"):
            for family, spec in specs.items():
                row = family_lookup[(cohort_id, source, family)]
                columns = spec["columns"]
                finite = np.isfinite(current[columns].to_numpy(float)) if columns else np.zeros((len(current), 0), bool)
                any_observed = finite.any(axis=1) if columns else np.zeros(len(current), bool)
                complete = finite.all(axis=1) if columns else np.zeros(len(current), bool)
                qualified = spec["state"] == "available" and float(any_observed.mean()) >= float(spec["minimum_observed_fraction"])
                audit.check(f"family_coverage_{cohort_id}_{source}_{family}",
                            int(row["eligible_rows"]) == len(current)
                            and int(row["any_observed_rows"]) == int(any_observed.sum())
                            and int(row["complete_rows"]) == int(complete.sum())
                            and core.close(row["any_observed_fraction"], any_observed.mean(), 1e-12)
                            and core.close(row["complete_fraction"], complete.mean(), 1e-12)
                            and core.as_bool(row["coverage_qualified"]) == qualified)

    candidate_coverage = pd.read_csv(results_dir / "candidate_coverage.csv", dtype=str, keep_default_na=False)
    expected_candidate_rows = sum(
        plan.loc[plan["cohort_id"] == cohort_id, "candidate_key"].nunique()
        for cohort_id in cohort_tables
    )
    audit.check("candidate_coverage_rows_derived", len(candidate_coverage) == expected_candidate_rows)
    audit.check("candidate_coverage_keys_unique", candidate_coverage[["cohort_id", "candidate_key"]].drop_duplicates().shape[0] == len(candidate_coverage))
    for row in candidate_coverage.to_dict(orient="records"):
        cohort = cohort_tables[row["cohort_id"]]
        codes = tuple(row["combination"].split("+"))
        columns = [column for code in codes for column in specs[code]["columns"]]
        finite = np.isfinite(cohort[columns].to_numpy(float))
        any_observed, complete = finite.any(axis=1), finite.all(axis=1)
        qualified = all(
            specs[code]["state"] == "available"
            and float(np.isfinite(cohort[specs[code]["columns"]].to_numpy(float)).any(axis=1).mean())
            >= float(specs[code]["minimum_observed_fraction"])
            for code in codes
        )
        audit.check(f"candidate_coverage_{row['cohort_id']}_{row['combination']}",
                    row["candidate_key"] == candidate_key(row["cohort_id"], row["combination"])
                    and int(row["eligible_rows"]) == len(cohort)
                    and int(row["feature_count"]) == len(columns)
                    and int(row["rows_with_any_observed_feature"]) == int(any_observed.sum())
                    and int(row["rows_complete_all_selected_features"]) == int(complete.sum())
                    and core.close(row["any_observed_fraction"], any_observed.mean(), 1e-12)
                    and core.close(row["complete_case_fraction_diagnostic_only"], complete.mean(), 1e-12)
                    and core.as_bool(row["coverage_qualified"]) == qualified)

    audit.counts.update({
        "active_old_families": list(lattice["active_old"]),
        "planned_old_families": list(lattice["planned_old"]),
        "active_new_families": list(lattice["active_new"]),
        "planned_new_families": list(lattice["planned_new"]),
        "plan_rows": len(plan),
        "unique_candidates": len(lattice["all_candidates"]),
        "incremental_comparisons": lattice["incremental_comparisons"],
        "cohorts": len(cohort_tables),
        "cohort_source_counts": {
            cohort_id: {
                "rows": len(cohort),
                "human_rows": int((cohort["__label"] == 0).sum()),
                "ai_rows": int((cohort["__label"] == 1).sum()),
                "human_sources": int(cohort.loc[cohort["__label"] == 0, "__source"].nunique()),
                "ai_sources": int(cohort.loc[cohort["__label"] == 1, "__source"].nunique()),
                "human_groups": int(cohort.loc[cohort["__label"] == 0, "__group"].nunique()),
                "ai_groups": int(cohort.loc[cohort["__label"] == 1, "__group"].nunique()),
            }
            for cohort_id, cohort in cohort_tables.items()
        },
    })
    return plan, registry, cohort_tables, lattice


def hash_fold(value: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def reconstruct_base_folds(
    cohort_id: str, table: pd.DataFrame, group_folds: int, seed: int
) -> dict[tuple[str, str, int], dict[str, Any]]:
    assignment = table["__group"].map(lambda value: hash_fold(str(value), group_folds, seed))
    humans = sorted(table.loc[table["__label"] == 0, "__source"].unique())
    ais = sorted(table.loc[table["__label"] == 1, "__source"].unique())
    output: dict[tuple[str, str, int], dict[str, Any]] = {}
    fold_index = 0
    for held_label, sources, opposite_label, fold_type in (
        (0, humans, 1, "human_source_holdout"),
        (1, ais, 0, "generator_holdout"),
    ):
        for held_source in sources:
            for inner in range(group_folds):
                held = (table["__label"] == held_label) & (table["__source"] == held_source) & (assignment == inner)
                opposite = (table["__label"] == opposite_label) & (assignment == inner)
                test = table[held | opposite].copy()
                test_groups = set(test["__group"])
                train = table[
                    ~((table["__label"] == held_label) & (table["__source"] == held_source))
                    & (assignment != inner)
                    & ~table["__group"].isin(test_groups)
                ].copy()
                if train["__label"].nunique() != 2 or test["__label"].nunique() != 2:
                    continue
                output[(fold_type, str(held_source), inner)] = {
                    "cohort_id": cohort_id,
                    "fold_index": fold_index,
                    "train_ids": set(train["__id"].astype(str)),
                    "test_ids": set(test["__id"].astype(str)),
                }
                fold_index += 1
    for inner in range(group_folds):
        test = table[assignment == inner].copy()
        train = table[assignment != inner].copy()
        if train["__label"].nunique() != 2 or test["__label"].nunique() != 2:
            continue
        output[("ordinary_group_holdout_descriptive", "__all_sources__", inner)] = {
            "cohort_id": cohort_id,
            "fold_index": fold_index,
            "train_ids": set(train["__id"].astype(str)),
            "test_ids": set(test["__id"].astype(str)),
        }
        fold_index += 1
    return output


def load_fold_registry(
    results_dir: Path,
    metadata_table: pd.DataFrame,
    cohort_tables: Mapping[str, pd.DataFrame],
    parameters: Mapping[str, Any],
    audit: Audit,
) -> tuple[dict[str, dict[str, Any]], set[tuple[str, str]]]:
    records = [
        json.loads(line)
        for line in (results_dir / "fold_registry.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    group_folds = int(parameters["group_folds"])
    seed = int(parameters["seed"])
    quantities = list(parameters["quantities"])
    quantity_strings = {str(value) for value in quantities}
    base_folds = {
        cohort_id: reconstruct_base_folds(cohort_id, cohort, group_folds, seed)
        for cohort_id, cohort in cohort_tables.items()
    }
    expected_cells = {
        (cohort_id, fold_type, heldout, inner, str(quantity))
        for cohort_id, folds in base_folds.items()
        for fold_type, heldout, inner in folds
        for quantity in quantities
    }
    observed_cells: set[tuple[str, str, str, int, str]] = set()
    by_uid: dict[str, dict[str, Any]] = {}
    minimum_train_ok: dict[str, bool] = {}
    metadata_lookup = metadata_table.set_index("__id")[["__role", "__label", "__source", "__group"]].to_dict(orient="index")
    type_counts: dict[str, int] = defaultdict(int)
    for record in records:
        uid = str(record["fold_uid"])
        audit.check(f"fold_uid_unique_{uid}", uid not in by_uid)
        by_uid[uid] = record
        quantity = str(record["quantity"])
        fold_type = str(record["fold_type"])
        heldout = str(record["heldout_source"])
        inner = int(record["opposite_group_fold"])
        cohort_id = str(record["cohort_id"])
        cell = (cohort_id, fold_type, heldout, inner, quantity)
        audit.check(f"fold_cell_unique_{uid}", cell not in observed_cells)
        observed_cells.add(cell)
        audit.check(f"fold_quantity_{uid}", quantity in quantity_strings)
        audit.check(f"fold_cohort_{uid}", cohort_id in cohort_tables)
        identity = (fold_type, heldout, inner)
        audit.check(f"fold_identity_{uid}", identity in base_folds[cohort_id], repr(identity))
        expected = base_folds[cohort_id][identity]
        audit.check(f"fold_index_{uid}", int(record["fold_index"]) == expected["fold_index"])
        train_ids = list(map(str, record["train_ids"]))
        test_ids = list(map(str, record["test_ids"]))
        train_set, test_set = set(train_ids), set(test_ids)
        audit.check(f"train_ids_unique_{uid}", len(train_ids) == len(train_set))
        audit.check(f"test_ids_unique_{uid}", len(test_ids) == len(test_set))
        audit.check(f"train_test_ids_disjoint_{uid}", not (train_set & test_set))
        audit.check(f"test_ids_exact_{uid}", test_set == expected["test_ids"])
        audit.check(f"train_ids_within_base_fold_{uid}", train_set.issubset(expected["train_ids"]))
        if quantity == "all":
            audit.check(f"all_quantity_train_ids_exact_{uid}", train_set == expected["train_ids"])
        audit.check(f"train_hash_{uid}", id_set_hash(train_ids) == record["train_id_set_sha256"])
        audit.check(f"test_hash_{uid}", id_set_hash(test_ids) == record["test_id_set_sha256"])
        audit.check(f"fold_ids_known_{uid}", all(item in metadata_lookup for item in train_set | test_set))
        roles = {metadata_lookup[item]["__role"] for item in train_set | test_set}
        audit.check(f"development_role_only_{uid}", roles == {parameters["development_role"]}, repr(roles))
        audit.check(f"no_disallowed_roles_{uid}", not (roles & DISALLOWED_ROLES), repr(roles))
        train_groups = set(map(str, record["train_group_ids"]))
        test_groups = set(map(str, record["test_group_ids"]))
        audit.check(f"train_test_groups_disjoint_{uid}", not (train_groups & test_groups))
        audit.check(f"train_groups_match_ids_{uid}", train_groups == {metadata_lookup[item]["__group"] for item in train_set})
        audit.check(f"test_groups_match_ids_{uid}", test_groups == {metadata_lookup[item]["__group"] for item in test_set})
        train_class_groups = defaultdict(set)
        for item in train_set:
            train_class_groups[int(metadata_lookup[item]["__label"])].add(metadata_lookup[item]["__group"])
        minimum_train_ok[uid] = all(
            len(train_class_groups[label]) >= int(parameters["min_train_groups_per_class"])
            for label in (0, 1)
        )
        if quantity != "all":
            cap = int(quantity)
            per_source_groups: dict[str, set[str]] = defaultdict(set)
            for item in train_set:
                per_source_groups[metadata_lookup[item]["__source"]].add(metadata_lookup[item]["__group"])
            audit.check(f"quantity_cap_{uid}", all(len(groups) <= cap for groups in per_source_groups.values()))
        quantity_value: Any = record["quantity"]
        expected_uid = canonical_hash({
            "cohort_id": cohort_id,
            "fold_index": int(record["fold_index"]),
            "fold_type": fold_type,
            "quantity": quantity_value,
            "seed": seed,
            "train_ids": sorted(train_ids),
            "test_ids": sorted(test_ids),
        })[:24]
        audit.check(f"fold_uid_hash_{uid}", uid == expected_uid)
        type_counts[fold_type] += 1

    audit.check("fold_registry_cells_exact", observed_cells == expected_cells,
                f"observed={len(observed_cells)} expected={len(expected_cells)}")
    audit.check("fold_registry_rows_exact", len(records) == len(expected_cells))
    derived_by_type = defaultdict(int)
    for _, fold_type, _, _, _ in expected_cells:
        derived_by_type[fold_type] += 1
    for fold_type, count in derived_by_type.items():
        audit.check(f"fold_type_count_{fold_type}", type_counts[fold_type] == count,
                    f"observed={type_counts[fold_type]} expected={count}")

    skipped: set[tuple[str, str]] = set()
    skipped_path = results_dir / "skipped_folds.csv"
    if skipped_path.stat().st_size > 1:
        with skipped_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                matches = [record for record in records if record["cohort_id"] == row["cohort_id"]
                           and str(record["fold_index"]) == str(row["fold_index"])
                           and record["fold_type"] == row["fold_type"]
                           and str(record["quantity"]) == str(row["quantity"])]
                audit.check("skipped_fold_resolves_unique", len(matches) == 1)
                skipped.add((str(matches[0]["fold_uid"]), str(row["quantity"])))
    expected_skipped = {
        (uid, str(record["quantity"]))
        for uid, record in by_uid.items()
        if not minimum_train_ok[uid]
    }
    audit.check(
        "minimum_train_group_failures_match_skips",
        skipped == expected_skipped,
        f"saved={len(skipped)} independently_expected={len(expected_skipped)}",
    )

    per_quantity = len(base_folds[next(iter(base_folds))]) if len(base_folds) == 1 else None
    audit.counts.update({
        "fold_registry_rows": len(records),
        "folds_per_quantity": per_quantity,
        "fold_type_quantity_cells": dict(type_counts),
        "skipped_fold_quantity_cells": len(skipped),
    })
    return by_uid, skipped


def verify_models(
    plan: pd.DataFrame,
    config: Mapping[str, Any],
    bundle: Mapping[str, Any],
    registry: pd.DataFrame,
    parameters: Mapping[str, Any],
    audit: Audit,
) -> dict[str, Mapping[str, Any]]:
    models = bundle["models"]
    unique_plan = plan.drop_duplicates("candidate_key").set_index("candidate_key")
    audit.check("frozen_model_count_derived", len(models) == len(unique_plan))
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
        audit.check(f"model_type_{key}", record["model"].get("model_type") == "weighted_ridge_linear_probability")
        audit.check(f"model_link_{key}", record["model"].get("prediction_link") == "identity")
        audit.check(f"model_threshold_{key}", float(record["threshold"]) == 0.5)
        audit.check(f"model_hash_{key}", canonical_hash({"model": record["model"], "threshold": record["threshold"]}) == record["model_sha256"])
        audit.check(f"model_contract_{key}", training["model"] == parameters["model"]
                    and training["threshold_policy"] == parameters["threshold_policy"]
                    and int(training["seed"]) == int(parameters["seed"]))
    return models


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    audit = Audit()
    try:
        context = load_contract_and_bundle(args, audit)
        parameters = context["parameters"]
        core.EXPECTED_PARAMETERS = dict(parameters)
        core.DISALLOWED_ROLES = set(DISALLOWED_ROLES)
        results_dir = args.results_dir.resolve()
        table, config = load_inputs(context, audit)
        plan, registry, cohort_tables, lattice = audit_plan_and_coverage(
            results_dir, table, config, context, audit
        )
        fold_by_uid, skipped = load_fold_registry(
            results_dir, table, cohort_tables, parameters, audit
        )
        models = verify_models(plan, config, context["bundle"], registry, parameters, audit)
        unique_plan = plan.drop_duplicates("candidate_key").set_index("candidate_key").to_dict(orient="index")
        metadata_lookup = table.set_index("__id")[["__role", "__label", "__source", "__group"]].to_dict(orient="index")

        source_metrics = core.load_metrics(results_dir / "development_source_holdout_metrics_by_source_pair.csv", "main", audit)
        group_metrics = core.load_metrics(results_dir / "development_group_cv_metrics_by_source_pair.csv", "main", audit)
        source_macro: dict[Any, Any] = {}
        group_macro: dict[Any, Any] = {}
        source_cells, source_hashes, source_rows, source_points = core.process_prediction_file(
            results_dir / "development_source_holdout_predictions.csv", source_metrics,
            fold_by_uid, unique_plan, metadata_lookup, models, audit, args.metric_tolerance,
            False, source_macro,
        )
        group_cells, group_hashes, group_rows, group_points = core.process_prediction_file(
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
        observed_candidate_quantities = {
            (unique_plan[cell[1]]["combination"], str(fold_by_uid[cell[0]]["quantity"]))
            for cell in main_cells
        }
        expected_candidate_quantities = {
            (combination, str(quantity))
            for combination in lattice["all_candidates"]
            for quantity in parameters["quantities"]
        }
        audit.check("candidate_quantity_grid_derived", observed_candidate_quantities == expected_candidate_quantities,
                    f"observed={len(observed_candidate_quantities)} expected={len(expected_candidate_quantities)}")
        core.verify_matched_arm_hashes(plan, fold_by_uid, main_hashes, audit)
        source_summary_rows = core.verify_source_summary(
            results_dir / "development_source_holdout_summary.csv", source_macro, audit, args.metric_tolerance
        )
        group_summary_rows = core.verify_source_summary(
            results_dir / "development_group_cv_summary.csv", group_macro, audit, args.metric_tolerance
        )
        source_delta_rows = core.verify_matched_deltas(
            results_dir / "development_source_holdout_matched_deltas.csv", source_metrics, plan, audit, args.metric_tolerance
        )
        group_delta_rows = core.verify_matched_deltas(
            results_dir / "development_group_cv_matched_deltas.csv", group_metrics, plan, audit, args.metric_tolerance
        )

        diagnostic_source_metrics = core.load_metrics(results_dir / "development_missingness_source_holdout_metrics.csv", "", audit)
        diagnostic_group_metrics = core.load_metrics(results_dir / "development_missingness_group_cv_metrics.csv", "", audit)
        ds_cells, ds_hashes, ds_rows, ds_points = core.process_prediction_file(
            results_dir / "development_missingness_source_holdout_predictions.csv",
            diagnostic_source_metrics, fold_by_uid, unique_plan, metadata_lookup, models,
            audit, args.metric_tolerance, True, None,
        )
        dg_cells, dg_hashes, dg_rows, dg_points = core.process_prediction_file(
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
            audit.check(f"diagnostic_test_ids_match_main_{uid}_{candidate}_{mode}",
                        digest == main_hashes[(uid, candidate, "main")])

        audit.counts.update({
            "candidate_quantity_cells": len(expected_candidate_quantities),
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
        audit.warn("This 10-second run is development-only evidence; historical locked, stress, pilot and provisional roles do not enter fitting or scoring here.")
        audit.warn("The 10-second and 30-second source compositions differ, so score differences do not identify a causal duration effect.")
        audit.warn("Ordinary group CV is descriptive and remains separate from source-held-out transfer evidence.")
        audit.warn("The candidate-by-quantity grid and incremental comparisons are dependent and are not multiplicity-adjusted discoveries.")
        audit.warn("F/H are direct original-mix DSP representations; classifier association is not scalar measurement validity or causal evidence.")
        audit.warn("Missingness-only and median-only models are diagnostics and must never select a candidate.")
        audit.warn("P is unavailable at 10 seconds and is correctly absent from the SDRFH lattice.")
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
