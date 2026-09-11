#!/usr/bin/env python3
"""Schema v3 evaluator with an explicit 60-second global-group-only mode.

The default mode preserves schema-v2 source-holdout plus descriptive group-CV
behavior.  ``group_cv_only`` is intentionally restricted to the exact 60-second
pure-F/H/M contract.  It uses global ``group_id`` folds, performs no source
holdout, records why generator holdout is ineligible when Suno is the only
development AI source, and leaves source-transfer J undefined.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import evaluate_new_phenomena_v2 as V2


SCHEMA_VERSION = 3
EVALUATION_MODES = ("source_holdout_and_group_cv", "group_cv_only")
OLD_FAMILY_ORDER = V2.OLD_FAMILY_ORDER
NEW_FAMILY_ORDER = V2.NEW_FAMILY_ORDER
FAMILY_ORDER = V2.FAMILY_ORDER
DISALLOWED_EVALUATION_ROLES = V2.DISALLOWED_EVALUATION_ROLES
BASE = V2.BASE


sha256_file = V2.sha256_file
canonical_hash = V2.canonical_hash
id_set_hash = V2.id_set_hash
parse_quantities = V2.parse_quantities
active_old_families = V2.active_old_families
active_new_families = V2.active_new_families
build_evaluation_plan = V2.build_evaluation_plan
validate_feature_columns = V2.validate_feature_columns
load_table = V2.load_table


def load_family_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"families JSON must declare schema_version={SCHEMA_VERSION}")
    old_raw = payload.get("old_families")
    new_raw = payload.get("new_families")
    if not isinstance(old_raw, dict) or set(old_raw) != set(OLD_FAMILY_ORDER):
        raise ValueError(f"old_families must define exactly {OLD_FAMILY_ORDER}")
    if not isinstance(new_raw, dict) or set(new_raw) != set(NEW_FAMILY_ORDER):
        raise ValueError("new_families must explicitly define F,H,M,V,B,A,T")
    old = {
        code: V2._validate_family_spec(code, old_raw[code], new=False)
        for code in OLD_FAMILY_ORDER
    }
    new = {
        code: V2._validate_family_spec(code, new_raw[code], new=True)
        for code in NEW_FAMILY_ORDER
    }
    if not any(new[code]["state"] == "available" for code in NEW_FAMILY_ORDER):
        raise ValueError("At least one new family must be available")
    flattened = [column for code in FAMILY_ORDER for column in (old | new)[code]["columns"]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Feature columns must not overlap across phenomenon families")
    forbidden = sorted(set(flattened) & V2.KNOWN_NON_PREDICTOR_COLUMNS)
    if forbidden:
        raise ValueError(f"Quality/status/coverage metadata cannot be predictors: {forbidden}")
    eligibility = payload.get("eligibility", {})
    status_columns = payload.get("status_columns", [])
    if not isinstance(eligibility, dict):
        raise ValueError("Top-level eligibility must be an object")
    if not isinstance(status_columns, list) or any(not isinstance(value, str) for value in status_columns):
        raise ValueError("Top-level status_columns must be a list")
    return {
        "schema_version": SCHEMA_VERSION,
        "cohort": str(payload.get("cohort", "configured cohort")),
        "eligibility": eligibility,
        "status_columns": status_columns,
        "old_families": old,
        "new_families": new,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("preregistration-draft", "dev", "historical-descriptive"),
        required=True,
    )
    parser.add_argument("--contract-target", choices=("dev", "historical-descriptive"), default="dev")
    parser.add_argument("--evaluation-mode", choices=EVALUATION_MODES,
                        default="source_holdout_and_group_cv")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, action="append", required=True)
    parser.add_argument("--families-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--frozen-dev-bundle", type=Path)
    parser.add_argument("--synthetic-test-only", action="store_true")
    parser.add_argument("--quantities", default=V2.DEFAULT_QUANTITIES)
    parser.add_argument("--group-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=V2.DEFAULT_SEED)
    parser.add_argument("--model", choices=("ridge", "logistic"), default="ridge")
    parser.add_argument("--threshold-policy", choices=("train_ba", "fixed_0.5"), default="train_ba")
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


def _metadata_duration_views(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "duration_view" not in reader.fieldnames:
            raise ValueError("group_cv_only metadata must contain duration_view")
        return {row["duration_view"].strip() for row in reader}


def validate_group_cv_only_contract(
    args: argparse.Namespace, config: dict[str, Any], target: str,
) -> None:
    if target != "dev" or args.stage == "historical-descriptive":
        raise ValueError("group_cv_only is restricted to development evaluation")
    if active_old_families(config):
        raise ValueError("60-second group_cv_only must not expose old-family baselines")
    if active_new_families(config) != ("F", "H", "M"):
        raise ValueError("60-second group_cv_only requires exactly F/H/M available")
    if _metadata_duration_views(args.metadata) != {"60s"}:
        raise ValueError("group_cv_only is restricted to exact duration_view=60s metadata")


def build_contract(
    args: argparse.Namespace, target: str, config: dict[str, Any],
) -> dict[str, Any]:
    if args.evaluation_mode == "group_cv_only":
        validate_group_cv_only_contract(args, config, target)
    contract = V2.build_contract(args, target)
    contract["schema_version"] = SCHEMA_VERSION
    contract["evaluator_sha256"] = sha256_file(Path(__file__).resolve())
    contract["schema_v2_dependency_sha256"] = sha256_file(Path(V2.__file__).resolve())
    contract["parameters"]["evaluation_mode"] = args.evaluation_mode
    return contract


def verify_authorization(args: argparse.Namespace, contract: dict[str, Any]) -> dict[str, Any]:
    if args.synthetic_test_only:
        return {"status": "synthetic_test_only", "contract_sha256": canonical_hash(contract)}
    if args.preregistration is None:
        raise ValueError("Real-corpus scoring is disabled until a frozen matching receipt is supplied")
    receipt = json.loads(args.preregistration.read_text(encoding="utf-8"))
    expected = canonical_hash(contract)
    if receipt.get("status") != "frozen" or receipt.get("contract_sha256") != expected:
        raise ValueError("Preregistration receipt is not frozen for this exact contract")
    if receipt.get("authorized_stage") != contract["authorized_stage"]:
        raise ValueError("Preregistration receipt authorizes a different stage")
    return {
        "status": "frozen_verified", "contract_sha256": expected,
        "receipt_path": str(args.preregistration.resolve()),
        "receipt_sha256": sha256_file(args.preregistration),
    }


def write_preregistration_draft(
    args: argparse.Namespace, config: dict[str, Any],
) -> None:
    contract = build_contract(args, args.contract_target, config)
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


def _check_group_cohort(table: pd.DataFrame, group_folds: int) -> None:
    if table["__label"].nunique() != 2:
        raise ValueError("Global group CV requires both classes")
    for label in (0.0, 1.0):
        if table.loc[table["__label"] == label, "__group"].nunique() < group_folds:
            raise ValueError(f"Class {int(label)} has fewer than {group_folds} global groups")


def materialize_group_plan(
    plan: pd.DataFrame, table: pd.DataFrame, config: dict[str, Any], role: str,
    group_folds: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    role_table = table[table["__role"] == role].copy()
    role_table = role_table[np.isfinite(role_table["__label"])].copy()
    cohorts: dict[str, pd.DataFrame] = {}
    registry: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for raw_codes, current in plan.groupby("eligibility_family_codes", sort=False):
        codes = json.loads(raw_codes)
        contract = V2.eligibility_contract(config, codes)
        eligible = V2.apply_contract(role_table, contract)
        _check_group_cohort(eligible, group_folds)
        digest = id_set_hash(eligible["__id"])
        cohort_id = f"dev-{digest[:16]}"
        cohorts.setdefault(cohort_id, eligible)
        contract_hash = canonical_hash(contract)
        registry.setdefault(cohort_id, {
            "cohort_id": cohort_id, "eligible_id_set_sha256": digest,
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
                "cohort_id": cohort_id, "eligibility_contract_sha256": contract_hash,
                "eligible_rows": int(len(eligible)), "eligible_id_set_sha256": digest,
            })
            record["candidate_key"] = V2.candidate_key(cohort_id, record["combination"])
            records.append(record)
    registry_rows = []
    for value in registry.values():
        output = dict(value)
        output["eligibility_contract_hashes"] = json.dumps(
            sorted(output["eligibility_contract_hashes"])
        )
        registry_rows.append(output)
    return pd.DataFrame(records), cohorts, pd.DataFrame(registry_rows)


def make_global_group_folds(
    table: pd.DataFrame, folds: int, seed: int,
) -> list[dict[str, Any]]:
    if folds < 2:
        raise ValueError("group-folds must be at least 2")
    assigned = table["__group"].map(lambda value: BASE.hash_fold(str(value), folds, seed))
    output: list[dict[str, Any]] = []
    for index in range(folds):
        test = table[assigned == index].copy()
        train = table[assigned != index].copy()
        if train["__label"].nunique() != 2 or test["__label"].nunique() != 2:
            raise ValueError(f"Global group fold {index} lacks a class")
        output.append({
            "fold_type": "ordinary_group_holdout_descriptive",
            "heldout_source": "__all_sources__",
            "opposite_group_fold": index,
            "train": train, "test": test,
        })
    return output


def _source_holdout_audit(table: pd.DataFrame, role: str) -> dict[str, Any]:
    development = table[(table["__role"] == role) & np.isfinite(table["__label"])].copy()
    ai_sources = sorted(development.loc[development["__label"] == 1, "__source"].unique())
    human_sources = sorted(development.loc[development["__label"] == 0, "__source"].unique())
    excluded = table[table["__role"] != role]
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_mode": "group_cv_only",
        "human_source_holdout": {
            "status": "not_run_by_predeclared_contract",
            "observed_development_sources": human_sources,
        },
        "generator_holdout": {
            "status": "ineligible_not_run",
            "reason": "leave-one-AI-source-out requires at least two development AI sources",
            "observed_development_ai_sources": ai_sources,
            "observed_development_ai_source_count": len(ai_sources),
        },
        "source_transfer_J": None,
        "source_transfer_J_status": "undefined_not_filled",
        "excluded_non_development_rows": int(len(excluded)),
        "excluded_sources_by_role": {
            str(role_name): sorted(current["__source"].astype(str).unique().tolist())
            for role_name, current in excluded.groupby("__role", sort=True)
        },
        "diffrhythm_pilot_training_rows": int((
            (development["__source"] == "ai_diffrhythm_pilot")
        ).sum()),
    }


def run_group_cv_only(
    args: argparse.Namespace, table: pd.DataFrame, config: dict[str, Any],
    contract: dict[str, Any], authorization: dict[str, Any], resolved: dict[str, str],
) -> None:
    audit = _source_holdout_audit(table, args.development_role)
    if audit["generator_holdout"]["observed_development_ai_source_count"] != 1:
        raise ValueError("60-second contract expects exactly one development AI source")
    if audit["generator_holdout"]["observed_development_ai_sources"] != ["Suno"]:
        raise ValueError("60-second contract expects Suno as the sole development AI source")
    if audit["diffrhythm_pilot_training_rows"] != 0:
        raise AssertionError("DiffRhythm pilot rows entered development training")

    original_materialize = V2.materialize_plan
    original_make_folds = V2.BASE.make_folds
    original_schema = V2.SCHEMA_VERSION
    try:
        V2.materialize_plan = materialize_group_plan
        V2.BASE.make_folds = make_global_group_folds
        V2.SCHEMA_VERSION = SCHEMA_VERSION
        V2.run_dev(args, table, config, contract, authorization, resolved)
    finally:
        V2.materialize_plan = original_materialize
        V2.BASE.make_folds = original_make_folds
        V2.SCHEMA_VERSION = original_schema

    (args.output_dir / "source_holdout_ineligibility.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    bundle_path = args.output_dir / "frozen_dev_models.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle.update({
        "schema_version": SCHEMA_VERSION,
        "evaluation_mode": "group_cv_only",
        "source_transfer_J": None,
        "source_transfer_J_status": "undefined_not_filled",
    })
    bundle_path.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest_path = args.output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "schema_version": SCHEMA_VERSION,
        "interpretation": (
            "development_only_global_group_cv_descriptive_recurrence_association; "
            "not source transfer and not future-generator generalization"
        ),
        "evaluation_mode": "group_cv_only",
        "source_holdout_status": "ineligible_not_run",
        "source_transfer_J": None,
        "source_transfer_J_status": "undefined_not_filled",
        "frozen_dev_bundle_sha256": sha256_file(bundle_path),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_family_config(args.families_json)
    if args.stage == "preregistration-draft":
        write_preregistration_draft(args, config)
        return
    contract = build_contract(args, args.stage, config)
    authorization = verify_authorization(args, contract)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output_dir / "run_manifest.json").exists():
            raise RuntimeError("Completed evaluation exists; use a new output directory")
        V2.write_process_state(args.output_dir, "running", stage=args.stage,
                               contract_sha256=canonical_hash(contract))
        try:
            table, resolved = load_table(args)
            validate_feature_columns(table, config)
            if args.stage == "dev" and args.evaluation_mode == "group_cv_only":
                run_group_cv_only(args, table, config, contract, authorization, resolved)
            elif args.stage == "dev":
                V2.run_dev(args, table, config, contract, authorization, resolved)
            elif args.evaluation_mode == "group_cv_only":
                raise ValueError("group_cv_only does not support historical scoring")
            else:
                V2.run_historical(args, table, config, contract, authorization)
        except BaseException as error:
            V2.write_process_state(args.output_dir, "failed", stage=args.stage, error=repr(error))
            raise
        V2.write_process_state(args.output_dir, "finished", stage=args.stage,
                               contract_sha256=canonical_hash(contract))


if __name__ == "__main__":
    main()
