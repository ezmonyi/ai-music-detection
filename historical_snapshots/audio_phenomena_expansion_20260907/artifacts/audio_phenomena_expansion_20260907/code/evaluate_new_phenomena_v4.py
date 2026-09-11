#!/usr/bin/env python3
"""Schema-v4 development-only exact60 seven-family descriptive global-group CV.

Separate, validated context proof is mandatory. Drafting cannot score. A real
run requires independent freezing of the exact data/code/runtime/config receipt.
No historical scoring, source-transfer score, winner selection or neural audio inference.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

import numpy as np
import pandas as pd

import prepare_evaluation_inputs_v4 as PREP
import evaluate_new_phenomena_v2 as V2
import evaluate_new_phenomena_v3 as V3

SCHEMA_VERSION = 4
SEED = 20260907
QUANTITIES = (25, 50, 100, 200, "all")
GROUP_FOLDS = 5
BASE = V2.BASE


def load_family_config(path):
    config = PREP.read_json(path)
    PREP.require(config == PREP.build_config(), "Schema-v4 family config must equal the fixed seven-family contract; no eligibility/status predictors")
    return config


def build_plan(table, config):
    PREP.require(set(table["__role"]) == {"development"}, "Only development rows may enter the plan")
    plan = V2.build_evaluation_plan(config)
    plan, cohorts, registry = V3.materialize_group_plan(plan, table, config, "development", GROUP_FOLDS)
    PREP.require(len(cohorts) == 1 and plan["candidate_key"].nunique() == 127, "Expected one cohort and all127 candidates")
    PREP.require(plan["eligible_id_set_sha256"].nunique() == 1 and plan["eligible_rows"].eq(len(table)).all(), "Candidate cohort shrinkage forbidden")
    comparisons = plan.loc[plan["plan_type"] == "incremental_matched", "comparison_id"].unique()
    PREP.require(len(comparisons) == 105 and sum(c.endswith("__plus__M") for c in comparisons) == 15, "Incomplete matched-comparison plan")
    return plan, cohorts, registry


def load_table(package):
    # The package validator already compared every output string to its evidence.
    meta = pd.read_csv(Path(package) / "metadata_60s.csv", dtype=str, keep_default_na=False)
    features = pd.read_csv(Path(package) / "features_60s.csv", dtype=str, keep_default_na=False)
    PREP.require(set(meta["role"]) == {"development"}, "Non-development metadata forbidden")
    PREP.require(set(meta["id"]) == set(features["id"]) and not meta["id"].duplicated().any() and not features["id"].duplicated().any(), "Model input IDs mismatch")
    table = meta.merge(features, on="id", validate="one_to_one")
    table = table.rename(columns={k: "__" + v for k, v in {"id": "id", "label": "label", "source_group": "source", "group_id": "group", "role": "role"}.items()})
    table["__label"] = pd.to_numeric(table["__label"], errors="raise")
    PREP.require(set(table["__label"]) == {0, 1}, "Both binary labels required")
    PREP.require(table.groupby("__source")["__label"].nunique().max() == 1, "A source must not cross labels")
    for columns in PREP.COLUMNS.values():
        for column in columns:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def build_contract(package, synthetic=False):
    proof = PREP.validate_package(package, synthetic)
    code_paths = [Path(__file__), Path(PREP.__file__), Path(V2.__file__), Path(V3.__file__), V2.BASE_EVALUATOR_PATH]
    return {"schema_version": 4, "authorized_stage": "dev", "synthetic_test_only": synthetic,
            "package_files_sha256": proof["files_sha256"],
            "preparation_audit_sha256": PREP.sha(Path(package) / "preparation_audit.json"),
            "code_sha256": {p.name: PREP.sha(p) for p in code_paths},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "thread_settings": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
            "parameters": {"quantities": list(QUANTITIES), "group_folds": GROUP_FOLDS, "seed": SEED,
                           "model": "weighted_ridge_linear_probability", "ridge": BASE.RIDGE, "threshold": 0.5,
                           "feature_mode": "values_plus_missing", "all_cap_diagnostics": ["median_only", "missingness_only"],
                           "scope": "all_development_same_native_60s_intervals", "fold_unit": "global_group_id",
                           "quantity_unit": "groups_per_source_within_training_fold",
                           "candidates": 127, "matched_comparisons": 105, "model_selection": False,
                           "source_holdout": False, "source_transfer_J": None,
                           "historical_locked_or_pilot_scoring": False}}


def verify_authorization(contract, receipt_path=None):
    if contract["synthetic_test_only"]:
        return {"status": "synthetic_test_only", "contract_sha256": PREP.digest(contract)}
    PREP.require(receipt_path is not None, "Real scoring refused: matching frozen receipt is required")
    receipt = PREP.read_json(receipt_path)
    PREP.require(receipt.get("status") == "frozen" and receipt.get("authorized_stage") == "dev"
                 and receipt.get("contract") == contract and receipt.get("contract_sha256") == PREP.digest(contract),
                 "Real scoring refused: receipt is not frozen for this exact input/code/config/runtime contract")
    return {"status": "frozen_verified", "contract_sha256": PREP.digest(contract), "receipt_sha256": PREP.sha(receipt_path)}


def make_schedule(table):
    """One materialized schedule shared by every candidate and diagnostic mode."""
    PREP.require(set(table["__role"]) == {"development"}, "Role leakage into schedule")
    folds = V3.make_global_group_folds(table, GROUP_FOLDS, SEED)
    schedule = []
    for index, fold in enumerate(folds):
        for quantity in QUANTITIES:
            train = BASE.deterministic_quantity(fold["train"], quantity, SEED)
            test = fold["test"]
            valid, reason = V2.valid_train(train, 2)
            PREP.require(valid, "Invalid train schedule: " + reason)
            PREP.require(not set(train["__group"]) & set(test["__group"]), "Global group leakage")
            PREP.require(not set(train["__id"]) & set(test["__id"]), "ID leakage")
            record = {"fold_index": index, "quantity": quantity, "fold_type": fold["fold_type"],
                      "train_ids": sorted(train["__id"]), "test_ids": sorted(test["__id"]),
                      "train_group_ids": sorted(set(train["__group"])), "test_group_ids": sorted(set(test["__group"])),
                      "train_id_set_sha256": V2.id_set_hash(train["__id"]), "test_id_set_sha256": V2.id_set_hash(test["__id"])}
            record["fold_uid"] = PREP.digest(record)[:24]
            schedule.append((record, train, test, fold))
    return schedule


def run_group_cv(output, table, config, contract, authorization):
    """Called only after package and frozen-receipt verification by main()."""
    expected_status = "synthetic_test_only" if contract.get("synthetic_test_only") else "frozen_verified"
    PREP.require(isinstance(authorization, dict) and authorization.get("status") == expected_status
                 and authorization.get("contract_sha256") == PREP.digest(contract), "Missing verified scoring authorization")
    output = Path(output)
    plan, cohorts, registry = build_plan(table, config)
    cohort_id = next(iter(cohorts))
    schedule = make_schedule(table)
    family_coverage, candidate_coverage = V2.coverage_rows(cohorts, config, plan)
    for name, frame in (("evaluation_plan.csv", plan), ("cohort_registry.csv", registry), ("family_coverage_by_source.csv", family_coverage), ("candidate_coverage.csv", candidate_coverage)):
        frame.to_csv(output / name, index=False)
    candidates = list(dict.fromkeys(plan["combination"]))
    predictions, metrics, aggregate, diagnostics, diagnostic_metrics = [], [], [], [], []
    models = {}
    for record, train, test, fold in schedule:
        print(json.dumps({"event": "group_fold_quantity", "fold": record["fold_index"], "quantity": record["quantity"], "candidates": 127}), flush=True)
        for combination in candidates:
            columns = V2.columns_for_combination(config, combination)
            modes = ["values_plus_missing"] + (["median_only", "missingness_only"] if record["quantity"] == "all" else [])
            for mode in modes:
                model = V2.fit_candidate(train, columns, "ridge", feature_mode=mode)
                # Frozen base reports this descriptive string identically for all
                # modes; correct the diagnostic record without changing its fit.
                model["missing_value_policy"] = {"values_plus_missing": "train-only median plus feature-missing indicators", "median_only": "train-only median; no indicators", "missingness_only": "feature-missing indicators only"}[mode]
                scores = V2.predict_candidate(test, model)
                PREP.require(np.isfinite(scores).all(), "Nonfinite model predictions")
                key = PREP.digest({"model": model, "threshold": 0.5})
                models.setdefault(key, model)
                common = {"cohort_id": cohort_id, "candidate_key": V2.candidate_key(cohort_id, combination),
                          "combination": combination, "quantity": record["quantity"], "fold_uid": record["fold_uid"],
                          "fold_index": record["fold_index"], "fold_type": fold["fold_type"],
                          "heldout_source": "__all_sources__", "opposite_group_fold": fold["opposite_group_fold"],
                          "model_sha256": key, "feature_mode": mode,
                          "train_id_set_sha256": record["train_id_set_sha256"], "test_id_set_sha256": record["test_id_set_sha256"]}
                pred = V2._prediction_rows(test, scores, 0.5, common)
                pairs = [{**common, **m} for m in V2.pair_metrics(test, scores, fold, 0.5, 1)]
                if mode == "values_plus_missing":
                    predictions.extend(pred)
                    metrics.extend(pairs)
                    aggregate.append({**common, "threshold": 0.5, **V2.metrics_at_threshold(test["__label"].to_numpy(int), scores, 0.5)})
                else:
                    diagnostics.extend(pred)
                    diagnostic_metrics.extend(pairs)
    metric_frame = pd.DataFrame(metrics)
    pd.DataFrame(predictions).to_csv(output / "development_group_cv_predictions.csv", index=False)
    metric_frame.to_csv(output / "development_group_cv_metrics_by_source_pair.csv", index=False)
    pd.DataFrame(aggregate).to_csv(output / "development_group_cv_pooled_fold_metrics.csv", index=False)
    V2.matched_metric_deltas(metric_frame, plan).to_csv(output / "development_group_cv_matched_deltas.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "development_group_cv_diagnostic_predictions.csv", index=False)
    pd.DataFrame(diagnostic_metrics).to_csv(output / "development_group_cv_diagnostic_metrics.csv", index=False)
    with (output / "fold_registry.jsonl").open("x") as f:
        for record, *_ in schedule:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    (output / "fold_models.json").write_text(json.dumps(models, allow_nan=False) + "\n")
    manifest = {"schema_version": 4, "stage": "dev", "authorization": authorization, "contract": contract,
                "rows": len(table), "unique_candidates": 127, "plan_rows": len(plan), "matched_comparisons": 105,
                "cohorts": 1, "source_transfer_J": None, "source_transfer_J_status": "undefined_not_filled",
                "source_holdout_status": "not_run_single_development_AI_source_Suno", "winner_selected": False,
                "interpretation": "descriptive within this Human/Suno source composition; not unseen-generator generalization",
                "missingness_caveat": "train-only feature missing indicators may encode source or measurement failure; all-cap diagnostics are non-selecting",
                "files_sha256": {p.name: PREP.sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preregistration-draft", "dev"), required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--synthetic-test-only", action="store_true")
    args = parser.parse_args(argv)
    PREP.require(not args.output_dir.exists(), "Refusing existing output directory")
    contract = build_contract(args.package_dir, args.synthetic_test_only)
    config = load_family_config(args.package_dir / "families_60s_v4.json")
    table = load_table(args.package_dir)
    # Scheduling/plan construction is score-free and checked before a draft can
    # be frozen, so an impossible fold can never become an authorized result.
    build_plan(table, config)
    make_schedule(table)
    PREP.require(build_contract(args.package_dir, args.synthetic_test_only) == contract,
                 "Package/code changed while loading model inputs")
    if args.stage == "preregistration-draft":
        authorization = None
    else:
        authorization = verify_authorization(contract, args.preregistration)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=args.output_dir.name + ".tmp.", dir=args.output_dir.parent))
    try:
        if args.stage == "preregistration-draft":
            receipt = {"schema_version": 4, "status": "draft", "authorized_stage": "dev", "contract_sha256": PREP.digest(contract), "contract": contract,
                       "freeze_instruction": "Independent root review must set status=frozen with no other changes; draft never authorizes scoring."}
            (temp / "preregistration_draft.json").write_text(json.dumps(receipt, indent=2) + "\n")
        else:
            run_group_cv(temp, table, config, contract, authorization)
            PREP.require(build_contract(args.package_dir, args.synthetic_test_only) == contract,
                         "Package/code changed during scoring; no result is published")
        PREP.require(not args.output_dir.exists(), "Output appeared during run")
        os.rename(temp, args.output_dir)
    except BaseException:
        shutil.rmtree(temp)
        raise


if __name__ == "__main__":
    main()
