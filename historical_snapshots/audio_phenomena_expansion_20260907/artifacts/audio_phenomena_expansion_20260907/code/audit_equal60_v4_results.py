#!/usr/bin/env python3
"""Read-only independent result audit for the exact60 schema-v4 evaluator.

Uses the preparer for physical-evidence admission only. Does not import an
evaluator, fit a model, decode media, select a candidate or score locked rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import prepare_evaluation_inputs_v4 as P

SEED = 20260907
CAPS = (25, 50, 100, 200, "all")
MODES = ("values_plus_missing", "median_only", "missingness_only")
FOLD_TYPE = "ordinary_group_holdout_descriptive"
FILES = {
    "evaluation_plan.csv", "cohort_registry.csv", "family_coverage_by_source.csv",
    "candidate_coverage.csv", "fold_registry.jsonl", "fold_models.json",
    "development_group_cv_predictions.csv", "development_group_cv_metrics_by_source_pair.csv",
    "development_group_cv_pooled_fold_metrics.csv", "development_group_cv_matched_deltas.csv",
    "development_group_cv_diagnostic_predictions.csv", "development_group_cv_diagnostic_metrics.csv",
}
COMMON = ["cohort_id", "candidate_key", "combination", "quantity", "fold_uid", "fold_index",
          "fold_type", "heldout_source", "opposite_group_fold", "model_sha256", "feature_mode",
          "train_id_set_sha256", "test_id_set_sha256"]
MEASURES = ["roc_auc", "balanced_accuracy", "ai_sensitivity", "human_specificity"]
METRICS = MEASURES + ["tp", "tn", "fp", "fn"]
PAIR = ["human_source", "ai_source", "test_human", "test_ai", "test_human_groups", "test_ai_groups"]
POLICIES = dict(zip(MODES, ["train-only median plus feature-missing indicators",
                          "train-only median; no indicators", "feature-missing indicators only"]))
WEIGHTING = "classes equal; sources equal within class; groups equal within source; samples equal within group"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def id_hash(ids):
    return hashlib.sha256("".join(i + "\n" for i in sorted(set(ids))).encode()).hexdigest()


def strict_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, f"Duplicate JSON key {key}: {path}")
            out[key] = value
        return out
    def constant(value):
        raise ValueError(f"Nonfinite JSON value {value}: {path}")
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=constant)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        rows = csv.reader(handle)
        header = next(rows, [])
        require(bool(header) and len(header) == len(set(header)), f"Missing/duplicate CSV header: {path}")
        values = list(rows)
        require(all(len(row) == len(header) for row in values), f"Malformed CSV row: {path}")
    return pd.DataFrame(values, columns=header)


def equal_records(actual, expected, keys, context):
    """Exact schema/keys/text; numeric values compared with CSV round-trip tolerance."""
    expected = pd.DataFrame(expected)
    require(set(actual.columns) == set(expected.columns), f"{context}: column schema mismatch")
    require(len(actual) == len(expected) and not actual.duplicated(keys).any(), f"{context}: missing/duplicate rows")
    actual = actual.set_index(keys).sort_index()
    expected = expected.astype({k: str for k in keys}).set_index(keys).sort_index()
    require(actual.index.equals(expected.index), f"{context}: key set mismatch")
    for col in expected:
        wanted = expected[col]
        if pd.api.types.is_numeric_dtype(wanted) and not pd.api.types.is_bool_dtype(wanted):
            got = pd.to_numeric(actual[col].replace("", np.nan), errors="raise").to_numpy(float)
            matches = np.array_equal(got, wanted.to_numpy(float)) if pd.api.types.is_integer_dtype(wanted) else np.allclose(got, wanted.to_numpy(float), rtol=1e-10, atol=1e-12, equal_nan=True)
            require(matches, f"{context}: numeric mismatch {col}")
        else:
            require(actual[col].astype(str).equals(wanted.astype(str)), f"{context}: mismatch {col}")


def subsets(codes):
    return ["+".join(c) for n in range(1, len(codes) + 1) for c in itertools.combinations(codes, n)]


def expected_plan(meta):
    cohort_hash = id_hash(meta.index)
    cohort = "dev-" + cohort_hash[:16]
    rows = []
    old, new = subsets("SDRP"), subsets("FHM")
    def add(plan_id, plan_type, comparison, arm, combination, eligible):
        codes = eligible.split("+")
        eligibility = {"global": {}, "families": [{"code": c, "eligibility": {}} for c in codes],
                       "missingness_policy": "never_complete_case_filter; training-fold imputation plus indicators"}
        rows.append(dict(plan_id=plan_id, plan_type=plan_type, comparison_id=comparison, arm=arm,
                         combination=combination, family_codes=json.dumps(combination.split("+")),
                         eligibility_family_codes=json.dumps(codes), cohort_id=cohort,
                         eligibility_contract_sha256=digest(eligibility), eligible_rows=len(meta),
                         eligible_id_set_sha256=cohort_hash,
                         candidate_key=hashlib.sha256(f"{cohort}|{combination}".encode()).hexdigest()[:20]))
    for combo in old:
        add("baseline::" + combo, "old_baseline_standalone", "", "standalone", combo, combo)
    for combo in new:
        add("new::" + combo, "new_standalone", "", "standalone", combo, combo)
    for left, right in itertools.product(old, new):
        comparison = f"incremental::{left}__plus__{right}"
        for arm, combination in (("baseline", left), ("added", left + "+" + right)):
            add(comparison + "::" + arm, "incremental_matched", comparison, arm, combination, left + "+" + right)
    return pd.DataFrame(rows), cohort


def schedule(meta):
    folds = meta.group_id.map(lambda g: int.from_bytes(hashlib.sha256(f"{SEED}|{g}".encode()).digest()[:8], "big") % 5)
    output = []
    for fold in range(5):
        test = meta[folds == fold]
        pool = meta[folds != fold]
        require(set(test.label) == {0, 1}, f"Fold {fold} lacks test class")
        for cap in CAPS:
            parts = []
            for _, source in pool.groupby(["label", "source_group"], sort=True):
                groups = sorted(source.group_id.unique(), key=lambda g: hashlib.sha256(f"{SEED}|{g}".encode()).hexdigest())
                chosen = groups if cap == "all" else groups[:cap]
                parts.append(source[source.group_id.isin(chosen)])
            train = pd.concat(parts) if cap != "all" else pool
            require(all(train[train.label == label].group_id.nunique() >= 2 for label in (0, 1)), "Invalid training group count")
            require(not set(train.group_id) & set(test.group_id), "Global group leakage")
            record = dict(fold_index=fold, quantity=cap, fold_type=FOLD_TYPE,
                          train_ids=sorted(train.index), test_ids=sorted(test.index),
                          train_group_ids=sorted(set(train.group_id)), test_group_ids=sorted(set(test.group_id)),
                          train_id_set_sha256=id_hash(train.index), test_id_set_sha256=id_hash(test.index))
            record["fold_uid"] = digest(record)[:24]
            output.append((record, train, test))
    return output


def metrics(labels, scores):
    """Independent Mann-Whitney average-rank AUC and unweighted confusion counts."""
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    require(np.isfinite(scores).all() and set(labels) <= {0, 1}, "Invalid metric inputs")
    pos, neg = labels == 1, labels == 0
    predicted = scores >= 0.5
    tp, tn = int(np.sum(predicted & pos)), int(np.sum(~predicted & neg))
    fp, fn = int(np.sum(predicted & neg)), int(np.sum(~predicted & pos))
    sensitivity = tp / pos.sum() if pos.any() else math.nan
    specificity = tn / neg.sum() if neg.any() else math.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    auc = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()) if pos.any() and neg.any() else math.nan
    return dict(roc_auc=float(auc), balanced_accuracy=(sensitivity + specificity) / 2,
                ai_sensitivity=sensitivity, human_specificity=specificity, tp=tp, tn=tn, fp=fp, fn=fn)


def design(values, mode, medians):
    missing = ~np.isfinite(values)
    imputed = np.where(missing, medians, values)
    return np.concatenate([imputed, missing], axis=1) if mode == MODES[0] else imputed if mode == MODES[1] else missing.astype(float)


def training_weights(train):
    sources = train.groupby("label").source_group.transform("nunique").to_numpy(float)
    groups = train.groupby(["label", "source_group"]).group_id.transform("nunique").to_numpy(float)
    samples = train.groupby(["label", "source_group", "group_id"]).group_id.transform("size").to_numpy(float)
    weights = 0.5 / (sources * groups * samples)
    return weights * (len(train) / weights.sum())


def check_model(model, train, features, columns, mode, weights=None):
    """Verify train-only transforms and ridge stationarity without fitting/solving."""
    expected_keys = {"columns", "medians", "mean", "scale", "coefficients_with_intercept", "ridge", "threshold",
                     "feature_mode", "training_rows", "training_source_counts", "weighting", "missing_value_policy",
                     "observed_fraction_by_column", "model_type", "prediction_link"}
    require(set(model) == expected_keys, "Model schema mismatch")
    for key, value in dict(columns=columns, ridge=10.0, threshold=0.5, feature_mode=mode,
                           training_rows=len(train), weighting=WEIGHTING, missing_value_policy=POLICIES[mode],
                           model_type="weighted_ridge_linear_probability", prediction_link="identity").items():
        require(model[key] == value, f"Model restriction mismatch: {key}")
    source_counts = (train.label.astype(str) + ":" + train.source_group).value_counts().sort_index().to_dict()
    require(model["training_source_counts"] == source_counts, "Model source counts mismatch")
    x = features.loc[train.index, columns].to_numpy(float)
    medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in x.T])
    observed = dict(zip(columns, np.isfinite(x).mean(axis=0)))
    require(set(model["observed_fraction_by_column"]) == set(columns), "Model observed column mismatch")
    require(all(abs(model["observed_fraction_by_column"][k] - v) < 1e-12 for k, v in observed.items()), "Model observed fraction mismatch")
    w = training_weights(train) if weights is None else weights
    z = design(x, mode, medians)
    mean = np.average(z, axis=0, weights=w)
    scale = np.sqrt(np.average((z - mean) ** 2, axis=0, weights=w))
    scale[scale < 1e-8] = 1.0
    for key, value in (("medians", medians), ("mean", mean), ("scale", scale)):
        got = np.asarray(model[key], float)
        require(got.shape == value.shape and np.isfinite(got).all() and np.allclose(got, value, rtol=1e-10, atol=1e-12), f"Train-only preprocessing mismatch: {key}")
    beta = np.asarray(model["coefficients_with_intercept"], float)
    require(beta.shape == (len(mean) + 1,) and np.isfinite(beta).all(), "Model coefficients invalid")
    matrix = np.column_stack([np.ones(len(train)), (z - mean) / scale])
    lhs = matrix.T @ (w * (matrix @ beta)) + np.r_[0.0, 10 * beta[1:]]
    rhs = matrix.T @ (w * train.label.to_numpy(float))
    require(np.linalg.norm(lhs - rhs, ord=np.inf) <= 1e-8 * max(1.0, np.linalg.norm(rhs, ord=np.inf)), "Ridge normal-equation residual mismatch")


def verify_contract(root, package, synthetic, preregistration):
    manifest = strict_json(root / "run_manifest.json")
    require(isinstance(manifest, dict) and set(manifest) == {"schema_version", "stage", "authorization", "contract", "rows",
            "unique_candidates", "plan_rows", "matched_comparisons", "cohorts", "source_transfer_J", "source_transfer_J_status",
            "source_holdout_status", "winner_selected", "interpretation", "missingness_caveat", "files_sha256"}, "Manifest schema mismatch")
    require(set(p.name for p in root.iterdir()) == FILES | {"run_manifest.json"}, "Missing/extra output files; historical/source-holdout outputs forbidden")
    require(all(p.is_file() and not p.is_symlink() for p in root.iterdir()), "Output must contain regular files only")
    require(set(manifest["files_sha256"]) == FILES, "Incomplete result hash manifest")
    for name in FILES:
        require(P.sha(root / name) == manifest["files_sha256"][name], f"Result hash mismatch: {name}")
    proof = P.validate_package(package, synthetic)
    contract = manifest["contract"]
    require(set(contract) == {"schema_version", "authorized_stage", "synthetic_test_only", "package_files_sha256",
                             "preparation_audit_sha256", "code_sha256", "runtime", "parameters"}, "Contract schema mismatch")
    require(contract["schema_version"] == 4 and contract["authorized_stage"] == "dev" and contract["synthetic_test_only"] is synthetic, "Wrong contract stage/mode")
    require(contract["package_files_sha256"] == proof["files_sha256"] and contract["preparation_audit_sha256"] == P.sha(package / "preparation_audit.json"), "Stale package contract")
    code_names = {"evaluate_new_phenomena_v4.py", "prepare_evaluation_inputs_v4.py", "evaluate_new_phenomena_v2.py", "evaluate_new_phenomena_v3.py", "frozen_evaluate_expanded_20260905.py"}
    require(set(contract["code_sha256"]) == code_names, "Contract code coverage mismatch")
    for name in code_names:
        require(contract["code_sha256"][name] == P.sha(Path(__file__).with_name(name)), f"Stale code contract: {name}")
    expected = dict(quantities=list(CAPS), group_folds=5, seed=SEED, model="weighted_ridge_linear_probability", ridge=10.0,
                    threshold=0.5, feature_mode=MODES[0], all_cap_diagnostics=list(MODES[1:]),
                    scope="all_development_same_native_60s_intervals", fold_unit="global_group_id",
                    quantity_unit="groups_per_source_within_training_fold", candidates=127, matched_comparisons=105,
                    model_selection=False, source_holdout=False, source_transfer_J=None, historical_locked_or_pilot_scoring=False)
    require(contract["parameters"] == expected, "Contract parameter restriction mismatch")
    runtime = contract["runtime"]
    require(set(runtime) == {"python", "numpy", "pandas", "thread_settings"} and all(isinstance(runtime[k], str) and runtime[k] for k in ("python", "numpy", "pandas")), "Malformed runtime contract")
    require(runtime["thread_settings"] == {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}, "Unfrozen thread settings")
    authorization = manifest["authorization"]
    require(authorization["contract_sha256"] == digest(contract), "Authorization digest mismatch")
    if synthetic:
        require(authorization == {"status": "synthetic_test_only", "contract_sha256": digest(contract)}, "Synthetic authorization mismatch")
    else:
        require(preregistration is not None, "Frozen receipt required for real results")
        receipt = strict_json(preregistration)
        require(receipt.get("status") == "frozen" and receipt.get("authorized_stage") == "dev" and receipt.get("schema_version") == 4, "Receipt not frozen schema-v4 dev")
        require(receipt["contract"] == contract and receipt["contract_sha256"] == digest(contract), "Stale frozen receipt")
        require(authorization == dict(status="frozen_verified", contract_sha256=digest(contract), receipt_sha256=P.sha(preregistration)), "Receipt authorization hash mismatch")
    for key, value in dict(schema_version=4, stage="dev", unique_candidates=127, plan_rows=232, matched_comparisons=105,
                           cohorts=1, source_transfer_J=None, source_transfer_J_status="undefined_not_filled",
                           source_holdout_status="not_run_single_development_AI_source_Suno", winner_selected=False,
                           interpretation="descriptive within this Human/Suno source composition; not unseen-generator generalization",
                           missingness_caveat="train-only feature missing indicators may encode source or measurement failure; all-cap diagnostics are non-selecting").items():
        require(manifest.get(key) == value, f"Manifest restriction mismatch: {key}")
    return manifest


def audit(result_dir, package_dir, preregistration=None, synthetic=False):
    root, package = Path(result_dir), Path(package_dir)
    manifest = verify_contract(root, package, synthetic, preregistration)
    meta = read_csv(package / "metadata_60s.csv").set_index("id")
    require(meta.index.is_unique and set(meta.role) == {"development"}, "Duplicate IDs/role leakage")
    meta["label"] = pd.to_numeric(meta.label, errors="raise").astype(int)
    require(set(meta.label) == {0, 1} and meta.groupby("source_group").label.nunique().max() == 1, "Invalid label/source mapping")
    require(len(meta) == manifest["rows"] and (synthetic or len(meta) == 1604), "Development row count mismatch")
    native = read_csv(package / "evidence/native.csv")
    require(native.groupby("group_id").role.nunique().max() == 1, "Global role leakage")
    require(set(native.loc[native.role == "development", "id"]) == set(meta.index), "Development cohort filtering")
    features = read_csv(package / "features_60s.csv").set_index("id")
    require(features.index.is_unique and set(features.index) == set(meta.index) and set(features) == set(sum(P.COLUMNS.values(), [])), "Predictor/ID schema mismatch")
    features = features.apply(pd.to_numeric, errors="coerce")
    plan, cohort = expected_plan(meta)
    equal_records(read_csv(root / "evaluation_plan.csv"), plan, ["plan_id"], "plan")
    registry = dict(cohort_id=cohort, eligible_id_set_sha256=id_hash(meta.index), rows=len(meta),
                    human_rows=int((meta.label == 0).sum()), ai_rows=int((meta.label == 1).sum()),
                    human_sources=meta[meta.label == 0].source_group.nunique(), ai_sources=meta[meta.label == 1].source_group.nunique(),
                    human_groups=meta[meta.label == 0].group_id.nunique(), ai_groups=meta[meta.label == 1].group_id.nunique(),
                    eligibility_contract_hashes=json.dumps(sorted(set(plan.eligibility_contract_sha256))))
    equal_records(read_csv(root / "cohort_registry.csv"), [registry], ["cohort_id"], "cohort registry")
    check_coverage(root, features, meta, plan, cohort)
    schedules = schedule(meta)
    lines = (root / "fold_registry.jsonl").read_text().splitlines()
    require(len(lines) == 25 and all(lines), "Incomplete/blank fold registry")
    # Reuse strict JSON parser on individual lines without creating files.
    def unique_pairs(items):
        require(len({k for k, _ in items}) == len(items), "Duplicate fold JSON key")
        return dict(items)
    actual_schedule = [json.loads(line, object_pairs_hook=unique_pairs) for line in lines]
    require(actual_schedule == [record for record, _, _ in schedules], "Fold/cap schedule mismatch")
    models = strict_json(root / "fold_models.json")
    require(isinstance(models, dict) and bool(models), "Missing model registry")
    for key, model in models.items():
        require(key == digest({"model": model, "threshold": 0.5}), "Model hash mismatch")
    used_models, validated_models = set(), set()
    pairs, pooled, diag_pairs = [], [], []
    candidates = plan.drop_duplicates("combination").set_index("combination")
    for filename, modes, output_pairs in (("development_group_cv_predictions.csv", [MODES[0]], pairs),
                                          ("development_group_cv_diagnostic_predictions.csv", list(MODES[1:]), diag_pairs)):
        pred = read_csv(root / filename)
        require(set(pred) == set(COMMON + ["row_id", "label", "source_group", "group_id", "score", "threshold", "predicted_label"]), "Prediction schema mismatch")
        require(not pred.duplicated(["combination", "fold_uid", "feature_mode", "row_id"]).any(), "Duplicate prediction")
        expected_count = 127 * len(meta) * (5 if modes == [MODES[0]] else 2)
        require(len(pred) == expected_count, "Incomplete prediction grid")
        groups = {k: v for k, v in pred.groupby(["fold_uid", "combination", "feature_mode"], sort=False)}
        wanted_keys = {(record["fold_uid"], combo, mode) for record, _, _ in schedules
                       if modes == [MODES[0]] or record["quantity"] == "all" for combo in candidates.index for mode in modes}
        require(set(groups) == wanted_keys, "Incomplete candidate/cap/diagnostic grid")
        for record, train, test in schedules:
            if modes != [MODES[0]] and record["quantity"] != "all":
                continue
            weights = training_weights(train)
            for combo in candidates.index:
                columns = [c for family in combo.split("+") for c in P.COLUMNS[family]]
                for mode in modes:
                    frame = groups[(record["fold_uid"], combo, mode)].set_index("row_id")
                    require(set(frame.index) == set(test.index), "Prediction test IDs mismatch")
                    frame = frame.loc[test.index]
                    key = frame.model_sha256.iloc[0]
                    require(key in models, "Missing referenced model")
                    common = dict(cohort_id=cohort, candidate_key=candidates.loc[combo, "candidate_key"], combination=combo,
                                  quantity=record["quantity"], fold_uid=record["fold_uid"], fold_index=record["fold_index"],
                                  fold_type=FOLD_TYPE, heldout_source="__all_sources__", opposite_group_fold=record["fold_index"],
                                  model_sha256=key, feature_mode=mode, train_id_set_sha256=record["train_id_set_sha256"], test_id_set_sha256=record["test_id_set_sha256"])
                    for name, value in common.items():
                        require(frame[name].eq(str(value)).all(), f"Prediction common field mismatch: {name}")
                    for name in ("label", "source_group", "group_id"):
                        require(frame[name].equals(test[name].astype(str)), f"Prediction identity mismatch: {name}")
                    scores = pd.to_numeric(frame.score, errors="coerce").to_numpy(float)
                    require(np.isfinite(scores).all() and frame.threshold.eq("0.5").all(), "Invalid score/threshold")
                    require(frame.predicted_label.tolist() == (scores >= 0.5).astype(int).astype(str).tolist(), "Predicted label threshold mismatch")
                    model = models[key]
                    validation_key = (key, record["train_id_set_sha256"], combo, mode)
                    if validation_key not in validated_models:
                        check_model(model, train, features, columns, mode, weights)
                        validated_models.add(validation_key)
                    used_models.add(key)
                    z = design(features.loc[test.index, columns].to_numpy(float), mode, np.asarray(model["medians"]))
                    matrix = np.column_stack([np.ones(len(test)), (z - model["mean"]) / model["scale"]])
                    require(np.allclose(scores, matrix @ model["coefficients_with_intercept"], rtol=1e-10, atol=1e-12), "Raw score/model replay mismatch")
                    if mode == MODES[0]:
                        pooled.append({**common, "threshold": 0.5, **metrics(test.label, scores)})
                    for human in sorted(test.loc[test.label == 0, "source_group"].unique()):
                        pair = test.source_group.isin([human, "Suno"]).to_numpy()
                        current = test.iloc[np.flatnonzero(pair)]
                        output_pairs.append({**common, "human_source": human, "ai_source": "Suno", "threshold": 0.5,
                                             "test_human": int((current.label == 0).sum()), "test_ai": int((current.label == 1).sum()),
                                             "test_human_groups": current[current.label == 0].group_id.nunique(),
                                             "test_ai_groups": current[current.label == 1].group_id.nunique(), **metrics(current.label, scores[pair])})
    require(used_models == set(models), "Unreferenced/full-development model forbidden")
    metric_keys = ["fold_uid", "candidate_key", "feature_mode"]
    equal_records(read_csv(root / "development_group_cv_pooled_fold_metrics.csv"), pooled, metric_keys, "pooled metrics")
    equal_records(read_csv(root / "development_group_cv_metrics_by_source_pair.csv"), pairs, metric_keys + ["human_source", "ai_source"], "source-pair metrics")
    equal_records(read_csv(root / "development_group_cv_diagnostic_metrics.csv"), diag_pairs, metric_keys + ["human_source", "ai_source"], "diagnostic metrics")
    deltas = expected_deltas(pairs, plan)
    equal_records(read_csv(root / "development_group_cv_matched_deltas.csv"), deltas, ["comparison_id", "fold_uid", "human_source", "ai_source"], "matched deltas")
    # Detect changing inputs/results across a long audit, including manifest replacement.
    require(verify_contract(root, package, synthetic, preregistration) == manifest, "Contract/results changed during audit")
    return dict(status="passed", schema_version=4, synthetic_test_only=synthetic, rows=len(meta), candidates=127,
                caps=list(CAPS), primary_predictions=127 * 5 * len(meta), diagnostic_predictions=127 * 2 * len(meta),
                pooled_metrics=len(pooled), source_pair_metrics=len(pairs), diagnostic_metrics=len(diag_pairs),
                matched_comparisons=105, matched_delta_rows=len(deltas), models=len(models), source_transfer_J=None,
                result_manifest_sha256=P.sha(root / "run_manifest.json"), auditor_sha256=P.sha(Path(__file__)),
                model_fitting_performed=False, metrics_independently_recomputed=True)


def expected_deltas(pairs, plan):
    indexed = {(r["combination"], r["fold_uid"], r["human_source"], r["ai_source"]): r for r in pairs}
    keys = ["cohort_id", "quantity", "fold_uid", "fold_index", "fold_type", "heldout_source", "human_source", "ai_source",
            "opposite_group_fold", "test_human", "test_ai", "test_human_groups", "test_ai_groups"]
    result = []
    for comparison, arms in plan[plan.plan_type == "incremental_matched"].groupby("comparison_id"):
        baseline = arms[arms.arm == "baseline"].iloc[0]
        added = arms[arms.arm == "added"].iloc[0]
        for (combo, fold, human, ai), left in indexed.items():
            if combo != baseline.combination:
                continue
            right = indexed[(added.combination, fold, human, ai)]
            require(all(left[k] == right[k] for k in keys), "Unmatched comparison metric keys")
            row = dict(comparison_id=comparison, baseline_combination=combo, added_combination=added.combination,
                       eligible_id_set_sha256=baseline.eligible_id_set_sha256, **{k: left[k] for k in keys})
            for measure in MEASURES:
                row[measure + "__baseline"] = left[measure]
                row[measure + "__added"] = right[measure]
                row["delta_" + measure + "__added_minus_baseline"] = right[measure] - left[measure]
            result.append(row)
    return result


def check_coverage(root, features, meta, plan, cohort):
    config = P.build_config()
    specs = config["old_families"] | config["new_families"]
    family, candidates = [], []
    for source, current in meta.groupby("source_group"):
        for code, spec in specs.items():
            finite = np.isfinite(features.loc[current.index, spec["columns"]].to_numpy(float))
            any_rows = finite.any(axis=1) if spec["columns"] else np.zeros(len(current), bool)
            complete = finite.all(axis=1) if spec["columns"] else np.zeros(len(current), bool)
            family.append(dict(cohort_id=cohort, source_group=source, family=code, state=spec["state"], phenomenon=spec["phenomenon"],
                               eligible_rows=len(current), any_observed_rows=int(any_rows.sum()), complete_rows=int(complete.sum()),
                               any_observed_fraction=float(any_rows.mean()), complete_fraction=float(complete.mean()),
                               minimum_observed_fraction=0.05, coverage_qualified=bool(spec["state"] == "available" and any_rows.mean() >= .05)))
    for _, item in plan.drop_duplicates("combination").iterrows():
        columns = [c for family in item.combination.split("+") for c in P.COLUMNS[family]]
        finite = np.isfinite(features[columns].to_numpy(float))
        candidates.append(dict(cohort_id=cohort, candidate_key=item.candidate_key, combination=item.combination,
                               eligible_rows=len(meta), feature_count=len(columns), rows_with_any_observed_feature=int(finite.any(axis=1).sum()),
                               rows_complete_all_selected_features=int(finite.all(axis=1).sum()), any_observed_fraction=float(finite.any(axis=1).mean()),
                               complete_case_fraction_diagnostic_only=float(finite.all(axis=1).mean()),
                               missingness_policy="no complete-case deletion; train-fold median plus indicators",
                               coverage_qualified=bool(all(np.isfinite(features[P.COLUMNS[c]].to_numpy(float)).any(axis=1).mean() >= .05 for c in item.combination.split("+")))))
    equal_records(read_csv(root / "family_coverage_by_source.csv"), family, ["source_group", "family"], "family coverage")
    equal_records(read_csv(root / "candidate_coverage.csv"), candidates, ["candidate_key"], "candidate coverage")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--synthetic-test-only", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.result_dir, args.package_dir, args.preregistration, args.synthetic_test_only), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
