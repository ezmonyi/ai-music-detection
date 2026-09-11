#!/usr/bin/env python3
"""Aggregate independently audited exploratory-v5 results for presentation.

This module never fits, scores, tunes, selects, or launches an evaluation.  It
streams accepted saved predictions only after a passed independent audit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile


FAMILIES = ("S", "D", "R", "P", "F", "H", "M")
FAMILY_COUNTS = {"S": 15, "D": 3, "R": 3, "P": 6, "F": 15, "H": 6, "M": 6}
COMBINATIONS = tuple("+".join(c) for n in range(1, 8) for c in itertools.combinations(FAMILIES, n))
CAPS = ("25", "50", "100", "200", "all")
PRIMARY_MODE = "values_plus_missing"
DIAGNOSTIC_MODES = ("median_only", "missingness_only")
FOLD_TYPES = ("human_source_holdout", "generator_holdout", "ordinary_group_holdout_descriptive")
FIXED12 = ("S", "D", "R", "P", "F", "H", "M", "S+D+R+P", "S+D+R+P+F",
           "S+D+R+P+H", "S+D+R+P+M", "S+D+R+P+F+H+M")
INCREMENTS = (("S+D+R+P", "S+D+R+P+F"), ("S+D+R+P", "S+D+R+P+H"),
              ("S+D+R+P", "S+D+R+P+M"), ("S+D+R+P", "S+D+R+P+F+H+M"))
INDEX = ("model_uid", "model_sha256", "model_jsonl_line", "combination", "feature_mode", "quantity",
         "fold_uid", "fold_index", "fold_type", "heldout_source", "opposite_group_fold",
         "train_id_set_sha256", "test_id_set_sha256", "training_rows", "test_rows")
PRED = ("model_uid", "row_id", "label", "source_group", "group_id", "role", "score", "threshold", "predicted_label")
PAIR = ("model_uid", "human_source", "ai_source", "test_human", "test_ai", "test_human_groups",
        "test_ai_groups", "threshold", "roc_auc", "balanced_accuracy", "ai_sensitivity",
        "human_specificity", "tp", "tn", "fp", "fn")
POOLED = ("model_uid", "test_rows", "test_groups", "threshold", "roc_auc", "balanced_accuracy",
          "ai_sensitivity", "human_specificity", "tp", "tn", "fp", "fn")
SOURCE = ("model_uid", "source_group", "label", "rows", "components", "threshold", "tp", "tn", "fp", "fn",
          "recording_positive_rate", "recording_negative_rate", "recording_ai_sensitivity",
          "recording_human_specificity", "equal_component_positive_rate", "equal_component_negative_rate",
          "equal_component_ai_sensitivity", "equal_component_human_specificity")
RESULT_FILES = frozenset({"fold_registry.jsonl", "omitted_fold_cells.json", "frozen_authorization.json",
                          "run_manifest.json", "fold_models.jsonl", "primary_model_index.csv",
                          "primary_predictions.csv", "primary_pair_metrics.csv", "primary_pooled_metrics.csv",
                          "primary_per_source_endpoints.csv", "diagnostic_model_index.csv",
                          "diagnostic_predictions.csv", "diagnostic_pair_metrics.csv",
                          "diagnostic_pooled_metrics.csv", "diagnostic_per_source_endpoints.csv"})
HEX64 = re.compile(r"[0-9a-f]{64}")
STAGE = "exploratory_v5_development_cv_only"
EXPECTED_REAL = {"valid_folds": 43, "primary_fits": 27305, "primary_prediction_rows": 7633970,
                 "diagnostic_fits": 10922, "diagnostic_prediction_rows": 3053588}
ROOT = Path(__file__).resolve().parent.parent
PARENT_PROTOCOL = ROOT / "EQUAL60_EXPLORATORY_V5_PROTOCOL_EN.md"
PRESENTATION_PROTOCOL = ROOT / "EQUAL60_V5_PRESENTATION_PROTOCOL_EN.md"
AUDITOR = Path(__file__).with_name("audit_equal60_v5_results.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    out = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            out.update(block)
    return out.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(value, dict), "Expected JSON object: " + str(path))
    return value


def strict_json_array(path):
    def pairs(items):
        return _unique_pairs(items, path)

    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError("Nonfinite JSON constant: " + token)))
    require(isinstance(value, list), "Expected JSON array: " + str(path))
    return value


def strict_jsonl(path):
    values = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            require(line.strip(), f"Blank JSONL line {number}: {path}")
            value = json.loads(line, object_pairs_hook=lambda pairs: _unique_pairs(pairs, path),
                               parse_constant=lambda token: (_ for _ in ()).throw(ValueError("Nonfinite JSON constant: " + token)))
            require(isinstance(value, dict), f"Expected JSON object on line {number}: {path}")
            values.append(value)
    require(values, "Empty JSONL: " + str(path))
    return values


def _unique_pairs(pairs, path):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key {key}: {path}")
        result[key] = value
    return result


def read_csv(path, fields):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == tuple(fields), "CSV schema changed: " + str(path))
        rows = list(reader)
    require(rows and all(None not in row and None not in row.values() for row in rows), "Empty or malformed CSV: " + str(path))
    return rows


def finite(row, name):
    try:
        value = float(row[name])
    except (KeyError, ValueError) as exc:
        raise ValueError("Invalid numeric field: " + name) from exc
    require(math.isfinite(value), "Nonfinite numeric field: " + name)
    return value


def integer(row, name):
    text = str(row.get(name, ""))
    require(re.fullmatch(r"0|[1-9][0-9]*", text) is not None, "Invalid nonnegative integer: " + name)
    return int(text)


def verify_publication(results_dir):
    commit_path = results_dir / "COMMIT.json"
    require(commit_path.is_file() and not commit_path.is_symlink(), "Missing final COMMIT marker")
    commit = strict_json(commit_path)
    require(commit.get("status") == "committed" and commit.get("publication") ==
            "exclusive directory reservation; hardlink COMMIT last", "Uncommitted v5 publication")
    require(set(commit.get("files", {})) == RESULT_FILES, "Committed result inventory changed")
    require({p.name for p in results_dir.iterdir()} == RESULT_FILES | {"COMMIT.json"},
            "Unexpected or missing v5 publication artifact")
    hashes = {}
    for name, record in commit["files"].items():
        path = results_dir / name
        require(path.is_file() and not path.is_symlink() and path.resolve() == path,
                "Noncanonical committed result: " + str(path))
        observed = sha(path)
        require(set(record) == {"sha256", "bytes"} and record["sha256"] == observed
                and record["bytes"] == path.stat().st_size, "Committed result hash/size changed: " + name)
        hashes[str(path)] = observed
    hashes[str(commit_path)] = sha(commit_path)
    return commit, hashes


def validate_gate(results_dir, audit_path, audit_sha256, synthetic):
    require(results_dir.is_absolute() and results_dir.is_dir() and results_dir.resolve() == results_dir,
            "Results directory must be canonical and absolute")
    require(audit_path.is_absolute() and audit_path.is_file() and not audit_path.is_symlink()
            and audit_path.resolve() == audit_path, "Audit must be a canonical absolute file")
    commit, hashes = verify_publication(results_dir)
    require(isinstance(audit_sha256, str) and HEX64.fullmatch(audit_sha256)
            and sha(audit_path) == audit_sha256, "Explicit independent audit SHA-256 mismatch")
    audit = strict_json(audit_path)
    manifest = strict_json(results_dir / "run_manifest.json")
    authorization = strict_json(results_dir / "frozen_authorization.json")
    contract = manifest.get("contract")
    require(isinstance(contract, dict) and contract.get("schema_version") == 5
            and contract.get("authorized_stage") == STAGE and contract.get("synthetic_test_only") is synthetic,
            "Frozen v5 contract scope changed")
    require(manifest.get("schema_version") == 5 and manifest.get("stage") == STAGE
            and manifest.get("status") == "completed_exploratory_cv"
            and manifest.get("synthetic_test_only") is synthetic,
            "Run manifest is not a completed v5 evaluation")
    require(authorization.get("status") == "frozen" and authorization.get("authorized_stage") == STAGE
            and authorization.get("contract") == contract and authorization.get("contract_sha256") == digest(contract),
            "Published frozen authorization does not match the run contract")
    review = authorization.get("independent_review", {})
    require(review.get("approved") is True and review.get("reviewer") == "root"
            and isinstance(review.get("reviewed_utc"), str) and review["reviewed_utc"].strip(),
            "Root-frozen contract review is missing")
    run_authorization = manifest.get("authorization")
    require(isinstance(run_authorization, dict)
            and run_authorization.get("status") == "frozen_verified"
            and run_authorization.get("contract_sha256") == digest(contract)
            and isinstance(run_authorization.get("receipt_sha256"), str)
            and HEX64.fullmatch(run_authorization["receipt_sha256"]),
            "Run manifest does not preserve the original frozen-receipt hash")
    accounting = contract.get("accounting")
    require(isinstance(accounting, dict), "Missing v5 accounting")
    if synthetic:
        require(type(contract.get("rows")) is int and 0 < contract["rows"] <= 16,
                "Synthetic fixture must contain 1 to 16 marked rows")
    else:
        require(contract.get("rows") == 2207 and contract.get("label_counts") == {"0": 1311, "1": 896}
                and accounting == EXPECTED_REAL, "Real v5 cohort or accounting changed")
    require(contract.get("families") and tuple(contract.get("combinations", ())) == COMBINATIONS
            and tuple(str(x) for x in contract.get("quantities", ())) == CAPS
            and contract.get("primary_feature_mode") == PRIMARY_MODE
            and tuple(contract.get("all_cap_diagnostics", ())) == DIAGNOSTIC_MODES
            and contract.get("fixed_threshold") == .5 and contract.get("selection") is False
            and contract.get("threshold_tuning") is False and contract.get("full_cohort_refit") is False,
            "Frozen family, cap, mode, threshold, or fit scope changed")
    expected_audit = {"status": "passed", "schema_version": 5, "stage": STAGE,
                      "synthetic_test_only": synthetic, "rows": contract["rows"],
                      "label_counts": contract["label_counts"], "source_counts": contract["source_counts"],
                      "families": FAMILY_COUNTS, "candidates": 127,
                      "caps": [25, 50, 100, 200, "all"], "valid_folds": accounting["valid_folds"],
                      "contract_sha256": digest(contract), "schedule_sha256": contract["schedule_sha256"],
                      "result_commit_sha256": hashes[str(results_dir / "COMMIT.json")],
                      "run_manifest_sha256": hashes[str(results_dir / "run_manifest.json")],
                      "frozen_receipt_sha256": run_authorization["receipt_sha256"],
                      "model_fitting_performed": False, "scorer_predict_called": False,
                      "optimizer_solution_independently_reproduced": False,
                      "training_stationarity_checked": True, "metrics_independently_recomputed": True}
    for key, wanted in expected_audit.items():
        require(audit.get(key) == wanted, "Independent v5 audit gate mismatch: " + key)
    require(audit.get("omitted_fold_cells") == len(strict_json_array(results_dir / "omitted_fold_cells.json"))
            and type(audit.get("checked_binding_count")) is int and audit["checked_binding_count"] > 0
            and isinstance(audit.get("package_commit_sha256"), str)
            and HEX64.fullmatch(audit["package_commit_sha256"]), "Independent audit omission/input binding scope changed")
    expected_file_hashes = {Path(path).name: value for path, value in hashes.items()}
    require(audit.get("result_files_sha256") == expected_file_hashes,
            "Independent audit does not bind the exact committed result files")
    expected_models = {"total": accounting["primary_fits"] + accounting["diagnostic_fits"],
                       "primary": accounting["primary_fits"], "diagnostic": accounting["diagnostic_fits"],
                       "model_hashes_checked": accounting["primary_fits"] + accounting["diagnostic_fits"],
                       "training_transforms_checked": accounting["primary_fits"] + accounting["diagnostic_fits"],
                       "normal_equation_residual_checked": accounting["primary_fits"] + accounting["diagnostic_fits"],
                       "scores_replayed": accounting["primary_prediction_rows"] + accounting["diagnostic_prediction_rows"],
                       "score_threshold_decisions_checked": accounting["primary_prediction_rows"] + accounting["diagnostic_prediction_rows"]}
    models = audit.get("models")
    numerical_fields = {"maximum_normal_equation_absolute_residual", "maximum_absolute_score_replay_error",
                        "minimum_absolute_saved_score_distance_from_threshold"}
    require(isinstance(models, dict) and set(models) == set(expected_models) | numerical_fields
            and all(models.get(key) == value for key, value in expected_models.items())
            and all(isinstance(models[key], (int, float)) and math.isfinite(models[key]) and models[key] >= 0
                    for key in numerical_fields),
            "Independent model replay/transform/stationarity accounting changed")
    streams = audit.get("streams")
    require(streams == manifest.get("stream_counts") and set(streams or {}) == {"primary", "diagnostic"}
            and all(set(streams[category]) == {"model_index", "predictions", "pair_metrics", "pooled_metrics", "per_source_endpoints"}
                    for category in streams), "Independent stream accounting changed")
    metrics = audit.get("metrics")
    expected_metrics = {"primary_pair_rows": streams["primary"]["pair_metrics"],
                        "primary_pooled_rows": streams["primary"]["pooled_metrics"],
                        "primary_per_source_rows": streams["primary"]["per_source_endpoints"],
                        "diagnostic_pair_rows": streams["diagnostic"]["pair_metrics"],
                        "diagnostic_pooled_rows": streams["diagnostic"]["pooled_metrics"],
                        "diagnostic_per_source_rows": streams["diagnostic"]["per_source_endpoints"]}
    require(metrics == expected_metrics, "Independent metric-row accounting changed")
    oof = audit.get("oof_source_endpoints")
    require(isinstance(oof, dict) and set(oof) == {"endpoint_cells", "endpoint_rows_summed_across_cells",
            "endpoint_components_summed_across_cells", "duplicate_oof_ids_within_arm_source",
            "denominator_checks_passed", "canonical_endpoints_sha256"}
            and all(type(oof[key]) is int and oof[key] > 0 for key in
                    ("endpoint_cells", "endpoint_rows_summed_across_cells", "endpoint_components_summed_across_cells"))
            and oof["duplicate_oof_ids_within_arm_source"] == 0
            and oof["denominator_checks_passed"] == oof["endpoint_cells"]
            and isinstance(oof["canonical_endpoints_sha256"], str)
            and HEX64.fullmatch(oof["canonical_endpoints_sha256"]), "Independent OOF source endpoint checks incomplete")
    require(audit.get("aggregation_contract") == {
            "unique_oof_recording_rates_within_arm_source": True, "equal_global_component_rates": True,
            "equal_fold_component_mean": False, "fold_pair_macro_hierarchy": True,
            "pool_across_held_source_arms": False, "pooled_out_of_fold_auc": False},
            "Independent aggregation contract changed")
    require(AUDITOR.is_file() and audit.get("auditor_sha256") == sha(AUDITOR),
            "Independent audit was not produced by the current auditor")
    hashes[str(audit_path)] = sha(audit_path)
    hashes[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    hashes[str(PARENT_PROTOCOL.resolve())] = sha(PARENT_PROTOCOL.resolve())
    hashes[str(PRESENTATION_PROTOCOL.resolve())] = sha(PRESENTATION_PROTOCOL.resolve())
    return audit, manifest, contract, hashes


def arm_key(row):
    return row["fold_type"], row["heldout_source"]


def validate_schedule(contract):
    schedule = contract.get("schedule")
    require(isinstance(schedule, list) and schedule, "Frozen schedule missing")
    by_uid, keys = {}, set()
    for record in schedule:
        require(isinstance(record, dict), "Malformed schedule record")
        cap = str(record.get("quantity"))
        key = (record.get("fold_type"), record.get("heldout_source"), cap, record.get("opposite_group_fold"))
        require(record.get("fold_type") in FOLD_TYPES and cap in CAPS and key not in keys,
                "Duplicate or invalid schedule cell")
        require(record.get("fold_uid") and HEX64.fullmatch(record.get("train_id_set_sha256", ""))
                and HEX64.fullmatch(record.get("test_id_set_sha256", "")), "Missing schedule identity hash")
        require(type(record.get("train", {}).get("rows")) is int and type(record.get("test", {}).get("rows")) is int,
                "Schedule population counts missing")
        keys.add(key)
        require(record["fold_uid"] not in by_uid, "Duplicate fold UID")
        by_uid[record["fold_uid"]] = record
    for fold_type, heldout in {(r["fold_type"], r["heldout_source"]) for r in schedule}:
        for inner in {r["opposite_group_fold"] for r in schedule if arm_key(r) == (fold_type, heldout)}:
            caps = {str(r["quantity"]) for r in schedule if arm_key(r) == (fold_type, heldout)
                    and r["opposite_group_fold"] == inner}
            require(caps == set(CAPS), "A valid fold must retain all five caps")
    return schedule, by_uid


def validate_indexes(results_dir, contract, schedule_by_uid):
    outputs = {}
    identity_refs = {}
    for category, modes in (("primary", (PRIMARY_MODE,)), ("diagnostic", DIAGNOSTIC_MODES)):
        rows = read_csv(results_dir / f"{category}_model_index.csv", INDEX)
        lookup = {}
        for row in rows:
            require(row["model_uid"] not in lookup and row["fold_uid"] in schedule_by_uid,
                    "Duplicate model UID or unknown fold UID")
            schedule = schedule_by_uid[row["fold_uid"]]
            cap = str(schedule["quantity"])
            require(row["combination"] in COMBINATIONS and row["feature_mode"] in modes
                    and row["quantity"] == cap and row["fold_type"] == schedule["fold_type"]
                    and row["heldout_source"] == schedule["heldout_source"]
                    and row["opposite_group_fold"] == str(schedule["opposite_group_fold"])
                    and row["train_id_set_sha256"] == schedule["train_id_set_sha256"]
                    and row["test_id_set_sha256"] == schedule["test_id_set_sha256"]
                    and integer(row, "training_rows") == schedule["train"]["rows"]
                    and integer(row, "test_rows") == schedule["test"]["rows"]
                    and HEX64.fullmatch(row["model_sha256"]), "Model index differs from frozen schedule")
            require((category == "primary") or cap == "all", "Diagnostics must remain all-cap only")
            identity_key = (category, row["feature_mode"], cap, row["fold_uid"])
            refs = (row["train_id_set_sha256"], row["test_id_set_sha256"], row["training_rows"], row["test_rows"])
            require(identity_refs.setdefault(identity_key, refs) == refs,
                    "Train/test identities vary between candidates")
            lookup[row["model_uid"]] = row
        expected = {(combo, mode, str(record["quantity"]), record["fold_uid"])
                    for record in schedule_by_uid.values() if category == "primary" or str(record["quantity"]) == "all"
                    for combo in COMBINATIONS for mode in modes}
        actual = {(r["combination"], r["feature_mode"], r["quantity"], r["fold_uid"]) for r in rows}
        require(actual == expected and len(rows) == len(expected),
                f"Incomplete {category} model grid")
        outputs[category] = (rows, lookup)
    return outputs


def aggregate_predictions(path, index_rows, index_lookup):
    """Stream models; pool exact recording counts and each component's own rate."""
    aggregates = defaultdict(lambda: {"rows": 0, "correct": 0, "components": 0,
                                      "component_rate_sum": 0.0, "folds": set()})
    canonical_tests = {}
    current_uid, current_rows = None, []

    def finish(uid, rows):
        if uid is None:
            return
        model = index_lookup[uid]
        require(len(rows) == integer(model, "test_rows") and len({r["row_id"] for r in rows}) == len(rows),
                "Prediction model does not contain the exact test denominator")
        identities = tuple((r["row_id"], r["label"], r["source_group"], r["group_id"], r["role"]) for r in rows)
        require(canonical_tests.setdefault(model["fold_uid"], identities) == identities,
                "Candidate predictions do not preserve exact fold test identities")
        by_source = defaultdict(list)
        for row in rows:
            require(row["label"] in ("0", "1") and row["role"] == "development"
                    and finite(row, "threshold") == .5 and row["predicted_label"] in ("0", "1")
                    and int(row["predicted_label"]) == int(finite(row, "score") >= .5),
                    "Prediction label, role, or threshold changed")
            by_source[row["source_group"]].append(row)
        for source, source_rows in by_source.items():
            labels = {r["label"] for r in source_rows}
            require(len(labels) == 1, "A source crosses labels")
            label = next(iter(labels))
            groups = defaultdict(list)
            for row in source_rows:
                groups[row["group_id"]].append(int(row["predicted_label"]) == int(label))
            key = (model["fold_type"], model["heldout_source"], model["combination"],
                   model["feature_mode"], model["quantity"], source, label)
            out = aggregates[key]
            out["rows"] += len(source_rows)
            out["correct"] += sum(int(row["predicted_label"]) == int(label) for row in source_rows)
            out["components"] += len(groups)
            out["component_rate_sum"] += math.fsum(sum(values) / len(values) for values in groups.values())
            out["folds"].add(model["fold_uid"])

    expected_uids = [row["model_uid"] for row in index_rows]
    expected_position = 0
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(tuple(reader.fieldnames or ()) == PRED, "Prediction CSV schema changed")
        for row in reader:
            require(None not in row and None not in row.values(), "Malformed prediction row")
            uid = row["model_uid"]
            if uid != current_uid:
                finish(current_uid, current_rows)
                require(expected_position < len(expected_uids) and uid == expected_uids[expected_position],
                        "Prediction models are missing, duplicated, or reordered")
                expected_position += 1
                current_uid, current_rows = uid, []
            current_rows.append(row)
    finish(current_uid, current_rows)
    require(expected_position == len(expected_uids), "Prediction stream ended before every model")

    # Test partitions are identical across candidates/modes and disjoint across folds within each arm and cap.
    seen = defaultdict(set)
    seen_groups = defaultdict(set)
    representative = {}
    for model in index_rows:
        key = (model["fold_type"], model["heldout_source"], model["quantity"], model["fold_uid"])
        representative.setdefault(key, model["fold_uid"])
    for fold_type, heldout, cap, fold_uid in representative:
        by_source = defaultdict(list)
        for row_id, label, source, group, role in canonical_tests[fold_uid]:
            by_source[source].append(row_id)
        for source, ids in by_source.items():
            key = (fold_type, heldout, cap, source)
            require(not (seen[key] & set(ids)), "Out-of-fold recording repeats within one reporting arm")
            seen[key].update(ids)
            groups = {group for row_id, label, row_source, group, role in canonical_tests[fold_uid]
                      if row_source == source}
            require(not (seen_groups[key] & groups), "Global component repeats across folds within one reporting arm")
            seen_groups[key].update(groups)

    result = []
    for key in sorted(aggregates, key=_source_sort_key):
        fold_type, heldout, combo, mode, cap, source, label = key
        value = aggregates[key]
        result.append({"fold_type": fold_type, "heldout_source": heldout, "combination": combo,
                       "feature_mode": mode, "quantity": cap, "source_group": source, "label": int(label),
                       "endpoint": "ai_sensitivity" if label == "1" else "human_specificity",
                       "unique_oof_recordings": value["rows"], "global_components": value["components"],
                       "correct_recordings": value["correct"],
                       "recording_rate": value["correct"] / value["rows"],
                       "equal_component_rate": value["component_rate_sum"] / value["components"],
                       "equal_component_minus_recording_pp": 100 * (value["component_rate_sum"] / value["components"]
                                                                    - value["correct"] / value["rows"]),
                       "valid_folds": len(value["folds"])})
    return result


def _source_sort_key(key):
    fold_type, heldout, combo, mode, cap, source, label = key
    return (FOLD_TYPES.index(fold_type), heldout, COMBINATIONS.index(combo),
            (PRIMARY_MODE, *DIAGNOSTIC_MODES).index(mode), CAPS.index(cap), label, source)


def aggregate_fold_metrics(path, fields, index_lookup, pair):
    rows = read_csv(path, fields)
    by_model = defaultdict(list)
    for row in rows:
        require(row["model_uid"] in index_lookup and finite(row, "threshold") == .5,
                "Metric row has unknown model or changed threshold")
        for metric in ("roc_auc", "balanced_accuracy"):
            require(0 <= finite(row, metric) <= 1, "Two-class metric outside [0,1]")
        if pair:
            require(integer(row, "test_human") > 0 and integer(row, "test_ai") > 0
                    and integer(row, "test_human_groups") > 0 and integer(row, "test_ai_groups") > 0,
                    "Empty source-pair denominator")
        else:
            require(integer(row, "test_rows") == integer(index_lookup[row["model_uid"]], "test_rows")
                    and integer(row, "test_groups") > 0, "Pooled fold denominator changed")
        by_model[row["model_uid"]].append(row)
    require(set(by_model) == set(index_lookup), "Metric table omits or adds model instances")
    folds = []
    for uid, model in index_lookup.items():
        values = by_model[uid]
        if not pair:
            require(len(values) == 1, "Expected one pooled metric row per model")
        folds.append({**{k: model[k] for k in ("fold_uid", "fold_type", "heldout_source", "combination",
                                               "feature_mode", "quantity")},
                      "cells": len(values), "roc_auc": math.fsum(finite(r, "roc_auc") for r in values) / len(values),
                      "balanced_accuracy": math.fsum(finite(r, "balanced_accuracy") for r in values) / len(values)})
    arms = []
    grouped = defaultdict(list)
    for row in folds:
        grouped[(row["fold_type"], row["heldout_source"], row["combination"],
                 row["feature_mode"], row["quantity"])].append(row)
    for key in sorted(grouped, key=_metric_sort_key):
        values = grouped[key]
        arms.append({"summary_level": "held_source_arm", "fold_type": key[0], "heldout_source": key[1],
                     "combination": key[2], "feature_mode": key[3], "quantity": key[4],
                     "held_source_arms": 1, "valid_folds": len(values),
                     "metric_cells": sum(r["cells"] for r in values),
                     "roc_auc": math.fsum(r["roc_auc"] for r in values) / len(values),
                     "balanced_accuracy": math.fsum(r["balanced_accuracy"] for r in values) / len(values)})
    macros = []
    macro_groups = defaultdict(list)
    for row in arms:
        macro_groups[(row["fold_type"], row["combination"], row["feature_mode"], row["quantity"])].append(row)
    for key in sorted(macro_groups, key=lambda k: (FOLD_TYPES.index(k[0]), COMBINATIONS.index(k[1]),
                                                   (PRIMARY_MODE, *DIAGNOSTIC_MODES).index(k[2]), CAPS.index(k[3]))):
        values = macro_groups[key]
        macros.append({"summary_level": "fold_type_macro", "fold_type": key[0], "heldout_source": "__equal_arms__",
                       "combination": key[1], "feature_mode": key[2], "quantity": key[3],
                       "held_source_arms": len(values), "valid_folds": sum(r["valid_folds"] for r in values),
                       "metric_cells": sum(r["metric_cells"] for r in values),
                       "roc_auc": math.fsum(r["roc_auc"] for r in values) / len(values),
                       "balanced_accuracy": math.fsum(r["balanced_accuracy"] for r in values) / len(values)})
    return arms + macros


def _metric_sort_key(key):
    fold_type, heldout, combo, mode, cap = key
    return (FOLD_TYPES.index(fold_type), heldout, COMBINATIONS.index(combo),
            (PRIMARY_MODE, *DIAGNOSTIC_MODES).index(mode), CAPS.index(cap))


def training_ranges(schedule):
    arm_rows = []
    groups = defaultdict(list)
    for record in schedule:
        train = record["train"]
        for source in ("__total__", *sorted(train.get("source_rows", {}))):
            rows = train["rows"] if source == "__total__" else train["source_rows"][source]
            components = len(train["groups"]) if source == "__total__" else train["source_groups"][source]
            groups[(record["fold_type"], record["heldout_source"], str(record["quantity"]), source)].append((rows, components))
    for key in sorted(groups, key=lambda k: (FOLD_TYPES.index(k[0]), k[1], CAPS.index(k[2]), k[3])):
        values = groups[key]
        arm_rows.append({"summary_level": "held_source_arm", "fold_type": key[0], "heldout_source": key[1],
                         "quantity": key[2], "training_source": key[3], "valid_folds": len(values),
                         "min_training_rows": min(v[0] for v in values), "max_training_rows": max(v[0] for v in values),
                         "min_training_groups": min(v[1] for v in values), "max_training_groups": max(v[1] for v in values)})
    macro_groups = defaultdict(list)
    for row in arm_rows:
        macro_groups[(row["fold_type"], row["quantity"], row["training_source"])].append(row)
    macros = []
    for key, values in macro_groups.items():
        macros.append({"summary_level": "fold_type_macro", "fold_type": key[0], "heldout_source": "__all_arms__",
                       "quantity": key[1], "training_source": key[2],
                       "valid_folds": sum(v["valid_folds"] for v in values),
                       "min_training_rows": min(v["min_training_rows"] for v in values),
                       "max_training_rows": max(v["max_training_rows"] for v in values),
                       "min_training_groups": min(v["min_training_groups"] for v in values),
                       "max_training_groups": max(v["max_training_groups"] for v in values)})
    return arm_rows + sorted(macros, key=lambda r: (FOLD_TYPES.index(r["fold_type"]), CAPS.index(r["quantity"]), r["training_source"]))


def fixed_overview(pair_rows):
    return [row for row in pair_rows if row["combination"] in FIXED12]


def validate_presentation_grids(results):
    for category in ("primary", "diagnostic"):
        expected_caps = set(CAPS) if category == "primary" else {"all"}
        expected_modes = (PRIMARY_MODE,) if category == "primary" else DIAGNOSTIC_MODES
        for table_name in ("pair", "pooled"):
            rows = results[category][table_name]
            scopes = defaultdict(list)
            for row in rows:
                scopes[(row["summary_level"], row["fold_type"], row["heldout_source"], row["feature_mode"])].append(row)
            require(scopes, f"Missing {category} {table_name} reporting scopes")
            for key, values in scopes.items():
                require(key[-1] in expected_modes
                        and {(row["combination"], row["quantity"]) for row in values}
                        == set(itertools.product(COMBINATIONS, expected_caps)),
                        f"Incomplete 127-by-cap {category} {table_name} reporting arm")
        source_scopes = defaultdict(list)
        for row in results[category]["source"]:
            source_scopes[(row["fold_type"], row["heldout_source"], row["feature_mode"],
                           row["source_group"], row["label"])].append(row)
        require(source_scopes, f"Missing {category} source reporting scopes")
        for key, values in source_scopes.items():
            require(key[2] in expected_modes
                    and {(row["combination"], row["quantity"]) for row in values}
                    == set(itertools.product(COMBINATIONS, expected_caps)),
                    f"Incomplete 127-by-cap {category} source reporting arm")


def fixed_increments(pair_rows, source_rows, index_rows):
    # Establish exact matching independently of the metric tables.
    refs = {(r["fold_type"], r["heldout_source"], r["feature_mode"], r["quantity"], r["fold_uid"], r["combination"]):
            (r["train_id_set_sha256"], r["test_id_set_sha256"]) for r in index_rows}
    for baseline, added in INCREMENTS:
        for key in list(refs):
            if key[-1] == baseline:
                other = (*key[:-1], added)
                require(other in refs and refs[other] == refs[key], "Fixed increment train/test identities differ")
    identity_digest = digest(sorted((k, v) for k, v in refs.items() if k[-1] in {x for pair in INCREMENTS for x in pair}))
    output = []
    pair_lookup = {(r["summary_level"], r["fold_type"], r["heldout_source"], r["combination"], r["quantity"]): r
                   for r in pair_rows if r["feature_mode"] == PRIMARY_MODE}
    for baseline, added in INCREMENTS:
        for key, left in pair_lookup.items():
            if key[3] != baseline:
                continue
            right = pair_lookup[(key[0], key[1], key[2], added, key[4])]
            for metric in ("roc_auc", "balanced_accuracy"):
                output.append({"scope": "two_class_pair_macro", "summary_level": key[0], "fold_type": key[1],
                               "heldout_source": key[2], "source_group": "__two_class__", "endpoint": metric,
                               "quantity": key[4], "baseline_combination": baseline, "added_combination": added,
                               "baseline_rate": left[metric], "added_rate": right[metric],
                               "added_minus_baseline_pp": 100 * (right[metric] - left[metric]),
                               "direction": _direction(right[metric] - left[metric]),
                               "paired_identity_registry_sha256": identity_digest})
    source_lookup = {(r["fold_type"], r["heldout_source"], r["combination"], r["quantity"], r["source_group"]): r
                     for r in source_rows if r["feature_mode"] == PRIMARY_MODE}
    for baseline, added in INCREMENTS:
        for key, left in source_lookup.items():
            if key[2] != baseline:
                continue
            right = source_lookup[(key[0], key[1], added, key[3], key[4])]
            deltas = {metric: right[metric] - left[metric] for metric in ("recording_rate", "equal_component_rate")}
            for metric in deltas:
                output.append({"scope": "source_endpoint", "summary_level": "held_source_arm",
                               "fold_type": key[0], "heldout_source": key[1], "source_group": key[4],
                               "endpoint": left["endpoint"] + "_" + metric, "quantity": key[3],
                               "baseline_combination": baseline, "added_combination": added,
                               "baseline_rate": left[metric], "added_rate": right[metric],
                               "added_minus_baseline_pp": 100 * deltas[metric],
                               "direction": _direction(deltas[metric]),
                               "paired_identity_registry_sha256": identity_digest})
            output[-2]["paired_direction_comparison"] = _paired_direction(deltas)
            output[-1]["paired_direction_comparison"] = _paired_direction(deltas)
    for row in output:
        row.setdefault("paired_direction_comparison", "not_applicable")
    return output


def _direction(value):
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def _paired_direction(values):
    directions = {_direction(v) for v in values.values()}
    return "same_direction" if len(directions) == 1 else "different_direction"


def write_csv(path, rows):
    require(rows, "Cannot write an empty presentation table")
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_text(primary_pair, source_rows, ranges, synthetic):
    prefix = "Synthetic fixture. Values below are test data only.\n\n" if synthetic else ""
    macro = [r for r in primary_pair if r["summary_level"] == "fold_type_macro"
             and r["combination"] in FIXED12]
    lines = ["# Exploratory v5 presentation tables", "", prefix.rstrip(), "" if prefix else "",
             "Two-class values use the fixed fold-pair macro. Source endpoints pool unique out-of-fold recordings within one held-source arm. Equal-component values give each global group one equal contribution. Human-source, generator, and ordinary group holdouts remain separate.", "",
             "## Fixed 12-candidate macro overview", "",
             "| Fold type | Combination | Cap | AUC | Balanced accuracy |",
             "|---|---|---:|---:|---:|"]
    for row in macro:
        lines.append(f"| {row['fold_type']} | {row['combination']} | {row['quantity']} | {row['roc_auc']:.6f} | {row['balanced_accuracy']:.6f} |")
    lines.extend(["", "## Interpretation limits", "",
             "The cap changes both the training composition and effective regularization. Results remain exploratory reuse. Fold ranges are descriptive. Macro rows average held-source-arm metrics without pooling their predictions. The tables do not calculate pooled out-of-fold AUC.", ""])
    return "\n".join(line for i, line in enumerate(lines) if line or i == 0 or lines[i - 1]) + "\n"


def render_latex(primary_pair, synthetic):
    rows = [r for r in primary_pair if r["summary_level"] == "fold_type_macro" and r["combination"] in FIXED12]
    lines = [r"% Requires booktabs and longtable.",
             r"% Labels: H-source = human source holdout; Generator = generator holdout; Group = ordinary group holdout."]
    if synthetic:
        lines.append(r"% SYNTHETIC FIXTURE. VALUES ARE TEST DATA ONLY.")
    lines.extend([r"\begin{longtable}{@{}llrrrr@{}}",
                  r"\caption{Exploratory v5 fixed-candidate fold-pair macro overview.}\label{tab:equal60-v5-overview}\\",
                  r"\toprule Type & Combination & Cap & Folds & AUC & BA \\ \midrule\endfirsthead",
                  r"\toprule Type & Combination & Cap & Folds & AUC & BA \\ \midrule\endhead",
                  r"\bottomrule\endlastfoot"])
    for row in rows:
        combo = row["combination"].replace("+", r"{+}")
        fold_type = {"human_source_holdout": "H-source", "generator_holdout": "Generator",
                     "ordinary_group_holdout_descriptive": "Group"}[row["fold_type"]]
        lines.append(f"{fold_type} & {combo} & {row['quantity']} & {row['valid_folds']} & {row['roc_auc']:.6f} & {row['balanced_accuracy']:.6f} " + r"\\")
    lines.extend([r"\end{longtable}", ""])
    return "\n".join(lines)


def commit_output(root):
    paths = sorted(root.iterdir())
    require(paths and not (root / "COMMIT.json").exists() and all(p.is_file() and not p.is_symlink() for p in paths),
            "Invalid presentation publication state")
    for path in paths:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    marker = {"status": "committed", "publication": "exclusive directory reservation; hardlink COMMIT last",
              "files": {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in paths}}
    fd, temporary = tempfile.mkstemp(prefix=".COMMIT.", dir=root)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(canonical(marker).decode() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, root / "COMMIT.json")
    finally:
        Path(temporary).unlink(missing_ok=True)
    return marker


def export(results_dir, audit_path, audit_sha256, output_dir, synthetic=False):
    results_dir, audit_path, output_dir = map(Path, (results_dir, audit_path, output_dir))
    require(output_dir.is_absolute() and output_dir.parent.is_dir() and output_dir.parent.resolve() == output_dir.parent
            and not output_dir.exists() and not output_dir.is_symlink(), "New canonical output directory required")
    audit, manifest, contract, hashes = validate_gate(results_dir, audit_path, audit_sha256, synthetic)
    schedule, schedule_by_uid = validate_schedule(contract)
    indexes = validate_indexes(results_dir, contract, schedule_by_uid)
    results = {}
    for category in ("primary", "diagnostic"):
        index_rows, lookup = indexes[category]
        results[category] = {
            "source": aggregate_predictions(results_dir / f"{category}_predictions.csv", index_rows, lookup),
            "pair": aggregate_fold_metrics(results_dir / f"{category}_pair_metrics.csv", PAIR, lookup, True),
            "pooled": aggregate_fold_metrics(results_dir / f"{category}_pooled_metrics.csv", POOLED, lookup, False),
        }
    validate_presentation_grids(results)
    overview = fixed_overview(results["primary"]["pair"])
    increments = fixed_increments(results["primary"]["pair"], results["primary"]["source"], indexes["primary"][0])
    ranges = training_ranges(schedule)
    output_dir.mkdir()
    try:
        write_csv(output_dir / "primary_pair_macro_all_arms.csv", results["primary"]["pair"])
        write_csv(output_dir / "primary_pooled_fold_companion_all_arms.csv", results["primary"]["pooled"])
        write_csv(output_dir / "primary_source_endpoints_all_arms.csv", results["primary"]["source"])
        write_csv(output_dir / "diagnostic_all_cap_pair_macro_all_arms.csv", results["diagnostic"]["pair"])
        write_csv(output_dir / "diagnostic_all_cap_pooled_fold_companion_all_arms.csv", results["diagnostic"]["pooled"])
        write_csv(output_dir / "diagnostic_all_cap_source_endpoints_all_arms.csv", results["diagnostic"]["source"])
        write_csv(output_dir / "primary_fixed12_overview.csv", overview)
        write_csv(output_dir / "primary_fixed_increments.csv", increments)
        write_csv(output_dir / "training_row_group_ranges.csv", ranges)
        (output_dir / "EQUAL60_V5_PRESENTATION_TABLES_EN.md").write_text(
            render_text(results["primary"]["pair"], results["primary"]["source"], ranges, synthetic))
        (output_dir / "equal60_v5_presentation_tables_en.tex").write_text(render_latex(results["primary"]["pair"], synthetic))
        current_hashes = {str(name): sha(name) for name in map(Path, hashes)}
        require(current_hashes == hashes, "Audited input changed during presentation aggregation")
        outputs_before_receipt = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                                  for path in sorted(output_dir.iterdir())}
        receipt = {"schema_version": 1, "status": "synthetic_fixture_complete" if synthetic else "complete",
                   "synthetic_test_only": synthetic, "contract_sha256": digest(contract),
                   "input_commit_sha256": hashes[str(results_dir / "COMMIT.json")],
                   "independent_audit": {"path": str(audit_path), "sha256": hashes[str(audit_path)]},
                   "scope": "Cross-fold descriptive aggregation under the frozen exploratory-v5 estimands. No fitting, scoring, threshold changes, source-arm pooling, candidate selection, or statistical inference.",
                   "primary_family_cap_cells_per_reporting_arm": 635,
                   "diagnostic_family_cells_per_reporting_arm_and_mode": 127,
                   "fixed_overview_candidates": list(FIXED12),
                   "fixed_matched_increments": [list(x) for x in INCREMENTS],
                   "source_endpoint_estimator": "correct unique out-of-fold recordings divided by unique recordings; equal-component is the mean of each global-group out-of-fold correct rate",
                   "two_class_estimator": "mean source pairs within fold, mean valid folds within held-source arm, equal mean arms within fold type",
                   "inputs_sha256": dict(sorted(hashes.items())), "exporter_sha256": sha(Path(__file__).resolve()),
                   "outputs_before_receipt_sha256": outputs_before_receipt}
        (output_dir / "presentation_receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
        require({str(name): sha(name) for name in map(Path, hashes)} == hashes,
                "Audited input changed before COMMIT publication")
        commit_output(output_dir)
    except BaseException:
        # Exclusive publication leaves a visible uncommitted orphan for audit.
        raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic-test-only", action="store_true")
    args = parser.parse_args(argv)
    require(all(path.is_absolute() for path in (args.results_dir, args.audit, args.output_dir)), "All paths must be absolute")
    print(json.dumps(export(args.results_dir, args.audit, args.audit_sha256, args.output_dir, args.synthetic_test_only),
                     sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
