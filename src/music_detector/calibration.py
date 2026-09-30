"""A new retrospective deployment experiment on development data only.

Freeze metadata partition before fitting. Whole dependency closures remain in
one partition, including relationships through records outside development.
This is neither a replay of historical estimates nor external validation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import platform
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import scipy

from .scoring import FAMILIES, canonical_hash

SCHEMA = "music-detector-deployment-v1"
SEED = "music-detector-retrospective-20260930-v1"
FAMILY_ORDER = tuple(name for name in FAMILIES if name != "BC")
PARTITIONS = ("train", "calibration", "test")
FRACTIONS = dict(zip(PARTITIONS, (0.6, 0.2, 0.2)))
PLANNER_SHA = "ff6965b3f9c834ee9d39bbf0f994595f44fa2e261d9c7c6b60a153cb9f72ce24"
SCOPE = "new_retrospective_internal_development_experiment_not_external_validation"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def id_hash(values):
    return hashlib.sha256("".join(f"{v}\n" for v in sorted(set(values))).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load_dependencies(planner_path):
    require(sha_file(planner_path) == PLANNER_SHA, "Historical dependency graph implementation changed")
    spec = importlib.util.spec_from_file_location("_frozen_native30_dependencies", planner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Dependencies


def partition_metadata(rows, screen, dependencies):
    """Outcome-blind deterministic allocation of full graph closures.

    First give every source one closure in each partition, rare sources first.
    Remaining closures minimize squared imbalance in source closure counts and
    source recording counts, normalized by each partition's target. A closure
    containing multiple generators is indivisible. Seeded hashes break ties.
    """
    require(rows and len({r["id"] for r in rows}) == len(rows), "Unique development IDs required")
    require(all(r["role"] == "development" and str(r["label"]) in {"0", "1"} for r in rows),
            "Only labeled development rows may enter the new experiment")
    graph = dependencies(screen, rows)
    excluded = [r for r in rows if graph.row_roots[r["id"]] in graph.protected_roots]
    groups = defaultdict(list)
    for row in rows:
        if graph.row_roots[row["id"]] not in graph.protected_roots:
            groups[graph.row_roots[row["id"]]].append(row)
    sources = sorted({r["source_group"] for members in groups.values() for r in members})
    source_labels = {source: {str(r["label"]) for r in rows if r["source_group"] == source} for source in sources}
    require(all(len(labels) == 1 for labels in source_labels.values()), "A source must have one immutable label")
    counts = {key: Counter(r["source_group"] for r in members) for key, members in groups.items()}
    totals_rows = Counter(r["source_group"] for members in groups.values() for r in members)
    totals_groups = Counter(source for count in counts.values() for source in count)
    require(all(totals_groups[s] >= 3 for s in sources), "Every source needs at least three independent closures")
    assigned = {name: [] for name in PARTITIONS}
    group_counts = {name: Counter() for name in PARTITIONS}
    row_counts = {name: Counter() for name in PARTITIONS}
    remaining = set(groups)

    def ordering(group):
        token = json.dumps(group, separators=(",", ":"))
        return hashlib.sha256(f"{SEED}|{token}".encode()).hexdigest()

    def cost(group, part):
        value = 0.0
        for source, n in counts[group].items():
            for current, total, increment in ((group_counts[part][source], totals_groups[source], 1),
                                              (row_counts[part][source], totals_rows[source], n)):
                target = total * FRACTIONS[part]
                value += ((current + increment - target) ** 2 - (current - target) ** 2) / target
        return value

    def assign(group, part):
        remaining.remove(group)
        assigned[part].append(group)
        group_counts[part].update(counts[group].keys())
        row_counts[part].update(counts[group])

    # Coverage anchors are based only on metadata, not features or outcomes.
    for source in sorted(sources, key=lambda source: (totals_groups[source], source)):
        for part in PARTITIONS:
            if group_counts[part][source]:
                continue
            candidates = [g for g in remaining if source in counts[g]]
            require(candidates, "Cannot give each partition source coverage without breaking dependency groups")
            group = min(candidates, key=lambda g: (cost(g, part), ordering(g)))
            assign(group, part)
    for group in sorted(remaining, key=lambda g: (-sum(1 / totals_groups[s] for s in counts[g]),
                                                 -len(groups[g]), ordering(g))):
        part = min(PARTITIONS, key=lambda p: (cost(group, p), PARTITIONS.index(p)))
        assign(group, part)
    split = {part: sorted([r for group in assigned[part] for r in groups[group]], key=lambda r: r["id"])
             for part in PARTITIONS}
    for first, second in itertools.combinations(PARTITIONS, 2):
        for key in ("id", "group_id", "component_id"):
            require({r[key] for r in split[first]}.isdisjoint(r[key] for r in split[second]), "Partition leakage: " + key)
        require(graph.roots(split[first]).isdisjoint(graph.roots(split[second])), "Transitive dependency leakage")
    for part, current in split.items():
        require({str(r["label"]) for r in current} == {"0", "1"}, part + " is missing a class")
        require({r["source_group"] for r in current} == set(sources), part + " is missing a source")
    return split, excluded, graph


def split_record(rows, graph):
    return {"rows": len(rows), "ids": [r["id"] for r in rows], "id_set_sha256": id_hash(r["id"] for r in rows),
        "group_set_sha256": id_hash(r["group_id"] for r in rows),
        "component_set_sha256": id_hash(r["component_id"] for r in rows),
        "dependency_roots_sha256": canonical_hash(sorted(graph.roots(rows))),
        "dependency_closures": len(graph.roots(rows)),
        "class_counts": dict(Counter(str(r["label"]) for r in rows)),
        "source_counts": dict(sorted(Counter(r["source_group"] for r in rows).items()))}


def freeze_protocol(metadata_path, features_path, screen_path, planner_path, output):
    """Read metadata and hash feature bytes, then persist protocol before fit."""
    upstream = {}
    for label, path in (("metadata", metadata_path), ("features", features_path), ("screen", screen_path)):
        path = Path(path)
        commit_path = path.parent / "COMMIT.json"
        commit = json.loads(commit_path.read_text())
        binding = commit["products"][path.name]
        require(binding["sha256"] == sha_file(path) and binding["bytes"] == path.stat().st_size,
                "Historical package product binding mismatch: " + label)
        upstream[label] = {"commit_sha256": sha_file(commit_path), "product_sha256": binding["sha256"]}
    metadata = json.loads(Path(metadata_path).read_text())
    screen = json.loads(Path(screen_path).read_text())
    split, excluded, graph = partition_metadata(metadata, screen, load_dependencies(planner_path))
    combinations = ["+".join(c) for n in range(1, 8) for c in itertools.combinations(FAMILY_ORDER, n)]
    protocol = {
        "schema_version": "music-detector-calibration-protocol-v1", "scope": SCOPE,
        "frozen_before_fitting": True, "seed": SEED, "target_partition_fractions": FRACTIONS,
        "partition_algorithm": "whole-full-graph-closure; source coverage anchors rarest first; greedy source closure/row squared target imbalance; seeded SHA256 ties",
        "sources": {"metadata": {"sha256": sha_file(metadata_path), "file": Path(metadata_path).name},
                    "features": {"sha256": sha_file(features_path), "file": Path(features_path).name},
                    "screen": {"sha256": sha_file(screen_path), "file": Path(screen_path).name}},
        "graph_implementation_sha256": PLANNER_SHA,
        "verified_upstream_package_bindings": upstream,
        "builder_sha256": sha_file(__file__),
        "scoring_sha256": sha_file(Path(__file__).with_name("scoring.py")),
        "split": {part: split_record(current, graph) for part, current in split.items()},
        "excluded_protected_ids": sorted(r["id"] for r in excluded), "input_development_rows": len(metadata),
        "all_combinations": combinations, "families": {key: FAMILIES[key] for key in FAMILY_ORDER},
        "classifier": {"kind": "weighted_ridge_identity", "ridge": 10.0, "intercept_penalized": False,
            "feature_mode": "values_plus_missing", "imputation": "train_only_column_median_or_zero_if_all_missing",
            "scaling": "train_only_weighted_mean_population_std; scale_below_1e-8_replaced_by_one",
            "weighting": "equal classes; sources within class; groups within source; rows within group; sum=n"},
        "calibration": {"kind": "weighted_monotonic_platt", "fit_partition": "calibration",
            "score_standardization": "calibration weighted mean and std; std<1e-8 replaced by1",
            "objective": "weighted_mean_logloss + 0.5e-6*slope_squared",
            "slope_bounds": [0.0, 10000.0], "intercept_bounds": [-50.0, 50.0],
            "optimizer": "L-BFGS-B", "initial": [1.0, 0.0], "maxiter": 1000, "ftol": 1e-12, "gtol": 1e-9,
            "minimum_groups_per_class_per_partition": 2,
            "unsupported_if": "partition class/group coverage fails; nonfinite fit; optimizer failure"},
        "evaluation": {"partition": "test", "never_used_for_fit_or_calibration": True,
            "metrics": ["Brier", "logloss", "ECE_10_equal_width_bins", "AUC", "balanced_accuracy", "source_counts"],
            "both_recording_and_equal_class_source_group_weighted": True,
            "probability_threshold": 0.5, "historical_style_raw_threshold": 0.5,
            "select_winner_or_reject_based_on_test_metric": False, "locked_labels_used": False},
        "probability_interpretation": "estimate for balanced class/source/group reference mixture; unknown upload prevalence and domain shift are not calibrated",
        "BC_is_research_only_and_excluded": True, "historical_reproduction_claimed": False,
        "external_validation_claimed": False,
    }
    write_new(output, protocol)
    return protocol


def sample_weights(rows):
    result = np.zeros(len(rows), dtype=np.float64)
    for label in ("0", "1"):
        sources = sorted({r["source_group"] for r in rows if str(r["label"]) == label})
        require(sources, "Both classes required")
        for source in sources:
            groups = defaultdict(list)
            for index, row in enumerate(rows):
                if str(row["label"]) == label and row["source_group"] == source:
                    groups[row["group_id"]].append(index)
            for indices in groups.values():
                result[indices] = 0.5 / (len(sources) * len(groups) * len(indices))
    return result * (len(rows) / result.sum())


def fit_ridge(rows, matrix, columns):
    """Numerically the archived ridge10 train-only transform/normal equations."""
    # The historical Pandas table presents homogeneous feature columns in
    # Fortran order; preserve reduction order as well as the equations.
    matrix = np.asfortranarray(matrix, dtype=np.float64)
    missing = ~np.isfinite(matrix)
    medians = np.asarray([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0 for c in matrix.T])
    imputed = np.where(missing, medians, matrix)
    values = np.concatenate((imputed, missing.astype(float)), axis=1)
    weights = sample_weights(rows)
    mean = np.average(values, axis=0, weights=weights)
    scale = np.sqrt(np.average((values - mean) ** 2, axis=0, weights=weights))
    scale[scale < 1e-8] = 1.0
    design = np.column_stack((np.ones(len(rows)), (values - mean) / scale))
    root_weights = np.sqrt(weights)
    weighted = design * root_weights[:, None]
    penalty = np.eye(design.shape[1]) * 10.0
    penalty[0, 0] = 0.0
    labels = np.asarray([int(row["label"]) for row in rows])
    coefficients = np.linalg.solve(weighted.T @ weighted + penalty, weighted.T @ (labels * root_weights))
    require(np.isfinite(coefficients).all(), "Nonfinite ridge coefficients")
    return {"columns": columns, "medians": medians.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
        "coefficients_with_intercept": coefficients.tolist(), "ridge": 10.0, "feature_mode": "values_plus_missing",
        "training_rows": len(rows), "missing_fractions": {column: float(missing[:, i].mean()) for i, column in enumerate(columns)}}


def ridge_scores(matrix, model):
    missing = ~np.isfinite(matrix)
    values = np.concatenate((np.where(missing, np.asarray(model["medians"]), matrix), missing.astype(float)), axis=1)
    standardized = (values - np.asarray(model["mean"])) / np.asarray(model["scale"])
    beta = np.asarray(model["coefficients_with_intercept"])
    return beta[0] + standardized @ beta[1:]


def fit_platt(scores, rows):
    weights = sample_weights(rows)
    weights /= weights.sum()
    labels = np.asarray([int(row["label"]) for row in rows])
    center = float(np.average(scores, weights=weights))
    scale = float(np.sqrt(np.average((scores - center) ** 2, weights=weights)))
    if scale < 1e-8:
        scale = 1.0
    z = (scores - center) / scale

    def objective(parameters):
        slope, intercept = parameters
        linear = slope * z + intercept
        loss = float(np.sum(weights * (np.logaddexp(0, linear) - labels * linear)) + 0.5e-6 * slope ** 2)
        residual = weights * (expit(linear) - labels)
        return loss, np.array([np.sum(residual * z) + 1e-6 * slope, np.sum(residual)])

    fitted = minimize(objective, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
        bounds=((0, 10000), (-50, 50)), options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-9})
    require(fitted.success and np.isfinite(fitted.x).all(), "Platt optimizer failed: " + str(fitted.message))
    return {"method": "weighted_monotonic_platt", "score_center": center, "score_scale": scale,
        "slope": float(fitted.x[0]), "intercept": float(fitted.x[1]), "fit_rows": len(rows),
        "optimizer_success": True, "optimizer_iterations": int(fitted.nit), "fit_partition": "calibration"}


def probabilities(scores, calibration):
    return expit(calibration["slope"] * (scores - calibration["score_center"]) / calibration["score_scale"] + calibration["intercept"])


def binary_metrics(labels, probability, weights=None):
    """Recording or explicitly weighted metrics; AUC handles tied scores."""
    labels, probability = np.asarray(labels, int), np.asarray(probability, float)
    require(set(labels.tolist()) == {0, 1} and np.isfinite(probability).all() and
            np.all((probability >= 0) & (probability <= 1)), "Two finite probability classes required")
    w = np.ones(len(labels), float) if weights is None else np.asarray(weights, float).copy()
    require(w.shape == labels.shape and np.isfinite(w).all() and np.all(w > 0), "Positive metric weights required")
    w /= w.sum()
    brier = float(np.sum(w * (probability - labels) ** 2))
    p = np.clip(probability, 1e-15, 1 - 1e-15)
    logloss = float(-np.sum(w * (labels * np.log(p) + (1 - labels) * np.log1p(-p))))
    ece, bins = 0.0, []
    bucket = np.minimum((probability * 10).astype(int), 9)
    for index in range(10):
        use = bucket == index
        mass = float(w[use].sum())
        if not mass:
            bins.append({"bin": index, "rows": 0, "weight": 0.0, "mean_probability": None, "positive_rate": None})
            continue
        confidence = float(np.average(probability[use], weights=w[use]))
        positive = float(np.average(labels[use], weights=w[use]))
        ece += mass * abs(confidence - positive)
        bins.append({"bin": index, "rows": int(use.sum()), "weight": mass,
                     "mean_probability": confidence, "positive_rate": positive})
    # Weighted Mann-Whitney rank statistic, processing ties together.
    order = np.argsort(probability, kind="stable")
    negative_before, numerator, start = 0.0, 0.0, 0
    while start < len(order):
        end = start + 1
        while end < len(order) and probability[order[end]] == probability[order[start]]:
            end += 1
        ids = order[start:end]
        negatives, positives = float(w[ids][labels[ids] == 0].sum()), float(w[ids][labels[ids] == 1].sum())
        numerator += positives * (negative_before + 0.5 * negatives)
        negative_before += negatives
        start = end
    auc = numerator / (float(w[labels == 0].sum()) * float(w[labels == 1].sum()))
    decision = probability >= 0.5
    human = float(np.average(~decision[labels == 0], weights=w[labels == 0]))
    ai = float(np.average(decision[labels == 1], weights=w[labels == 1]))
    return {"rows": len(labels), "brier_score": brier, "logloss": logloss, "ece": float(ece),
        "auc": float(auc), "balanced_accuracy": (human + ai) / 2,
        "human_specificity": human, "ai_sensitivity": ai, "reliability_bins": bins}


def train_deployment(protocol_path, metadata_path, features_path, screen_path, planner_path,
                     bundle_path, report_dir):
    """Fit frozen 127 combinations; never read locked feature/label files."""
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    require(protocol["schema_version"] == "music-detector-calibration-protocol-v1" and protocol["frozen_before_fitting"] is True,
            "Freeze a protocol before fitting")
    require(protocol["builder_sha256"] == sha_file(__file__) and
            protocol["scoring_sha256"] == sha_file(Path(__file__).with_name("scoring.py")), "Code changed after protocol freeze")
    for key, path in (("metadata", metadata_path), ("features", features_path), ("screen", screen_path)):
        require(protocol["sources"][key]["sha256"] == sha_file(path), "Frozen input changed: " + key)
    rows = json.loads(Path(metadata_path).read_text())
    screen = json.loads(Path(screen_path).read_text())
    split, excluded, graph = partition_metadata(rows, screen, load_dependencies(planner_path))
    require(protocol["split"] == {part: split_record(current, graph) for part, current in split.items()}, "Partition replay mismatch")
    # Feature values are read only after freeze and all input/split hashes pass.
    features = json.loads(Path(features_path).read_text())
    lookup = {row["id"]: row for row in features}
    require(len(lookup) == len(features) and set(lookup) == {row["id"] for row in rows}, "Feature/development ID mismatch")
    allowed = {name for names in FAMILIES.values() for name in names}
    require(all(set(row) == {"id", *allowed} and all(value is None or
        (type(value) in (float, int) and math.isfinite(value)) for key, value in row.items() if key != "id") for row in features),
        "Expected finite-or-null frozen 55-feature package")
    for part, current in split.items():
        require(all(len({row["group_id"] for row in current if str(row["label"]) == label}) >= 2 for label in ("0", "1")),
                "Insufficient independent class groups in " + part)
    protocol_hash = canonical_hash(protocol)
    records, models = {}, {}
    for combination in protocol["all_combinations"]:
        selected = combination.split("+")
        columns = [column for family in selected for column in FAMILIES[family]]
        matrices = {part: np.asarray([[lookup[row["id"]][column] for column in columns] for row in current], float)
                    for part, current in split.items()}
        try:
            model = fit_ridge(split["train"], matrices["train"], columns)
            calibrator = fit_platt(ridge_scores(matrices["calibration"], model), split["calibration"])
            test_scores = ridge_scores(matrices["test"], model)
            test_probability = probabilities(test_scores, calibrator)
            labels = [int(row["label"]) for row in split["test"]]
            weights = sample_weights(split["test"])
            weighted = binary_metrics(labels, test_probability, weights)
            unweighted = binary_metrics(labels, test_probability)
            raw_decision = (test_scores >= 0.5).astype(float)
            historical_style = binary_metrics(labels, raw_decision, weights)
            new_model = {"schema_version": "music-detector-deployment-model-v1", "families": selected,
                "status": "supported", "scope": SCOPE, "protocol_sha256": protocol_hash,
                "training_id_set_sha256": protocol["split"]["train"]["id_set_sha256"],
                "calibration_id_set_sha256": protocol["split"]["calibration"]["id_set_sha256"],
                "test_id_set_sha256": protocol["split"]["test"]["id_set_sha256"],
                "ridge_model": model, "calibration": calibrator,
                "calibration_status": "fitted_dev_calibration_evaluated_untouched_internal_dev_test",
                "probability_threshold": 0.5, "historical_style_raw_threshold": 0.5,
                "internal_test_metrics": {key: val for key, val in weighted.items() if key != "reliability_bins"},
                "probability_interpretation": protocol["probability_interpretation"],
                "missingness_is_model_input": True, "external_validation": False}
            models[combination] = {"payload": new_model, "sha256": canonical_hash(new_model)}
            records[combination] = {"status": "supported", "model_sha256": canonical_hash(new_model),
                "weighted": weighted, "recording_weighted": unweighted,
                "raw_threshold_0_5_balanced_accuracy": historical_style["balanced_accuracy"],
                "test_prediction_sha256": canonical_hash([{"id": row["id"], "raw_score": float(score),
                    "probability": float(probability), "label": int(row["label"])}
                    for row, score, probability in zip(split["test"], test_scores, test_probability)])}
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            records[combination] = {"status": "unsupported", "reason": str(error)}
        if len(records) % 16 == 0:
            print(f"New development experiment: {len(records)}/127 combinations", flush=True)
    report = {"scope": SCOPE, "protocol_sha256": protocol_hash, "protocol_file_sha256": sha_file(protocol_path),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "split": {part: {k: v for k, v in summary.items() if k != "ids"} for part, summary in protocol["split"].items()},
        "combinations": records, "supported_combinations": len(models), "intended_combinations": 127,
        "locked_rows_used": 0, "BC_probability_models": 0, "winner_selection": False,
        "warning": "Internal probabilities estimate a balanced reference mixture; source shift and unknown generators remain unvalidated."}
    bundle = {"schema_version": SCHEMA, "scope": SCOPE, "protocol_sha256": protocol_hash,
        "report_sha256": canonical_hash(report), "models": models,
        "unsupported": {key: record for key, record in records.items() if record["status"] != "supported"}}
    write_new(Path(report_dir) / "internal_test_report.json", report)
    write_new(bundle_path, bundle)
    write_new(Path(report_dir) / "ACCEPTANCE.json", {"scope": SCOPE,
        "protocol_file_sha256": sha_file(protocol_path), "bundle_file_sha256": sha_file(bundle_path),
        "report_file_sha256": sha_file(Path(report_dir) / "internal_test_report.json"),
        "supported": len(models), "locked_labels_used": False, "BC_excluded": True})
    return bundle, report


def predict_deployment(bundle: dict, features: dict, families: list[str]) -> dict:
    """Score a new schema model; calibrated p>=.5 is the deployment decision."""
    require(isinstance(bundle, dict) and bundle.get("schema_version") == SCHEMA and bundle.get("scope") == SCOPE,
            "Unsupported deployment bundle")
    require(isinstance(families, list) and families and len(set(families)) == len(families)
            and all(f in FAMILY_ORDER for f in families), "Select admitted families; BC has no probability model")
    selected = [family for family in FAMILY_ORDER if family in families]
    key = "+".join(selected)
    require(key in bundle["models"], "Selected combination is unsupported")
    envelope = bundle["models"][key]
    payload = envelope["payload"]
    require(envelope["sha256"] == canonical_hash(payload) and payload["families"] == selected and
            payload["protocol_sha256"] == bundle["protocol_sha256"] and payload["scope"] == SCOPE,
            "Deployment model identity mismatch")
    require(payload.get("schema_version") == "music-detector-deployment-model-v1"
            and payload.get("status") == "supported" and payload.get("external_validation") is False,
            "Unsupported deployment model scope")
    model, calibration = payload["ridge_model"], payload["calibration"]
    columns = [column for family in selected for column in FAMILIES[family]]
    require(model["columns"] == columns and set(features) == set(columns), "Features must exactly match selected-family columns")
    require(model["ridge"] == 10.0 and model["feature_mode"] == "values_plus_missing", "Frozen estimator changed")
    require(all(value is None or type(value) in (float, int) and math.isfinite(value) for value in features.values()),
            "Expected finite values or explicit null missing measurements")
    matrix = np.asarray([[features[column] for column in columns]], float)
    missing = ~np.isfinite(matrix[0])
    n = len(columns)
    for field, count in (("medians", n), ("mean", 2*n), ("scale", 2*n), ("coefficients_with_intercept", 2*n+1)):
        require(len(model[field]) == count and np.isfinite(np.asarray(model[field], float)).all(), "Malformed model parameter: " + field)
    require(np.all(np.asarray(model["scale"]) > 0), "Nonpositive model scale")
    require(calibration["method"] == "weighted_monotonic_platt" and calibration["fit_partition"] == "calibration"
            and calibration["optimizer_success"] is True and calibration["score_scale"] > 0 and calibration["slope"] >= 0
            and all(math.isfinite(calibration[k]) for k in ("score_center", "score_scale", "slope", "intercept")),
            "Invalid deployment calibration")
    raw = float(ridge_scores(matrix, model)[0])
    probability = float(probabilities(np.asarray([raw]), calibration)[0])
    require(math.isfinite(raw) and math.isfinite(probability), "Nonfinite deployment score")
    values = np.concatenate((np.where(missing, np.asarray(model["medians"]), matrix[0]), missing.astype(float)))
    z = (values - np.asarray(model["mean"])) / np.asarray(model["scale"])
    terms = z * np.asarray(model["coefficients_with_intercept"])[1:]
    contributions = []
    for index, name in enumerate(columns):
        family = next(f for f in selected if name in FAMILIES[f])
        contributions.append({"feature": name, "family": family, "value_contribution": float(terms[index]),
            "missingness_contribution": float(terms[n+index]), "total": float(terms[index] + terms[n+index]),
            "missing": bool(missing[index])})
    return {"probability": probability, "ai_probability": probability, "raw_score": raw,
        "decision": "likely_ai" if probability >= 0.5 else "likely_human",
        "decision_threshold": 0.5, "decision_scale": "new_deployment_calibrated_probability",
        "historical_style_raw_decision": int(raw >= 0.5), "historical_style_raw_threshold": 0.5,
        "calibration_status": payload["calibration_status"], "scope": SCOPE,
        "families": selected, "model_sha256": envelope["sha256"], "protocol_sha256": bundle["protocol_sha256"],
        "intercept": model["coefficients_with_intercept"][0], "contributions": contributions,
        "family_contributions": {family: math.fsum(item["total"] for item in contributions if item["family"] == family) for family in selected},
        "missing_features": [column for column, absent in zip(columns, missing) if absent],
        "internal_test_metrics": payload["internal_test_metrics"],
        "probability_interpretation": payload["probability_interpretation"],
        "explanation_scope": "contributions explain raw decision score, not causal authorship or percentage probability shares"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "train"))
    for name in ("metadata", "features", "screen", "planner", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        result = freeze_protocol(args.metadata, args.features, args.screen, args.planner, args.protocol)
        print(json.dumps({"protocol_sha256": canonical_hash(result), "split": {p: len(s["ids"]) for p, s in result["split"].items()}}))
    else:
        require(args.bundle and args.report_dir, "Training requires --bundle and --report-dir")
        bundle, _ = train_deployment(args.protocol, args.metadata, args.features, args.screen, args.planner, args.bundle, args.report_dir)
        print(json.dumps({"supported": len(bundle["models"]), "scope": bundle["scope"]}))


if __name__ == "__main__":
    main()
