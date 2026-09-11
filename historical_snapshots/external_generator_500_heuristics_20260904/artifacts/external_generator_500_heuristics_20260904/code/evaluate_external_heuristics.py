#!/usr/bin/env python3
"""Fit frozen heuristic scores on old development data and evaluate AIME only once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from workflow_features import DYNAMICS_FEATURES, RHYTHM_FEATURES, STRUCTURE_FEATURES


SEED = 20260904
N_BOOT = 2_000
RIDGE = 10.0
THRESHOLD = 0.5
PRIMARY = [
    "MusicGen Small", "MusicGen Medium", "MusicGen Large",
    "AudioLDM 2 Large", "AudioLDM 2 Music", "Mustango",
]
SECONDARY = ["Udio", "Riffusion", "Stable Audio v1", "Stable Audio v2"]
GENERATORS = PRIMARY + SECONDARY


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    if not n0 or not n1:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    pred = np.asarray(scores, dtype=float) >= THRESHOLD
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1_ai = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_human = 2 * tn / max(2 * tn + fp + fn, 1)
    return {
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "roc_auc": auc(labels, scores),
        "ai_sensitivity": sensitivity,
        "human_specificity": specificity,
        "macro_f1": 0.5 * (f1_ai + f1_human),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def fit_model(table: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    human = table[table.label == 0]
    ai = table[table.label == 1]
    if len(human) != 400 or len(ai) != 1200:
        raise RuntimeError(f"Unexpected training balance: Human={len(human)}, AI={len(ai)}")
    balanced_human = pd.concat([human, human, human], ignore_index=True)
    train = pd.concat([balanced_human, ai], ignore_index=True)
    x = train[columns].to_numpy(float)
    y = train.label.to_numpy(int)
    missing = ~np.isfinite(x)
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x = np.where(missing, medians, x)
    x = np.concatenate((x, missing.astype(float)), axis=1)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    params = {
        "columns": columns,
        "medians": medians.tolist(),
        "mean_with_missing_indicators": mean.tolist(),
        "scale_with_missing_indicators": scale.tolist(),
        "weights_with_intercept": weights.tolist(),
        "ridge_penalty": RIDGE,
        "threshold": THRESHOLD,
        "n_train_human_unique": len(human),
        "n_train_human_effective": len(balanced_human),
        "n_train_ai": len(ai),
    }
    return weights, params


def predict(table: pd.DataFrame, params: dict[str, object]) -> np.ndarray:
    columns = list(params["columns"])
    x = table[columns].to_numpy(float)
    missing = ~np.isfinite(x)
    medians = np.asarray(params["medians"], dtype=float)
    x = np.where(missing, medians, x)
    x = np.concatenate((x, missing.astype(float)), axis=1)
    mean = np.asarray(params["mean_with_missing_indicators"], dtype=float)
    scale = np.asarray(params["scale_with_missing_indicators"], dtype=float)
    weights = np.asarray(params["weights_with_intercept"], dtype=float)
    z = (x - mean) / scale
    return np.column_stack((np.ones(len(z)), z)) @ weights


def scenario_table(test: pd.DataFrame, generators: list[str], strict: bool = False) -> pd.DataFrame:
    subset = test[(test.label == 0) | test.generator.isin(generators)].copy()
    if strict:
        touched = set(subset.loc[subset.prior_external_exact_track_overlap == 1, "condition_id"])
        subset = subset[~subset.condition_id.isin(touched)].copy()
    expected_ai_per_condition = len(generators)
    counts = subset.groupby("condition_id").label.agg(["count", "sum"])
    good = counts[(counts["sum"] == expected_ai_per_condition) & (counts["count"] == expected_ai_per_condition + 1)].index
    subset = subset[subset.condition_id.isin(good)].copy()
    if subset.empty:
        raise RuntimeError("Scenario has no complete paired conditions")
    return subset


def bootstrap(table: pd.DataFrame, score_column: str, rng: np.random.Generator) -> dict[str, float]:
    condition_ids = sorted(table.condition_id.unique())
    blocks = {value: np.flatnonzero(table.condition_id.to_numpy() == value) for value in condition_ids}
    bas, aucs = np.empty(N_BOOT), np.empty(N_BOOT)
    labels = table.label.to_numpy(int)
    scores = table[score_column].to_numpy(float)
    for iteration in range(N_BOOT):
        sampled = rng.choice(condition_ids, len(condition_ids), replace=True)
        indices = np.concatenate([blocks[value] for value in sampled])
        result = classification_metrics(labels[indices], scores[indices])
        bas[iteration] = result["balanced_accuracy"]
        aucs[iteration] = result["roc_auc"]
    return {
        "balanced_accuracy_ci_low": float(np.quantile(bas, 0.025)),
        "balanced_accuracy_ci_high": float(np.quantile(bas, 0.975)),
        "roc_auc_ci_low": float(np.quantile(aucs, 0.025)),
        "roc_auc_ci_high": float(np.quantile(aucs, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--external-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.features)
    if len(table) != 7100 or table.track.nunique() != 7100:
        raise RuntimeError("Feature table is not the frozen 7100-track cohort")
    train = table[table.split == "development"].copy()
    test = table[table.split == "external_test_frozen"].copy()
    if len(train) != 1600 or len(test) != 5500:
        raise RuntimeError("Unexpected train/test row count")
    observed = sorted(test.loc[test.label == 1, "generator"].unique())
    if sorted(GENERATORS) != observed:
        raise RuntimeError(f"Unexpected generators: {observed}")

    external = json.loads(args.external_summary.read_text(encoding="utf-8"))
    family_status = {
        family: bool(result["sensitivity_pass"])
        for family, result in external["family_results"].items()
    }
    feature_sets = {
        "dynamics": list(DYNAMICS_FEATURES),
        "rhythm": list(RHYTHM_FEATURES),
        "structure": list(STRUCTURE_FEATURES),
        "all_three": list(DYNAMICS_FEATURES + RHYTHM_FEATURES + STRUCTURE_FEATURES),
    }

    parameters: dict[str, object] = {}
    for name, columns in feature_sets.items():
        _, params = fit_model(train, columns)
        parameters[name] = params
        table[f"score__{name}"] = predict(table, params)
    test = table[table.split == "external_test_frozen"].copy()
    table.to_csv(args.output_dir / "features_with_scores.csv", index=False)
    with (args.output_dir / "model_parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(parameters, handle, indent=2)
        handle.write("\n")
    coefficient_rows: list[dict[str, object]] = []
    for family, params in parameters.items():
        columns = list(params["columns"])
        weights = np.asarray(params["weights_with_intercept"], dtype=float)
        for index, feature in enumerate(columns):
            coefficient_rows.append({
                "feature_set": family,
                "feature": feature,
                "standardized_value_weight": float(weights[1 + index]),
                "missingness_indicator_weight": float(weights[1 + len(columns) + index]),
            })
    pd.DataFrame(coefficient_rows).to_csv(args.output_dir / "model_coefficients.csv", index=False)

    scenarios: list[tuple[str, list[str], bool]] = []
    scenarios.extend((f"generator__{name}", [name], False) for name in GENERATORS)
    scenarios.extend([
        ("primary_six", PRIMARY, False),
        ("secondary_four", SECONDARY, False),
        ("all_ten", GENERATORS, False),
        ("primary_six_strict_no_prior_condition", PRIMARY, True),
        ("secondary_four_strict_no_prior_condition", SECONDARY, True),
    ])
    rng = np.random.default_rng(SEED)
    metric_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    group_metric_rows: list[dict[str, object]] = []
    for scenario, generators, strict in scenarios:
        subset = scenario_table(test, generators, strict=strict)
        for name in feature_sets:
            score_column = f"score__{name}"
            metrics = classification_metrics(subset.label.to_numpy(int), subset[score_column].to_numpy(float))
            metrics.update(bootstrap(subset, score_column, rng))
            metric_rows.append({
                "scenario": scenario,
                "feature_set": name,
                "status": "sensitivity_qualified" if name == "dynamics" and family_status.get("dynamics") else "exploratory",
                "n_conditions": subset.condition_id.nunique(),
                "n_human": int((subset.label == 0).sum()),
                "n_ai": int((subset.label == 1).sum()),
                "generators": "|".join(generators),
                **metrics,
            })
            score_rows.extend({
                "scenario": scenario,
                "feature_set": name,
                "track": row.track,
                "condition_id": row.condition_id,
                "generator": row.generator,
                "label": int(row.label),
                "score": float(getattr(row, score_column)),
            } for row in subset.itertuples(index=False))
            for group_id, group in subset.groupby("group_id"):
                group_metrics = classification_metrics(
                    group.label.to_numpy(int), group[score_column].to_numpy(float)
                )
                group_metric_rows.append({
                    "scenario": scenario,
                    "group_id": group_id,
                    "feature_set": name,
                    "n_conditions": group.condition_id.nunique(),
                    "n_human": int((group.label == 0).sum()),
                    "n_ai": int((group.label == 1).sum()),
                    **group_metrics,
                })
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "external_metrics.csv", index=False)
    pd.DataFrame(score_rows).to_csv(args.output_dir / "external_scores.csv", index=False)
    pd.DataFrame(group_metric_rows).to_csv(args.output_dir / "frozen_group_metrics.csv", index=False)

    univariate_rows: list[dict[str, object]] = []
    all_features = list(DYNAMICS_FEATURES + RHYTHM_FEATURES + STRUCTURE_FEATURES)
    for scenario, generators, strict in scenarios:
        subset = scenario_table(test, generators, strict=strict)
        for feature in all_features:
            finite = subset[np.isfinite(subset[feature].to_numpy(float))]
            human_values = finite.loc[finite.label == 0, feature].to_numpy(float)
            ai_values = finite.loc[finite.label == 1, feature].to_numpy(float)
            univariate_rows.append({
                "scenario": scenario,
                "feature": feature,
                "n_human_finite": len(human_values),
                "n_ai_finite": len(ai_values),
                "human_median": float(np.median(human_values)) if len(human_values) else float("nan"),
                "ai_median": float(np.median(ai_values)) if len(ai_values) else float("nan"),
                "raw_higher_means_ai_auc": auc(finite.label.to_numpy(int), finite[feature].to_numpy(float)),
            })
    pd.DataFrame(univariate_rows).to_csv(args.output_dir / "univariate_external_metrics.csv", index=False)

    coverage_rows: list[dict[str, object]] = []
    for (model, label), group in test.groupby(["model", "label"]):
        coverage_rows.append({
            "model": model,
            "class_name": "ai" if int(label) else "human",
            "n": len(group),
            "dynamics_complete_fraction": float(group[list(DYNAMICS_FEATURES)].notna().all(axis=1).mean()),
            "rhythm_complete_fraction": float(group[list(RHYTHM_FEATURES)].notna().all(axis=1).mean()),
            "structure_complete_fraction": float(group[list(STRUCTURE_FEATURES)].notna().all(axis=1).mean()),
            "any_structure_fraction": float(group.structure_eligible.mean()),
            "median_beats": float(group.n_beats.median()),
            "median_sections": float(group.n_sections.median()),
        })
    pd.DataFrame(coverage_rows).to_csv(args.output_dir / "feature_coverage.csv", index=False)

    distribution_rows: list[dict[str, object]] = []
    for (model, label), group in test.groupby(["model", "label"]):
        for feature in all_features:
            values = group[feature].to_numpy(float)
            values = values[np.isfinite(values)]
            distribution_rows.append({
                "model": model,
                "class_name": "ai" if int(label) else "human",
                "feature": feature,
                "n_finite": len(values),
                "median": float(np.median(values)) if len(values) else float("nan"),
                "q25": float(np.quantile(values, 0.25)) if len(values) else float("nan"),
                "q75": float(np.quantile(values, 0.75)) if len(values) else float("nan"),
            })
    pd.DataFrame(distribution_rows).to_csv(args.output_dir / "feature_distributions.csv", index=False)

    per_generator = metrics[metrics.scenario.str.startswith("generator__")]
    macro = per_generator.groupby("feature_set").agg(
        mean_generator_auc=("roc_auc", "mean"),
        min_generator_auc=("roc_auc", "min"),
        max_generator_auc=("roc_auc", "max"),
        mean_generator_balanced_accuracy=("balanced_accuracy", "mean"),
    ).reset_index()
    macro.to_csv(args.output_dir / "per_generator_macro_summary.csv", index=False)
    summary = {
        "seed": SEED,
        "bootstrap_replicates": N_BOOT,
        "ridge_penalty": RIDGE,
        "threshold": THRESHOLD,
        "training_rows": len(train),
        "external_test_rows": len(test),
        "primary_generators": PRIMARY,
        "secondary_generators": SECONDARY,
        "external_family_status": family_status,
        "interpretation": {
            "dynamics": "sensitivity-qualified",
            "rhythm": "exploratory because the external family gate failed",
            "structure": "exploratory because the external family gate failed",
            "all_three": "exploratory because it contains failed families",
        },
        "macro_per_generator": macro.to_dict(orient="records"),
    }
    (args.output_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
