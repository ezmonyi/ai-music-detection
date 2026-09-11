#!/usr/bin/env python3
"""Evaluate frozen change-sensitive features as incremental AI/Human evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from workflow_features import DYNAMICS_FEATURES, RHYTHM_FEATURES, STRUCTURE_FEATURES


SEED = 20260903
N_BOOT = 2_000
BASE_COLUMNS = {
    "id", "label", "class_name", "source", "split", "track", "variant",
    "raw_vocal_to_mix_rms_db", "raw_vocal_activity_frame_ratio",
    "passes_frozen_activity_threshold",
}


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    ranks = rankdata(np.concatenate((negative, positive)), method="average")
    n0, n1 = len(negative), len(positive)
    return float((ranks[n0:].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    predictions = np.asarray(scores) >= 0.5
    labels = np.asarray(labels, dtype=int)
    return float(0.5 * (np.mean(predictions[labels == 0] == 0) + np.mean(predictions[labels == 1] == 1)))


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    train_x = np.asarray(train_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    medians = np.nanmedian(train_x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    train_missing = ~np.isfinite(train_x)
    test_missing = ~np.isfinite(test_x)
    train_x = np.where(train_missing, medians, train_x)
    test_x = np.where(test_missing, medians, test_x)
    train_x = np.concatenate((train_x, train_missing.astype(float)), axis=1)
    test_x = np.concatenate((test_x, test_missing.astype(float)), axis=1)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z_train = (train_x - mean) / scale
    z_test = (test_x - mean) / scale
    design = np.column_stack((np.ones(len(z_train)), z_train))
    penalty = np.eye(design.shape[1]) * 10.0
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return np.column_stack((np.ones(len(z_test)), z_test)) @ weights


def load_spectral(path: Path, include_label: int | None = None) -> tuple[dict[str, dict[str, str]], list[str]]:
    table = pd.read_csv(path)
    table = table[table["variant"] == "bias_corrected"]
    if include_label is not None:
        table = table[table["label"] == include_label]
    features = [column for column in table.columns if column not in BASE_COLUMNS]
    lookup = {str(row["track"]): row.to_dict() for _, row in table.iterrows()}
    return lookup, features


def bootstrap_metrics(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    unique_groups = sorted(set(groups))
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    bas, aucs = np.empty(N_BOOT), np.empty(N_BOOT)
    for iteration in range(N_BOOT):
        selected = np.concatenate([
            rng.choice(indices[group], len(indices[group]), replace=True) for group in unique_groups
        ])
        bas[iteration] = balanced_accuracy(labels[selected], scores[selected])
        aucs[iteration] = auc(labels[selected], scores[selected])
    return {
        "balanced_accuracy_ci_low": float(np.quantile(bas, 0.025)),
        "balanced_accuracy_ci_high": float(np.quantile(bas, 0.975)),
        "roc_auc_ci_low": float(np.quantile(aucs, 0.025)),
        "roc_auc_ci_high": float(np.quantile(aucs, 0.975)),
    }


def bootstrap_auc_delta(labels: np.ndarray, base: np.ndarray, augmented: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    unique_groups = sorted(set(groups))
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    draws = np.empty(N_BOOT)
    for iteration in range(N_BOOT):
        selected = np.concatenate([
            rng.choice(indices[group], len(indices[group]), replace=True) for group in unique_groups
        ])
        draws[iteration] = auc(labels[selected], augmented[selected]) - auc(labels[selected], base[selected])
    point = auc(labels, augmented) - auc(labels, base)
    return {
        "delta_auc": float(point),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "probability_delta_positive": float(np.mean(draws > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-features", type=Path, required=True)
    parser.add_argument("--suno-spectral", type=Path, required=True)
    parser.add_argument("--heartmula-spectral", type=Path, required=True)
    parser.add_argument("--acestep-spectral", type=Path, required=True)
    parser.add_argument("--external-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    external = json.loads(args.external_summary.read_text(encoding="utf-8"))
    family_status = {
        family: bool(result["sensitivity_pass"])
        for family, result in external["family_results"].items()
    }

    table = pd.read_csv(args.new_features)
    suno, spectral_columns = load_spectral(args.suno_spectral)
    heart, heart_columns = load_spectral(args.heartmula_spectral, include_label=1)
    ace, ace_columns = load_spectral(args.acestep_spectral, include_label=1)
    if spectral_columns != heart_columns or spectral_columns != ace_columns:
        raise RuntimeError("Spectral feature columns differ across domains")
    spectral = {**suno, **heart, **ace}
    missing_spectral = sorted(set(table.track) - set(spectral))
    if missing_spectral:
        raise RuntimeError(f"Missing spectral rows: {missing_spectral[:5]} ({len(missing_spectral)})")
    for column in spectral_columns:
        table[f"spectral__{column}"] = [float(spectral[track][column]) for track in table.track]
    spectral_columns = [f"spectral__{column}" for column in spectral_columns]

    feature_sets = {
        "dynamics": list(DYNAMICS_FEATURES),
        "rhythm": list(RHYTHM_FEATURES),
        "structure": list(STRUCTURE_FEATURES),
        "all_new": list(DYNAMICS_FEATURES + RHYTHM_FEATURES + STRUCTURE_FEATURES),
        "spectral": spectral_columns,
        "spectral_plus_dynamics": spectral_columns + list(DYNAMICS_FEATURES),
        "spectral_plus_rhythm": spectral_columns + list(RHYTHM_FEATURES),
        "spectral_plus_structure": spectral_columns + list(STRUCTURE_FEATURES),
        "spectral_plus_all_new": spectral_columns + list(DYNAMICS_FEATURES + RHYTHM_FEATURES + STRUCTURE_FEATURES),
    }
    qualified_columns = []
    if family_status.get("dynamics", False):
        qualified_columns.extend(DYNAMICS_FEATURES)
    if family_status.get("rhythm", False):
        qualified_columns.extend(RHYTHM_FEATURES)
    if family_status.get("structure", False):
        qualified_columns.extend(STRUCTURE_FEATURES)
    if qualified_columns:
        feature_sets["sensitivity_qualified"] = list(qualified_columns)
        feature_sets["spectral_plus_sensitivity_qualified"] = spectral_columns + list(qualified_columns)
    generators = ["suno", "heartmula", "acestep"]
    source_to_generator = {
        "humair_suno": "suno", "suno_unknown": "suno",
        "heartmula": "heartmula", "acestep": "acestep",
    }
    table["generator"] = ["human" if label == 0 else source_to_generator[source] for label, source in zip(table.label, table.source)]
    table.to_csv(args.output_dir / "joined_features.csv", index=False)

    scenarios = []
    for generator in generators:
        scenarios.append((f"within_{generator}", [generator], [generator]))
        scenarios.append((f"leave_out_{generator}", [name for name in generators if name != generator], [generator]))
    scenarios.append(("pooled_all", generators, generators))

    rows, score_rows = [], []
    rng = np.random.default_rng(SEED)
    for scenario, train_generators, test_generators in scenarios:
        human_train = table[(table.label == 0) & (table.split == "development")]
        ai_train = table[(table.label == 1) & (table.split == "development") & table.generator.isin(train_generators)]
        repeats = int(np.ceil(len(ai_train) / len(human_train)))
        balanced_human = pd.concat([human_train] * repeats, ignore_index=True).iloc[: len(ai_train)]
        train = pd.concat([balanced_human, ai_train], ignore_index=True)
        test = table[((table.label == 0) | table.generator.isin(test_generators)) & (table.split == "locked_test")].copy()
        test = test[(test.label == 0) | test.generator.isin(test_generators)]
        labels = test.label.to_numpy(int)
        groups = np.asarray(["human" if value == 0 else generator for value, generator in zip(test.label, test.generator)])
        for feature_set, columns in feature_sets.items():
            scores = fit_predict(train[columns].to_numpy(float), train.label.to_numpy(int), test[columns].to_numpy(float))
            result = {
                "scenario": scenario,
                "feature_set": feature_set,
                "n_train_human_unique": len(human_train),
                "n_train_human_effective": len(balanced_human),
                "n_train_ai": len(ai_train),
                "n_test_human": int((labels == 0).sum()),
                "n_test_ai": int((labels == 1).sum()),
                "dimension_before_missing_indicators": len(columns),
                "balanced_accuracy": balanced_accuracy(labels, scores),
                "roc_auc": auc(labels, scores),
            }
            result.update(bootstrap_metrics(labels, scores, groups, rng))
            rows.append(result)
            score_rows.extend({
                "scenario": scenario, "feature_set": feature_set,
                "track": track, "label": int(label), "source": source,
                "score": float(score),
            } for track, label, source, score in zip(test.track, labels, test.source, scores))

    metrics = pd.DataFrame(rows)
    scores = pd.DataFrame(score_rows)
    metrics.to_csv(args.output_dir / "detector_metrics.csv", index=False)
    scores.to_csv(args.output_dir / "detector_scores.csv", index=False)

    deltas = []
    for scenario in metrics.scenario.unique():
        subset = scores[scores.scenario == scenario]
        base = subset[subset.feature_set == "spectral"].set_index("track")
        augmented_names = [
            "spectral_plus_dynamics", "spectral_plus_rhythm",
            "spectral_plus_structure", "spectral_plus_all_new",
        ]
        if "spectral_plus_sensitivity_qualified" in feature_sets:
            augmented_names.append("spectral_plus_sensitivity_qualified")
        for augmented_name in augmented_names:
            augmented = subset[subset.feature_set == augmented_name].set_index("track")
            common = base.index.intersection(augmented.index)
            labels = base.loc[common, "label"].to_numpy(int)
            groups = np.asarray(["human" if label == 0 else source for label, source in zip(labels, base.loc[common, "source"])])
            stats = bootstrap_auc_delta(labels, base.loc[common, "score"].to_numpy(float), augmented.loc[common, "score"].to_numpy(float), groups, rng)
            deltas.append({"scenario": scenario, "augmented_feature_set": augmented_name, "n": len(common), **stats})
    pd.DataFrame(deltas).to_csv(args.output_dir / "incremental_auc_deltas.csv", index=False)

    coverage = []
    for (source, split), group in table.groupby(["source", "split"]):
        row = {"source": source, "split": split, "n": len(group)}
        for family, columns in (("dynamics", DYNAMICS_FEATURES), ("rhythm", RHYTHM_FEATURES), ("structure", STRUCTURE_FEATURES)):
            row[f"{family}_complete_fraction"] = float(np.isfinite(group[list(columns)].to_numpy(float)).all(axis=1).mean())
        coverage.append(row)
    pd.DataFrame(coverage).to_csv(args.output_dir / "feature_coverage.csv", index=False)
    metadata = {
        "seed": SEED, "bootstrap_replicates": N_BOOT,
        "feature_sets": feature_sets,
        "n_rows": len(table),
        "domain_counts": {str(key): int(value) for key, value in Counter(table.generator).items()},
        "external_sensitivity_required": True,
        "external_all_families_pass": bool(external["all_families_pass"]),
        "external_family_status": family_status,
        "sensitivity_qualified_columns": list(qualified_columns),
        "interpretation": (
            "All requested families are evaluated, but only externally sensitivity-qualified "
            "families may be treated as validated detector extensions. Non-qualified family "
            "ablations are exploratory and cannot rescue the failed external gate."
        ),
    }
    (args.output_dir / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_metrics": len(metrics), "n_deltas": len(deltas), **metadata}, indent=2))


if __name__ == "__main__":
    main()
