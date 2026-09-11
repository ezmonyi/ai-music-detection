#!/usr/bin/env python3
"""Evaluate all 15 four-family combinations and data diversity/quantity curves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


SEED = 20_260_904
RIDGE = 10.0
THRESHOLD = 0.5
GROUPS = [f"group_{index:02d}" for index in range(1, 6)]
GENERATORS = [
    "MusicGen Small",
    "MusicGen Medium",
    "MusicGen Large",
    "AudioLDM 2 Large",
    "AudioLDM 2 Music",
    "Mustango",
    "Udio",
    "Riffusion",
    "Stable Audio v1",
    "Stable Audio v2",
]
AMOUNTS = [25, 50, 100, 200, 400]
GENERATOR_COUNTS = [1, 2, 4, 6, 8, 10]

FAMILIES = {
    "S": [
        "spectral__tilt_1_5k_db_oct",
        "spectral__hf_tilt_5_16k_db_oct",
        "spectral__hf_ratio_5_16_db",
        "spectral__air_ratio_12_20_db",
        "spectral__sibilance_ratio_5_10_db",
        "spectral__hf_flatness",
        "spectral__hf_entropy",
        "spectral__hf_crest_db",
        "spectral__fakeprint_peak_density",
        "spectral__fakeprint_periodicity",
        "spectral__hf_flux",
        "spectral__hf_frame_similarity",
        "spectral__hf_power_sd_db",
        "spectral__hf_mod_4_12_share",
        "spectral__sibilance_contrast_db",
        "spectral__sibilance_burst_rate_hz",
    ],
    "D": ["dynamics_span", "dynamics_iqr", "dynamics_adjacent_change"],
    "R": ["ibi_cv", "tempo_tv", "tempo_entropy"],
    "P": [
        "section_duration_cv",
        "section_duration_entropy",
        "section_bars_cv",
        "section_bars_offmode_fraction",
        "section_duration_median",
        "section_bars_median",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heuristics", type=Path, required=True)
    parser.add_argument("--spectral", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
    if not n0 or not n1:
        return math.nan
    ranks = rankdata(scores, method="average")
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    prediction = np.asarray(scores, dtype=float) >= THRESHOLD
    tp = int(((prediction == 1) & (labels == 1)).sum())
    tn = int(((prediction == 0) & (labels == 0)).sum())
    fp = int(((prediction == 1) & (labels == 0)).sum())
    fn = int(((prediction == 0) & (labels == 1)).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "roc_auc": auc(labels, scores),
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "ai_sensitivity": sensitivity,
        "human_specificity": specificity,
    }


def combinations() -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    family_names = list(FAMILIES)
    for size in range(1, len(family_names) + 1):
        for names in itertools.combinations(family_names, size):
            output["+".join(names)] = sum((FAMILIES[name] for name in names), [])
    return output


def source_group(row: pd.Series) -> str:
    if int(row.label) == 0:
        return "Human:FMA" if row.split == "development" else "Human:MTG-Jamendo"
    if row.split == "development":
        if row.source in ("humair_suno", "suno_unknown"):
            return "AI:Suno"
        if row.source == "heartmula":
            return "AI:HeartMuLa"
        if row.source == "acestep":
            return "AI:ACE-Step"
    return f"AI:{row.generator}"


def sample_weights(table: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(table), dtype=float)
    labels = table.label.to_numpy(int)
    groups = table.training_source_group.to_numpy(str)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        current_groups = sorted(set(groups[indices]))
        for group in current_groups:
            selected = indices[groups[indices] == group]
            weights[selected] = 0.5 / (len(current_groups) * len(selected))
    weights *= len(table) / weights.sum()
    return weights


def fit(table: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    x = table[columns].to_numpy(float)
    y = table.label.to_numpy(float)
    missing = ~np.isfinite(x)
    # Avoid an expected warning for the deliberately retained, completely
    # unobservable 10 s phrase/section columns.
    medians = np.asarray([
        float(np.median(column[np.isfinite(column)])) if np.any(np.isfinite(column)) else 0.0
        for column in x.T
    ])
    x = np.where(missing, medians, x)
    x = np.concatenate((x, missing.astype(float)), axis=1)
    weights = sample_weights(table)
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_y = y * np.sqrt(weights)
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_y,
    )
    return {
        "columns": columns,
        "medians": medians.tolist(),
        "mean_with_missing": mean.tolist(),
        "scale_with_missing": scale.tolist(),
        "coefficients_with_intercept": coefficients.tolist(),
        "ridge": RIDGE,
        "threshold": THRESHOLD,
        "n_rows": len(table),
        "n_human": int((table.label == 0).sum()),
        "n_ai": int((table.label == 1).sum()),
        "training_source_counts": table.training_source_group.value_counts().sort_index().to_dict(),
        "weighting": "binary classes equal; sources equal within class; samples equal within source",
    }


def predict(table: pd.DataFrame, model: dict[str, object]) -> np.ndarray:
    columns = list(model["columns"])
    x = table[columns].to_numpy(float)
    missing = ~np.isfinite(x)
    medians = np.asarray(model["medians"], dtype=float)
    x = np.where(missing, medians, x)
    x = np.concatenate((x, missing.astype(float)), axis=1)
    mean = np.asarray(model["mean_with_missing"], dtype=float)
    scale = np.asarray(model["scale_with_missing"], dtype=float)
    coefficients = np.asarray(model["coefficients_with_intercept"], dtype=float)
    design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    return design @ coefficients


def ordered_conditions(external_train: pd.DataFrame, fold: int) -> list[str]:
    values = sorted(external_train.condition_id.dropna().unique())
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{SEED}|fold={fold}|{value}".encode()).hexdigest(),
    )


def generator_order() -> list[str]:
    rng = np.random.default_rng(SEED)
    return list(rng.permutation(GENERATORS))


def cyclic_subset(order: list[str], start: int, count: int) -> list[str]:
    return [order[(start + offset) % len(order)] for offset in range(count)]


def evaluate_one(
    train: pd.DataFrame,
    test: pd.DataFrame,
    combo_name: str,
    columns: list[str],
    fold: int,
    amount: int,
    generator_count: int,
    repeat: int,
    selected_generators: list[str],
) -> list[dict[str, object]]:
    model = fit(train, columns)
    scores = predict(test, model)
    scored = test[["track", "label", "generator"]].copy()
    scored["score"] = scores
    output = []
    human = scored[scored.label == 0]
    for generator in GENERATORS:
        current = pd.concat((human, scored[(scored.label == 1) & (scored.generator == generator)]))
        result = metrics(current.label.to_numpy(int), current.score.to_numpy(float))
        output.append({
            "fold": fold,
            "test_group": GROUPS[fold],
            "amount_per_new_source": amount,
            "external_training_generator_count": generator_count,
            "generator_subset_repeat": repeat,
            "training_generators": "|".join(selected_generators),
            "test_generator": generator,
            "transfer_status": "seen" if generator in selected_generators else "unseen",
            "combination": combo_name,
            "families": combo_name,
            "feature_count": len(columns),
            "train_rows": len(train),
            "train_human": int((train.label == 0).sum()),
            "train_ai": int((train.label == 1).sum()),
            "test_human": int((current.label == 0).sum()),
            "test_ai": int((current.label == 1).sum()),
            **result,
        })
    return output


def aggregate(raw: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    fold_config = (
        raw.groupby(group_columns + ["fold", "generator_subset_repeat"], dropna=False)
        .agg(
            roc_auc=("roc_auc", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            ai_sensitivity=("ai_sensitivity", "mean"),
            human_specificity=("human_specificity", "mean"),
        )
        .reset_index()
    )
    return (
        fold_config.groupby(group_columns, dropna=False)
        .agg(
            configurations=("roc_auc", "size"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            roc_auc_min=("roc_auc", "min"),
            roc_auc_max=("roc_auc", "max"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            ai_sensitivity_mean=("ai_sensitivity", "mean"),
            human_specificity_mean=("human_specificity", "mean"),
        )
        .reset_index()
    )


def plot_combination_table(path: Path, quantity: pd.DataFrame) -> None:
    order = sorted(
        quantity.combination.unique(),
        key=lambda name: (name.count("+") + 1, name),
    )
    amounts = [0, *AMOUNTS]
    matrix = np.full((len(order), len(amounts)), np.nan)
    for row in quantity.itertuples(index=False):
        matrix[order.index(row.combination), amounts.index(int(row.amount_per_new_source))] = row.roc_auc_mean
    figure, axis = plt.subplots(figsize=(10.8, 8.2), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.45, vmax=0.80)
    axis.set_xticks(np.arange(len(amounts)), ["0\nlegacy", *[str(x) for x in AMOUNTS]])
    axis.set_yticks(np.arange(len(order)), order)
    axis.set_xlabel("New AIME training conditions per source (5-fold estimate; max 400/fold)")
    axis.set_ylabel("Feature-family combination (S=spectral, D=dynamics, R=rhythm, P=phrase/section)")
    axis.set_title("Four-family pipeline combinations by training quantity: mean generator-macro ROC-AUC")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if np.isfinite(matrix[row, column]):
                axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", fontsize=8)
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("ROC-AUC")
    figure.savefig(path, dpi=190)
    plt.close(figure)


def plot_diversity(path: Path, grid: pd.DataFrame, best_combo: str) -> None:
    selected = grid[(grid.combination == best_combo) & (grid.transfer_status == "unseen")]
    amounts = [25, 100, 400]
    figure, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    for amount in amounts:
        current = selected[selected.amount_per_new_source == amount].sort_values(
            "external_training_generator_count"
        )
        axis.plot(
            current.external_training_generator_count,
            current.roc_auc_mean,
            marker="o",
            linewidth=2,
            label=f"{amount}/source",
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xticks([1, 2, 4, 6, 8])
    axis.set_ylim(0.35, 0.85)
    axis.set_xlabel("Number of new AI generator families in training")
    axis.set_ylabel("Unseen-generator macro ROC-AUC")
    axis.set_title(f"Generator-diversity transfer for CV-leading combination {best_combo}")
    axis.grid(alpha=0.25)
    axis.legend(title="AIME conditions")
    figure.savefig(path, dpi=190)
    plt.close(figure)


def write_csv(path: Path, table: pd.DataFrame) -> None:
    table.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir.parent.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    heuristics = pd.read_csv(args.heuristics)
    spectral = pd.read_csv(args.spectral)
    if len(heuristics) != 7_100 or heuristics.track.nunique() != 7_100:
        raise RuntimeError("Expected the frozen 7,100-row heuristic table")
    if len(spectral) != 7_100 or spectral.track.nunique() != 7_100:
        raise RuntimeError("Expected exactly one spectral row per frozen track")
    table = heuristics.merge(spectral, on="track", how="left", validate="one_to_one")
    table["training_source_group"] = table.apply(source_group, axis=1)
    observed = sorted(table.loc[table.split == "external_test_frozen", "generator"].unique())
    if observed != sorted(["human", *GENERATORS]):
        raise RuntimeError(f"Unexpected external sources: {observed}")

    combos = combinations()
    all_columns = sorted(set(sum(combos.values(), [])))
    missing_columns = [name for name in all_columns if name not in table]
    if missing_columns:
        raise RuntimeError(f"Missing feature columns: {missing_columns}")
    table.to_csv(args.output_dir / "four_family_features.csv", index=False)

    legacy = table[table.split == "development"].copy()
    external = table[table.split == "external_test_frozen"].copy()
    order = generator_order()
    raw_rows: list[dict[str, object]] = []

    # Legacy-only baseline: all external generators are unseen.
    for fold, group in enumerate(GROUPS):
        test = external[external.group_id == group].copy()
        for combo_name, columns in combos.items():
            raw_rows.extend(
                evaluate_one(legacy, test, combo_name, columns, fold, 0, 0, 0, [])
            )

    # Paired conditions, nested quantity samples, balanced cyclic generator subsets.
    for fold, group in enumerate(GROUPS):
        pool = external[external.group_id != group].copy()
        test = external[external.group_id == group].copy()
        condition_order = ordered_conditions(pool, fold)
        for generator_count in GENERATOR_COUNTS:
            starts = [fold] if generator_count == 10 else [fold, fold + 5]
            for repeat, start in enumerate(starts):
                selected_generators = cyclic_subset(order, start, generator_count)
                for amount in AMOUNTS:
                    selected_conditions = set(condition_order[:amount])
                    added = pool[
                        pool.condition_id.isin(selected_conditions)
                        & ((pool.label == 0) | pool.generator.isin(selected_generators))
                    ].copy()
                    expected = amount * (generator_count + 1)
                    if len(added) != expected:
                        raise RuntimeError(
                            f"Paired sample mismatch fold={fold} k={generator_count} n={amount}: "
                            f"{len(added)} != {expected}"
                        )
                    train = pd.concat((legacy, added), ignore_index=True)
                    for combo_name, columns in combos.items():
                        raw_rows.extend(
                            evaluate_one(
                                train,
                                test,
                                combo_name,
                                columns,
                                fold,
                                amount,
                                generator_count,
                                repeat,
                                selected_generators,
                            )
                        )

    raw = pd.DataFrame(raw_rows)
    write_csv(args.output_dir / "per_generator_cv_results.csv", raw)

    baseline_raw = raw[(raw.amount_per_new_source == 0) & (raw.external_training_generator_count == 0)]
    baseline = aggregate(baseline_raw.assign(transfer_status="all"), ["combination", "amount_per_new_source", "transfer_status"])
    k10_raw = raw[(raw.external_training_generator_count == 10)].copy()
    quantity = aggregate(k10_raw.assign(transfer_status="seen"), ["combination", "amount_per_new_source", "transfer_status"])
    quantity = pd.concat((baseline, quantity), ignore_index=True)
    spectral_baseline = quantity[quantity.combination == "S"][["amount_per_new_source", "roc_auc_mean"]].rename(
        columns={"roc_auc_mean": "spectral_only_auc"}
    )
    quantity = quantity.merge(spectral_baseline, on="amount_per_new_source", how="left")
    quantity["delta_auc_vs_spectral_only"] = quantity.roc_auc_mean - quantity.spectral_only_auc
    write_csv(args.output_dir / "combination_by_quantity_k10.csv", quantity)

    positive = raw[raw.amount_per_new_source > 0].copy()
    grid_parts = []
    for status in ("seen", "unseen"):
        current = positive[positive.transfer_status == status]
        if not current.empty:
            grid_parts.append(
                aggregate(
                    current,
                    [
                        "combination",
                        "amount_per_new_source",
                        "external_training_generator_count",
                        "transfer_status",
                    ],
                )
            )
    grid = pd.concat(grid_parts, ignore_index=True)
    write_csv(args.output_dir / "diversity_quantity_grid.csv", grid)

    full = quantity[quantity.amount_per_new_source == 400].copy()
    full["family_count"] = full.combination.str.count(r"\+") + 1
    full = full.sort_values(["roc_auc_mean", "family_count"], ascending=[False, True])
    best_combo = str(full.iloc[0].combination)
    best_columns = combos[best_combo]
    write_csv(args.output_dir / "combination_full_400_ranked.csv", full)
    best_grid = grid[grid.combination == best_combo].copy()
    write_csv(args.output_dir / "best_combination_diversity_table.csv", best_grid)

    # Deployment candidate consumes all 500 rows/source. It has no AIME test estimate.
    deployment_train = pd.concat((legacy, external), ignore_index=True)
    deployment_model = fit(deployment_train, best_columns)
    deployment_model.update({
        "combination": best_combo,
        "status": "exploratory deployment candidate; all AIME rows consumed; requires a new untouched test",
        "external_rows_per_source": 500,
        "external_human_sources": ["MTG-Jamendo"],
        "external_ai_generators": GENERATORS,
        "generator_order_for_cv": order,
        "feature_family_definitions": FAMILIES,
    })
    (args.output_dir / "deployment_model_all_500.json").write_text(
        json.dumps(deployment_model, indent=2) + "\n", encoding="utf-8"
    )

    plot_combination_table(figures / "four_family_combination_quantity_table.png", quantity)
    plot_diversity(figures / "unseen_generator_diversity_curves.png", grid, best_combo)

    summary = {
        "seed": SEED,
        "ridge": RIDGE,
        "threshold": THRESHOLD,
        "rows": len(table),
        "legacy_rows": len(legacy),
        "aime_rows": len(external),
        "feature_family_count": len(FAMILIES),
        "combination_count": len(combos),
        "cv_folds": GROUPS,
        "amounts_per_new_source_evaluated": AMOUNTS,
        "deployment_amount_per_new_source": 500,
        "generator_counts": GENERATOR_COUNTS,
        "generator_order": order,
        "cv_leading_combination": best_combo,
        "cv_leading_k10_n400_auc": float(full.iloc[0].roc_auc_mean),
        "cv_leading_k10_n400_ba": float(full.iloc[0].balanced_accuracy_mean),
        "spectral_only_k10_n400_auc": float(full[full.combination == "S"].iloc[0].roc_auc_mean),
        "all_four_k10_n400_auc": float(full[full.combination == "S+D+R+P"].iloc[0].roc_auc_mean),
        "structure_complete_feature_coverage": float(table.structure_eligible.mean()),
        "interpretation": "development/CV evidence after consuming the former AIME external test; a new external confirmation is required",
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
