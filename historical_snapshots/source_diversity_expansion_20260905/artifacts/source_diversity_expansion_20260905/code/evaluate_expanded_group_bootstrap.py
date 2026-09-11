#!/usr/bin/env python3
"""Group-block bootstrap for frozen leader versus S-only locked source pairs.

The bootstrap unit is ``group_id``, never an individual file variant. Groups
shared across classes are sampled jointly. Human-only and AI-only group pools are
sampled separately so every replicate preserves both classes. This is especially
important for DiffRhythm: 50 rendered files remain 10 independent AI groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_expanded import metrics


DEFAULT_SLICES = (
    "new_catalogue_plus_legacy_locked_ai",
    "new_catalogue_plus_diff_rhythm_pilot",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True,
                        help="evaluate_expanded_locked_scores.csv")
    parser.add_argument("--frozen-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slices", default=",".join(DEFAULT_SLICES))
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20_260_905)
    return parser.parse_args()


def sampled_indices(pair: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    labels = pair["__label"].to_numpy(int)
    groups = pair["__group"].astype(str).to_numpy()
    human_groups = set(groups[labels == 0])
    ai_groups = set(groups[labels == 1])
    shared = sorted(human_groups & ai_groups)
    human_only = sorted(human_groups - set(shared))
    ai_only = sorted(ai_groups - set(shared))
    blocks = {group: np.flatnonzero(groups == group) for group in set(groups)}
    chosen: list[str] = []
    if shared:
        chosen.extend(rng.choice(shared, len(shared), replace=True).tolist())
    if human_only:
        chosen.extend(rng.choice(human_only, len(human_only), replace=True).tolist())
    if ai_only:
        chosen.extend(rng.choice(ai_only, len(ai_only), replace=True).tolist())
    if not chosen:
        raise ValueError("Source pair has no group_id blocks")
    return np.concatenate([blocks[group] for group in chosen])


def quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {f"{prefix}_ci_low": math.nan, f"{prefix}_ci_high": math.nan}
    return {
        f"{prefix}_ci_low": float(np.quantile(finite, 0.025)),
        f"{prefix}_ci_high": float(np.quantile(finite, 0.975)),
    }


def bootstrap_pair(
    pair: pd.DataFrame,
    leader: str,
    baseline: str,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    labels = pair["__label"].to_numpy(int)
    leader_scores = pair[leader].to_numpy(float)
    baseline_scores = pair[baseline].to_numpy(float)
    if set(labels) != {0, 1}:
        raise ValueError("Each source pair must contain both known classes")
    human_groups = set(pair.loc[pair.__label == 0, "__group"].astype(str))
    ai_groups = set(pair.loc[pair.__label == 1, "__group"].astype(str))
    shared_groups = human_groups & ai_groups
    point_leader = metrics(labels, leader_scores)
    point_baseline = metrics(labels, baseline_scores)
    identity = "|".join([
        str(pair["slice"].iloc[0]), str(pair["human_source"].iloc[0]),
        str(pair["ai_source"].iloc[0]), leader, baseline,
    ])
    rng_seed = seed ^ int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    rng = np.random.default_rng(rng_seed)
    bootstrap_rows: list[dict[str, float | int]] = []
    for replicate in range(replicates):
        indices = sampled_indices(pair, rng)
        current_labels = labels[indices]
        leader_result = metrics(current_labels, leader_scores[indices])
        baseline_result = metrics(current_labels, baseline_scores[indices])
        bootstrap_rows.append({
            "replicate": replicate,
            "leader_roc_auc": leader_result["roc_auc"],
            "s_only_roc_auc": baseline_result["roc_auc"],
            "delta_roc_auc__leader_minus_s": leader_result["roc_auc"] - baseline_result["roc_auc"],
            "leader_balanced_accuracy": leader_result["balanced_accuracy"],
            "s_only_balanced_accuracy": baseline_result["balanced_accuracy"],
            "delta_balanced_accuracy__leader_minus_s": (
                leader_result["balanced_accuracy"] - baseline_result["balanced_accuracy"]
            ),
            "leader_human_specificity": leader_result["human_specificity"],
            "s_only_human_specificity": baseline_result["human_specificity"],
            "delta_human_specificity__leader_minus_s": (
                leader_result["human_specificity"] - baseline_result["human_specificity"]
            ),
            "leader_ai_sensitivity": leader_result["ai_sensitivity"],
            "s_only_ai_sensitivity": baseline_result["ai_sensitivity"],
        })
    boot = pd.DataFrame(bootstrap_rows)
    summary: dict[str, Any] = {
        "slice": pair["slice"].iloc[0],
        "feature_set": pair["feature_set"].iloc[0],
        "leader_combination": leader,
        "baseline_combination": baseline,
        "human_source": pair["human_source"].iloc[0],
        "ai_source": pair["ai_source"].iloc[0],
        "human_rows": int((labels == 0).sum()),
        "ai_rows": int((labels == 1).sum()),
        "human_groups": len(human_groups),
        "ai_groups": len(ai_groups),
        "shared_cross_class_groups": len(shared_groups),
        "independent_group_units": len(human_groups | ai_groups),
        "replicates": replicates,
        "bootstrap_unit": "group_id block; shared cross-class groups sampled jointly",
        "leader_roc_auc": point_leader["roc_auc"],
        "s_only_roc_auc": point_baseline["roc_auc"],
        "delta_roc_auc__leader_minus_s": point_leader["roc_auc"] - point_baseline["roc_auc"],
        "leader_balanced_accuracy": point_leader["balanced_accuracy"],
        "s_only_balanced_accuracy": point_baseline["balanced_accuracy"],
        "delta_balanced_accuracy__leader_minus_s": (
            point_leader["balanced_accuracy"] - point_baseline["balanced_accuracy"]
        ),
        "leader_human_specificity": point_leader["human_specificity"],
        "s_only_human_specificity": point_baseline["human_specificity"],
        "delta_human_specificity__leader_minus_s": (
            point_leader["human_specificity"] - point_baseline["human_specificity"]
        ),
        "leader_ai_sensitivity": point_leader["ai_sensitivity"],
        "s_only_ai_sensitivity": point_baseline["ai_sensitivity"],
    }
    for column in (
        "leader_roc_auc", "s_only_roc_auc", "delta_roc_auc__leader_minus_s",
        "leader_balanced_accuracy", "s_only_balanced_accuracy",
        "delta_balanced_accuracy__leader_minus_s", "leader_human_specificity",
        "s_only_human_specificity", "delta_human_specificity__leader_minus_s",
        "leader_ai_sensitivity", "s_only_ai_sensitivity",
    ):
        summary.update(quantiles(boot[column].to_numpy(float), column))
    return summary, boot


def main() -> None:
    args = parse_args()
    if args.replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    frozen = json.loads(args.frozen_selection.read_text(encoding="utf-8"))
    primary = frozen["primary_feature_set"]
    leader = frozen["leader"]["combination"]
    scores = pd.read_csv(args.scores, low_memory=False)
    selected_slices = {value.strip() for value in args.slices.split(",") if value.strip()}
    scores = scores[(scores.feature_set == primary) & scores.slice.isin(selected_slices)
                    & scores.combination.isin({leader, "S"})].copy()
    if scores.empty:
        raise ValueError("No requested primary-feature-set score rows found")
    known = scores[np.isfinite(pd.to_numeric(scores["__label"], errors="coerce"))].copy()
    index_columns = ["slice", "feature_set", "__id", "__label", "__source", "__group"]
    wide = known.pivot(index=index_columns, columns="combination", values="score").reset_index()
    if leader not in wide or "S" not in wide:
        raise ValueError("Scores must contain both the frozen leader and S-only for identical rows")
    summaries: list[dict[str, Any]] = []
    replicate_parts: list[pd.DataFrame] = []
    for slice_name, current in wide.groupby("slice", sort=True):
        humans = sorted(current.loc[current.__label == 0, "__source"].unique())
        ais = sorted(current.loc[current.__label == 1, "__source"].unique())
        for human_source in humans:
            for ai_source in ais:
                pair = current[((current.__label == 0) & (current.__source == human_source))
                               | ((current.__label == 1) & (current.__source == ai_source))].copy()
                pair["human_source"] = human_source
                pair["ai_source"] = ai_source
                summary, boot = bootstrap_pair(pair, leader, "S", args.replicates, args.seed)
                summaries.append(summary)
                boot.insert(0, "ai_source", ai_source)
                boot.insert(0, "human_source", human_source)
                boot.insert(0, "slice", slice_name)
                replicate_parts.append(boot)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(args.output_dir / "evaluate_expanded_group_bootstrap_pairs.csv", index=False)
    pd.concat(replicate_parts, ignore_index=True).to_csv(
        args.output_dir / "evaluate_expanded_group_bootstrap_replicates.csv", index=False
    )
    metadata = {
        "primary_feature_set": primary,
        "frozen_leader": leader,
        "baseline": "S",
        "slices": sorted(selected_slices),
        "source_pairs": len(summary_table),
        "replicates_per_pair": args.replicates,
        "seed": args.seed,
        "bootstrap_unit": "group_id; never file variants",
        "selection_changed": False,
    }
    (args.output_dir / "evaluate_expanded_group_bootstrap_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
