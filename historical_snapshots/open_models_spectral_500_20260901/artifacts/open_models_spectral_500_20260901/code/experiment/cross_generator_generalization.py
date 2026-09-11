#!/usr/bin/env python3
"""Train on one/open-model generator domain and test on another locked domain."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_demucs_artifacts as artifactlib
import stem_spectral_ablation as stemlib


GENERATORS = ("Suno", "HeartMuLa", "ACE-Step")
SLUGS = {"Suno": "suno", "HeartMuLa": "heartmula", "ACE-Step": "acestep"}
REPRESENTATIONS = (
    "raw__vocal_spectral",
    "raw__vocals_full_0_3_20k",
    "raw__vocals_only_5_10k",
    "raw__vocals_without_5_10k",
    "raw__vocals_only_5_16k",
    "raw__vocals_low_mid_0_3_5k",
    "bias_corrected__vocal_spectral",
    "bias_corrected__vocals_full_0_3_20k",
    "bias_corrected__vocals_only_5_10k",
    "bias_corrected__vocals_without_5_10k",
    "bias_corrected__vocals_only_5_16k",
    "bias_corrected__vocals_low_mid_0_3_5k",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for slug in SLUGS.values():
        parser.add_argument(f"--{slug}-manifest", type=Path, required=True)
        parser.add_argument(f"--{slug}-metrics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Restrict every train/test class to the raw-frozen vocal-active sensitivity subset.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_domain(manifest_path: Path, metrics_dir: Path) -> dict[str, object]:
    manifest = read_jsonl(manifest_path)
    metric_rows = read_csv(metrics_dir / "vocal_metrics_raw_and_corrected.csv")
    metric_lookup = {
        (str(row["track"]), str(row["variant"])): row for row in metric_rows
    }
    activity_rows = read_csv(metrics_dir / "vocal_activity.csv")
    activity_lookup = {
        str(row["track"]): str(row["passes_frozen_activity_threshold"]).lower()
        in {"1", "true", "yes"}
        for row in activity_rows
    }
    with np.load(metrics_dir / "frequency_vectors_raw_and_corrected.npz", allow_pickle=False) as payload:
        vectors = {key: np.asarray(payload[key]) for key in payload.files}
    matrices = {}
    for variant in ("raw", "bias_corrected"):
        matrices[f"{variant}__vocal_spectral"] = np.asarray(
            [
                [
                    float(metric_lookup[(artifactlib.track_name(row), variant)][metric])
                    for metric in stemlib.ALL_METRICS
                ]
                for row in manifest
            ]
        )
        for name in artifactlib.FREQUENCY_RANGES:
            matrices[f"{variant}__{name}"] = vectors[f"{variant}__{name}"]
    return {
        "manifest": manifest,
        "labels": np.asarray([int(row["label"]) for row in manifest]),
        "splits": np.asarray([str(row["split"]) for row in manifest]),
        "tracks": np.asarray([artifactlib.track_name(row) for row in manifest]),
        "vocal_active": np.asarray(
            [activity_lookup[artifactlib.track_name(row)] for row in manifest], dtype=bool
        ),
        "matrices": matrices,
    }


def indices(
    domain: dict[str, object], split: str, label: int, active_only: bool = False
) -> np.ndarray:
    labels = np.asarray(domain["labels"])
    splits = np.asarray(domain["splits"])
    mask = (labels == label) & (splits == split)
    if active_only:
        mask &= np.asarray(domain["vocal_active"])
    return np.flatnonzero(mask)


def balanced_fit_predict(
    human_x: np.ndarray,
    ai_x: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    repeats = int(np.ceil(ai_x.shape[0] / human_x.shape[0]))
    balanced_human = np.tile(human_x, (repeats, 1))[: ai_x.shape[0]]
    train_x = np.concatenate((balanced_human, ai_x), axis=0)
    train_y = np.concatenate(
        (np.zeros(balanced_human.shape[0], dtype=int), np.ones(ai_x.shape[0], dtype=int))
    )
    return artifactlib.fit_predict(train_x, train_y, test_x)


def evaluate(
    representation: str,
    train_domain: str,
    test_domain: str,
    human_train: np.ndarray,
    ai_train: np.ndarray,
    human_test: np.ndarray,
    ai_test: np.ndarray,
    seed: int,
) -> dict[str, object]:
    test_x = np.concatenate((human_test, ai_test), axis=0)
    labels = np.concatenate(
        (np.zeros(human_test.shape[0], dtype=int), np.ones(ai_test.shape[0], dtype=int))
    )
    scores = balanced_fit_predict(human_train, ai_train, test_x)
    balanced_accuracy, roc_auc = artifactlib.metrics(labels, scores)
    tag = f"{representation}|{train_domain}|{test_domain}"
    offset = sum((index + 1) * ord(char) for index, char in enumerate(tag))
    ba_low, ba_high, auc_low, auc_high = artifactlib.bootstrap_intervals(
        labels, scores, seed + offset, repeats=1_000
    )
    return {
        "representation": representation,
        "train_domain": train_domain,
        "test_domain": test_domain,
        "n_train_human_effective": ai_train.shape[0],
        "n_train_human_unique": human_train.shape[0],
        "n_train_ai": ai_train.shape[0],
        "n_test_human": human_test.shape[0],
        "n_test_ai": ai_test.shape[0],
        "balanced_accuracy": balanced_accuracy,
        "balanced_accuracy_ci_low": ba_low,
        "balanced_accuracy_ci_high": ba_high,
        "roc_auc": roc_auc,
        "roc_auc_ci_low": auc_low,
        "roc_auc_ci_high": auc_high,
    }


def plot_single_source_heatmaps(path: Path, rows: list[dict[str, object]]) -> None:
    corrected = [name for name in REPRESENTATIONS if name.startswith("bias_corrected__")]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, representation in zip(axes.flat, corrected):
        matrix = np.full((3, 3), np.nan)
        for row in rows:
            if (
                row["representation"] == representation
                and row["train_domain"] in GENERATORS
                and row["test_domain"] in GENERATORS
            ):
                i = GENERATORS.index(str(row["train_domain"]))
                j = GENERATORS.index(str(row["test_domain"]))
                matrix[i, j] = float(row["roc_auc"])
        image = axis.imshow(matrix, cmap="viridis", vmin=0.5, vmax=1.0)
        axis.set_xticks(np.arange(3), GENERATORS, rotation=15, ha="right")
        axis.set_yticks(np.arange(3), GENERATORS)
        axis.set_xlabel("Locked test generator")
        axis.set_ylabel("Development train generator")
        axis.set_title(representation.removeprefix("bias_corrected__"))
        for i in range(3):
            for j in range(3):
                axis.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white" if matrix[i, j] < 0.75 else "black")
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), pad=0.01)
    colorbar.set_label("ROC-AUC")
    figure.suptitle("Cross-generator Human-vs-AI generalization")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(
    path: Path, rows: list[dict[str, object]], active_only: bool
) -> None:
    corrected = [row for row in rows if str(row["representation"]).startswith("bias_corrected__")]
    lines = [
        "# Cross-generator generalization",
        "",
        "## Design",
        "",
        "- Fit uses development only; evaluation uses each target generator's locked test plus the same locked Human cohort.",
        "- Single-source cells train on one generator. `leave_out_X` trains on the other two; `pooled_all` trains on all three.",
        "- When two or three AI domains are pooled, Human development rows are repeated only as regression weights so the training loss remains class-balanced; no duplicated Human appears in evaluation.",
        "- AUC is the primary result because the score threshold can shift across generator domains.",
        f"- Cohort: {'raw-frozen vocal-active sensitivity subset' if active_only else 'all tracks'}.",
        "",
        "## Leave-one-generator-out corrected result",
        "",
        "| Target | Best representation trained without target | Locked AUC | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for target in GENERATORS:
        train_name = f"leave_out_{target}"
        candidates = [
            row
            for row in corrected
            if row["train_domain"] == train_name and row["test_domain"] == target
        ]
        best = max(candidates, key=lambda row: float(row["roc_auc"]))
        lines.append(
            f"| {target} | {str(best['representation']).removeprefix('bias_corrected__')} | "
            f"{float(best['roc_auc']):.3f} | {float(best['roc_auc_ci_low']):.3f}-{float(best['roc_auc_ci_high']):.3f} |"
        )
    lines += [
        "",
        "## Pooled-all corrected result",
        "",
        "| Target | Best representation | Locked AUC | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for target in GENERATORS:
        candidates = [
            row
            for row in corrected
            if row["train_domain"] == "pooled_all" and row["test_domain"] == target
        ]
        best = max(candidates, key=lambda row: float(row["roc_auc"]))
        lines.append(
            f"| {target} | {str(best['representation']).removeprefix('bias_corrected__')} | "
            f"{float(best['roc_auc']):.3f} | {float(best['roc_auc_ci_low']):.3f}-{float(best['roc_auc_ci_high']):.3f} |"
        )
    pooled_candidates = [
        row
        for row in corrected
        if row["train_domain"] == "pooled_all" and row["test_domain"] == "pooled_all"
    ]
    pooled_best = max(pooled_candidates, key=lambda row: float(row["roc_auc"]))
    lines += [
        "",
        f"With the locked Human cohort included once and all three locked AI cohorts pooled, the best corrected representation is `{str(pooled_best['representation']).removeprefix('bias_corrected__')}` at AUC {float(pooled_best['roc_auc']):.3f} ({float(pooled_best['roc_auc_ci_low']):.3f}-{float(pooled_best['roc_auc_ci_high']):.3f}).",
    ]
    lines += [
        "",
        "## Interpretation",
        "",
        "High diagonal AUC with weak off-diagonal AUC indicates generator fingerprints. Strong leave-one-generator-out AUC is more persuasive evidence for a reusable AI-music heuristic, although all three AI prompt domains still share Muse text conditions and must not be called universal.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    domains = {}
    for generator in GENERATORS:
        slug = SLUGS[generator]
        domains[generator] = load_domain(
            getattr(args, f"{slug}_manifest"), getattr(args, f"{slug}_metrics_dir")
        )

    def selected(domain: dict[str, object], split: str, label: int) -> np.ndarray:
        return indices(domain, split, label, active_only=args.active_only)

    rows = []
    canonical = domains["Suno"]
    for representation in REPRESENTATIONS:
        human_train = np.asarray(canonical["matrices"][representation])[
            selected(canonical, "development", 0)
        ]
        for train_generator in GENERATORS:
            train = domains[train_generator]
            ai_train = np.asarray(train["matrices"][representation])[
                selected(train, "development", 1)
            ]
            for test_generator in GENERATORS:
                test = domains[test_generator]
                rows.append(
                    evaluate(
                        representation,
                        train_generator,
                        test_generator,
                        human_train,
                        ai_train,
                        np.asarray(test["matrices"][representation])[
                            selected(test, "locked_test", 0)
                        ],
                        np.asarray(test["matrices"][representation])[
                            selected(test, "locked_test", 1)
                        ],
                        args.seed,
                    )
                )
        for held_out in GENERATORS:
            included = [name for name in GENERATORS if name != held_out]
            ai_train = np.concatenate(
                [
                    np.asarray(domains[name]["matrices"][representation])[
                        selected(domains[name], "development", 1)
                    ]
                    for name in included
                ],
                axis=0,
            )
            test = domains[held_out]
            rows.append(
                evaluate(
                    representation,
                    f"leave_out_{held_out}",
                    held_out,
                    human_train,
                    ai_train,
                    np.asarray(test["matrices"][representation])[
                        selected(test, "locked_test", 0)
                    ],
                    np.asarray(test["matrices"][representation])[
                        selected(test, "locked_test", 1)
                    ],
                    args.seed,
                )
            )
        pooled_ai = np.concatenate(
            [
                np.asarray(domains[name]["matrices"][representation])[
                    selected(domains[name], "development", 1)
                ]
                for name in GENERATORS
            ],
            axis=0,
        )
        for test_generator in GENERATORS:
            test = domains[test_generator]
            rows.append(
                evaluate(
                    representation,
                    "pooled_all",
                    test_generator,
                    human_train,
                    pooled_ai,
                    np.asarray(test["matrices"][representation])[
                        selected(test, "locked_test", 0)
                    ],
                    np.asarray(test["matrices"][representation])[
                        selected(test, "locked_test", 1)
                    ],
                    args.seed,
                )
            )
        pooled_locked_ai = np.concatenate(
            [
                np.asarray(domains[name]["matrices"][representation])[
                    selected(domains[name], "locked_test", 1)
                ]
                for name in GENERATORS
            ],
            axis=0,
        )
        rows.append(
            evaluate(
                representation,
                "pooled_all",
                "pooled_all",
                human_train,
                pooled_ai,
                np.asarray(canonical["matrices"][representation])[
                    selected(canonical, "locked_test", 0)
                ],
                pooled_locked_ai,
                args.seed,
            )
        )

    write_csv(args.output_dir / "cross_generator_evaluation.csv", rows)
    plot_single_source_heatmaps(args.output_dir / "cross_generator_auc_heatmaps.png", rows)
    write_report(args.output_dir / "REPORT.md", rows, args.active_only)
    metadata = {
        "generators": list(GENERATORS),
        "representations": list(REPRESENTATIONS),
        "nominal_development_per_generator": {"human": 400, "ai": 400},
        "nominal_locked_per_generator": {"human": 100, "ai": 100},
        "bootstrap_repeats": 1000,
        "active_only": args.active_only,
        "actual_counts": {
            generator: {
                f"{split}_{class_name}": int(
                    selected(domain, split, label).size
                )
                for split in ("development", "locked_test")
                for label, class_name in ((0, "human"), (1, "ai"))
            }
            for generator, domain in domains.items()
        },
        "pooled_human_policy": "repeat unique development Human rows as class weights only",
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
