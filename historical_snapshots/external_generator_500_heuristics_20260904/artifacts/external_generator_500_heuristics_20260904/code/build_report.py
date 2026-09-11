#!/usr/bin/env python3
"""Build bilingual reports and a dashboard for the external heuristic evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURES = ["dynamics", "rhythm", "structure", "all_three"]
LABELS = {"dynamics": "Dynamics", "rhythm": "Rhythm", "structure": "Structure", "all_three": "All three"}
COLORS = {"dynamics": "#2a9d8f", "rhythm": "#e9c46a", "structure": "#e76f51", "all_three": "#577590"}


def interval(row: pd.Series, metric: str = "roc_auc") -> str:
    return f"{row[metric]:.3f} [{row[f'{metric}_ci_low']:.3f}, {row[f'{metric}_ci_high']:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    metrics = pd.read_csv(args.evaluation_dir / "external_metrics.csv")
    coverage = pd.read_csv(args.evaluation_dir / "feature_coverage.csv")
    univariate = pd.read_csv(args.evaluation_dir / "univariate_external_metrics.csv")
    groups = pd.read_csv(args.evaluation_dir / "frozen_group_metrics.csv")
    summary = json.loads((args.evaluation_dir / "evaluation_summary.json").read_text())
    reports = args.output_root / "reports"
    figures = args.output_root / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    generator_rows = metrics[metrics.scenario.str.startswith("generator__")].copy()
    generator_rows["generator"] = generator_rows.scenario.str.removeprefix("generator__")
    generator_order = summary["primary_generators"] + summary["secondary_generators"]
    x = np.arange(len(generator_order))
    fig, axes = plt.subplots(3, 1, figsize=(13, 14), constrained_layout=True)
    width = 0.2
    for index, feature in enumerate(FEATURES):
        values = [float(generator_rows[(generator_rows.generator == model) & (generator_rows.feature_set == feature)].roc_auc.iloc[0]) for model in generator_order]
        axes[0].bar(x + (index - 1.5) * width, values, width, label=LABELS[feature], color=COLORS[feature])
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("A. Fixed 10 s heuristic transfer by external generator")
    axes[0].set_xticks(x, generator_order, rotation=25, ha="right")
    axes[0].legend(ncol=4, fontsize=9)

    aggregate_names = ["primary_six", "secondary_four", "all_ten"]
    aggregate_labels = ["Primary six", "Secondary four", "All ten"]
    agg_x = np.arange(len(aggregate_names))
    for index, feature in enumerate(FEATURES):
        rows = [metrics[(metrics.scenario == scenario) & (metrics.feature_set == feature)].iloc[0] for scenario in aggregate_names]
        values = np.asarray([row.roc_auc for row in rows])
        errors = np.asarray([[row.roc_auc - row.roc_auc_ci_low for row in rows], [row.roc_auc_ci_high - row.roc_auc for row in rows]])
        axes[1].bar(agg_x + (index - 1.5) * width, values, width, yerr=errors, capsize=2, label=LABELS[feature], color=COLORS[feature])
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("ROC-AUC (95% paired bootstrap CI)")
    axes[1].set_title("B. Aggregate external transfer; Human controls included once")
    axes[1].set_xticks(agg_x, aggregate_labels)

    coverage_order = ["MTG-Jamendo"] + generator_order
    coverage_matrix = []
    for model in coverage_order:
        row = coverage[coverage.model == model].iloc[0]
        coverage_matrix.append([row.dynamics_complete_fraction, row.rhythm_complete_fraction, row.structure_complete_fraction])
    image = axes[2].imshow(np.asarray(coverage_matrix).T, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[2].set_title("C. Complete-feature coverage")
    axes[2].set_xticks(np.arange(len(coverage_order)), coverage_order, rotation=25, ha="right")
    axes[2].set_yticks(np.arange(3), ["Dynamics", "Rhythm", "Structure"])
    for row in range(3):
        for column in range(len(coverage_order)):
            value = coverage_matrix[column][row]
            axes[2].text(column, row, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=8)
    fig.colorbar(image, ax=axes[2], label="fraction")
    dashboard = figures / "external_500_heuristics_dashboard.png"
    fig.savefig(dashboard, dpi=180)
    plt.close(fig)

    dynamics = generator_rows[generator_rows.feature_set == "dynamics"].set_index("generator")
    macro = pd.DataFrame(summary["macro_per_generator"]).set_index("feature_set")
    primary_dyn = metrics[(metrics.scenario == "primary_six") & (metrics.feature_set == "dynamics")].iloc[0]
    secondary_dyn = metrics[(metrics.scenario == "secondary_four") & (metrics.feature_set == "dynamics")].iloc[0]
    all_dyn = metrics[(metrics.scenario == "all_ten") & (metrics.feature_set == "dynamics")].iloc[0]
    strict_primary = metrics[(metrics.scenario == "primary_six_strict_no_prior_condition") & (metrics.feature_set == "dynamics")].iloc[0]
    strict_secondary = metrics[(metrics.scenario == "secondary_four_strict_no_prior_condition") & (metrics.feature_set == "dynamics")].iloc[0]
    group_dynamics = groups[groups.feature_set == "dynamics"].groupby("scenario").roc_auc.agg(["min", "max"])
    worst = dynamics.roc_auc.idxmin()
    best = dynamics.roc_auc.idxmax()

    english = [
        "# Three heuristic families on the 500-per-generator external test set",
        "",
        "## Design",
        "",
        "The score models were fit once on 1,600 pre-existing 10-second development clips (400 Human and 1,200 Suno/HeartMuLa/ACE-Step) and then applied without refitting to 5,500 frozen AIME clips. Each external generator contributes 500 tracks and is compared with the same 500 prompt-matched MTG-Jamendo Human controls. All confidence intervals use 2,000 prompt-condition bootstrap replicates.",
        "",
        "Only dynamics passed the earlier controlled-change family gate. Rhythm, structure, and their union remain exploratory even when their external classification scores are favorable.",
        "",
        "## Confirmatory dynamics result",
        "",
        f"The sensitivity-qualified dynamics score has mean per-generator AUC {macro.loc['dynamics', 'mean_generator_auc']:.3f}. The pristine six-generator aggregate is {interval(primary_dyn)}, the family-seen four-generator aggregate is {interval(secondary_dyn)}, and the full ten-generator aggregate is {interval(all_dyn)}. The weakest generator is {worst} at AUC {dynamics.loc[worst, 'roc_auc']:.3f}; the strongest is {best} at {dynamics.loc[best, 'roc_auc']:.3f}.",
        "",
        f"Dropping every prompt condition touched by the earlier 100-track AIME diagnostic gives primary AUC {strict_primary.roc_auc:.3f} over {int(strict_primary.n_conditions)} conditions and secondary AUC {strict_secondary.roc_auc:.3f} over {int(strict_secondary.n_conditions)} conditions.",
        "",
        f"Across the five pre-frozen 100-condition groups, dynamics AUC ranges from {group_dynamics.loc['primary_six','min']:.3f} to {group_dynamics.loc['primary_six','max']:.3f} for the primary six and from {group_dynamics.loc['secondary_four','min']:.3f} to {group_dynamics.loc['secondary_four','max']:.3f} for the secondary four. This group dispersion is reported descriptively and was not used to select a subset.",
        "",
        "## Per-generator metrics",
        "",
        "| Generator | Dynamics AUC | Dynamics BA | Rhythm AUC | Structure AUC | All-three AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in generator_order:
        rows = generator_rows[generator_rows.generator == model].set_index("feature_set")
        english.append(f"| {model} | {rows.loc['dynamics','roc_auc']:.3f} | {rows.loc['dynamics','balanced_accuracy']:.3f} | {rows.loc['rhythm','roc_auc']:.3f} | {rows.loc['structure','roc_auc']:.3f} | {rows.loc['all_three','roc_auc']:.3f} |")
    english += [
        "",
        "## Aggregate metrics",
        "",
        "| Cohort | Feature family | Status | ROC-AUC [95% CI] | Balanced accuracy [95% CI] | Human specificity | AI sensitivity |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for scenario, cohort in zip(aggregate_names, aggregate_labels):
        for feature in FEATURES:
            row = metrics[(metrics.scenario == scenario) & (metrics.feature_set == feature)].iloc[0]
            english.append(f"| {cohort} | {LABELS[feature]} | {row.status} | {interval(row)} | {interval(row, 'balanced_accuracy')} | {row.human_specificity:.3f} | {row.ai_sensitivity:.3f} |")
    english += [
        "",
        "## Feature coverage by source",
        "",
        "| Source | Dynamics complete | Rhythm complete | Structure complete | Median beats | Median sections |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in coverage_order:
        row = coverage[coverage.model == model].iloc[0]
        english.append(f"| {model} | {row.dynamics_complete_fraction:.3f} | {row.rhythm_complete_fraction:.3f} | {row.structure_complete_fraction:.3f} | {row.median_beats:.1f} | {row.median_sections:.1f} |")

    all_ten_univariate = univariate[univariate.scenario == "all_ten"].copy()
    all_ten_univariate["distance_from_chance"] = (all_ten_univariate.raw_higher_means_ai_auc - 0.5).abs()
    strongest_raw = all_ten_univariate.sort_values("distance_from_chance", ascending=False).iloc[0]
    english += [
        "",
        "## Coverage and interpretation",
        "",
        "Feature missingness is part of deployment behavior and was not filtered away. Ten-second rhythm requires at least 16 detected beats, while structure requires at least three predicted segments; coverage must therefore be interpreted together with classification. AUC below 0.5 is a score-direction failure on the external generator, not evidence of useful inverse prediction, because the direction was frozen from the old development set.",
        "",
        f"As a descriptive, non-selective check across all ten generators, the individual measurement farthest from chance is `{strongest_raw.feature}` with raw higher-means-AI AUC {strongest_raw.raw_higher_means_ai_auc:.3f} (finite Human n={int(strongest_raw.n_human_finite)}, AI n={int(strongest_raw.n_ai_finite)}). This is not a new threshold or a confirmatory feature selection.",
        "",
        "The result tests transfer of the three frozen heuristic workflows. It does not validate MIDI authorship, and it does not allow rhythm or structure into the final applied filter because those families failed their independent change-sensitivity gates.",
        "",
        f"![Dashboard](../figures/{dashboard.name})",
    ]
    (reports / "REPORT_EN.md").write_text("\n".join(english) + "\n", encoding="utf-8")

    chinese = [
        "# 三组 heuristic 在每模型 500 首外部测试集上的结果",
        "",
        "固定分类器只使用旧的 1,600 条 development 音频拟合；新的 5,500 条 AIME 音频不参与特征选择、归一化、缺失值统计、阈值选择或重新训练。训练和测试均使用严格 10 秒视图。",
        "",
        "只有非人声 dynamics 通过了此前的 controlled-change gate，可以作为 sensitivity-qualified 结果。Rhythm、structure 和三组联合结果均只能视为 exploratory。",
        "",
        f"Dynamics 的六个全新生成器 aggregate AUC 为 {interval(primary_dyn)}；四个 family-seen 生成器为 {interval(secondary_dyn)}；十模型合并为 {interval(all_dyn)}。十个模型的平均 per-generator AUC 为 {macro.loc['dynamics', 'mean_generator_auc']:.3f}。",
        "",
        "低于 0.5 的 AUC 表示旧 development 集确定的 score 方向在该外部生成器上发生反转，不能事后翻转方向后再宣称有效。10 秒片段对于 rhythm 和 section variety 的可观测性也明显受限，因此 coverage 必须和性能一起报告。",
        "",
        f"![结果图](../figures/{dashboard.name})",
    ]
    (reports / "REPORT_CN.md").write_text("\n".join(chinese) + "\n", encoding="utf-8")

    report_summary = {
        "dynamics_primary_auc": float(primary_dyn.roc_auc),
        "dynamics_primary_auc_ci": [float(primary_dyn.roc_auc_ci_low), float(primary_dyn.roc_auc_ci_high)],
        "dynamics_secondary_auc": float(secondary_dyn.roc_auc),
        "dynamics_all_ten_auc": float(all_dyn.roc_auc),
        "dynamics_macro_generator_auc": float(macro.loc["dynamics", "mean_generator_auc"]),
        "dynamics_worst_generator": worst,
        "dynamics_best_generator": best,
        "dashboard": str(dashboard),
    }
    (reports / "report_summary.json").write_text(json.dumps(report_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report_summary, indent=2))


if __name__ == "__main__":
    main()
