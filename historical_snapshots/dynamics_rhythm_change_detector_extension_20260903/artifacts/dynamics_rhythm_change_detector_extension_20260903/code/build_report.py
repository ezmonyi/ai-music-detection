#!/usr/bin/env python3
"""Build auditable reports and a compact dashboard from frozen result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DISPLAY = {
    "dynamics": "Dynamics",
    "rhythm": "Rhythm change",
    "structure": "Section/bar variety",
    "spectral": "Spectral baseline",
    "all_new": "All new",
    "spectral_plus_dynamics": "Spectral + dynamics",
    "spectral_plus_rhythm": "Spectral + rhythm",
    "spectral_plus_structure": "Spectral + structure",
    "spectral_plus_all_new": "Spectral + all new",
    "sensitivity_qualified": "Qualified only",
    "spectral_plus_sensitivity_qualified": "Spectral + qualified",
}


def f3(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def metrics_table(frame: pd.DataFrame, scenarios: list[str]) -> str:
    rows = ["| Scenario | Representation | BA | ROC-AUC | 95% AUC CI |", "|---|---|---:|---:|---:|"]
    wanted = [
        "dynamics", "rhythm", "structure", "all_new", "spectral",
        "spectral_plus_dynamics", "spectral_plus_rhythm",
        "spectral_plus_structure", "spectral_plus_all_new",
        "spectral_plus_sensitivity_qualified",
    ]
    for scenario in scenarios:
        part = frame[frame.scenario == scenario].set_index("feature_set")
        for name in wanted:
            if name not in part.index:
                continue
            row = part.loc[name]
            rows.append(
                f"| {scenario} | {DISPLAY.get(name, name)} | {f3(row.balanced_accuracy)} | "
                f"{f3(row.roc_auc)} | [{f3(row.roc_auc_ci_low)}, {f3(row.roc_auc_ci_high)}] |"
            )
    return "\n".join(rows)


def delta_table(frame: pd.DataFrame) -> str:
    rows = ["| Scenario | Addition | Delta AUC | 95% CI | P(delta>0) |", "|---|---|---:|---:|---:|"]
    for _, row in frame.iterrows():
        rows.append(
            f"| {row.scenario} | {DISPLAY.get(row.augmented_feature_set, row.augmented_feature_set)} | "
            f"{f3(row.delta_auc)} | [{f3(row.ci_low)}, {f3(row.ci_high)}] | "
            f"{f3(row.probability_delta_positive)} |"
        )
    return "\n".join(rows)


def sensitivity_table(frame: pd.DataFrame) -> str:
    rows = [
        "| Family | Comparison | Role | Feature | n | Median delta | P(superiority) | 95% CI |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        rows.append(
            f"| {DISPLAY.get(row.family, row.family)} | {row.comparison} | {row.role} | "
            f"{row.feature} | {int(row.n)} | {f3(row.median_delta)} | "
            f"{f3(row.probability_superiority)} | [{f3(row.ci_low)}, {f3(row.ci_high)}] |"
        )
    return "\n".join(rows)


def coverage_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---:" if column not in ("source", "split") else "---" for column in columns) + "|",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            cells.append(f3(float(value)) if isinstance(value, (float, np.floating)) else str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def build_figure(summary: dict, metrics: pd.DataFrame, deltas: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    families = ["dynamics", "rhythm", "structure"]
    point = np.asarray([summary["family_results"][name]["probability_superiority"] for name in families])
    low = np.asarray([summary["family_results"][name]["ci_low"] for name in families])
    high = np.asarray([summary["family_results"][name]["ci_high"] for name in families])
    colors = ["#2a9d8f" if summary["family_results"][name]["sensitivity_pass"] else "#e76f51" for name in families]
    axes[0].bar(range(3), point, color=colors)
    axes[0].errorbar(range(3), point, yerr=np.vstack((point - low, high - point)), fmt="none", color="black", capsize=4)
    axes[0].axhline(0.75, color="#264653", linestyle="--", linewidth=1.3, label="point gate = 0.75")
    axes[0].set_xticks(range(3), [DISPLAY[name] for name in families], rotation=15, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Probability of superiority")
    axes[0].set_title("A. External controlled-change sensitivity")
    axes[0].legend(frameon=False, fontsize=8)

    pooled = metrics[metrics.scenario == "pooled_all"].set_index("feature_set")
    names = [name for name in [
        "dynamics", "rhythm", "structure", "all_new", "spectral",
        "spectral_plus_dynamics", "spectral_plus_rhythm",
        "spectral_plus_structure", "spectral_plus_all_new",
    ] if name in pooled.index]
    vals = [pooled.loc[name, "roc_auc"] for name in names]
    axes[1].barh(range(len(names)), vals, color=["#457b9d" if "spectral" in name else "#a8dadc" for name in names])
    axes[1].set_yticks(range(len(names)), [DISPLAY.get(name, name) for name in names], fontsize=8)
    axes[1].set_xlim(0.45, 1.01)
    axes[1].set_xlabel("Locked-test ROC-AUC")
    axes[1].set_title("B. Pooled Human vs. all AI")
    axes[1].invert_yaxis()

    focus = deltas[deltas.augmented_feature_set == "spectral_plus_all_new"].copy()
    focus["order"] = focus.scenario.map({
        "within_suno": 0, "within_heartmula": 1, "within_acestep": 2,
        "pooled_all": 3, "leave_out_suno": 4, "leave_out_heartmula": 5,
        "leave_out_acestep": 6,
    })
    focus = focus.sort_values("order")
    y = np.arange(len(focus))
    values = focus.delta_auc.to_numpy(float)
    lo = focus.ci_low.to_numpy(float)
    hi = focus.ci_high.to_numpy(float)
    axes[2].errorbar(values, y, xerr=np.vstack((values - lo, hi - values)), fmt="o", color="#6d597a", capsize=3)
    axes[2].axvline(0, color="black", linewidth=1)
    axes[2].set_yticks(y, focus.scenario.str.replace("_", " "), fontsize=8)
    axes[2].set_xlabel("Delta ROC-AUC vs. spectral")
    axes[2].set_title("C. Exploratory all-new increment")
    axes[2].invert_yaxis()

    fig.suptitle("Dynamics, rhythm, and section/bar-length detector extension", fontsize=14, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-results", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.external_results / "controlled_change_summary.json").read_text())
    sensitivity = pd.read_csv(args.external_results / "controlled_change_sensitivity.csv")
    metrics = pd.read_csv(args.evaluation_dir / "detector_metrics.csv")
    deltas = pd.read_csv(args.evaluation_dir / "incremental_auc_deltas.csv")
    coverage = pd.read_csv(args.evaluation_dir / "feature_coverage.csv")
    metadata = json.loads((args.evaluation_dir / "evaluation_metadata.json").read_text())

    fig_path = args.output_root / "figures" / "change_detector_extension_dashboard.png"
    build_figure(summary, metrics, deltas, fig_path)

    external_rows = []
    for family in ("dynamics", "rhythm", "structure"):
        row = summary["family_results"][family]
        external_rows.append(
            f"| {DISPLAY[family]} | {row['n']} | {f3(row['probability_superiority'])} | "
            f"[{f3(row['ci_low'])}, {f3(row['ci_high'])}] | {'PASS' if row['sensitivity_pass'] else 'FAIL'} |"
        )
    external_table = "\n".join([
        "| Family | Pairs | Probability of superiority | 95% CI | Frozen gate |",
        "|---|---:|---:|---:|---:|", *external_rows,
    ])
    qualified = ", ".join(metadata["external_family_status"].keys()) if summary["all_families_pass"] else ", ".join(
        name for name, passed in metadata["external_family_status"].items() if passed
    )

    en = f"""# Controlled-change validation and AI/Human detector extension

## Executive result

The two questions must remain separate. The external controlled-change gate passed only for **dynamics**. Rhythm and section/bar-length variety failed their predeclared family-level sensitivity gates, although tempo entropy was individually responsive. All three families were nevertheless applied to the frozen 2,000-track Human/Suno/HeartMuLa/ACE-Step cohort as requested; results involving non-qualified families are explicitly exploratory and cannot retroactively validate them.

Sensitivity-qualified families: **{qualified or 'none'}**.

## External controlled-change validation

{external_table}

The frozen family gate required point probability-of-superiority >= 0.75 and bootstrap lower 95% bound > 0.60. Constant-gain and constant-tempo negative controls, individual features, and raw paired outputs are retained in `external_results/controlled_change_sensitivity.csv` and `controlled_change_features.csv`.

{sensitivity_table(sensitivity)}

## Application to the existing AI/Human cohort

{metrics_table(metrics, ['within_suno', 'within_heartmula', 'within_acestep', 'pooled_all'])}

## Cross-generator and incremental evidence

{metrics_table(metrics, ['leave_out_suno', 'leave_out_heartmula', 'leave_out_acestep'])}

{delta_table(deltas)}

## Measurement coverage

{coverage_table(coverage)}

## Interpretation boundary

- A positive locked-test delta shows incremental association in this cohort, not universal authorship evidence.
- Dynamics is the only externally sensitivity-qualified family in this run.
- Rhythm and structure results are exploratory because their family gates failed before AI/Human evaluation.
- All-In-One frequently returned too few sections in 30-second excerpts; missingness was median-imputed from development data and explicitly indicated to the ridge model.
- Within-generator performance may exploit production or generator fingerprints. Leave-one-generator-out results are the stronger generalization check.
- The pre-existing independent external AI/Human test was not used for representation selection and was not reopened in this extension.

![Dashboard](../figures/change_detector_extension_dashboard.png)
"""

    zh = f"""# 可控变化验证与 AI/Human 判别扩展

## 核心结论

本实验严格拆分两个问题：外部数据上能否测到人为施加的变化，以及这些数值在既有 AI/Human 音乐中是否带来增量判别信息。外部预注册门槛只有 **Dynamics** 通过；Rhythm 和乐句/乐段小节长度的 family composite 未通过。按照需求，三类仍全部应用到固定的 2,000 首 Human/Suno/HeartMuLa/ACE-Step 音乐，但未通过外部门槛的结果只作探索性分析，不能用 AI/Human 上的高分反向证明测量有效。

通过外部敏感性门槛的 family：**{qualified or '无'}**。

## 外部可控变化检验

{external_table}

门槛为优势概率点估计 >= 0.75 且 bootstrap 95% CI 下界 > 0.60。固定增益、固定速度负对照、各单项指标及配对原始结果均保留。

{sensitivity_table(sensitivity)}

## 既有 AI/Human 音乐上的应用

{metrics_table(metrics, ['within_suno', 'within_heartmula', 'within_acestep', 'pooled_all'])}

## 跨生成器与相对频谱基线的增量

{metrics_table(metrics, ['leave_out_suno', 'leave_out_heartmula', 'leave_out_acestep'])}

{delta_table(deltas)}

## 特征覆盖率

{coverage_table(coverage)}

## 解释边界

- 锁定测试集上的正增量只代表本 cohort 内的额外相关信息，不是通用作者身份信号。
- 本轮只有 Dynamics 获得外部变化敏感性验证。
- Rhythm 与 Structure 的 family gate 预先失败，因此其 AI/Human 消融只能标为 exploratory。
- All-In-One 在 30 秒窗口中经常只返回很少的段落；缺失值只用 development split 的中位数填补，并给分类器加入缺失指示变量。
- 单生成器高分可能依赖制作风格或生成器指纹，leave-one-generator-out 更能反映泛化。
- 本轮未重开、也未据此选择既有的独立外部 AI/Human 测试集。

![Dashboard](../figures/change_detector_extension_dashboard.png)
"""

    reports = args.output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "REPORT_EN.md").write_text(en, encoding="utf-8")
    (reports / "REPORT_ZH.md").write_text(zh, encoding="utf-8")
    print(json.dumps({
        "report_en": str(reports / "REPORT_EN.md"),
        "report_zh": str(reports / "REPORT_ZH.md"),
        "figure": str(fig_path),
        "n_metrics": len(metrics), "n_deltas": len(deltas),
    }, indent=2))


if __name__ == "__main__":
    main()
