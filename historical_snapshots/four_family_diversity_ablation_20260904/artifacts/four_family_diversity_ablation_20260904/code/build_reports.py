#!/usr/bin/env python3
"""Build English and Chinese audit reports from the frozen four-family outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SELECTED = ["S", "D", "R", "P", "S+D", "S+R", "S+D+R", "S+D+R+P"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def f(value: float) -> str:
    return f"{float(value):.3f}"


def quantity_table(quantity: pd.DataFrame) -> list[str]:
    lines = [
        "| Combination | Legacy 0 | 25/source | 50/source | 100/source | 200/source | 400/source |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for combo in SELECTED:
        row = quantity[quantity.combination == combo].set_index("amount_per_new_source")
        lines.append(
            f"| {combo} | " + " | ".join(f(row.loc[n, "roc_auc_mean"]) for n in (0, 25, 50, 100, 200, 400)) + " |"
        )
    return lines


def full_combination_table(rank: pd.DataFrame) -> list[str]:
    lines = [
        "| Rank | Combination | Mean AUC | Fold SD | Balanced accuracy | Delta vs S |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for number, row in enumerate(rank.itertuples(index=False), 1):
        lines.append(
            f"| {number} | {row.combination} | {f(row.roc_auc_mean)} | {f(row.roc_auc_sd)} | "
            f"{f(row.balanced_accuracy_mean)} | {float(row.delta_auc_vs_spectral_only):+.3f} |"
        )
    return lines


def diversity_table(grid: pd.DataFrame, combo: str, amount: int) -> list[str]:
    selected = grid[(grid.combination == combo) & (grid.amount_per_new_source == amount)]
    lines = [
        "| New AI generators in training | Seen AUC | Unseen AUC | Unseen BA |",
        "|---:|---:|---:|---:|",
    ]
    for k in (1, 2, 4, 6, 8, 10):
        seen = selected[(selected.external_training_generator_count == k) & (selected.transfer_status == "seen")]
        unseen = selected[(selected.external_training_generator_count == k) & (selected.transfer_status == "unseen")]
        seen_text = f(float(seen.iloc[0].roc_auc_mean)) if len(seen) else "-"
        unseen_text = f(float(unseen.iloc[0].roc_auc_mean)) if len(unseen) else "-"
        ba_text = f(float(unseen.iloc[0].balanced_accuracy_mean)) if len(unseen) else "-"
        lines.append(f"| {k} | {seen_text} | {unseen_text} | {ba_text} |")
    return lines


def per_generator_table(raw: pd.DataFrame) -> list[str]:
    selected = raw[
        (raw.amount_per_new_source == 400)
        & (raw.external_training_generator_count == 10)
        & raw.combination.isin(["S", "S+D+R"])
    ]
    pivot = selected.groupby(["test_generator", "combination"]).roc_auc.mean().unstack()
    pivot["delta"] = pivot["S+D+R"] - pivot["S"]
    lines = [
        "| Test generator | S AUC | S+D+R AUC | Delta |",
        "|---|---:|---:|---:|",
    ]
    for generator, row in pivot.sort_values("S+D+R", ascending=False).iterrows():
        lines.append(
            f"| {generator} | {f(row['S'])} | {f(row['S+D+R'])} | {float(row['delta']):+.3f} |"
        )
    return lines


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank = pd.read_csv(args.evaluation / "combination_full_400_ranked.csv")
    quantity = pd.read_csv(args.evaluation / "combination_by_quantity_k10.csv")
    grid = pd.read_csv(args.evaluation / "diversity_quantity_grid.csv")
    raw = pd.read_csv(args.evaluation / "per_generator_cv_results.csv")
    summary = json.loads((args.evaluation / "evaluation_summary.json").read_text())
    best = summary["cv_leading_combination"]

    en = [
        "# Four-family combination and data-diversity ablation",
        "",
        "## Outcome",
        "",
        "Adding the high-frequency spectral family changes the result materially. The CV-leading "
        f"pipeline is `{best}` (spectral + non-vocal dynamics + rhythm): generator-macro AUC "
        f"{summary['cv_leading_k10_n400_auc']:.3f} and balanced accuracy "
        f"{summary['cv_leading_k10_n400_ba']:.3f} when trained on 400 condition-matched AIME rows "
        "per source in each fold. Spectral alone reaches AUC "
        f"{summary['spectral_only_k10_n400_auc']:.3f}; dynamics and rhythm add only +"
        f"{summary['cv_leading_k10_n400_auc']-summary['spectral_only_k10_n400_auc']:.3f} in aggregate.",
        "",
        "This is exploratory five-fold development evidence, not a new external confirmation. The "
        "former AIME external test has now been consumed for training/model selection at the user's "
        "request. A fresh untouched corpus is required before deployment claims.",
        "",
        "## Protocol",
        "",
        "- 7,100 tracks: 1,600 legacy development tracks plus 5,500 AIME tracks.",
        "- Five condition-disjoint folds; one frozen 100-condition group is held out per fold.",
        "- New training amounts: 25, 50, 100, 200 or 400 paired conditions per source.",
        "- New AI-generator counts: 1, 2, 4, 6, 8 or 10; deterministic balanced cyclic subsets.",
        "- Fifteen non-empty combinations of `S`, `D`, `R`, and `P`.",
        "- Source-balanced ridge score, fixed penalty 10 and threshold 0.5; no hyperparameter tuning.",
        "- Primary metric: mean Human-vs-generator ROC-AUC, macro-averaged across generators.",
        "",
        "`S` contains 16 MUSDB-bias-corrected high-frequency metrics measured on the Demucs vocal "
        "stem. Frame membership is frozen from raw-vocal RMS. `D`, `R`, and `P` retain the earlier "
        "definitions.",
        "",
        "## Quantity comparison (all ten new generators in training)",
        "",
        *quantity_table(quantity),
        "",
        "The largest jump is domain/generator adaptation, not raw sample count: `S+D+R` rises from "
        "0.441 with legacy-only training to 0.660 with only 25 new conditions per source. Increasing "
        "25 to 400 adds another 0.035 AUC. The CV leader stabilises by about 100-200 conditions per "
        "source.",
        "",
        "![All four-family combinations by quantity](../figures/four_family_combination_quantity_table.png)",
        "",
        "## All 15 combinations at 400/source",
        "",
        *full_combination_table(rank),
        "",
        "The phrase/section family has zero complete-feature coverage at ten seconds. Therefore "
        "every combination containing `P` duplicates the corresponding combination without it; this "
        "is an observability failure, not evidence that musical structure is irrelevant.",
        "",
        "## Generator diversity is more important than more examples from one generator",
        "",
        *diversity_table(grid, best, 400),
        "",
        "At 400/source, unseen-generator AUC rises from 0.472 with one new AI generator family in "
        "training to 0.630 with eight. By contrast, at eight training generators, increasing from "
        "25 to 400 conditions/source changes unseen AUC only from 0.618 to 0.630. This supports the "
        "data-diversity diagnosis: adding generator families matters much more than adding more "
        "tracks from one or two families.",
        "",
        "![Unseen-generator diversity curves](../figures/unseen_generator_diversity_curves.png)",
        "",
        "## Per-generator effect of adding dynamics and rhythm",
        "",
        *per_generator_table(raw),
        "",
        "The +0.010 macro gain is not uniform. Dynamics/rhythm help AudioLDM 2 Large and the three "
        "MusicGen variants, but reduce AUC for Riffusion and both Stable Audio variants. The combined "
        "pipeline still contains generator-specific directions and must expose per-generator results.",
        "",
        "## Diversity limitation and decision",
        "",
        "The AI side is now substantially broader, but Human diversity remains inadequate: all 500 "
        "new Human controls come from MTG-Jamendo, while the only other Human source is legacy FMA. "
        "AIME also packages all models in one benchmark and derives prompts from a shared tag system. "
        "Consequently AUC 0.695 is an AIME-domain cross-validation estimate, not a production-wide "
        "AI-music detector result.",
        "",
        "The saved `deployment_model_all_500.json` consumes all 500 AIME rows per source and uses "
        f"`{best}`. It is a deployment candidate only: because no AIME row remains untouched, no "
        "test score is attached to that refit. The next confirmatory collection must add independent "
        "Human catalogues, codecs/mastering chains, genres and unseen generator families.",
        "",
    ]
    (args.output_dir / "REPORT_EN.md").write_text("\n".join(en), encoding="utf-8")

    zh = [
        "# 四类特征组合与数据多样性消融",
        "",
        "## 结论",
        "",
        f"加入高频频谱后，CV 最优组合为 `{best}`：每折每来源使用 400 条时，generator-macro "
        f"ROC-AUC={summary['cv_leading_k10_n400_auc']:.3f}，balanced accuracy="
        f"{summary['cv_leading_k10_n400_ba']:.3f}。频谱单独为 "
        f"{summary['spectral_only_k10_n400_auc']:.3f}，dynamics+rhythm 的总体增量只有 +"
        f"{summary['cv_leading_k10_n400_auc']-summary['spectral_only_k10_n400_auc']:.3f}。",
        "",
        "原 AIME 外部测试集在本实验中已经被转为开发/CV 数据，不能再称为 untouched external "
        "test。全部 500 条/来源通过五折实验参与，但每一折严格使用 400 训练、100 测试。",
        "",
        "## 训练数量表（十个新生成器均进入训练）",
        "",
        *quantity_table(quantity),
        "",
        "`S+D+R` 从 legacy-only 的 0.441，在每来源加入 25 条后跃升到 0.660；继续从 25 "
        "增加到 400 只再增加 0.035。主要收益来自跨域/跨生成器覆盖，而不是单来源继续堆量。",
        "",
        "## 多样性表（S+D+R，400/来源）",
        "",
        *diversity_table(grid, best, 400),
        "",
        "训练只覆盖 1 个新生成器时，未见生成器 AUC=0.472；覆盖 8 个时提高到 0.630。"
        "这直接说明当前 pipeline 的主要瓶颈是 generator diversity。",
        "",
        "## 边界",
        "",
        "AI 侧多样性已明显增加，但 Human 侧仍只有 MTG-Jamendo 加旧 FMA 两个来源，而且 "
        "AIME 的十个模型共享 benchmark packaging 和 tag prompt 体系。因此 0.695 只是 AIME "
        "域内五折估计。全 500 条重拟合模型已保存，但必须在新的 Human catalogue 和新生成器 "
        "holdout 上再次验证。",
        "",
    ]
    (args.output_dir / "REPORT_CN.md").write_text("\n".join(zh), encoding="utf-8")


if __name__ == "__main__":
    main()
