#!/usr/bin/env python3
"""Analyze frozen Music Flamingo pairwise dynamics/rhythm judgments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, spearmanr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def load_latest(path: Path) -> list[dict]:
    latest: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                latest[row["query_id"]] = row
    return list(latest.values())


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def chosen_window(row: dict) -> str | None:
    choice = row.get("parsed_choice")
    if choice == "A":
        return row["audio_a_window_id"]
    if choice == "B":
        return row["audio_b_window_id"]
    return None


def task_summary(rows: list[dict], task: str) -> dict:
    subset = [x for x in rows if x["task"] == task]
    primary = [x for x in subset if int(x.get("order_reversal", -1)) == 0]
    n = len(primary)
    parsed_primary = [x for x in primary if x.get("exact_json_parse")]
    parsed_all = [x for x in subset if x.get("exact_json_parse")]
    correct = sum(bool(x.get("correct")) for x in primary)
    a_or_b = [x for x in parsed_all if x.get("parsed_choice") in {"A", "B"}]
    ci_low, ci_high = wilson(correct, n)
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in subset:
        by_pair[row["pair_id"]].append(row)
    consistent = 0
    both_correct = 0
    complete_pairs = 0
    for pair_rows in by_pair.values():
        if len(pair_rows) != 2:
            continue
        complete_pairs += 1
        picks = [chosen_window(x) for x in pair_rows]
        consistent += int(picks[0] is not None and picks[0] == picks[1])
        both_correct += int(all(bool(x.get("correct")) for x in pair_rows))
    gaps = np.asarray([float(x["target_gap"]) for x in primary]) if primary else np.asarray([])
    gap_median = float(np.median(gaps)) if len(gaps) else math.nan
    low_gap = [x for x in primary if float(x["target_gap"]) <= gap_median]
    nuisance = np.asarray([float(x["nuisance_gap"]) for x in primary]) if primary else np.asarray([])
    nuisance_median = float(np.median(nuisance)) if len(nuisance) else math.nan
    low_nuisance = [x for x in primary if float(x["nuisance_gap"]) <= nuisance_median]
    corr = spearmanr(
        [float(x["target_gap"]) for x in primary],
        [int(bool(x.get("correct"))) for x in primary],
    ) if len(primary) >= 3 else None
    return {
        "task": task,
        "queries": len(subset),
        "primary_queries": n,
        "physical_pairs": len(by_pair),
        "accuracy": correct / n if n else math.nan,
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
        "correct": correct,
        "binomial_p_vs_half": float(binomtest(correct, n, 0.5, alternative="greater").pvalue) if n else math.nan,
        "json_parse_rate": len(parsed_all) / len(subset) if subset else math.nan,
        "primary_json_parse_rate": len(parsed_primary) / n if n else math.nan,
        "tie_rate": sum(x.get("parsed_choice") == "TIE" for x in parsed_all) / len(subset) if subset else math.nan,
        "first_position_rate_among_ab": sum(x.get("parsed_choice") == "A" for x in a_or_b) / len(a_or_b) if a_or_b else math.nan,
        "order_consistency": consistent / complete_pairs if complete_pairs else math.nan,
        "both_orders_correct_rate": both_correct / complete_pairs if complete_pairs else math.nan,
        "median_target_gap": gap_median,
        "small_target_gap_accuracy": sum(bool(x.get("correct")) for x in low_gap) / len(low_gap) if low_gap else math.nan,
        "median_nuisance_gap": nuisance_median,
        "small_nuisance_gap_accuracy": sum(bool(x.get("correct")) for x in low_nuisance) / len(low_nuisance) if low_nuisance else math.nan,
        "target_gap_correctness_spearman": float(corr.statistic) if corr is not None else math.nan,
        "target_gap_correctness_p": float(corr.pvalue) if corr is not None else math.nan,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_latest(args.state)
    summaries = [task_summary(rows, task) for task in ("dynamics", "rhythm")]
    write_csv(args.output_dir / "task_summary.csv", summaries)

    style_rows = []
    primary_rows = [x for x in rows if int(x.get("order_reversal", -1)) == 0]
    for (task, style), subset in sorted(
        ((key, list(group)) for key, group in _group(primary_rows, lambda x: (x["task"], x["style"].split("/")[0]))),
        key=lambda x: x[0],
    ):
        n = len(subset)
        k = sum(bool(x.get("correct")) for x in subset)
        lo, hi = wilson(k, n)
        style_rows.append({"task": task, "style": style, "n": n, "accuracy": k / n, "ci_low": lo, "ci_high": hi})
    write_csv(args.output_dir / "style_summary.csv", style_rows)

    error_counts = Counter(
        "runtime_error" if x.get("status") != "ok" else
        "invalid_json" if not x.get("exact_json_parse") else
        "tie" if x.get("parsed_choice") == "TIE" else
        "wrong_ab" if not x.get("correct") else "correct"
        for x in rows
    )
    result = {
        "tasks": summaries,
        "total_latest_queries": len(rows),
        "outcome_counts": dict(error_counts),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    x = np.arange(len(summaries))
    values = np.asarray([s["accuracy"] for s in summaries])
    yerr = np.asarray([
        [s["accuracy"] - s["accuracy_ci_low"] for s in summaries],
        [s["accuracy_ci_high"] - s["accuracy"] for s in summaries],
    ])
    ax.bar(x, values, color=["#2b6cb0", "#dd6b20"], width=0.58)
    ax.errorbar(x, values, yerr=yerr, fmt="none", ecolor="black", capsize=5)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x, ["Dynamics", "Rhythm"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Order-balanced pairwise accuracy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "pairwise_accuracy.png", dpi=180)
    plt.close(fig)

    report = [
        "# Music Flamingo MIDI-Dynamics and Rhythm-Variation Probe",
        "",
        "## Primary results",
        "",
        "| Task | Primary / all queries | Physical pairs | Primary-order accuracy (95% Wilson CI) | Order consistency | JSON parse (all orders) | First-position rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        report.append(
            f"| {s['task']} | {s['primary_queries']} / {s['queries']} | {s['physical_pairs']} | "
            f"{fmt(s['accuracy'])} ({fmt(s['accuracy_ci_low'])}--{fmt(s['accuracy_ci_high'])}) | "
            f"{fmt(s['order_consistency'])} | {fmt(s['json_parse_rate'])} | "
            f"{fmt(s['first_position_rate_among_ab'])} |"
        )
    report += [
        "",
        "The primary accuracy uses only the seed-randomized first ordering of each physical pair. TIE, invalid JSON, and runtime errors are counted as incorrect. The reversed ordering is used only for order consistency and position-bias diagnostics.",
        "",
        "## Confound checks",
        "",
        "| Task | Accuracy at smaller target gaps | Accuracy at smaller nuisance gaps | Target-gap/correctness Spearman | p vs. chance |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in summaries:
        report.append(
            f"| {s['task']} | {fmt(s['small_target_gap_accuracy'])} | "
            f"{fmt(s['small_nuisance_gap_accuracy'])} | "
            f"{fmt(s['target_gap_correctness_spearman'])} | {s['binomial_p_vs_half']:.4g} |"
        )
    report += [
        "",
        "## Frozen admission-rule decision",
        "",
    ]
    for s in summaries:
        passed = (
            s["accuracy_ci_low"] > 0.5
            and s["order_consistency"] >= 0.75
            and s["json_parse_rate"] >= 0.95
            and s["small_nuisance_gap_accuracy"] > 0.5
        )
        report.append(f"- **{s['task']}**: {'PASS' if passed else 'FAIL'} the numerical portions of the frozen admission rule.")
    report += [
        "",
        "Style-direction stability must also be inspected in `style_summary.csv`; small style cells are descriptive only.",
        "",
        "## Interpretation boundary",
        "",
        "This controlled GMD result measures pairwise perception on non-vocal electronic-drum performances. It does not establish continuous velocity estimation, pitched-instrument generalization, or AI-versus-Human discrimination. A model-derived judgment may enter the next detector ablation only when the complete frozen admission rule passes.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def _group(rows: list[dict], key_fn):
    groups: dict[object, list[dict]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return groups.items()


if __name__ == "__main__":
    main()
