#!/usr/bin/env python3
"""Compare preregistered and post-hoc forced-choice Music Flamingo runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--primary", type=Path, required=True)
    p.add_argument("--forced", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def load_latest(path: Path) -> list[dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["query_id"]] = row
    return list(rows.values())


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def physical_choice(row: dict) -> str | None:
    if row.get("parsed_choice") == "A":
        return row["audio_a_window_id"]
    if row.get("parsed_choice") == "B":
        return row["audio_b_window_id"]
    return None


def summarize(rows: list[dict], condition: str, task: str) -> dict:
    subset = [x for x in rows if x["task"] == task]
    primary = [x for x in subset if int(x["order_reversal"]) == 0]
    correct = sum(bool(x.get("correct")) for x in primary)
    lo, hi = wilson(correct, len(primary))
    choices = Counter(x.get("parsed_choice") or "INVALID" for x in subset)
    by_pair = defaultdict(list)
    for row in subset:
        by_pair[row["pair_id"]].append(row)
    consistent = sum(
        len(pair_rows) == 2
        and physical_choice(pair_rows[0]) is not None
        and physical_choice(pair_rows[0]) == physical_choice(pair_rows[1])
        for pair_rows in by_pair.values()
    )
    ab = choices["A"] + choices["B"]
    return {
        "condition": condition,
        "task": task,
        "pairs": len(primary),
        "accuracy": correct / len(primary),
        "ci_low": lo,
        "ci_high": hi,
        "order_consistency": consistent / len(by_pair),
        "first_position_rate": choices["A"] / ab if ab else math.nan,
        "tie_rate": choices["TIE"] / len(subset),
        "choice_a": choices["A"],
        "choice_b": choices["B"],
        "choice_tie": choices["TIE"],
        "queries": len(subset),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary = load_latest(args.primary)
    forced = load_latest(args.forced)
    summaries = [
        summarize(rows, condition, task)
        for condition, rows in (("Primary", primary), ("Forced A/B", forced))
        for task in ("dynamics", "rhythm")
    ]
    with (args.output_dir / "condition_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    x = np.arange(2)
    width = 0.34
    colors = {"Primary": "#2b6cb0", "Forced A/B": "#dd6b20"}
    for offset, condition in ((-width / 2, "Primary"), (width / 2, "Forced A/B")):
        rows = [next(x for x in summaries if x["condition"] == condition and x["task"] == task) for task in ("dynamics", "rhythm")]
        values = np.asarray([x["accuracy"] for x in rows])
        errors = np.asarray([
            [x["accuracy"] - x["ci_low"] for x in rows],
            [x["ci_high"] - x["accuracy"] for x in rows],
        ])
        axes[0].bar(x + offset, values, width, label=condition, color=colors[condition])
        axes[0].errorbar(x + offset, values, yerr=errors, fmt="none", ecolor="black", capsize=4)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, ["Dynamics", "Rhythm"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy on one randomized order / pair")
    axes[0].legend(frameon=False)

    labels = ["Primary\nDynamics", "Primary\nRhythm", "Forced\nDynamics", "Forced\nRhythm"]
    order = [
        next(x for x in summaries if x["condition"] == condition and x["task"] == task)
        for condition, task in (("Primary", "dynamics"), ("Primary", "rhythm"), ("Forced A/B", "dynamics"), ("Forced A/B", "rhythm"))
    ]
    bottom = np.zeros(4)
    for key, label, color in (("choice_a", "A (first)", "#4c78a8"), ("choice_b", "B (second)", "#f58518"), ("choice_tie", "TIE", "#bab0ac")):
        vals = np.asarray([row[key] / row["queries"] for row in order])
        axes[1].bar(np.arange(4), vals, bottom=bottom, label=label, color=color)
        bottom += vals
    axes[1].set_xticks(np.arange(4), labels)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Response fraction across both orders")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "primary_vs_forced_choice.png", dpi=200)
    plt.close(fig)

    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
