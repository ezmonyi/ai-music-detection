#!/usr/bin/env python3
"""HeartMuLa-vs-Human sensitivity analysis excluding padded AI generations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_demucs_artifacts as artifactlib
import cross_generator_generalization as crosslib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--standardization-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-native-duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20_260_901)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def deterministic_human(indices: np.ndarray, tracks: np.ndarray, n: int) -> np.ndarray:
    ranked = sorted(
        indices,
        key=lambda index: hashlib.sha256(
            f"heartmula-duration-robust-v1|{tracks[index]}".encode()
        ).hexdigest(),
    )
    return np.asarray(ranked[:n], dtype=int)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    domain = crosslib.load_domain(args.manifest, args.metrics_dir)
    standardization = read_jsonl(args.standardization_jsonl)
    native_duration = {
        f"ai_heartmula_{row['id']}": float(row["source_duration_s"])
        for row in standardization
        if row.get("status") == "ok"
    }
    labels = np.asarray(domain["labels"])
    splits = np.asarray(domain["splits"])
    tracks = np.asarray(domain["tracks"])
    long_ai = np.asarray(
        [native_duration.get(str(track), 0.0) >= args.minimum_native_duration for track in tracks]
    )

    selected: dict[tuple[str, int], np.ndarray] = {}
    counts: dict[str, int] = {}
    for split in ("development", "locked_test"):
        ai = np.flatnonzero((labels == 1) & (splits == split) & long_ai)
        human_candidates = np.flatnonzero((labels == 0) & (splits == split))
        human = deterministic_human(human_candidates, tracks, ai.size)
        selected[(split, 0)] = human
        selected[(split, 1)] = ai
        counts[f"{split}_human"] = int(human.size)
        counts[f"{split}_ai"] = int(ai.size)

    rows = []
    for representation in crosslib.REPRESENTATIONS:
        matrix = np.asarray(domain["matrices"][representation])
        human_train = matrix[selected[("development", 0)]]
        ai_train = matrix[selected[("development", 1)]]
        human_test = matrix[selected[("locked_test", 0)]]
        ai_test = matrix[selected[("locked_test", 1)]]
        test_x = np.concatenate((human_test, ai_test), axis=0)
        test_y = np.concatenate(
            (np.zeros(human_test.shape[0], dtype=int), np.ones(ai_test.shape[0], dtype=int))
        )
        scores = crosslib.balanced_fit_predict(human_train, ai_train, test_x)
        balanced_accuracy, roc_auc = artifactlib.metrics(test_y, scores)
        tag_offset = sum(
            (index + 1) * ord(char) for index, char in enumerate(representation)
        )
        ba_low, ba_high, auc_low, auc_high = artifactlib.bootstrap_intervals(
            test_y, scores, args.seed + tag_offset, repeats=1_000
        )
        rows.append(
            {
                "representation": representation,
                **counts,
                "balanced_accuracy": balanced_accuracy,
                "balanced_accuracy_ci_low": ba_low,
                "balanced_accuracy_ci_high": ba_high,
                "roc_auc": roc_auc,
                "roc_auc_ci_low": auc_low,
                "roc_auc_ci_high": auc_high,
            }
        )
    write_csv(args.output_dir / "classification_ablation.csv", rows)
    corrected = [row for row in rows if str(row["representation"]).startswith("bias_corrected__")]
    best = max(corrected, key=lambda row: float(row["roc_auc"]))
    lines = [
        "# HeartMuLa native-duration sensitivity analysis",
        "",
        f"AI generations shorter than {args.minimum_native_duration:g} s are excluded before fitting. Human rows are deterministically downsampled within each split to the same count. This is a post-generation sensitivity analysis, not the confirmatory cohort.",
        "",
        f"Counts: development {counts['development_human']} Human + {counts['development_ai']} HeartMuLa; locked test {counts['locked_test_human']} + {counts['locked_test_ai']}.",
        "",
        f"Best corrected representation: `{str(best['representation']).removeprefix('bias_corrected__')}`, locked AUC {float(best['roc_auc']):.3f} ({float(best['roc_auc_ci_low']):.3f}-{float(best['roc_auc_ci_high']):.3f}), balanced accuracy {float(best['balanced_accuracy']):.3f}.",
        "",
        "If this remains high, trailing zero padding is not sufficient to explain the HeartMuLa result; it still does not remove genre, mastering, or separator-domain confounds.",
        "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    metadata = {
        "minimum_native_duration_s": args.minimum_native_duration,
        "counts": counts,
        "human_selection": "smallest salted SHA-256 track keys within split",
        "bootstrap_repeats": 1000,
        "confirmatory": False,
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
