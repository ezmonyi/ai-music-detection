#!/usr/bin/env python3
"""Validate row counts, split invariants, structural parity and key hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    evaluation = root / "results" / "evaluation"
    spectral_path = root / "results" / "spectral" / "spectral_features.csv"
    features_path = evaluation / "four_family_features.csv"
    raw_path = evaluation / "per_generator_cv_results.csv"
    quantity_path = evaluation / "combination_by_quantity_k10.csv"
    rank_path = evaluation / "combination_full_400_ranked.csv"
    grid_path = evaluation / "diversity_quantity_grid.csv"
    model_path = evaluation / "deployment_model_all_500.json"

    spectral = pd.read_csv(spectral_path)
    features = pd.read_csv(features_path)
    raw = pd.read_csv(raw_path)
    quantity = pd.read_csv(quantity_path)
    rank = pd.read_csv(rank_path)
    grid = pd.read_csv(grid_path)
    model = json.loads(model_path.read_text())

    assert len(spectral) == spectral.track.nunique() == 7_100
    spectral_values = spectral.filter(regex=r"^spectral__").to_numpy(float)
    assert spectral_values.shape == (7_100, 16)
    assert np.all(np.isfinite(spectral_values))
    assert len(features) == features.track.nunique() == 7_100
    assert len(raw) == 42_000
    assert raw.combination.nunique() == 15
    assert set(raw.test_group) == {f"group_{index:02d}" for index in range(1, 6)}
    assert len(quantity) == 90
    assert set(quantity.amount_per_new_source) == {0, 25, 50, 100, 200, 400}
    assert len(rank) == 15
    assert len(grid) == 825
    assert float(features.structure_eligible.mean()) == 0.0
    assert model["combination"] == "S+D+R"
    assert model["n_rows"] == 7_100
    assert model["external_rows_per_source"] == 500

    # Adding the completely unavailable P family must not alter the score.
    lookup = quantity.set_index(["combination", "amount_per_new_source"])
    for combination in ("S", "D", "R", "S+D", "S+R", "D+R", "S+D+R"):
        with_p = "+".join([*combination.split("+"), "P"])
        for amount in (0, 25, 50, 100, 200, 400):
            left = float(lookup.loc[(combination, amount), "roc_auc_mean"])
            right = float(lookup.loc[(with_p, amount), "roc_auc_mean"])
            assert abs(left - right) < 1e-5, (combination, amount, left, right)

    # Every CV fold has exactly 100 Human and 100 examples for each generator.
    external = features[features.split == "external_test_frozen"]
    per_group = external.groupby(["group_id", "generator"]).size()
    assert len(per_group) == 55 and set(per_group) == {100}

    paths = [spectral_path, features_path, raw_path, quantity_path, rank_path, grid_path, model_path]
    output = {
        "status": "pass",
        "checks": {
            "spectral_rows": 7_100,
            "spectral_feature_count": 16,
            "spectral_nonfinite_values": 0,
            "joined_feature_rows": 7_100,
            "per_generator_cv_rows": 42_000,
            "combination_count": 15,
            "quantity_rows": 90,
            "diversity_grid_rows": 825,
            "cv_group_source_cells": 55,
            "rows_per_cv_group_source_cell": 100,
            "phrase_complete_coverage": 0.0,
            "phrase_parity_tolerance": 1e-5,
            "deployment_rows": 7_100,
            "deployment_external_rows_per_source": 500,
        },
        "sha256": {str(path.relative_to(root)): sha256(path) for path in paths},
    }
    destination = root / "VALIDATION_SUMMARY.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
