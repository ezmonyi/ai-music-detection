#!/usr/bin/env python3
"""Synthetic end-to-end tests for evaluate_expanded.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_expanded as evaluator


HERE = Path(__file__).resolve().parent
PROGRAM = HERE / "evaluate_expanded.py"
REPORT_PROGRAM = HERE / "evaluate_expanded_report.py"
BOOTSTRAP_PROGRAM = HERE / "evaluate_expanded_group_bootstrap.py"


class ExpandedEvaluationTest(unittest.TestCase):
    def test_two_row_extractor_schema_coalesces_authoritative_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.csv"
            features = root / "features.csv"
            pd.DataFrame([
                {"id": "a", "label": 0, "role": "development", "source_id": "human",
                 "group_id": "artist:a", "native_sample_rate_hz": 44100, "duration_sec": 30,
                 "available": 1},
                {"id": "b", "label": 1, "role": "development", "source_id": "generator",
                 "group_id": "prompt:b", "native_sample_rate_hz": 48000, "duration_sec": 30,
                 "available": 1},
            ]).to_csv(metadata, index=False)
            pd.DataFrame([
                {"item_id": "a", "label": 0, "source_id": "human", "group_id": "artist:a",
                 "native_sample_rate_hz": 44100, "duration_sec": 30.0, "s8__x": 0.1},
                {"item_id": "b", "label": 1, "source_id": "generator", "group_id": "prompt:b",
                 "native_sample_rate_hz": 48000, "duration_sec": 30.0, "s8__x": 0.9},
            ]).to_csv(features, index=False)
            args = Namespace(
                metadata=metadata, features=[features], id_column=None, label_column=None,
                role_column=None, source_column=None, group_column=None,
            )
            table, resolved = evaluator.load_table(args)
            self.assertEqual(resolved["id"], "id")
            self.assertIn("extractor__native_sample_rate_hz", table)
            self.assertNotIn("native_sample_rate_hz_x", table)
            self.assertEqual(table.native_sample_rate_hz.tolist(), [44100, 48000])

    def make_inputs(self, root: Path) -> tuple[Path, Path, Path, Path]:
        rng = np.random.default_rng(20260905)
        metadata: list[dict[str, object]] = []
        features: list[dict[str, object]] = []

        def add_source(role: str, source: str, label: object, count: int) -> None:
            numeric_label = 0 if label == 0 else (1 if label == 1 else None)
            for index in range(count):
                row_id = f"{role}_{source}_{index:03d}"
                metadata.append({
                    "row_id": row_id,
                    "label": label,
                    "role": role,
                    "source_id": source,
                    "group_id": (f"shared_muse_prompt_{index:03d}"
                                 if source.startswith("generator_")
                                 else f"{source}_creator_{index:03d}"),
                    "native_duration_sec": 35.0,
                    "native_sample_rate_hz": 44_100,
                    "duration_sec": 30,
                    "available": 1,
                    "acquisition": ("expansion" if source.endswith("_c") else "prior"),
                    "feature_status": "ok",
                })
                centre = 0.2 if numeric_label == 0 else (0.8 if numeric_label == 1 else 0.55)
                features.append({
                    "item_id": row_id,
                    "label": label,
                    "source_id": source,
                    "group_id": (f"shared_muse_prompt_{index:03d}"
                                 if source.startswith("generator_")
                                 else f"{source}_creator_{index:03d}"),
                    "native_sample_rate_hz": 44_100,
                    "duration_sec": 30.0,
                    "s8__signal": centre + rng.normal(0, 0.08),
                    "d__signal": centre + rng.normal(0, 0.10),
                    "r__signal": centre + rng.normal(0, 0.12),
                    "p__signal": centre + rng.normal(0, 0.14),
                    "s16__signal": centre + rng.normal(0, 0.07),
                })

        for source, count in (("human_a", 11), ("human_b", 13), ("human_c", 17)):
            add_source("development", source, 0, count)
        for source, count in (("generator_a", 12), ("generator_b", 15), ("generator_c", 18)):
            add_source("development", source, 1, count)
        add_source("locked", "musicnet", 0, 8)
        add_source("locked", "diff_rhythm", 1, 8)
        add_source("stress", "gtzan", 0, 6)
        add_source("provisional", "audiox", "unknown", 6)
        # Unavailable rows are inventory records, not silent feature failures.
        metadata.append({
            "row_id": "pending_001", "label": 0, "role": "locked", "source_id": "pending",
            "group_id": "pending_group", "native_duration_sec": 35.0,
            "native_sample_rate_hz": 44_100, "available": 0, "feature_status": "pending",
        })

        metadata_path = root / "metadata.csv"
        features_path = root / "features.csv"
        families_path = root / "families.json"
        slices_path = root / "slices.json"
        pd.DataFrame(metadata).to_csv(metadata_path, index=False)
        pd.DataFrame(features).to_csv(features_path, index=False)
        families = {
            "primary_feature_set": "common8_30s",
            "feature_sets": {
                "common8_30s": {
                    "selection_eligible": True,
                    "cohort": "synthetic matched 30 s",
                    "eligibility": {"min": {"native_duration_sec": 30}},
                    "status_columns": ["feature_status"],
                    "families": {
                        "S": ["s8__signal"], "D": ["d__signal"],
                        "R": ["r__signal"], "P": ["p__signal"],
                    },
                },
                "native16_30s": {
                    "selection_eligible": False,
                    "cohort": "synthetic native bandwidth sensitivity",
                    "eligibility": {"min": {"native_duration_sec": 30,
                                              "native_sample_rate_hz": 40000}},
                    "status_columns": ["feature_status"],
                    "families": {
                        "S": ["s16__signal"], "D": ["d__signal"],
                        "R": ["r__signal"], "P": ["p__signal"],
                    },
                },
            },
        }
        families_path.write_text(json.dumps(families), encoding="utf-8")
        slices_path.write_text(json.dumps({
            "locked_catalogue_plus_pilot": ["locked"],
            "stress_humans": ["stress"],
            "audiox_exploratory": ["provisional"],
        }), encoding="utf-8")
        return metadata_path, features_path, families_path, slices_path

    def run_program(self, *args: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(PROGRAM), *map(str, args)],
            check=False, text=True, capture_output=True,
        )

    def test_cv_freeze_then_locked_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, features, families, slices = self.make_inputs(root)
            scopes = root / "scopes.json"
            scopes.write_text(json.dumps({
                "prior_only": {"equals": {"acquisition": "prior"}},
                "all_development": {},
            }), encoding="utf-8")
            cv_output = root / "cv"
            result = self.run_program(
                "--stage", "cv", "--metadata", metadata, "--features", features,
                "--families-json", families, "--output-dir", cv_output,
                "--quantities", "2,4,all", "--opposite-class-folds", "2",
                "--training-scopes-json", scopes,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            frozen_path = cv_output / "evaluate_expanded_frozen_selection.json"
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            self.assertIn(frozen["leader"]["combination"], {
                "S", "D", "R", "P", "S+D", "S+R", "S+P", "D+R", "D+P", "R+P",
                "S+D+R", "S+D+P", "S+R+P", "D+R+P", "S+D+R+P",
            })
            summary = pd.read_csv(cv_output / "evaluate_expanded_cv_summary.csv")
            primary_all = summary[(summary.feature_set == "common8_30s")
                                  & (summary.training_scope == "all_development")
                                  & (summary.quantity == "all")]
            self.assertEqual(len(primary_all), 15)
            self.assertTrue((primary_all.selection_score > 0.75).all())
            raw = pd.read_csv(cv_output / "evaluate_expanded_cv_raw.csv")
            cohort_keys = ["feature_set", "training_scope", "quantity", "fold_index",
                           "fold_type", "heldout_source", "opposite_source"]
            same_cohort = raw.groupby(cohort_keys)[["test_human", "test_ai"]].nunique()
            self.assertEqual(int(same_cohort.to_numpy().max()), 1)
            matched = pd.read_csv(cv_output / "evaluate_expanded_training_scope_matched.csv")
            self.assertTrue(len(matched) > 0)

            locked_output = root / "locked"
            result = self.run_program(
                "--stage", "locked", "--metadata", metadata, "--features", features,
                "--families-json", families, "--output-dir", locked_output,
                "--frozen-selection", frozen_path, "--evaluation-slices-json", slices,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            macro = pd.read_csv(locked_output / "evaluate_expanded_locked_macro_metrics.csv")
            locked_primary = macro[
                (macro.feature_set == "common8_30s")
                & (macro.slice == "locked_catalogue_plus_pilot")
            ]
            self.assertEqual(len(locked_primary), 15)
            self.assertEqual(int(locked_primary.frozen_leader.sum()), 1)
            scores = pd.read_csv(locked_output / "evaluate_expanded_locked_scores.csv")
            provisional = scores[scores.slice == "audiox_exploratory"]
            self.assertTrue(len(provisional) > 0)
            self.assertTrue(provisional["__label"].isna().all())
            report_path = root / "report.md"
            report = subprocess.run(
                [sys.executable, str(REPORT_PROGRAM), "--cv-dir", str(cv_output),
                 "--locked-dir", str(locked_output), "--output", str(report_path)],
                check=False, text=True, capture_output=True,
            )
            self.assertEqual(report.returncode, 0, report.stderr)
            self.assertIn("Frozen selection result", report_path.read_text(encoding="utf-8"))
            bootstrap_output = root / "bootstrap"
            bootstrap = subprocess.run(
                [sys.executable, str(BOOTSTRAP_PROGRAM),
                 "--scores", str(locked_output / "evaluate_expanded_locked_scores.csv"),
                 "--frozen-selection", str(frozen_path), "--output-dir", str(bootstrap_output),
                 "--slices", "locked_catalogue_plus_pilot", "--replicates", "100"],
                check=False, text=True, capture_output=True,
            )
            self.assertEqual(bootstrap.returncode, 0, bootstrap.stderr)
            bootstrap_pairs = pd.read_csv(
                bootstrap_output / "evaluate_expanded_group_bootstrap_pairs.csv"
            )
            self.assertEqual(len(bootstrap_pairs), 1)
            self.assertEqual(bootstrap_pairs.independent_group_units.iloc[0], 16)

    def test_expected_measurement_nan_is_imputed_but_missing_record_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, features, families, _ = self.make_inputs(root)
            frame = pd.read_csv(features)
            frame.loc[0, "s8__signal"] = np.nan
            frame.to_csv(features, index=False)
            result = self.run_program(
                "--stage", "cv", "--metadata", metadata, "--features", features,
                "--families-json", families, "--output-dir", root / "cv",
                "--quantities", "2,all", "--opposite-class-folds", "2",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            frozen = json.loads((root / "cv" / "evaluate_expanded_frozen_selection.json").read_text())
            model = frozen["models"]["common8_30s"]["S"]
            self.assertLess(model["observed_fraction_by_column"]["s8__signal"], 1.0)

            frame = frame[frame.item_id != "development_human_a_000"]
            frame.to_csv(features, index=False)
            result = self.run_program(
                "--stage", "cv", "--metadata", metadata, "--features", features,
                "--families-json", families, "--output-dir", root / "cv_missing_record",
                "--quantities", "2,all", "--opposite-class-folds", "2",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no feature record or explicit failure", result.stderr)

    def test_all_missing_structure_is_reported_but_cannot_win(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, features, families, _ = self.make_inputs(root)
            frame = pd.read_csv(features)
            frame["p__signal"] = np.nan
            frame.to_csv(features, index=False)
            result = self.run_program(
                "--stage", "cv", "--metadata", metadata, "--features", features,
                "--families-json", families, "--output-dir", root / "cv",
                "--quantities", "2,all", "--opposite-class-folds", "2",
                "--bootstrap-replicates", "100",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            frozen = json.loads((root / "cv" / "evaluate_expanded_frozen_selection.json").read_text())
            self.assertFalse(frozen["all_four"]["qualifies_for_selection"])
            self.assertNotIn("P", frozen["leader"]["combination"].split("+"))
            summary = pd.read_csv(root / "cv" / "evaluate_expanded_cv_summary.csv")
            primary_all = summary[(summary.feature_set == "common8_30s") & (summary.quantity == "all")]
            self.assertEqual(len(primary_all), 15)
            self.assertFalse(bool(primary_all.loc[primary_all.combination == "P", "qualifies_for_selection"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
