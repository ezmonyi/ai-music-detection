#!/usr/bin/env python3
"""Synthetic-only contract tests for evaluate_new_phenomena.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_new_phenomena as evaluator


HERE = Path(__file__).resolve().parent
PROGRAM = HERE / "evaluate_new_phenomena.py"


class NewPhenomenaEvaluationTest(unittest.TestCase):
    def make_inputs(self, root: Path, *, motif_eligibility: bool = True) -> tuple[Path, Path, Path]:
        rng = np.random.default_rng(20260907)
        metadata: list[dict[str, object]] = []
        features: list[dict[str, object]] = []

        def add(role: str, source: str, label: object, count: int) -> None:
            numeric = None if label == "unknown" else int(label)
            for index in range(count):
                row_id = f"{role}_{source}_{index:02d}"
                duration = 20.0 if role == "development" and index == 0 else 40.0
                group = f"global:{source}:{index:02d}"
                metadata.append({
                    "row_id": row_id, "label": label, "role": role,
                    "source_group": source, "group_id": group,
                    "duration_view": "30s", "native_duration_sec": duration,
                    "available": 1,
                })
                centre = 0.5 if numeric is None else (0.2 if numeric == 0 else 0.8)
                features.append({
                    "item_id": row_id, "feature_status": "complete",
                    "s": centre + rng.normal(0, 0.15),
                    "d": centre + rng.normal(0, 0.17),
                    "r": centre + rng.normal(0, 0.18),
                    # P is naturally wholly unavailable. It must not shrink any cohort.
                    "p": np.nan,
                    "f": centre + rng.normal(0, 0.14),
                    "h": np.nan if index % 5 == 0 else centre + rng.normal(0, 0.16),
                    "m": np.nan if duration < 30 else centre + rng.normal(0, 0.19),
                })

        for source in ("human_a", "human_b"):
            add("development", source, 0, 6)
        for source in ("generator_a", "generator_b"):
            add("development", source, 1, 6)
        add("locked", "old_human_test", 0, 4)
        add("locked", "old_ai_test", 1, 4)
        add("pilot", "pilot_generator", 1, 3)
        add("provisional", "unverified_audio", "unknown", 3)

        metadata_path = root / "metadata.csv"
        features_path = root / "features.csv"
        config_path = root / "families.json"
        pd.DataFrame(metadata).to_csv(metadata_path, index=False)
        pd.DataFrame(features).to_csv(features_path, index=False)
        config = {
            "schema_version": 1,
            "cohort": "synthetic 30-second contract test",
            "eligibility": {"equals": {"duration_view": "30s"}},
            "status_columns": ["feature_status"],
            "old_families": {
                "S": {"state": "available", "columns": ["s"]},
                "D": {"state": "available", "columns": ["d"]},
                "R": {"state": "available", "columns": ["r"]},
                "P": {"state": "available", "columns": ["p"]},
            },
            "new_families": {
                "F": {"state": "available", "columns": ["f"],
                      "phenomenon": "phase evolution and group delay"},
                "H": {"state": "available", "columns": ["h"],
                      "phenomenon": "tonal and harmonic path"},
                "M": {"state": "available", "columns": ["m"],
                      "phenomenon": "long-range motif recurrence",
                      "eligibility": ({"min": {"native_duration_sec": 30}}
                                      if motif_eligibility else {})},
                "V": {"state": "planned", "columns": [],
                      "phenomenon": "note-level pitch microstructure"},
                "B": {"state": "planned", "columns": [],
                      "phenomenon": "breath-event organization"},
                "A": {"state": "planned", "columns": [],
                      "phenomenon": "articulation and phoneme-note coordination"},
                "T": {"state": "planned", "columns": [],
                      "phenomenon": "source-identity timbral continuity"},
            },
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return metadata_path, features_path, config_path

    def command(
        self, stage: str, metadata: Path, features: Path, config: Path, output: Path,
        *extra: object,
    ) -> list[str]:
        return [
            sys.executable, str(PROGRAM), "--stage", stage,
            "--metadata", str(metadata), "--features", str(features),
            "--families-json", str(config), "--output-dir", str(output),
            "--group-folds", "2", "--quantities", "all",
            "--min-train-groups-per-class", "1", "--min-test-groups-per-class", "1",
            *map(str, extra),
        ]

    def test_plan_is_complete_and_future_families_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, _, config_path = self.make_inputs(Path(temporary), motif_eligibility=False)
            config = evaluator.load_family_config(config_path)
            plan = evaluator.build_evaluation_plan(config)
            self.assertEqual(len(plan[plan.plan_type == "old_baseline_standalone"]), 15)
            self.assertEqual(len(plan[plan.plan_type == "new_standalone"]), 7)
            self.assertEqual(len(plan[plan.plan_type == "incremental_matched"]), 210)
            self.assertEqual(set(plan.combination), {
                evaluator.combo_name(combo)
                for combo in evaluator.nonempty_combinations(("S", "D", "R", "P", "F", "H", "M"))
            })
            self.assertEqual(evaluator.active_new_families(config), ("F", "H", "M"))
            self.assertTrue(all(config["new_families"][code]["state"] == "planned"
                                for code in ("V", "B", "A", "T")))

    def test_fh_only_30s_wave_allows_m_to_remain_explicitly_planned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, config_path = self.make_inputs(root)
            payload = json.loads(config_path.read_text())
            payload["new_families"]["M"] = {
                "state": "planned", "columns": [],
                "phenomenon": "long-range motif recurrence",
            }
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            config = evaluator.load_family_config(config_path)
            self.assertEqual(evaluator.active_new_families(config), ("F", "H"))
            plan = evaluator.build_evaluation_plan(config)
            self.assertEqual(len(plan[plan.plan_type == "new_standalone"]), 3)
            self.assertEqual(len(plan[plan.plan_type == "incremental_matched"]), 90)
            self.assertNotIn("M", "+".join(plan.combination))

    def test_known_quality_metadata_is_rejected_as_predictor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, config_path = self.make_inputs(root)
            payload = json.loads(config_path.read_text())
            payload["new_families"]["F"]["columns"].append("F_quality_eligible")
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot be predictors"):
                evaluator.load_family_config(config_path)

    def test_development_outputs_are_matched_and_do_not_complete_case_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, features, config = self.make_inputs(root)
            output = root / "dev"
            result = subprocess.run(
                self.command("dev", metadata, features, config, output, "--synthetic-test-only"),
                check=False, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            plan = pd.read_csv(output / "evaluation_plan.csv", keep_default_na=False)
            registry = pd.read_csv(output / "cohort_registry.csv")
            self.assertEqual(len(registry), 2)  # M's duration gate, never P missingness.
            baseline_s = plan[plan.plan_id == "baseline::S"].iloc[0]
            baseline_p = plan[plan.plan_id == "baseline::P"].iloc[0]
            self.assertEqual(baseline_s.eligible_rows, 24)
            self.assertEqual(baseline_p.eligible_rows, 24)

            coverage = pd.read_csv(output / "family_coverage_by_source.csv")
            p_rows = coverage[coverage.family == "P"]
            self.assertTrue((p_rows.any_observed_rows == 0).all())
            planned = coverage[coverage.family.isin(["V", "B", "A", "T"])]
            self.assertTrue((planned.state == "planned").all())
            self.assertFalse(planned.coverage_qualified.any())

            deltas = pd.read_csv(output / "development_source_holdout_matched_deltas.csv")
            self.assertGreater(len(deltas), 0)
            self.assertTrue((deltas.test_human > 0).all())
            self.assertTrue((deltas.test_ai > 0).all())

            # Every matched baseline/added pair uses one exact cohort and fold ID.
            comparison = plan[plan.comparison_id == "incremental::S__plus__M"]
            self.assertEqual(comparison.cohort_id.nunique(), 1)
            self.assertEqual(comparison.eligible_id_set_sha256.nunique(), 1)
            self.assertEqual(set(comparison.arm), {"baseline", "added"})

            source_predictions = pd.read_csv(output / "development_source_holdout_predictions.csv")
            group_predictions = pd.read_csv(output / "development_group_cv_predictions.csv")
            self.assertGreater(len(source_predictions), 0)
            self.assertGreater(len(group_predictions), 0)
            for frame in (source_predictions, group_predictions):
                self.assertIn("group_id", frame)
                self.assertIn("model_sha256", frame)
            missingness = pd.read_csv(
                output / "development_missingness_source_holdout_metrics.csv"
            )
            self.assertEqual(set(missingness.feature_mode), {"median_only", "missingness_only"})
            self.assertEqual(set(missingness.diagnostic_status), {"sensitivity_only_never_select"})

            folds = [json.loads(line) for line in (output / "fold_registry.jsonl").read_text().splitlines()]
            self.assertGreater(len(folds), 0)
            for fold in folds:
                self.assertFalse(set(fold["train_group_ids"]) & set(fold["test_group_ids"]))

            bundle = json.loads((output / "frozen_dev_models.json").read_text())
            self.assertIsNone(bundle["selection"])
            self.assertIn("training_config_sha256", next(iter(bundle["models"].values())))
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["authorization"]["status"], "synthetic_test_only")

    def test_real_scoring_gate_and_historical_descriptive_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, features, config = self.make_inputs(root)
            blocked = subprocess.run(
                self.command("dev", metadata, features, config, root / "blocked"),
                check=False, text=True, capture_output=True,
            )
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("Real-corpus scoring is disabled", blocked.stderr)

            dev = root / "dev"
            passed = subprocess.run(
                self.command("dev", metadata, features, config, dev, "--synthetic-test-only"),
                check=False, text=True, capture_output=True,
            )
            self.assertEqual(passed.returncode, 0, passed.stderr)
            bundle = dev / "frozen_dev_models.json"
            historical = root / "historical"
            scored = subprocess.run(
                self.command(
                    "historical-descriptive", metadata, features, config, historical,
                    "--frozen-dev-bundle", bundle, "--synthetic-test-only",
                ),
                check=False, text=True, capture_output=True,
            )
            self.assertEqual(scored.returncode, 0, scored.stderr)
            manifest = json.loads((historical / "run_manifest.json").read_text())
            self.assertEqual(manifest["evaluation_status"],
                             "historical_descriptive_only_never_select")
            self.assertFalse(manifest["selection_performed"])
            predictions = pd.read_csv(historical / "historical_descriptive_predictions.csv")
            self.assertEqual(set(predictions.source_group), {"old_human_test", "old_ai_test"})
            self.assertFalse(predictions.source_group.str.contains("pilot|unverified").any())
            deltas = pd.read_csv(historical / "historical_descriptive_matched_deltas.csv")
            self.assertGreater(len(deltas), 0)

            rejected = subprocess.run(
                self.command(
                    "historical-descriptive", metadata, features, config, root / "pilot_rejected",
                    "--frozen-dev-bundle", bundle, "--historical-roles", "locked,pilot",
                    "--synthetic-test-only",
                ),
                check=False, text=True, capture_output=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("Pilot/provisional roles cannot enter", rejected.stderr)

    def test_threshold_is_derived_only_from_supplied_training_rows(self) -> None:
        table = pd.DataFrame({
            "__label": [0.0, 0.0, 1.0, 1.0],
            "__source": ["h", "h", "a", "a"],
            "__group": ["h1", "h2", "a1", "a2"],
            "x": [0.1, 0.3, 0.7, 0.9],
        })
        model = evaluator.fit_candidate(table, ["x"], "ridge")
        threshold = evaluator.threshold_for_training(table, model, "train_ba")
        self.assertTrue(np.isfinite(threshold))
        self.assertNotIn("test", model)


if __name__ == "__main__":
    unittest.main()
