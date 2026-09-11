#!/usr/bin/env python3
"""Synthetic contract tests for the schema-v3 60-second group-CV mode."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_new_phenomena_v3 as evaluator


HERE = Path(__file__).resolve().parent
PROGRAM = HERE / "evaluate_new_phenomena_v3.py"
PREPARER = HERE / "prepare_evaluation_inputs_v3.py"


class GroupCvOnlyEvaluationTest(unittest.TestCase):
    @staticmethod
    def _group_for_fold(prefix: str, fold: int, used: set[str]) -> str:
        index = 0
        while True:
            candidate = f"global:{prefix}:{index:04d}"
            index += 1
            if candidate not in used and evaluator.BASE.hash_fold(candidate, 5, 20260907) == fold:
                used.add(candidate)
                return candidate

    def make_inputs(self, root: Path) -> tuple[Path, Path, Path, set[str]]:
        used: set[str] = set()
        metadata: list[dict[str, object]] = []
        features: list[dict[str, object]] = []
        pilot_ids: set[str] = set()
        rng = np.random.default_rng(20260907)

        def add(row_id: str, label: int, role: str, source: str, group: str) -> None:
            metadata.append({
                "id": row_id,
                "label": label,
                "role": role,
                "source_group": source,
                "group_id": group,
                "duration_view": "60s",
                "available": 1,
            })
            centre = 0.25 if label == 0 else 0.75
            features.append({
                "id": row_id,
                "F_metric": centre + rng.normal(0, 0.08),
                "H_metric": centre + rng.normal(0, 0.10),
                "M_metric": centre + rng.normal(0, 0.12),
                "new_extraction_status": "ok",
                "new_extraction_contract_hash": "synthetic-v3",
            })

        # Every fold has at least two label-specific groups and one cross-label group.
        # The cross-label case specifically checks that group_id, not label/source, is split.
        counter = 0
        for fold in range(5):
            for repeat in range(2):
                human_group = self._group_for_fold(f"human-{fold}-{repeat}", fold, used)
                ai_group = self._group_for_fold(f"ai-{fold}-{repeat}", fold, used)
                add(f"dev_h_{counter:03d}", 0, "development",
                    "human_a" if repeat == 0 else "human_b", human_group)
                add(f"dev_a_{counter:03d}", 1, "development", "Suno", ai_group)
                counter += 1
            shared = self._group_for_fold(f"shared-{fold}", fold, used)
            add(f"dev_h_shared_{fold}", 0, "development", "human_a", shared)
            add(f"dev_a_shared_{fold}", 1, "development", "Suno", shared)

        for index in range(6):
            row_id = f"pilot_diffrhythm_{index:02d}"
            pilot_ids.add(row_id)
            add(row_id, 1, "pilot", "ai_diffrhythm_pilot", f"pilot:diff:{index}")
        for index in range(4):
            add(f"locked_h_{index:02d}", 0, "locked", "human_locked", f"locked:h:{index}")
            add(f"locked_a_{index:02d}", 1, "locked", "Suno", f"locked:a:{index}")

        metadata_path = root / "metadata_60s.csv"
        features_path = root / "features_60s.csv"
        config_path = root / "families_60s_v3.json"
        pd.DataFrame(metadata).to_csv(metadata_path, index=False)
        pd.DataFrame(features).to_csv(features_path, index=False)
        config = {
            "schema_version": 3,
            "cohort": "synthetic exact-60-second pure F/H/M",
            "eligibility": {"equals": {"duration_view": "60s", "available": 1}},
            "status_columns": ["new_extraction_status", "new_extraction_contract_hash"],
            "old_families": {
                code: {"state": "planned", "columns": []}
                for code in evaluator.OLD_FAMILY_ORDER
            },
            "new_families": {
                "F": {"state": "available", "columns": ["F_metric"],
                      "phenomenon": "phase evolution and group delay"},
                "H": {"state": "available", "columns": ["H_metric"],
                      "phenomenon": "tonal and harmonic path"},
                "M": {"state": "available", "columns": ["M_metric"],
                      "phenomenon": "long-range motif recurrence"},
                "V": {"state": "planned", "columns": [], "phenomenon": "pitch microstructure"},
                "B": {"state": "planned", "columns": [], "phenomenon": "breath events"},
                "A": {"state": "planned", "columns": [], "phenomenon": "articulation"},
                "T": {"state": "planned", "columns": [], "phenomenon": "timbre continuity"},
            },
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return metadata_path, features_path, config_path, pilot_ids

    @staticmethod
    def command(
        stage: str, metadata: Path, features: Path, config: Path, output: Path,
        *extra: str,
    ) -> list[str]:
        return [
            sys.executable, str(PROGRAM),
            "--stage", stage,
            "--metadata", str(metadata),
            "--features", str(features),
            "--families-json", str(config),
            "--output-dir", str(output),
            "--group-folds", "5",
            "--quantities", "all",
            "--seed", "20260907",
            "--model", "ridge",
            "--threshold-policy", "fixed_0.5",
            "--min-train-groups-per-class", "1",
            "--min-test-groups-per-class", "1",
            *extra,
        ]

    def test_group_cv_only_runs_seven_fhm_candidates_without_group_or_pilot_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, features, config, pilot_ids = self.make_inputs(root)
            output = root / "group_cv"
            result = subprocess.run(
                self.command(
                    "dev", metadata, features, config, output,
                    "--evaluation-mode", "group_cv_only", "--synthetic-test-only",
                ),
                check=False, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            plan = pd.read_csv(output / "evaluation_plan.csv", keep_default_na=False)
            self.assertEqual(len(plan), 7)
            self.assertEqual(plan["candidate_key"].nunique(), 7)
            self.assertEqual(set(plan["plan_type"]), {"new_standalone"})
            self.assertEqual(set(plan["combination"]), {"F", "H", "M", "F+H", "F+M", "H+M", "F+H+M"})

            folds = [json.loads(line) for line in
                     (output / "fold_registry.jsonl").read_text().splitlines()]
            self.assertEqual(len(folds), 5)
            for fold in folds:
                self.assertFalse(set(fold["train_group_ids"]) & set(fold["test_group_ids"]))
                self.assertFalse(pilot_ids & set(fold["train_ids"]))
                self.assertFalse(pilot_ids & set(fold["test_ids"]))

            predictions = pd.read_csv(output / "development_group_cv_predictions.csv")
            self.assertFalse(pilot_ids & set(predictions["row_id"]))
            self.assertEqual(set(predictions["fold_type"]), {"ordinary_group_holdout_descriptive"})
            audit = json.loads((output / "source_holdout_ineligibility.json").read_text())
            self.assertEqual(audit["generator_holdout"]["status"], "ineligible_not_run")
            self.assertEqual(audit["generator_holdout"]["observed_development_ai_sources"], ["Suno"])
            self.assertEqual(audit["diffrhythm_pilot_training_rows"], 0)
            self.assertIsNone(audit["source_transfer_J"])
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["evaluation_mode"], "group_cv_only")
            self.assertEqual(manifest["unique_candidates"], 7)
            self.assertIsNone(manifest["source_transfer_J"])
            self.assertEqual(manifest["source_transfer_J_status"], "undefined_not_filled")

    def test_default_source_holdout_still_rejects_one_development_ai_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, features, config, _ = self.make_inputs(root)
            result = subprocess.run(
                self.command(
                    "dev", metadata, features, config, root / "default_mode",
                    "--synthetic-test-only",
                ),
                check=False, text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("at least two source_group values per class", result.stderr)

    def test_draft_contract_hashes_v3_and_declares_group_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, features, config, _ = self.make_inputs(root)
            output = root / "draft"
            result = subprocess.run(
                self.command(
                    "preregistration-draft", metadata, features, config, output,
                    "--evaluation-mode", "group_cv_only",
                ),
                check=False, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads((output / "preregistration_draft.json").read_text())
            contract = receipt["contract"]
            self.assertEqual(receipt["schema_version"], 3)
            self.assertEqual(contract["schema_version"], 3)
            self.assertEqual(contract["parameters"]["evaluation_mode"], "group_cv_only")
            self.assertEqual(contract["evaluator_sha256"], evaluator.sha256_file(PROGRAM))
            self.assertEqual(
                contract["schema_v2_dependency_sha256"],
                evaluator.sha256_file(Path(evaluator.V2.__file__).resolve()),
            )
            self.assertEqual(receipt["contract_sha256"], evaluator.canonical_hash(contract))
            self.assertEqual(receipt["status"], "draft")

    def make_extraction_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        extraction = root / "extraction"
        extraction.mkdir()
        ids = ["h1", "h2", "a1", "a2"]
        metadata = root / "cohort_60s.csv"
        pd.DataFrame({
            "id": ids,
            "source_group": ["human_a", "human_b", "Suno", "Suno"],
            "group_id": [f"g{i}" for i in range(4)],
            "role": ["development"] * 4,
            "duration_view": ["60s"] * 4,
            "available": [1] * 4,
            "label": [0, 0, 1, 1],
        }).to_csv(metadata, index=False)
        contract_hash = "synthetic-extraction-v3"
        feature_path = extraction / "features.csv"
        pd.DataFrame({
            "id": ids,
            "F_x": [0.1, 0.2, 0.8, 0.9], "F_status": ["ok"] * 4,
            "H_x": [0.2, 0.3, 0.7, 0.8], "H_status": ["ok"] * 4,
            "M_x": [0.3, 0.4, 0.6, 0.7], "M_status": ["ok"] * 4,
            "extraction_status": ["ok"] * 4,
            "extraction_contract_hash": [contract_hash] * 4,
        }).to_csv(feature_path, index=False)
        feature_sha = hashlib.sha256(feature_path.read_bytes()).hexdigest()
        metadata_sha = hashlib.sha256(metadata.read_bytes()).hexdigest()
        (extraction / "contract.json").write_text(json.dumps({
            "duration": 60.0,
            "preflight_only": False,
            "contract_hash": contract_hash,
            "selected_ids": ids,
            "feature_names": {"F": ["F_x"], "H": ["H_x"], "M": ["M_x"]},
            "code_sha256": {},
            "metadata_sha256": metadata_sha,
        }), encoding="utf-8")
        (extraction / "summary.json").write_text(json.dumps({
            "complete_accounting": True,
            "contract_hash": contract_hash,
            "expected": 4,
            "features_csv_sha256": feature_sha,
        }), encoding="utf-8")
        audit = root / "extraction_audit.json"
        audit.write_text(json.dumps({
            "passed": True,
            "errors": [],
            "row_count": 4,
            "extraction_contract_hash": contract_hash,
        }), encoding="utf-8")
        return extraction, audit, metadata

    def test_v3_preparer_publishes_only_pure_fhm_and_refuses_old_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction, audit, metadata = self.make_extraction_fixture(root)
            output = root / "prepared"
            result = subprocess.run([
                sys.executable, str(PREPARER),
                "--extraction-dir", str(extraction),
                "--extraction-audit", str(audit),
                "--metadata", str(metadata),
                "--output-dir", str(output),
            ], check=False, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = evaluator.load_family_config(output / "families_60s_v3.json")
            self.assertEqual(evaluator.active_old_families(config), ())
            self.assertEqual(evaluator.active_new_families(config), ("F", "H", "M"))
            self.assertEqual(len(evaluator.build_evaluation_plan(config)), 7)
            prepared_audit = json.loads((output / "preparation_audit.json").read_text())
            self.assertTrue(prepared_audit["metadata_sha256_matches_extraction_contract"])
            self.assertFalse(prepared_audit["scoring_authorized"])

    def test_v3_preparer_rejects_metadata_bytes_not_frozen_by_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction, audit, metadata = self.make_extraction_fixture(root)
            frame = pd.read_csv(metadata)
            frame.loc[0, "source_group"] = "changed_after_extraction"
            frame.to_csv(metadata, index=False)
            result = subprocess.run([
                sys.executable, str(PREPARER),
                "--extraction-dir", str(extraction),
                "--extraction-audit", str(audit),
                "--metadata", str(metadata),
                "--output-dir", str(root / "must_not_exist"),
            ], check=False, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Metadata SHA-256 differs", result.stderr)
            self.assertFalse((root / "must_not_exist").exists())


if __name__ == "__main__":
    unittest.main()
