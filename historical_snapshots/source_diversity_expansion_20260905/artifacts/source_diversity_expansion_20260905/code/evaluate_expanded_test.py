#!/usr/bin/env python3
"""Synthetic end-to-end tests for evaluate_expanded.py."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_expanded as evaluator
import evaluate_expanded_latex as latex_exporter
import evaluate_expanded_plot as plotter
from evaluate_expanded_orchestrate import command_records_parallel, load_failure_manifest
from evaluate_expanded_reporting import family_availability_by_source


HERE = Path(__file__).resolve().parent
PROGRAM = HERE / "evaluate_expanded.py"
REPORT_PROGRAM = HERE / "evaluate_expanded_report.py"
BOOTSTRAP_PROGRAM = HERE / "evaluate_expanded_group_bootstrap.py"
PLOT_PROGRAM = HERE / "evaluate_expanded_plot.py"
ORCHESTRATE_PROGRAM = HERE / "evaluate_expanded_orchestrate.py"
LATEX_PROGRAM = HERE / "evaluate_expanded_latex.py"


class ExpandedEvaluationTest(unittest.TestCase):
    def test_full_accounting_jsonl_extracts_only_nested_serialization_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "accounting.jsonl"
            path.write_text("\n".join([
                json.dumps({"item_id": "complete", "serialization_failure": {}}),
                json.dumps({"item_id": "failed", "serialization_failure": {
                    "mapped": True, "reason": "beat_this_serializer_not_all_downbeats_are_beats",
                }}),
            ]) + "\n", encoding="utf-8")
            self.assertEqual(load_failure_manifest(path), {
                "failed": "beat_this_serializer_not_all_downbeats_are_beats",
            })

    def test_parallel_cv_scheduler_joins_all_jobs_before_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "peer_completed.txt"
            successful = [
                sys.executable, "-c",
                f"from pathlib import Path; Path({str(marker)!r}).write_text('done')",
            ]
            failing = [sys.executable, "-c", "raise SystemExit(7)"]
            with self.assertRaisesRegex(RuntimeError, "after all jobs joined"):
                command_records_parallel([
                    (failing, root / "failed.log"),
                    (successful, root / "successful.log"),
                ])
            self.assertEqual(marker.read_text(), "done")

    def test_family_availability_separates_natural_missing_from_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.csv"
            features = root / "features.csv"
            pd.DataFrame([
                {"track": "a", "source_group": "source", "role": "development"},
                {"track": "b", "source_group": "source", "role": "development"},
            ]).to_csv(metadata, index=False)
            pd.DataFrame([
                {"item_id": "a", "s16_feature_status": "computed", "s8_feature_status": "computed",
                 "d_feature_status": "eligible", "r_feature_status": "eligible",
                 "p_feature_status": "unavailable_fewer_than_3_sections_or_downbeat_spans",
                 "feature_status": "complete"},
                {"item_id": "b", "s16_feature_status": "computed", "s8_feature_status": "computed",
                 "d_feature_status": "eligible", "r_feature_status": "processing_failure",
                 "p_feature_status": "processing_failure", "feature_status": "complete"},
            ]).to_csv(features, index=False)
            result = family_availability_by_source(metadata, features, "10s").set_index("family")
            self.assertEqual(result.loc["P", "natural_unobservable_rate"], 0.5)
            self.assertEqual(result.loc["P", "processing_failure_rate"], 0.5)
            self.assertEqual(result.loc["R", "natural_unobservable_rate"], 0.0)
            self.assertEqual(result.loc["R", "processing_failure_rate"], 0.5)

    def test_beat_serialization_incident_is_r_failure_but_p_natural_with_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.csv"
            features = root / "features.csv"
            pd.DataFrame([{
                "track": "serializer_incident", "source_group": "source", "role": "development",
            }]).to_csv(metadata, index=False)
            pd.DataFrame([{
                "item_id": "serializer_incident", "s16_feature_status": "computed",
                "s8_feature_status": "computed", "d_feature_status": "eligible",
                "r_feature_status": "unavailable_fewer_than_16_beats_or_invalid_intervals",
                "p_feature_status": "unavailable_fewer_than_3_sections_or_downbeat_spans",
                "feature_status": "accounted_serialization_failure",
            }]).to_csv(features, index=False)
            result = family_availability_by_source(metadata, features, "10s").set_index("family")
            self.assertEqual(result.loc["R", "natural_unobservable_count"], 0)
            self.assertEqual(result.loc["R", "processing_failure_count"], 1)
            self.assertEqual(result.loc["R", "beat_dependency_failure_count"], 1)
            self.assertEqual(result.loc["P", "natural_unobservable_count"], 1)
            self.assertEqual(result.loc["P", "processing_failure_count"], 0)
            self.assertEqual(result.loc["P", "beat_dependency_failure_count"], 1)

    def test_canonical_family_combination_presentation_order(self) -> None:
        names = list(evaluator.combinations({name: [name.lower()] for name in evaluator.FAMILY_ORDER}))
        shuffled = sorted(names, reverse=True)
        expected = [
            "S", "D", "R", "P", "S+D", "S+R", "S+P", "D+R", "D+P", "R+P",
            "S+D+R", "S+D+P", "S+R+P", "D+R+P", "S+D+R+P",
        ]
        self.assertEqual(sorted(shuffled, key=plotter.combination_order), expected)
        self.assertEqual(sorted(shuffled, key=latex_exporter.combination_order), expected)

    def test_formal_preflight_requires_complete_exact_ids_and_source_group_units(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {"track": "prior_h", "label": 0, "source_id": "human_catalog_a",
                 "source_group": "Human A", "role": "development", "group_id": "artist:a",
                 "acquisition": "prior", "available": 1, "evaluation_allowed": np.nan},
                {"track": "prior_a1", "label": 1, "source_id": "humair_suno",
                 "source_group": "Suno", "role": "development", "group_id": "creator:a",
                 "acquisition": "prior", "available": 1, "evaluation_allowed": np.nan},
                {"track": "prior_a2", "label": 1, "source_id": "suno_unknown",
                 "source_group": "Suno", "role": "development", "group_id": "creator:b",
                 "acquisition": "prior", "available": 1, "evaluation_allowed": np.nan},
                {"track": "new_h", "label": 0, "source_id": "human_catalog_b",
                 "source_group": "Human B", "role": "development", "group_id": "artist:b",
                 "acquisition": "expansion", "available": 1, "evaluation_allowed": "yes"},
                {"track": "new_a", "label": 1, "source_id": "generator_b",
                 "source_group": "Generator B", "role": "development", "group_id": "prompt:b",
                 "acquisition": "expansion", "available": 1, "evaluation_allowed": "yes"},
                {"track": "new_locked", "label": 0, "source_id": "musicnet",
                 "source_group": "MusicNet", "role": "locked", "group_id": "composer:c",
                 "acquisition": "expansion", "available": 1, "evaluation_allowed": "locked_only"},
                {"track": "new_stress", "label": 0, "source_id": "gtzan",
                 "source_group": "GTZAN", "role": "stress", "group_id": "artist:d",
                 "acquisition": "expansion", "available": 1, "evaluation_allowed": "stress_only"},
            ]
            feature_rows = [{
                "item_id": row["track"], "feature_status": "complete",
                "s16_feature_status": "computed", "s8_feature_status": "computed",
                "d_feature_status": "eligible", "r_feature_status": "eligible",
                "p_feature_status": "eligible", "s8__x": 0.0,
            } for row in rows]
            paths = {}

            def file_sha256(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            def id_digest(ids: list[str]) -> str:
                raw = "".join(f"{value}\n" for value in sorted(set(ids))).encode("utf-8")
                return hashlib.sha256(raw).hexdigest()

            def verification_payload(duration: str, metadata_path: Path,
                                     features_path: Path) -> dict[str, object]:
                ids = [row["track"] for row in rows]
                digest = id_digest(ids)
                count = len(ids)
                return {
                    "schema_version": 1, "status": "passed", "passed": True,
                    "duration_view": duration,
                    "metadata": {"path": str(metadata_path.resolve()),
                                 "sha256": file_sha256(metadata_path), "row_count": count,
                                 "unique_id_count": count, "id_set_sha256": digest},
                    "features": {"path": str(features_path.resolve()),
                                 "sha256": file_sha256(features_path), "row_count": count,
                                 "unique_id_count": count, "id_set_sha256": digest},
                    "expected": {"metadata_row_count": count, "feature_row_count": count},
                    "identity": {"exact_id_set_match": True, "duplicate_metadata_ids": [],
                                 "duplicate_feature_ids": [], "missing_feature_ids": [],
                                 "unexpected_feature_ids": []},
                    "feature_status_counts": {
                        "s16": {"computed": count}, "s8": {"computed": count},
                        "d": {"eligible": count}, "r": {"eligible": count},
                        "p": {"eligible": count}, "overall": {"complete": count},
                    },
                    "failure_accounting": {
                        "unexplained_failure_count": 0, "unexplained_failure_ids": [],
                        "serialization_failure_count": 0,
                        "serialization_failure_reason_counts": {},
                        "serialization_failures_mapped_count": 0,
                        "serialization_failures_all_mapped": True,
                        "serialization_failure_manifest_path": None,
                        "serialization_failure_manifest_sha256": None,
                        "natural_missing_counts": {"d": 0, "r": 0, "p": 0},
                        "natural_missing_allowed": True,
                    },
                    "hash_definition": {
                        "id_set_sha256": "sha256 of UTF-8 sorted unique IDs joined by newline with trailing newline",
                        "file_sha256": "raw file bytes",
                    },
                    "generated_at_utc": "2026-09-05T00:00:00Z",
                }

            for duration in ("10s", "30s"):
                metadata_path = root / f"metadata_{duration}.csv"
                features_path = root / f"features_{duration}.csv"
                frame = pd.DataFrame(rows)
                frame["duration_view"] = duration
                frame.to_csv(metadata_path, index=False)
                pd.DataFrame(feature_rows).to_csv(features_path, index=False)
                verification = root / f"verification_{duration}.json"
                verification.write_text(json.dumps(
                    verification_payload(duration, metadata_path, features_path)
                ), encoding="utf-8")
                paths[duration] = (metadata_path, features_path, verification)
            output = root / "formal"
            materialization_audit = root / "materialization_validation.json"
            materialization_audit.write_text(json.dumps({
                "status": "passed", "passed": True, "expected": 2641, "verified": 2641,
                "materialized_or_excluded": 2641, "zero_unexplained_failures": True,
                "unexplained_failure_count": 0, "rehash_enabled": True, "valid": True,
                "canonical_records": 2641, "success_items": 2641,
                "materialization_manifest_sha256": "1" * 64,
                "frozen_manifest_sha256": "2" * 64,
                "duplicate_exclusion_manifest_path": None,
                "duplicate_exclusion_manifest_sha256": None, "errors": [],
            }), encoding="utf-8")
            legacy_audit = root / "legacy_native_1000.json"
            legacy_audit.write_text(json.dumps({
                "status": "passed", "passed": True, "expected": 1000, "verified": 1000,
                "unique_native_sha256": 1000,
                "details": [{"status": "verified"} for _ in range(1000)],
            }), encoding="utf-8")
            command = [
                sys.executable, str(ORCHESTRATE_PROGRAM), "--mode", "preflight",
                "--metadata-10s", str(paths["10s"][0]), "--features-10s", str(paths["10s"][1]),
                "--verification-10s", str(paths["10s"][2]),
                "--metadata-30s", str(paths["30s"][0]), "--features-30s", str(paths["30s"][1]),
                "--verification-30s", str(paths["30s"][2]), "--output-root", str(output),
                "--materialization-audit", str(materialization_audit),
                "--legacy-audit", str(legacy_audit),
                "--expected-metadata-10s", "7", "--expected-expansion-10s", "4",
                "--expected-development-10s", "5", "--expected-human-sources-10s", "2",
                "--expected-ai-sources-10s", "2", "--expected-locked-10s", "1",
                "--expected-stress-10s", "1", "--expected-pilot-10s", "0",
                "--expected-provisional-10s", "0", "--expected-metadata-30s", "7",
                "--expected-development-30s", "5", "--expected-locked-30s", "1",
                "--expected-stress-30s", "1", "--expected-pilot-30s", "0",
                "--expected-provisional-30s", "0",
            ]
            passed = subprocess.run(command, check=False, text=True, capture_output=True)
            self.assertEqual(passed.returncode, 0, passed.stderr)
            audit = json.loads((output / "evaluate_expanded_formal_orchestration_audit.json").read_text())
            self.assertEqual(audit["status"], "preflight_passed")
            self.assertEqual(audit["durations"]["10s"]["development_ai_sources"], 2)

            valid_verification = paths["10s"][2].read_text(encoding="utf-8")
            paths["10s"][2].write_text("{}", encoding="utf-8")
            failed = subprocess.run(command, check=False, text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("positive schema-v1 pass", failed.stderr)

            partial = json.loads(valid_verification)
            partial["status"] = "partial"
            partial["passed"] = False
            paths["10s"][2].write_text(json.dumps(partial), encoding="utf-8")
            failed = subprocess.run(command, check=False, text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("positive schema-v1 pass", failed.stderr)

            wrong_hash = json.loads(valid_verification)
            wrong_hash["features"]["sha256"] = "0" * 64
            paths["10s"][2].write_text(json.dumps(wrong_hash), encoding="utf-8")
            failed = subprocess.run(command, check=False, text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("hashes do not match", failed.stderr)

            paths["10s"][2].write_text(valid_verification, encoding="utf-8")
            pd.DataFrame(feature_rows[:-1]).to_csv(paths["10s"][1], index=False)
            failed = subprocess.run(command, check=False, text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("feature ID set mismatch", failed.stderr)

    def test_quantity_sampling_aligns_shared_conditions_across_sources(self) -> None:
        rows = []
        for source in ("generator_a", "generator_b"):
            for group in ("prompt_1", "prompt_2", "prompt_3", "prompt_4"):
                rows.append({"__label": 1.0, "__source": source, "__group": group})
        table = pd.DataFrame(rows)
        selected = evaluator.deterministic_quantity(table, 2, evaluator.DEFAULT_SEED)
        self.assertIsNotNone(selected)
        groups_by_source = {
            source: set(current["__group"])
            for source, current in selected.groupby("__source")
        }
        self.assertEqual(groups_by_source["generator_a"], groups_by_source["generator_b"])
        larger = evaluator.deterministic_quantity(table, 3, evaluator.DEFAULT_SEED)
        for source, current in larger.groupby("__source"):
            self.assertTrue(groups_by_source[source].issubset(set(current["__group"])))

    def test_roundoff_tie_from_constant_family_prefers_fewer_families(self) -> None:
        candidates = pd.DataFrame([
            {"combination": "S", "family_count": 1, "selection_score": 0.5},
            {"combination": "S+D", "family_count": 2,
             "selection_score": 0.5 + 0.5 * evaluator.SELECTION_TIE_TOLERANCE},
            {"combination": "S+D+R", "family_count": 3,
             "selection_score": 0.5 + 0.9 * evaluator.SELECTION_TIE_TOLERANCE},
        ])
        chosen = evaluator.choose_with_tolerance(candidates)
        self.assertEqual(chosen.combination, "S")

    def test_cached_preparation_is_numerically_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, features, families_path, _ = self.make_inputs(root)
            args = Namespace(
                metadata=metadata, features=[features], id_column=None, label_column=None,
                role_column=None, source_column=None, group_column=None,
            )
            table, _ = evaluator.load_table(args)
            config = evaluator.load_family_config(families_path)
            spec = config["feature_sets"]["common8_30s"]
            table, _ = evaluator.apply_eligibility(table[table["__role"] == "development"], spec)
            table.loc[table.index[::9], "r__signal"] = np.nan
            table.loc[table.index[::13], "p__signal"] = np.nan
            family_columns = spec["families"]
            union = sum((family_columns[name] for name in evaluator.FAMILY_ORDER), [])
            prepared = evaluator.prepare_training(table, union)
            for _, columns in evaluator.combinations(family_columns).items():
                direct = evaluator.fit_model(table, columns)
                cached = evaluator.fit_prepared(prepared, columns)
                self.assertTrue(np.allclose(
                    direct["coefficients_with_intercept"], cached["coefficients_with_intercept"],
                    rtol=1e-11, atol=1e-12,
                ))
                self.assertTrue(np.allclose(
                    evaluator.predict(table, direct), evaluator.predict(table, cached),
                    rtol=1e-11, atol=1e-12,
                ))

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
                    "source_group": source,
                    "generator_family": (
                        "shared_generator_family" if source in {"generator_a", "generator_b"}
                        else (source if numeric_label == 1 else "")
                    ),
                    "group_id": (f"shared_muse_prompt_{index:03d}"
                                 if source.startswith("generator_")
                                 else f"{source}_creator_{index:03d}"),
                    "native_duration_sec": 35.0,
                    "native_sample_rate_hz": 44_100,
                    "duration_sec": 30,
                    "available": 1,
                    "acquisition": ("expansion" if source.endswith("_c") else "prior"),
                    "feature_status": "complete",
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
                    "feature_status": "complete",
                    "s16_feature_status": "computed", "s8_feature_status": "computed",
                    "d_feature_status": "eligible", "r_feature_status": "eligible",
                    "p_feature_status": "eligible",
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
                "--quantities", "25,50,100,200,all", "--opposite-class-folds", "2",
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
            family_holdout = pd.read_csv(
                cv_output / "evaluate_expanded_generator_family_holdout_summary.csv"
            )
            self.assertEqual(set(family_holdout.heldout_generator_family),
                             {"shared_generator_family", "generator_c"})
            self.assertEqual(set(family_holdout.combination),
                             {frozen["leader"]["combination"], "S"})

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
                 "--slices", "locked_catalogue_plus_pilot", "--replicates", "2000"],
                check=False, text=True, capture_output=True,
            )
            self.assertEqual(bootstrap.returncode, 0, bootstrap.stderr)
            bootstrap_pairs = pd.read_csv(
                bootstrap_output / "evaluate_expanded_group_bootstrap_pairs.csv"
            )
            self.assertEqual(len(bootstrap_pairs), 1)
            self.assertEqual(bootstrap_pairs.independent_group_units.iloc[0], 16)
            plot_path = root / "official_table.png"
            plot = subprocess.run(
                [sys.executable, str(PLOT_PROGRAM),
                 "--cv-summary", str(cv_output / "evaluate_expanded_cv_summary.csv"),
                 "--frozen-selection", str(frozen_path), "--output", str(plot_path)],
                check=False, text=True, capture_output=True,
            )
            self.assertEqual(plot.returncode, 0, plot.stderr)
            self.assertGreater(plot_path.stat().st_size, 10_000)
            plot_pdf_path = root / "official_table.pdf"
            plot_pdf = subprocess.run(
                [sys.executable, str(PLOT_PROGRAM),
                 "--cv-summary", str(cv_output / "evaluate_expanded_cv_summary.csv"),
                 "--frozen-selection", str(frozen_path), "--output", str(plot_pdf_path)],
                check=False, text=True, capture_output=True,
            )
            self.assertEqual(plot_pdf.returncode, 0, plot_pdf.stderr)
            self.assertGreater(plot_pdf_path.stat().st_size, 10_000)
            formal_audit = root / "formal_audit.json"
            formal_audit.write_text(json.dumps({"status": "formal_complete"}), encoding="utf-8")
            latex_output = root / "latex"
            latex = subprocess.run(
                [sys.executable, str(LATEX_PROGRAM), "--duration", "30s",
                 "--cv-dir", str(cv_output), "--locked-dir", str(locked_output),
                 "--bootstrap-dir", str(bootstrap_output),
                 "--formal-audit", str(formal_audit), "--output-dir", str(latex_output),
                 "--metadata", str(metadata), "--features", str(features)],
                check=False, text=True, capture_output=True,
            )
            self.assertEqual(latex.returncode, 0, latex.stderr)
            self.assertEqual(len(list(latex_output.glob("results_30s_*.tex"))), 8)
            self.assertTrue((latex_output / "results_30s_spectral_sensitivity.csv").is_file())
            self.assertTrue((latex_output / "results_30s_s16_availability_by_source.csv").is_file())
            self.assertIn("S8 N/F", (
                latex_output / "results_30s_family_availability_compact.tex"
            ).read_text(encoding="utf-8"))
            self.assertIn("p{2.7cm}", (
                latex_output / "results_30s_locked_specificity.tex"
            ).read_text(encoding="utf-8"))
            self.assertIn("Actual training ranges", (
                latex_output / "results_30s_j_matrix.tex"
            ).read_text(encoding="utf-8"))
            j_matrix = pd.read_csv(latex_output / "results_30s_j_matrix.csv").set_index("families")
            expected_j = float(primary_all.loc[primary_all.combination == "S", "selection_score"].iloc[0])
            self.assertAlmostEqual(float(j_matrix.loc["S", "all"]), expected_j, places=14)
            self.assertNotIn("2\\,\\mathrm{BA}", (
                latex_output / "results_30s_j_matrix.tex"
            ).read_text(encoding="utf-8"))

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
