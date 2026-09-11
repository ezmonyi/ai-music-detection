#!/usr/bin/env python3
"""Synthetic-only tests for the exploratory-v5 scientific figure exporter."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest import mock

import plot_equal60_v5_results as F
import present_equal60_v5_results as P
import test_present_equal60_v5_results as PT


def put(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def png_size(path):
    data = Path(path).read_bytes()[:24]
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("Not a PNG")
    return struct.unpack(">II", data[16:24])


class FigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name).resolve()
        P.AUDITOR = cls.root / "audit_equal60_v5_results.py"
        cls.results, cls.audit = PT.fixture(cls.root / "evaluator_fixture")
        cls.presentation = cls.root / "presentation_fixture"
        P.export(cls.results, cls.audit, P.sha(cls.audit), cls.presentation, synthetic=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.case = self.root / self._testMethodName
        self.case.mkdir()
        self.presentation_copy = self.case / "presentation"
        shutil.copytree(self.presentation, self.presentation_copy)
        self.audit_copy = self.case / "audit.json"
        shutil.copy2(self.audit, self.audit_copy)

    def acceptance(self, extra=None):
        value = {"schema_version": 1, "status": "accepted_for_scientific_figure_export",
                 "reviewer": "root", "synthetic_test_only": True,
                 "presentation_commit_sha256": F.sha(self.presentation_copy / "COMMIT.json"),
                 "presentation_receipt_sha256": F.sha(self.presentation_copy / "presentation_receipt.json"),
                 "numerical_audit_sha256": F.sha(self.audit_copy),
                 "figure_export_authorized": True, "reviewed_utc": "synthetic-test"}
        if extra:
            value.update(extra)
        path = self.case / "acceptance.json"
        put(path, value)
        return path

    def run_export(self, name="figures", acceptance=None):
        acceptance = acceptance or self.acceptance()
        output = self.case / name
        receipt = F.export(self.presentation_copy, acceptance, F.sha(acceptance), self.audit_copy,
                           F.sha(self.audit_copy), output, synthetic=True)
        return output, receipt

    def refresh_presentation_commit(self, filename):
        commit_path = self.presentation_copy / "COMMIT.json"
        commit = F.strict_json(commit_path)
        path = self.presentation_copy / filename
        commit["files"][filename] = {"sha256": F.sha(path), "bytes": path.stat().st_size}
        put(commit_path, commit)

    def test_complete_six_renders_receipt_and_commit(self):
        before = {name: F.sha(self.presentation_copy / name) for name in F.PRESENTATION_FILES}
        output, receipt = self.run_export()
        expected_figures = {f"figure_{n}_{stem}.{suffix}"
                            for n, stem in ((1, "primary_macro_heatmaps"),
                                            (2, "source_endpoints_and_increments"),
                                            (3, "all_cap_diagnostics"))
                            for suffix in ("pdf", "png")}
        self.assertEqual({p.name for p in output.iterdir()}, expected_figures | {"figure_receipt.json", "COMMIT.json"})
        self.assertEqual(receipt["rate_color_scale"], [0, 1])
        self.assertAlmostEqual(receipt["increment_color_scale_pp"][0],
                               -receipt["increment_color_scale_pp"][1])
        self.assertGreater(receipt["increment_color_scale_pp"][1], 0)
        self.assertTrue(receipt["synthetic_test_only"])
        self.assertTrue(receipt["cap_curves_omitted_as_redundant"])
        self.assertRegex(receipt["render_runtime"]["python"], r"^\d+\.\d+")
        self.assertRegex(receipt["render_runtime"]["matplotlib"], r"^\d+\.\d+")
        self.assertRegex(receipt["render_runtime"]["numpy"], r"^\d+\.\d+")
        self.assertEqual(before, {name: F.sha(self.presentation_copy / name) for name in F.PRESENTATION_FILES})
        commit = F.strict_json(output / "COMMIT.json")
        self.assertEqual(set(commit["files"]), expected_figures | {"figure_receipt.json"})
        for filename in expected_figures:
            path = output / filename
            if path.suffix == ".pdf":
                self.assertEqual(path.read_bytes()[:5], b"%PDF-")
            else:
                width, height = png_size(path)
                self.assertGreater(width, 1000)
                self.assertGreater(height, 500)

    def test_missing_primary_cell_rejected_even_with_rehashed_commit(self):
        path = self.presentation_copy / "primary_pair_macro_all_arms.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        fieldnames = list(rows[0])
        removed = [row for row in rows if row["summary_level"] == "fold_type_macro"
                   and row["fold_type"] == F.FOLD_TYPES[0] and row["combination"] == F.COMBINATIONS[0]
                   and row["quantity"] == F.CAPS[0]]
        self.assertEqual(len(removed), 1)
        rows = [row for row in rows if row is not removed[0]]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(rows)
        self.refresh_presentation_commit(path.name)
        with self.assertRaisesRegex(ValueError, "Primary macro heatmap grid"):
            self.run_export()

    def test_presentation_tamper_and_stale_acceptance_rejected(self):
        acceptance = self.acceptance()
        path = self.presentation_copy / "primary_fixed12_overview.csv"
        path.write_text(path.read_text() + "tamper\n")
        with self.assertRaisesRegex(ValueError, "Presentation file changed"):
            self.run_export(acceptance=acceptance)

    def test_duplicate_plus_missing_increment_cell_rejected(self):
        path = self.presentation_copy / "primary_fixed_increments.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        fieldnames = list(rows[0])
        context = tuple(rows[0][name] for name in
                        ("scope", "summary_level", "fold_type", "heldout_source", "source_group", "endpoint"))
        matching = [row for row in rows if tuple(row[name] for name in
                    ("scope", "summary_level", "fold_type", "heldout_source", "source_group", "endpoint")) == context]
        self.assertGreaterEqual(len(matching), 2)
        matching[1]["added_combination"] = matching[0]["added_combination"]
        matching[1]["quantity"] = matching[0]["quantity"]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(rows)
        self.refresh_presentation_commit(path.name)
        with self.assertRaisesRegex(ValueError, "increment grid"):
            self.run_export()

    def test_real_shaped_synthetic_figure_two_layout(self):
        source_contexts = []
        for name in ("H1", "H2", "H3", "H4"):
            source_contexts.append(("human_source_holdout", name, name, "human_specificity"))
        for name in ("A1", "A2", "A3", "A4"):
            source_contexts.append(("generator_holdout", name, name, "ai_sensitivity"))
        for name in ("H1", "H2", "H3", "H4", "A1", "A2", "A3", "A4"):
            endpoint = "human_specificity" if name.startswith("H") else "ai_sensitivity"
            source_contexts.append(("ordinary_group_holdout_descriptive", "__all_sources__", name, endpoint))
        source_rows = []
        for fold, held, source, endpoint in source_contexts:
            for combo in F.FIXED12:
                for cap in F.CAPS:
                    source_rows.append({"fold_type": fold, "heldout_source": held, "source_group": source,
                                        "endpoint": endpoint, "combination": combo,
                                        "feature_mode": "values_plus_missing", "quantity": cap,
                                        "recording_rate": ".625", "equal_component_rate": ".60"})

        increment_contexts = []
        for fold, held, source, endpoint in source_contexts:
            for estimator in ("recording_rate", "equal_component_rate"):
                increment_contexts.append(("source_endpoint", "held_source_arm", fold, held, source,
                                           endpoint + "_" + estimator))
        for fold, held_values in (("human_source_holdout", ("H1", "H2", "H3", "H4")),
                                  ("generator_holdout", ("A1", "A2", "A3", "A4")),
                                  ("ordinary_group_holdout_descriptive", ("__all_sources__",))):
            for held in held_values:
                for metric in ("roc_auc", "balanced_accuracy"):
                    increment_contexts.append(("two_class_pair_macro", "held_source_arm", fold, held,
                                               "__two_class__", metric))
        for fold in F.FOLD_TYPES:
            for metric in ("roc_auc", "balanced_accuracy"):
                increment_contexts.append(("two_class_pair_macro", "fold_type_macro", fold, "__equal_arms__",
                                           "__two_class__", metric))
        self.assertEqual(len(source_contexts), 16)
        self.assertEqual(len(increment_contexts), 56)
        increment_rows = []
        for context in increment_contexts:
            for added in F.ADDED:
                for cap in F.CAPS:
                    increment_rows.append({"scope": context[0], "summary_level": context[1],
                                           "fold_type": context[2], "heldout_source": context[3],
                                           "source_group": context[4], "endpoint": context[5],
                                           "added_combination": added, "quantity": cap,
                                           "baseline_combination": "S+D+R+P",
                                           "added_minus_baseline_pp": "1.25"})
        output = self.case / "real_shaped_synthetic"
        output.mkdir()
        limit = F.figure_source_and_increments(source_rows, increment_rows, output, synthetic=True)
        self.assertEqual(limit, 1.25)
        width, height = png_size(output / "figure_2_source_endpoints_and_increments.png")
        self.assertGreaterEqual(width, 6000)  # At least 0.65 inch per one of 56 multiline context labels.
        self.assertGreater(height, 3000)
        self.assertEqual((output / "figure_2_source_endpoints_and_increments.pdf").read_bytes()[:5], b"%PDF-")

    def test_explicit_audit_and_acceptance_hashes_are_required(self):
        acceptance = self.acceptance()
        with self.assertRaisesRegex(ValueError, "acceptance SHA"):
            F.export(self.presentation_copy, acceptance, "0" * 64, self.audit_copy, F.sha(self.audit_copy),
                     self.case / "bad_acceptance", synthetic=True)
        with self.assertRaisesRegex(ValueError, "audit SHA"):
            F.export(self.presentation_copy, acceptance, F.sha(acceptance), self.audit_copy, "0" * 64,
                     self.case / "bad_audit", synthetic=True)

    def test_exact_acceptance_schema_and_mode_required(self):
        acceptance = self.acceptance({"undeclared_note": "not admitted"})
        with self.assertRaisesRegex(ValueError, "does not authorize"):
            self.run_export(acceptance=acceptance)
        acceptance = self.acceptance({"synthetic_test_only": False})
        with self.assertRaisesRegex(ValueError, "does not authorize"):
            self.run_export("wrong_mode", acceptance)

    def test_existing_output_rejected_without_writes(self):
        acceptance = self.acceptance()
        output = self.case / "exists"
        output.mkdir()
        with self.assertRaisesRegex(ValueError, "New canonical"):
            F.export(self.presentation_copy, acceptance, F.sha(acceptance), self.audit_copy,
                     F.sha(self.audit_copy), output, synthetic=True)

    def test_input_change_during_render_leaves_no_commit(self):
        acceptance = self.acceptance()
        original = F.figure_primary_heatmaps

        def mutate_after_first_figure(*args, **kwargs):
            original(*args, **kwargs)
            acceptance.write_text(acceptance.read_text() + " ")

        output = self.case / "changed_during_render"
        with mock.patch.object(F, "figure_primary_heatmaps", side_effect=mutate_after_first_figure):
            with self.assertRaisesRegex(ValueError, "input changed during rendering"):
                F.export(self.presentation_copy, acceptance, F.sha(acceptance), self.audit_copy,
                         F.sha(self.audit_copy), output, synthetic=True)
        self.assertTrue(output.is_dir())
        self.assertFalse((output / "COMMIT.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
