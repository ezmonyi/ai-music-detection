#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import audit_sdrfh_10s_results as audit10
import pandas as pd


def family(state: str = "available") -> dict[str, object]:
    return {
        "state": state,
        "columns": ["x"] if state == "available" else [],
        "eligibility": {},
        "status_columns": [],
        "minimum_observed_fraction": 0.05,
    }


def config_sdrfh() -> dict[str, object]:
    return {
        "old_families": {
            "S": family(), "D": family(), "R": family(), "P": family("planned"),
        },
        "new_families": {
            "F": family(), "H": family(), "M": family("planned"),
            "V": family("planned"), "B": family("planned"),
            "A": family("planned"), "T": family("planned"),
        },
    }


class Audit10Tests(unittest.TestCase):
    def test_schema_v2_sdrfh_lattice_is_derived(self) -> None:
        observed = audit10.lattice_expectations(config_sdrfh())
        self.assertEqual(observed["active_old"], ("S", "D", "R"))
        self.assertEqual(observed["planned_old"], ("P",))
        self.assertEqual(observed["active_new"], ("F", "H"))
        self.assertEqual(len(observed["old_combinations"]), 7)
        self.assertEqual(len(observed["new_combinations"]), 3)
        self.assertEqual(len(observed["all_candidates"]), 31)
        self.assertEqual(observed["incremental_comparisons"], 21)
        self.assertEqual(observed["plan_rows"], 52)
        self.assertNotIn("P", {code for combo in observed["all_candidates"] for code in combo.split("+")})

    def test_lattice_is_not_hard_coded_to_31(self) -> None:
        config = config_sdrfh()
        config["old_families"]["R"] = family("planned")
        observed = audit10.lattice_expectations(config)
        self.assertEqual(len(observed["old_combinations"]), 3)
        self.assertEqual(len(observed["new_combinations"]), 3)
        self.assertEqual(len(observed["all_candidates"]), 15)
        self.assertEqual(observed["incremental_comparisons"], 9)
        self.assertEqual(observed["plan_rows"], 24)

    def test_fold_counts_come_from_sources_and_contract(self) -> None:
        counts = audit10.expected_fold_counts(7, 13, 5, [25, 50, 100, 200, "all"])
        self.assertEqual(counts["human_source_holdout"], 175)
        self.assertEqual(counts["generator_holdout"], 325)
        self.assertEqual(counts["ordinary_group_holdout_descriptive"], 25)
        self.assertEqual(counts["per_quantity"], 105)
        self.assertEqual(counts["total"], 525)

    def test_frozen_inputs_yield_actual_10s_source_and_fold_counts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        family_config = json.loads(
            (root / "evaluation_inputs/sdrfh_10s_v2/families_10s_v2.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(len(audit10.lattice_expectations(family_config)["all_candidates"]), 31)
        metadata = pd.read_csv(
            root / "evaluation_inputs/sdrfh_10s_v2/metadata_10s.csv", low_memory=False
        )
        cohort = metadata[
            (metadata["role"] == "development")
            & (metadata["duration_view"] == "10s")
            & (metadata["eligible_common8"] == 1)
        ].rename(
            columns={
                "id": "__id", "label": "__label", "source_group": "__source",
                "group_id": "__group",
            }
        )
        self.assertEqual(len(cohort), 8361)
        self.assertEqual(cohort.loc[cohort["__label"] == 0, "__source"].nunique(), 7)
        self.assertEqual(cohort.loc[cohort["__label"] == 1, "__source"].nunique(), 13)
        folds = audit10.reconstruct_base_folds("synthetic-id", cohort, 5, 20260907)
        self.assertEqual(len(folds), 105)

    def test_canonical_code_order_rejects_reordering(self) -> None:
        self.assertEqual(audit10._canonical_codes('["S", "R", "F"]'), ("S", "R", "F"))
        with self.assertRaises(audit10.AuditFailure):
            audit10._canonical_codes('["F", "S"]')
        with self.assertRaises(audit10.AuditFailure):
            audit10._canonical_codes('["S", "S"]')

    def test_hash_fold_is_deterministic(self) -> None:
        self.assertEqual(audit10.hash_fold("group-17", 5, 20260907), audit10.hash_fold("group-17", 5, 20260907))
        self.assertIn(audit10.hash_fold("group-17", 5, 20260907), range(5))

    def test_partial_output_is_not_read_without_run_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            (results / "evaluation_plan.csv").write_text("malformed partial data\n", encoding="utf-8")
            args = audit10.parse_args(["--results-dir", str(results)])
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), self.assertRaises(SystemExit) as raised:
                audit10.load_contract_and_bundle(args, audit10.Audit())
            self.assertEqual(raised.exception.code, 3)
            receipt = json.loads(stream.getvalue())
            self.assertEqual(receipt["status"], "not_ready")
            self.assertIn("partial outputs were not inspected", receipt["reason"])


if __name__ == "__main__":
    unittest.main()
