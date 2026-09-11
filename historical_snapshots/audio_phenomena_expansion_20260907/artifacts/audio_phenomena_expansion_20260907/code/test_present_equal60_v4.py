#!/usr/bin/env python3
"""Synthetic-fixture-only tests for the exhaustive schema-v4 presenter."""
import csv
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import present_equal60_v4 as P


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_key(combo):
    return hashlib.sha256(("fixture-cohort|" + combo).encode()).hexdigest()[:20]


def make_fixture(root):
    """Build unmistakably synthetic saved-metric files; never call an evaluator."""
    root = Path(root)
    root.mkdir()
    combos = P.combinations()
    plan = []
    plan_id = 0
    for combo in P.subsets(P.OLD_ORDER):
        plan.append(dict(plan_id=f"fixture-{plan_id}", plan_type="old_baseline_standalone",
                         comparison_id="", arm="standalone", combination=combo,
                         cohort_id="fixture-cohort", eligible_rows=100,
                         eligible_id_set_sha256="f" * 64))
        plan_id += 1
    for combo in P.subsets(P.NEW_ORDER):
        plan.append(dict(plan_id=f"fixture-{plan_id}", plan_type="new_standalone",
                         comparison_id="", arm="standalone", combination=combo,
                         cohort_id="fixture-cohort", eligible_rows=100,
                         eligible_id_set_sha256="f" * 64))
        plan_id += 1
    for cid, left, added in P.comparisons():
        for arm, combo in (("baseline", left), ("added", added)):
            plan.append(dict(plan_id=f"fixture-{plan_id}", plan_type="incremental_matched",
                             comparison_id=cid, arm=arm, combination=combo,
                             cohort_id="fixture-cohort", eligible_rows=100,
                             eligible_id_set_sha256="f" * 64))
            plan_id += 1
    write(root / "evaluation_plan.csv", plan)

    pair, pooled = [], []
    for ci, combo in enumerate(combos):
        for qi, cap in enumerate(P.CAPS):
            for fold in range(5):
                common = dict(cohort_id="fixture-cohort", candidate_key=candidate_key(combo),
                              combination=combo, quantity=cap, fold_uid=f"fold-{fold}-{cap}",
                              fold_index=fold, fold_type=P.FOLD_TYPE,
                              heldout_source="__all_sources__", opposite_group_fold=fold,
                              model_sha256="m" * 64, feature_mode="values_plus_missing",
                              train_id_set_sha256="t" * 64, test_id_set_sha256="e" * 64,
                              threshold=0.5)
                base = 0.50 + ci * .0001 + qi * .001 + fold * .01
                pooled.append({**common, "roc_auc": base + .05,
                               "balanced_accuracy": base + .04,
                               "ai_sensitivity": base + .03,
                               "human_specificity": base + .05,
                               "tp": 4, "tn": 12, "fp": 3, "fn": 1})
                for human in range(5):
                    pair.append({**common, "human_source": f"Human{human}", "ai_source": "Suno",
                                 "test_human": 4, "test_ai": 3,
                                 "test_human_groups": 4, "test_ai_groups": 3,
                                 "roc_auc": base + human * .001,
                                 "balanced_accuracy": base - .01 + human * .001,
                                 "ai_sensitivity": base - .02 + human * .001,
                                 "human_specificity": base + human * .001,
                                 "tp": 2, "tn": 3, "fp": 1, "fn": 1})
    write(root / "development_group_cv_metrics_by_source_pair.csv", pair)
    write(root / "development_group_cv_pooled_fold_metrics.csv", pooled)
    lookup = {(r["combination"], r["quantity"], r["fold_uid"], r["human_source"]): r for r in pair}
    deltas = []
    for cid, left, added in P.comparisons():
        for cap in P.CAPS:
            for fold in range(5):
                for human in range(5):
                    key = (cap, f"fold-{fold}-{cap}", f"Human{human}")
                    a, b = lookup[(left,) + key], lookup[(added,) + key]
                    row = dict(comparison_id=cid, baseline_combination=left,
                               added_combination=added, eligible_id_set_sha256="f" * 64,
                               cohort_id="fixture-cohort", quantity=cap, fold_uid=key[1],
                               fold_index=fold, fold_type=P.FOLD_TYPE,
                               heldout_source="__all_sources__", human_source=key[2], ai_source="Suno",
                               opposite_group_fold=fold, test_human=4, test_ai=3,
                               test_human_groups=4, test_ai_groups=3)
                    for measure in P.MEASURES:
                        row[measure + "__baseline"] = a[measure]
                        row[measure + "__added"] = b[measure]
                        row["delta_" + measure + "__added_minus_baseline"] = b[measure] - a[measure]
                    deltas.append(row)
    write(root / "development_group_cv_matched_deltas.csv", deltas)
    hashes = {name: P.sha(root / name) for name in P.REQUIRED_RESULTS}
    manifest = {"schema_version": 4, "stage": "dev",
                "authorization": {"status": "synthetic_test_only"},
                "contract": {"synthetic_test_only": True},
                "source_transfer_J": None,
                "source_holdout_status": "not_run_single_development_AI_source_Suno",
                "winner_selected": False, "files_sha256": hashes}
    dump(root / "run_manifest.json", manifest)
    return manifest


def make_audit(path, result_dir, **changes):
    report = {"status": "passed", "schema_version": 4, "synthetic_test_only": True,
              "rows": 100, "candidates": 127, "caps": [25, 50, 100, 200, "all"],
              "primary_predictions": 63500, "diagnostic_predictions": 25400,
              "pooled_metrics": 3175, "source_pair_metrics": 15875,
              "diagnostic_metrics": 6350, "matched_comparisons": 105,
              "matched_delta_rows": 13125, "source_transfer_J": None,
              "model_fitting_performed": False, "metrics_independently_recomputed": True,
              "result_manifest_sha256": P.sha(Path(result_dir) / "run_manifest.json"),
              "auditor_sha256": P.sha(P.AUDITOR)}
    report.update(changes)
    dump(path, report)


class PresenterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.results = self.root / "SYNTHETIC_FIXTURE_RESULTS"
        make_fixture(self.results)
        self.audit = self.root / "SYNTHETIC_FIXTURE_AUDIT.json"
        make_audit(self.audit, self.results)

    def tearDown(self):
        self.temp.cleanup()

    def export(self):
        output = self.root / "SYNTHETIC_FIXTURE_PRESENTATION"
        receipt = P.export(self.results, self.audit, output, synthetic=True)
        return output, receipt

    def rewrite_and_rehash(self, name, mutate):
        path = self.results / name
        rows, _ = P.read_csv(path)
        mutate(rows)
        write(path, rows)
        manifest = P.strict_json(self.results / "run_manifest.json")
        manifest["files_sha256"][name] = P.sha(path)
        dump(self.results / "run_manifest.json", manifest)
        make_audit(self.audit, self.results)

    def test_complete_127x5_and_105x5_with_known_aggregate(self):
        output, receipt = self.export()
        pair, _ = P.read_csv(output / "all_127x5_source_pair_fold_mean.csv")
        pooled, _ = P.read_csv(output / "all_127x5_pooled_fold_mean.csv")
        delta, _ = P.read_csv(output / "all_105x5_matched_contrasts.csv")
        self.assertEqual((len(pair), len(pooled), len(delta)), (635, 635, 525))
        self.assertEqual({(r["combination"], r["quantity"]) for r in pair},
                         {(c, q) for c in P.combinations() for q in P.CAPS})
        self.assertEqual({(r["comparison_id"], r["quantity"]) for r in delta},
                         {(cid, q) for cid, _, _ in P.comparisons() for q in P.CAPS})
        first = next(r for r in pair if r["combination"] == "S" and r["quantity"] == "25")
        # Mean of fold offsets [0,.01,...,.04] and source offsets [0,.001,...,.004].
        self.assertAlmostEqual(float(first["source_pair_fold_mean_roc_auc"]), .522)
        self.assertEqual((first["pair_fold_cells"], first["folds"], first["human_sources"]), ("25", "5", "5"))
        contrast = next(r for r in delta if r["comparison_id"] == "incremental::S__plus__F" and r["quantity"] == "25")
        expected = (P.combinations().index("S+F") - P.combinations().index("S")) * .0001
        self.assertAlmostEqual(float(contrast["source_pair_fold_mean_delta_roc_auc"]), expected)
        markdown = (output / "EQUAL60_PRESENTATION_V4_TABLES_EN.md").read_text()
        latex = (output / "equal60_presentation_v4_tables_en.tex").read_text()
        self.assertIn("SYNTHETIC FIXTURE", markdown)
        self.assertNotIn("source-transfer J /", markdown.lower())
        self.assertEqual(latex.count(r"\begin{longtable}"), 4)
        self.assertIn(r"p{0.22\linewidth}*{5}{p{0.14\linewidth}}", latex)
        self.assertIn(r"\fbox{\parbox", latex)  # Visible warning, not only a comment.
        self.assertIn(r"S $\rightarrow$ S+F", latex)
        self.assertNotIn("incremental::S__plus__F", markdown)
        self.assertNotIn(r"incremental::S\_\_plus\_\_F", latex)
        self.assertTrue(any(r["comparison_id"] == "incremental::S__plus__F" for r in delta))
        manifest = P.strict_json(self.results / "run_manifest.json")
        for name, digest in manifest["files_sha256"].items():
            self.assertEqual(receipt["inputs"][str((self.results / name).resolve())], digest)
        self.assertEqual(receipt["candidate_cap_cells"], 635)
        self.assertIsNone(receipt["source_transfer_J"])

    def test_missing_and_duplicate_cells_fail(self):
        name = "development_group_cv_metrics_by_source_pair.csv"
        self.rewrite_and_rehash(name, lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "Missing source-pair|Incomplete"):
            self.export()
        shutil.rmtree(self.results)
        make_fixture(self.results)
        make_audit(self.audit, self.results)
        self.rewrite_and_rehash(name, lambda rows: rows.__setitem__(-1, dict(rows[0])))
        with self.assertRaisesRegex(ValueError, "Duplicate source-pair"):
            self.export()

    def test_tampered_manifest_or_result_and_stale_audit_fail(self):
        with (self.results / "development_group_cv_pooled_fold_metrics.csv").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "changed after audit"):
            self.export()
        shutil.rmtree(self.results)
        make_fixture(self.results)
        make_audit(self.audit, self.results)
        manifest = P.strict_json(self.results / "run_manifest.json")
        manifest["winner_selected"] = True
        dump(self.results / "run_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "Audit does not bind|winner"):
            self.export()

    def test_audit_gate_and_incorrect_j_claim_forbidden(self):
        make_audit(self.audit, self.results, status="failed")
        with self.assertRaisesRegex(ValueError, "audit gate"):
            self.export()
        make_audit(self.audit, self.results, source_transfer_J=.7)
        with self.assertRaisesRegex(ValueError, "source_transfer_J"):
            self.export()
        manifest = P.strict_json(self.results / "run_manifest.json")
        manifest["source_transfer_J"] = .7
        dump(self.results / "run_manifest.json", manifest)
        make_audit(self.audit, self.results)
        with self.assertRaisesRegex(ValueError, "J or winner"):
            self.export()

    def test_real_mode_refuses_synthetic_fixture(self):
        with self.assertRaisesRegex(ValueError, "synthetic_test_only"):
            P.export(self.results, self.audit, self.root / "real-looking", synthetic=False)

    def test_mutation_during_aggregation_is_detected_from_initial_snapshot(self):
        original = P.aggregate_inputs
        target = self.results / "development_group_cv_metrics_by_source_pair.csv"

        def mutate_after_read(*args):
            result = original(*args)
            with target.open("a") as handle:
                handle.write("\n")
            return result

        with patch.object(P, "aggregate_inputs", side_effect=mutate_after_read), \
             self.assertRaisesRegex(ValueError, "changed during presentation export"):
            self.export()


if __name__ == "__main__":
    unittest.main()
