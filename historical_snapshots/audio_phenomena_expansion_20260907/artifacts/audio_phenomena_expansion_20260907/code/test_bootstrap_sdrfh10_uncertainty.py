#!/usr/bin/env python3
"""10s adapter/invariants plus the preserved numerical core's arithmetic tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import bootstrap_sdrfh10_uncertainty as subject
from test_bootstrap_matched_deltas import WeightedMetricTests, GlobalGroupBootstrapTests, AuthorizationIntegrationTests


def prediction_rows():
    rows = []
    for combo in subject.COMBINATIONS:
        for direction, held in (("human_source_holdout","h1"),("generator_holdout","a1")):
            for i,(label,source,group,score) in enumerate(((0,"h1","shared",.1),(0,"h1","g1",.8),
                                                       (1,"a1","shared",.9),(1,"a1","g2",.2))):
                rows.append({"combination":combo,"quantity":"all","fold_uid":direction,"fold_index":"0",
                             "fold_type":direction,"heldout_source":held,"opposite_group_fold":"0",
                             "row_id":str(i),"label":str(label),"source_group":source,"group_id":group,
                             "score":str(score + (0.05*label if combo != subject.BASELINE else 0)),"threshold":"0.5"})
    return rows


class TenSecondAdapterTests(unittest.TestCase):
    def test_private_core_keeps_original_30s_module_unchanged(self):
        import bootstrap_matched_deltas
        self.assertEqual(bootstrap_matched_deltas.BASELINE,"S+D+R")
        self.assertEqual(subject.core.BASELINE,"S+D")

    def test_paired_identity_missing_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"Prediction identities differ"):
            subject.core.build_observations(prediction_rows()[:-1])

    def test_paired_metadata_changed_is_rejected(self):
        rows=prediction_rows()
        rows[-1]["group_id"]="unpaired"
        with self.assertRaisesRegex(ValueError,"metadata differs"):
            subject.core.build_observations(rows)

    def test_duplicate_prediction_id_is_rejected(self):
        rows=prediction_rows()
        with self.assertRaisesRegex(ValueError,"Duplicate prediction"):
            subject.core.build_observations(rows+[rows[0]])

    def test_cross_label_source_and_fold_group_is_one_unit(self):
        observations=subject.core.build_observations(prediction_rows())
        self.assertEqual(len(observations["groups"]),3)
        indices=[i for i,r in enumerate(observations["master"]) if r["group_id"]=="shared"]
        self.assertEqual(len(indices),4)
        self.assertEqual(len(set(observations["group_indices"][indices])),1)

    def test_scores_are_never_reoriented_using_held_labels(self):
        result=subject.core.weighted_metrics(np.array([0,1]),np.array([.9,.1]),np.ones(2))
        self.assertEqual(result["roc_auc"],0)
        self.assertEqual(result["balanced_accuracy"],0)

    def test_wrong_contract_rejected_even_with_matching_declared_hash(self):
        m={"stage":"dev","schema_version":2,"authorization":{"status":"frozen_verified","contract_sha256":subject.CONTRACT_SHA},"contract":{}}
        with self.assertRaisesRegex(ValueError,"Wrong exact10"):
            subject.check_contract(m)

    def test_bare_or_draft_authorization_rejected(self):
        for auth in ("frozen_verified", {"status":"frozen"}, {"status":"draft"}):
            with self.assertRaisesRegex(ValueError,"nested frozen_verified"):
                subject.check_contract({"stage":"dev","schema_version":2,"authorization":auth})

    def test_no_completion_manifest_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,"Incomplete evaluation"):
                subject.input_paths(Path(root))

    def test_full_cell_denominator_and_point_macro_reconstruction(self):
        core=subject.core
        observations=core.build_observations(prediction_rows())
        metrics,summary,deltas=[],[],[]
        for combo in subject.COMBINATIONS:
            for direction,held in (("human_source_holdout","h1"),("generator_holdout","a1")):
                row={"combination":combo,"quantity":"all","fold_uid":direction,"fold_index":"0",
                     "fold_type":direction,"heldout_source":held,"opposite_group_fold":"0",
                     "human_source":"h1","ai_source":"a1","test_human":"2","test_ai":"2",
                     "test_human_groups":"2","test_ai_groups":"2"}
                # Independent scalar, explicit count calculation verifies vectorized output.
                scores=np.array([.1,.8,.9,.2])
                if combo != subject.BASELINE:
                    scores+=np.array([0,0,.05,.05])
                values=core.weighted_metrics(np.array([0,0,1,1]),scores,np.ones(4))
                row.update({k:str(values[k]) for k in ("roc_auc","balanced_accuracy")})
                metrics.append(row)
                summary.append({"combination":combo,"fold_type":direction,
                                "roc_auc_source_macro":row["roc_auc"],"balanced_accuracy_source_macro":row["balanced_accuracy"]})
                if combo != subject.BASELINE:
                    deltas.append({**row,"comparison_id":combo,"baseline_combination":subject.BASELINE,
                                   "added_combination":combo,"delta_roc_auc__added_minus_baseline":"0",
                                   "delta_balanced_accuracy__added_minus_baseline":"0"})
        cells=core.build_cells(metrics,observations)
        with tempfile.TemporaryDirectory() as root:
            plan=Path(root)/"plan.csv"
            core.write_csv(plan,[{"plan_type":"incremental_matched","arm":"added","combination":c,"comparison_id":c} for c in subject.ADDED])
            points,contrasts=core.validate_point_estimates({"metrics":metrics,"summary":summary,"deltas":deltas,"plan_path":plan},observations,cells)
            self.assertEqual(len(points),4)
            self.assertTrue(all(r["delta_J"]==0 for r in contrasts))
        bad=copy.deepcopy(metrics)
        bad[0]["test_human_groups"]="3"
        with self.assertRaisesRegex(ValueError,"cell count mismatch"):
            core.build_cells(bad,observations)

    def test_source_uniform_and_concentrated_draw(self):
        matrix=np.arange(2*7*13*3*2,dtype=float).reshape(2,7,13,3,2)
        np.testing.assert_allclose(subject.source_reweight(matrix,np.ones(7),np.ones(13)),matrix.mean(axis=(0,1,2)))
        human=np.zeros(7); human[2]=7
        ai=np.zeros(13); ai[8]=13
        np.testing.assert_allclose(subject.source_reweight(matrix,human,ai),matrix[:,2,8].mean(axis=0))

    def test_source_empty_nonfinite_or_wrong_shape_refused(self):
        matrix=np.zeros((2,7,13,3,2))
        with self.assertRaises(ValueError):
            subject.source_reweight(matrix,np.zeros(7),np.ones(13))
        with self.assertRaises(ValueError):
            subject.source_reweight(matrix,np.ones(6),np.ones(13))
        matrix[0,0,0,0,0]=np.nan
        with self.assertRaises(ValueError):
            subject.source_reweight(matrix,np.ones(7),np.ones(13))


if __name__=="__main__":
    unittest.main()
