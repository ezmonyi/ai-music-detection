"""Synthetic tests for the new executor, not evidence of real-cohort completion."""
import tempfile
from pathlib import Path
import unittest
import pandas as pd
import evaluate_expanded_native30_v1 as run


class ExecutorTest(unittest.TestCase):
    def setUp(self):
        rows=[]
        for i in range(24):
            rows.append({'__id':str(i),'__label':i%2,'__source':'human' if i%2==0 else 'ai',
                         '__group':f'g{i}','__component':f'c{i}','__role':'development',
                         **{name:(None if i%7==0 else float((i+1)*(j+1)%17))
                            for j,name in enumerate(run.evaluator.FEATURES)}})
        run._TABLE=pd.DataFrame(rows)
        run._CONTRACT={'fixture':'synthetic_not_actual_evaluation'}
        self.tmp=tempfile.TemporaryDirectory()
        run._OUT=Path(self.tmp.name)
        (run._OUT/'models').mkdir()
        self.task={'train_ids':[str(i) for i in range(16)],'test_ids':[str(i) for i in range(16,24)],
                   'train_id_set_sha256':run.evaluator.planner.id_set_hash([str(i) for i in range(16)]),
                   'test_id_set_sha256':run.evaluator.planner.id_set_hash([str(i) for i in range(16,24)]),
                   'combination':'BC','feature_mode':'values_plus_missing','quantity':'all',
                   'schedule_uid':'synthetic','fold_type':'ordinary_group_holdout_descriptive',
                   'heldout_source':'__all_sources__'}

    def tearDown(self):self.tmp.cleanup()

    def test_fit_replay_and_resume_all_modes(self):
        for mode in ('values_plus_missing','median_only','missingness_only'):
            task={**self.task,'feature_mode':mode}
            first=run.one(task)
            self.assertEqual(first,run.one(task))

    def test_full_catalogue(self):
        task={**self.task,'combination':'+'.join(run.evaluator.FAMILIES)}
        uid,binding=run.one(task)
        self.assertTrue(Path(binding['path']).is_file())

    def test_reject_shared_train_test_group(self):
        run._TABLE.loc[16,'__group']='g0'
        with self.assertRaises(ValueError):run.one(self.task)

    def test_forked_workers_preserve_verified_results(self):
        tasks=[{**self.task,'feature_mode':mode} for mode in ('values_plus_missing','median_only','missingness_only')]
        with run.ProcessPoolExecutor(max_workers=2,mp_context=run.multiprocessing.get_context('fork')) as pool:
            results=list(pool.map(run.one,tasks))
        self.assertEqual(len({uid for uid,_ in results}),3)
        self.assertEqual(results,[run.one(task) for task in tasks])


if __name__=='__main__':unittest.main()
