"""Synthetic acceptance checks; no acquired media, extraction or model fitting."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import run_mureka60_fhm_v2 as M


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(M.prep.json_bytes(value))


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.features = self.root/'features'
        self.metadata = self.root/'metadata.csv'
        self.reference = dict(duration=60.0, preflight_only=False, input_config={}, F_config={},
                              feature_names={'F':['F_x'],'H':['H_x'],'M':['M_x']}, code_sha256={}, runtime={})
        rows = [dict(id='synthetic_'+str(i), audio_path='/synthetic/'+str(i)+'.mp3',
                     source_audio_sha256='a'*64, crop_start_frame=100, sf_header_frames=4000000,
                     group_id='synthetic_group_'+str(i), role=M.prep.ROLE) for i in range(500)]
        self.metadata.write_bytes(M.prep.csv_bytes(rows))
        rows = M.read_csv(self.metadata)
        self.contract = dict(self.reference, selected_ids=[r['id'] for r in rows],
                             metadata_sha256=M.prep.sha_file(self.metadata))
        self.contract['contract_hash'] = M.prep.canonical_hash(self.contract)
        self.records = []
        for row in rows:
            item = dict(row, input_row_hash=M.prep.canonical_hash(row),
                        extraction_contract_hash=self.contract['contract_hash'], extraction_status='ok',
                        source_audio_path=row['audio_path'], crop_start_frame=100, crop_frames=M.prep.FRAMES,
                        source_total_frames=4000000, analysis_frames=960000, analysis_sr=16000,
                        F_x=.25, H_x=.5, M_x=None, F_status='ok', H_status='ok', M_status='unavailable',
                        analysis_waveform_sha256='b'*64)
            path = self.features/'items'/(hashlib.sha256(row['id'].encode()).hexdigest()+'.json')
            put(path,item)
            self.records.append(item)
        self.publish()

    def publish(self):
        put(self.features/'contract.json',self.contract)
        (self.features/'features.csv').write_bytes(M.prep.csv_bytes(self.records))
        put(self.features/'summary.json',dict(expected=500, recorded=500, complete_accounting=True,
            contract_hash=self.contract['contract_hash'], features_csv_sha256=M.prep.sha_file(self.features/'features.csv'),
            status_counts={'ok':500}, family_status_counts={'F_status':{'ok':500},'H_status':{'ok':500},
                                                          'M_status':{'unavailable':500}}))
        put(self.features/'process.json',dict(state='finished',complete_accounting=True))

    def validate(self):
        return M.validate_results(self.features,self.metadata,self.reference)

    def test_all500_retained_with_unobservable_family(self):
        result = self.validate()
        self.assertEqual(result['rows'],500)
        self.assertEqual(result['family_status_counts']['M'],{'unavailable':500})
        self.assertFalse(result['classifier_fitted'])

    def test_descriptor_tampering_rejected(self):
        self.records[0]['F_x'] = .26
        self.publish()
        with self.assertRaisesRegex(ValueError,'Descriptor CSV/item'):
            self.validate()

    def test_missing_row_rejected(self):
        self.records.pop()
        self.publish()
        with self.assertRaisesRegex(ValueError,'Feature rows'):
            self.validate()

    def test_processing_error_not_silently_accepted(self):
        path = next((self.features/'items').iterdir())
        item = M.prep.read_json(path); item['extraction_status']='input_error'; put(path,item)
        with self.assertRaisesRegex(ValueError,'Processing failure retained'):
            self.validate()

    def test_header_observation_not_substituted_for_actual_crop(self):
        path = next((self.features/'items').iterdir())
        item = M.prep.read_json(path); item['crop_start_frame']=101; put(path,item)
        with self.assertRaisesRegex(ValueError,'interval/source/role'):
            self.validate()

    def test_runtime_parity_before_acceptance(self):
        reference = copy.deepcopy(self.reference); reference['runtime']={'numpy':'other'}
        with self.assertRaisesRegex(ValueError,'parity failed'):
            M.validate_results(self.features,self.metadata,reference)

    def test_summary_counts_are_verified(self):
        path=self.features/'summary.json'; summary=M.prep.read_json(path)
        summary['family_status_counts']['M_status']={'ok':500}; put(path,summary)
        with self.assertRaisesRegex(ValueError,'summary observability'):
            self.validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
