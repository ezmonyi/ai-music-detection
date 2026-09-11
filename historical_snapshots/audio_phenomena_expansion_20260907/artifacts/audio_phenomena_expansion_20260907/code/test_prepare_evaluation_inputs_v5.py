"""Small package fixtures only; no real preparation, classifier or inference."""
import ast
import copy
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import prepare_evaluation_inputs_v5 as P


def put(path,value):path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')


def write_csv(path,records):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def fixture(root):
    root.mkdir();put(root/'synthetic_fixture.json',{'synthetic_test_only':True,'purpose':'derived_v5_package_fixture_no_real_admission'})
    for kind in ('old','mureka','saraga'):
        metadata=[];features=[]
        for i in range(2):
            name=('Suno' if i else 'MTG-Jamendo') if kind=='old' else ('Mureka_v9' if kind=='mureka' else 'human_saraga_hindustani_v1')
            iid=f'synthetic_v5_{kind}_{i}'
            row=dict(id=iid,label='1' if name in ('Suno','Mureka_v9') else '0',source_group=name,
                group_id='synthetic_group_'+iid,role={'old':'development','mureka':'external_generator_unscored','saraga':'external_human_unscored'}[kind],
                duration_view='60s',native_sample_rate_hz='48000' if kind=='saraga' and i else '44100')
            if kind=='mureka':row.update(acquisition_role='reserved_unscored',classifier_admission_authorized='False')
            if kind=='saraga':row.update(evaluation_allowed='False',classifier_admission_authorized='False')
            metadata.append(row);features.append(dict(id=iid,**{c:['','nan','na','null','1.2300','-0.0'][(j+i)%6] for j,c in enumerate(P.DESCRIPTORS)}))
        write_csv(root/(kind+'_metadata.csv'),metadata);write_csv(root/(kind+'_features.csv'),features)


def rebind_package(root):
    proof=P.read_json(root/'preparation_audit.json');proof['files_sha256']={n:P.sha(root/n) for n in P.PRODUCTS};put(root/'preparation_audit.json',proof)
    put(root/'COMMIT.json',dict(status='committed',publication='exclusive hardlinks, COMMIT last',
        files={n:{'sha256':P.sha(root/n),'bytes':(root/n).stat().st_size} for n in P.PRODUCTS|{'preparation_audit.json'}}))


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name).resolve();cls.fixture=cls.root/'fixture';fixture(cls.fixture)
        cls.spec={'synthetic_fixture':str(cls.fixture)};cls.package=cls.root/'package'
        with patch.object(P,'old_source',side_effect=AssertionError('No real data admission')),patch.object(P,'external_source',side_effect=AssertionError('No real external admission')):
            cls.proof=P.prepare(cls.spec,cls.package,True)
        cls.sources=P.synthetic_sources(cls.fixture,P.Bindings())
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_package_exact_schema_rows_and_no_authorization(self):
        proof=P.validate_package(self.package,True)
        self.assertEqual(proof['rows'],6);self.assertFalse(proof['fitting_authorized']);self.assertFalse(proof['scoring_authorized'])
        self.assertEqual(set(proof['files_sha256']),P.PRODUCTS)
        self.assertEqual(P.read_json(self.package/'family_config.json'),P.P4.build_config())
        self.assertEqual(proof['contract']['descriptor_columns'],P.DESCRIPTORS)
        self.assertEqual(len(P.DESCRIPTORS),54)

    def test_every_value_and_missing_token_preserved(self):
        rows=P.rows(self.package/'features_60s.csv',P.Bindings(),['id',*P.DESCRIPTORS]);got={r['id']:r for r in rows}
        for source in self.sources:
            for row in source['features']:self.assertEqual(got[row['id']],row)
        self.assertEqual({v for r in rows for c,v in r.items() if c!='id'},{'','nan','na','null','1.2300','-0.0'})

    def test_origin_roles_flags_and_consumed_disclosure(self):
        meta=P.rows(self.package/'metadata_60s.csv',P.Bindings(),P.META);ledger=P.rows(self.package/'origin_ledger.csv',P.Bindings(),P.LEDGER)
        self.assertEqual({r['role'] for r in meta},{'development'})
        for row in ledger:
            external=row['origin_kind']!='old';self.assertEqual(row['consumed_external'],str(external))
            original=json.loads(row['original_metadata_json']);self.assertEqual(row['original_role'],original['role'])
            flags=json.loads(row['original_flags_json'])
            if row['origin_kind']=='saraga':self.assertEqual(flags,{'evaluation_allowed':'False','classifier_admission_authorized':'False'})
            if row['origin_kind']=='old':self.assertEqual(flags,{})
        self.assertEqual(self.proof['contract']['label_counts'],{'0':3,'1':3})
        self.assertEqual(self.proof['contract']['source_counts'],{'MTG-Jamendo':1,'Suno':1,'Mureka_v9':2,'human_saraga_hindustani_v1':2})

    def test_native_info_and_sorted_products(self):
        meta=P.rows(self.package/'metadata_60s.csv',P.Bindings());features=P.rows(self.package/'features_60s.csv',P.Bindings());ledger=P.rows(self.package/'origin_ledger.csv',P.Bindings())
        ids=[r['id'] for r in meta];self.assertEqual(ids,sorted(ids));self.assertEqual(ids,[r['id'] for r in features]);self.assertEqual(ids,[r['id'] for r in ledger])
        self.assertEqual({r['native_sample_rate_hz'] for r in meta},{'44100','48000'})

    def test_original_locked_pilot_and_development_relabel_refused(self):
        for origin,role in [(0,'locked'),(0,'pilot'),(1,'development'),(2,'development')]:
            data=copy.deepcopy(self.sources);data[origin]['metadata'][0]['role']=role
            with self.assertRaisesRegex(ValueError,'role leakage'):P.combine(data,True)

    def test_historical_id_and_group_exclusion_includes_external_rows(self):
        for origin in range(3):
            for collision in ('id','group'):
                data=copy.deepcopy(self.sources);row=data[origin]['metadata'][0]
                data[0]['excluded_id_groups']={row['id'] if collision=='id' else 'synthetic_v5_excluded':
                    row['group_id'] if collision=='group' else 'synthetic_group_excluded'}
                with self.assertRaisesRegex(ValueError,'locked/pilot IDs or groups'):P.combine(data,True)

    def test_original_native_flags_preserved_without_rewriting_package_row(self):
        data=copy.deepcopy(self.sources);original=data[0]['metadata'][0];iid=original['id']
        data[0]['flags'][iid]={'evaluation_allowed':''}
        _,_,ledger=P.combine(data,True);row=next(r for r in ledger if r['id']==iid)
        self.assertEqual(json.loads(row['original_flags_json']),{'evaluation_allowed':''})
        self.assertEqual(json.loads(row['original_metadata_json']),original)

    def test_nested_validator_code_is_contract_bound(self):
        path=Path(P.__file__).with_name('verify_extract_equal60.py').resolve()
        self.assertEqual(self.proof['contract']['input_files_sha256'][str(path)],P.sha(path))

    def test_duplicate_and_order_mismatch_refused(self):
        data=copy.deepcopy(self.sources);data[0]['features'].reverse()
        with self.assertRaisesRegex(ValueError,'order mismatch'):P.combine(data,True)
        data=copy.deepcopy(self.sources);data[0]['metadata'].append(data[0]['metadata'][0]);data[0]['features'].append(data[0]['features'][0])
        with self.assertRaisesRegex(ValueError,'duplicate'):P.combine(data,True)

    def test_cross_source_duplicate_and_group_collision_refused(self):
        data=copy.deepcopy(self.sources);old=data[1]['metadata'][0]['id'];new=data[0]['metadata'][0]['id']
        data[1]['metadata'][0]['id']=data[1]['features'][0]['id']=new;data[1]['rates'][new]=data[1]['rates'].pop(old);data[1]['flags'][new]=data[1]['flags'].pop(old)
        with self.assertRaisesRegex(ValueError,'Cross-origin duplicate'):P.combine(data,True)
        data=copy.deepcopy(self.sources);data[1]['metadata'][0]['group_id']=data[0]['metadata'][0]['group_id']
        with self.assertRaisesRegex(ValueError,'Global group crosses'):P.combine(data,True)

    def test_predictor_schema_infinity_and_bad_tokens_refused(self):
        for token in ('inf','-Infinity','nonsense'):
            data=copy.deepcopy(self.sources);data[0]['features'][0][P.DESCRIPTORS[0]]=token
            with self.assertRaises(ValueError):P.combine(data,True)
        data=copy.deepcopy(self.sources);data[0]['features'][0]['status']='good'
        with self.assertRaisesRegex(ValueError,'predictor schema'):P.combine(data,True)

    def test_source_label_origin_and_context_refused(self):
        for key,value in [('label','0'),('source_group','Suno'),('duration_view','30s')]:
            data=copy.deepcopy(self.sources);data[1]['metadata'][0][key]=value
            with self.assertRaises(ValueError):P.combine(data,True)

    def test_external_admission_flags_cannot_change(self):
        data=copy.deepcopy(self.sources);data[2]['metadata'][0]['evaluation_allowed']='True'
        with self.assertRaisesRegex(ValueError,'flags changed'):P.combine(data,True)
        data=copy.deepcopy(self.sources);iid=data[2]['metadata'][0]['id'];data[2]['flags'][iid]={'evaluation_allowed':'True'}
        with self.assertRaisesRegex(ValueError,'flags disagree'):P.combine(data,True)

    def test_real_and_synthetic_modes_cannot_mix(self):
        with self.assertRaises(ValueError):P.assemble(self.spec,False)
        with self.assertRaises(ValueError):P.validate_package(self.package,False)
        with self.assertRaises(ValueError):P.assemble(dict(self.spec,old_package='forbidden'),True)
        with self.assertRaisesRegex(ValueError,'synthetic identity'):P.combine(self.sources,False)

    def test_changed_product_and_resigned_value_tamper_refused(self):
        root=self.root/'changed_package';shutil.copytree(self.package,root)
        rows=P.rows(root/'features_60s.csv',P.Bindings());rows[0][P.DESCRIPTORS[0]]='999';write_csv(root/'features_60s.csv',rows)
        with self.assertRaises(ValueError):P.validate_package(root,True)
        rebind_package(root)
        with self.assertRaisesRegex(ValueError,'Derived values'):P.validate_package(root,True)

    def test_original_input_change_invalidates_frozen_package(self):
        root=self.root/'changed_input';shutil.copytree(self.fixture,root);spec={'synthetic_fixture':str(root)};out=self.root/'input_bound_package';P.prepare(spec,out,True)
        path=root/'old_features.csv';r=P.rows(path,P.Bindings());r[0][P.DESCRIPTORS[0]]='3.4';write_csv(path,r)
        with self.assertRaisesRegex(ValueError,'contract differs'):P.validate_package(out,True)

    def test_exclusive_commit_publication_and_orphan_refusal(self):
        with self.assertRaisesRegex(ValueError,'Existing output'):P.prepare(self.spec,self.package,True)
        orphan=self.root/'orphan';orphan.mkdir()
        with self.assertRaises(ValueError):P.prepare(self.spec,orphan,True)
        with self.assertRaises(ValueError):P.validate_package(orphan,True)

    def test_no_real_or_fit_decode_entrypoints(self):
        tree=ast.parse(Path(P.__file__).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(any(s in (n or '') for s in ('evaluate_new','score_','torch','soundfile','sklearn')) for n in imports))
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('fit','predict','decode','median','impute') for n in ast.walk(tree)))

    def test_real_count_contract_without_real_identities(self):
        # Arithmetic-only validation; no manufactured real Human IDs or admission.
        P.validate_real_counts(dict(P.COUNTS),{'0':1311,'1':896},[72,14,10,5,2])
        for sources,labels,groups in [(dict(P.COUNTS,Suno=397),{'0':1311,'1':896},[72,14,10,5,2]),
                                      (dict(P.COUNTS),{'0':1310,'1':897},[72,14,10,5,2]),
                                      (dict(P.COUNTS),{'0':1311,'1':896},[71,15,10,5,2])]:
            with self.assertRaises(ValueError):P.validate_real_counts(sources,labels,groups)

    def test_unaccepted_real_audit_cannot_enter_package(self):
        bad=self.root/'unaccepted_audit.json';put(bad,{'status':'passed','synthetic_test_only':True})
        spec=P.defaults(self.root);spec['old_audit']=str(bad)
        with self.assertRaisesRegex(ValueError,'hash mismatch'):P.old_source(spec,P.Bindings())
        for kind in ('mureka','saraga'):
            spec[kind+'_audit']=str(bad)
            with self.assertRaisesRegex(ValueError,'hash mismatch'):P.external_source(spec,kind,P.Bindings())

    def test_mirror_binding_requires_unique_exact_file_hash(self):
        path=self.root/'compact.csv';path.write_text('synthetic compact bytes')
        mapping={'/original/compact.csv':P.sha(path)}
        self.assertEqual(P.bound_by_basename(path,mapping,P.Bindings()),'/original/compact.csv')
        for changed in [{'/original/compact.csv':'0'*64},{'/original/other.csv':P.sha(path)},
                        {**mapping,'/another/compact.csv':P.sha(path)}]:
            with self.assertRaises(ValueError):P.bound_by_basename(path,changed,P.Bindings())

    def test_hardlink_failure_never_overwrites_or_accepts_orphan(self):
        output=self.root/'failed_publication'
        with patch.object(P.U.os,'link',side_effect=OSError('unsupported link')):
            with self.assertRaises(OSError):P.prepare(self.spec,output,True)
        self.assertFalse((output/'COMMIT.json').exists())
        with self.assertRaisesRegex(ValueError,'Existing output'):P.prepare(self.spec,output,True)


if __name__=='__main__':unittest.main()
