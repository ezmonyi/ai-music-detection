"""Synthetic metadata only; no numerical modules, features, audio or classifiers."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import plan_native30_evaluation_schedule_bc_v1 as m


def row(ident,source,label,group=None,component=None):
    return {'id':ident,'source_group':source,'label':label,'role':'development',
            'group_id':group or 'g_'+ident,'component_id':component or 'c_'+(group or ident)}


def screen_for(rows):
    components={}
    for r in rows:
        c=components.setdefault(r['component_id'],{'component_id':r['component_id'],'members':[],
            'candidate_screen_members':[],'protected_relationships':[],'link_tokens':[]})
        c['members'].append(r['id']); c['candidate_screen_members'].append(r['id'])
    return {'rows':copy.deepcopy(rows),'components':list(components.values())}


def fixture():
    rows=[]
    for source,label in [('H1','0'),('H2','0'),('A1','1'),('A2','1')]:
        rows += [row(f'{source}_{n}',source,label,('paired' if source in ('H1','A1') else source)+f'_{n}') for n in range(30)]
    rows.sort(key=lambda r:r['id']); return rows,screen_for(rows)


def config():
    path=Path(m.__file__).with_name('evaluate_native30_v3.py')
    m.binding(path,m.EVALUATOR_SHA)
    return m.family_config_from_source(path.read_text())


class CatalogueTests(unittest.TestCase):
    def test_exact_source_feature_order_and_all255_subsets(self):
        rows,screen=fixture(); old=m.b.make_schedule(rows,screen)
        result=m.derive_schedule(old,old,config())
        self.assertEqual(result['families'],['S','D','R','P','F','H','SC','BC'])
        self.assertEqual(len(result['combinations']),255); self.assertEqual(len(set(result['combinations'])),255)
        self.assertEqual([c for c in result['combinations'] if 'BC' not in c.split('+')],old['combinations'])
        self.assertEqual(len(result['feature_names']),55); self.assertEqual(result['feature_names'][-1],m.BC_FEATURE)
        self.assertEqual(result['feature_names'][:-1],sum(config().values(),[]))
        for combination in old['combinations']:
            names=sum((config()[f] for f in combination.split('+')),[])
            self.assertEqual(result['combination_feature_names'][combination],names)
            self.assertEqual(result['combination_feature_names'][combination+'+BC'],names+[m.BC_FEATURE])
        self.assertEqual(len(result['bc_matched_comparisons']),127); self.assertIsNone(result['BC_standalone_reference'])
        self.assertFalse(any(c['reference_combination']=='' or c['bc_combination']=='BC' for c in result['bc_matched_comparisons']))

    def test_fail_closed_catalogue_expression(self):
        for source in ("__import__('os')","x.member","x[0]","(lambda:1)()","[1]*2","{x for x in [1]}","(x:=1)"):
            with self.subTest(source=source),self.assertRaisesRegex(ValueError,'unsupported'):
                m.catalogue_expression(ast.parse(source,mode='eval').body)
        with self.assertRaisesRegex(ValueError,'unsupported'):
            m.catalogue_expression(ast.Import(names=[ast.alias(name='os')]))
        with self.assertRaises(ValueError): m.catalogue_expression(ast.parse('unknown',mode='eval').body)

    def test_assignment_is_not_silently_skipped_or_reinterpreted(self):
        literal=repr(config())
        for source in (f'FAMILIES={literal}\nFAMILIES={{}}',f'FAMILIES:dict={literal}',f'FAMILIES=alias={literal}',
                       f'def hidden():\n FAMILIES={literal}',f'FAMILIES={literal}\nFAMILIES += {{}}',
                       f'FAMILIES={literal}\nFAMILIES["S"]=["changed"]'):
            with self.subTest(source=source[:40]),self.assertRaises(ValueError): m.family_config_from_source(source)
        # Unrelated module instructions remain inert; this is not a module loader.
        source="raise RuntimeError('MUST NOT EXECUTE')\nFAMILIES="+literal
        self.assertEqual(m.family_config_from_source(source),config())
        reversed_config=dict(reversed(list(config().items())))
        with self.assertRaises(ValueError): m.family_config_from_source('FAMILIES='+repr(reversed_config))


class ProjectionTests(unittest.TestCase):
    def test_complete_old_cells_folds_caps_references_and_globals_preserved(self):
        rows,screen=fixture(); old=m.b.make_schedule(rows,screen); before=copy.deepcopy(old)
        globals_before={k:copy.deepcopy(getattr(m.b,k)) for k in ('COMBINATIONS','FAMILIES','CAPS','SCOPE')}
        result=m.derive_schedule(old,old,config())
        self.assertEqual(old,before)
        for key in ('fold_cells','cap_schedules','omitted_fold_cells','schedule_sha256','input_population','eligible_population','shared_groups'):
            self.assertEqual(result[key],old[key])
        projected=[x for x in result['combination_schedule_cells'] if 'BC' not in x['combination'].split('+')]
        self.assertEqual(projected,old['combination_schedule_cells'])
        for key,value in globals_before.items(): self.assertEqual(getattr(m.b,key),value)
        self.assertTrue(any(c['additional_dependency_purged_from_reference']['rows'] for c in result['fold_cells']))

    def test_omitted_cells_and_protected_transitive_exclusion_are_never_repaired(self):
        rows=[row('human','H','0'),row('ai','A','1')]
        screen=screen_for(rows); old=m.b.make_schedule(rows,screen); result=m.derive_schedule(old,old,config())
        self.assertEqual(result['accounting']['eligible_combination_cap_cells'],0)
        self.assertEqual(len(result['combination_schedule_cells']),15*7*255)
        self.assertEqual(result['omitted_fold_cells'],old['omitted_fold_cells'])
        self.assertTrue(all(s['omission_reasons'] for s in result['cap_schedules']))
        rows,screen=fixture(); selected=row('protected','H1','0','protected_alias','pc1'); rows.append(selected)
        outside=row('outside','H1','0','protected_alias','pc2'); outside['role']='locked'
        screen=screen_for(rows+[outside]); old=m.b.make_schedule(rows,screen); result=m.derive_schedule(old,old,config())
        self.assertEqual(result['protected_excluded_population']['ids'],['protected'])

    def test_primary_all_caps_diagnostics_only_all_and_recomputed_counts(self):
        rows,screen=fixture(); old=m.b.make_schedule(rows,screen); result=m.derive_schedule(old,old,config())
        schedules={s['schedule_uid']:s for s in result['cap_schedules']}
        for e in result['combination_schedule_cells']:
            if e['analysis_role']=='diagnostic': self.assertEqual(schedules[e['schedule_uid']]['quantity'],'all')
        old_count=m.cell_counts(old['combination_schedule_cells'])
        all_count=m.cell_counts(result['combination_schedule_cells'])
        added=result['additional_BC_projection']['accounting']
        for metric in ('intended_cells','eligible_cells','omitted_cells'):
            self.assertEqual(all_count['total'][metric],old_count['total'][metric]+added['total'][metric])
            self.assertEqual(all_count['total'][metric]*127,old_count['total'][metric]*255)
        self.assertEqual(result['additional_BC_projection']['combination_count'],128)

    def test_malformed_old_schedule_cannot_be_relabelled(self):
        rows,screen=fixture(); replay=m.b.make_schedule(rows,screen)
        for key in ('fold_cells','cap_schedules','omitted_fold_cells','combination_schedule_cells'):
            changed=copy.deepcopy(replay)
            if changed[key]: changed[key].pop()
            else: changed[key].append({'invalid':True})
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'replay mismatch'): m.derive_schedule(changed,replay,config())

    def test_no_availability_filter_or_file_access_and_metadata_features_are_inert(self):
        rows,screen=fixture(); original=m.b.make_schedule(rows,screen); family_config=config()
        for r in rows: r.update(input={'path':'/DO_NOT_OPEN/audio.wav'},feature_path='/DO_NOT_OPEN/features.json',BC_value=None)
        with patch.object(Path,'open',side_effect=AssertionError('pure schedule opened a file')):
            replay=m.b.make_schedule(rows,screen); result=m.derive_schedule(original,replay,family_config)
        self.assertEqual(result['input_population']['rows'],len(rows)); self.assertEqual(result['classifier_fits'],0)
        self.assertEqual(result['audio_files_opened'],0); self.assertEqual(result['feature_files_opened'],0)


class AuthorityTests(unittest.TestCase):
    def test_wrong_cohort_pin_fails_before_read(self):
        with patch.object(Path,'open',side_effect=AssertionError('unpinned open')):
            for invalid in (None,'','abc'):
                with self.assertRaisesRegex(ValueError,'mandatory'): m.build('/a',invalid,'/b','/c','/d','/e','/f','/g')

    def test_scope_and_scientific_failure_do_not_authorize_extension(self):
        audit_entry={'path':'/synthetic/audit.json','sha256':m.AUDIT_SHA}; integration={'sha256':m.INTEGRATION_SHA}
        decision={'version':'bc_native30_transfer_decision_v1','status':'external_measurement_gate_accepted_prospective_transfer_implementation_only',
            'scope':m.DECISION_SCOPE,'prospective_cohort_rows':3830,'prospective_feature':m.BC_FEATURE,'integration_draft_sha256':m.INTEGRATION_SHA,
            'independent_audit':audit_entry,'producer_commit_sha256':'producer','correction_authority_sha256':'correction'}
        audit={'version':'audit_bc_guitarset_reserved_admission_v2','passed':True,'all_scientific_requirements_satisfied_after_replay':True,
            'BC_admitted':False,'classifier_fits':0,'thresholds_changed':False,'result_COMMIT':{'sha256':'producer'},
            'correction_authority':{'sha256':'correction'},'measurement_counts':{'all':{'attempted':1440,'expected':1440,'failed_after_attempt':0,'skipped_before_attempt':0,'successful':1440}}}
        m.verify_transfer_metadata(decision,audit,audit_entry,integration)
        for key in ('passed','all_scientific_requirements_satisfied_after_replay'):
            changed=copy.deepcopy(audit); changed[key]=False
            with self.assertRaises(ValueError): m.verify_transfer_metadata(decision,changed,audit_entry,integration)
        changed=copy.deepcopy(decision); changed['scope']['classifier_fits_authorized_by_this_file']=True
        with self.assertRaises(ValueError): m.verify_transfer_metadata(changed,audit,audit_entry,integration)

    def test_synthetic_draft_publication_never_authority_and_rejects_mutation(self):
        rows,screen=fixture(); old=m.b.make_schedule(rows,screen); result=m.derive_schedule(old,old,config())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); meta=root/'metadata.json'; meta.write_bytes(m.canonical({'synthetic':True}))
            entry=m.binding(meta); result['input_bindings']={'original_schedule':entry,'original_commit':entry}
            result['historical_original_schedule_provenance']={'input_bindings':{str(meta):entry},'runtime':{'executable':entry}}
            target=root/'derived'; m.publish_draft(result,target)
            commit=m.read(target/'COMMIT.json'); self.assertFalse(commit['fitting_authorized']); self.assertFalse(commit['measurement_authorized'])
            with self.assertRaises(ValueError): m.publish_draft(result,target)
            meta.write_text('changed')
            with self.assertRaises(ValueError): m.preflight_summary(result)


if __name__=='__main__': unittest.main()
