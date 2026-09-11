"""Metadata-only eight-family schedule derivation; never reads features or fits.

The immutable seven-family schedule is independently replayed through its pinned
metadata planner. Only the catalogue expands. No old globals or receipts change.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import itertools
from pathlib import Path
import platform
import sys

VERSION = 'plan_native30_evaluation_schedule_bc_v1'
PLANNER_SHA = 'ff6965b3f9c834ee9d39bbf0f994595f44fa2e261d9c7c6b60a153cb9f72ce24'
PLANNER_TEST_SHA = '1dab206d40cc624a4dcb3377a94b596b94873ec9c9020375f8cb4e3808ff2924'
EVALUATOR_SHA = 'c2544fe4ce98d88b906f502a576f788da46a4e28117ea7f3ce41802495fe7d1c'
SCHEDULE_SHA = '4e12300a7fe43b4f3024c1cd9e675c25b0ade3af25dded0d688b8d8f2fcab7a6'
SCHEDULE_COMMIT_SHA = '5d0e329d9ef0a794e32acbf5bee06a4a09a58b453e1741ba6232fbfb32347233'
DECISION_SHA = '5f49573923d9ccbc1fcfc6e5b04a32b90b85c3f929e589d2f69d2969d020ca82'
AUDIT_SHA = '637f07ec893ed64a85e4fcd4971da3b177ec4f9efa83315b76e4441cae5ec30e'
INTEGRATION_SHA = '2c8682eca7f93947de6e2fe961e6f1a06afc6085a39806a480d90b0c19600330'
BC_FEATURE = 'BC_b2_500_750_1250hz_center8s_median'
DECISION_SCOPE = {'external_measurement_gate_accepted': True,
    'prospective_native30_implementation_and_synthetic_tests_authorized': True,
    'actual_cohort_measurement_authorized_by_this_file': False,
    'classifier_fits_authorized_by_this_file': False, 'model_scoring_authorized_by_this_file': False,
    'threshold_changes_authorized': False, 'separate_measurement_freeze_required': True, 'separate_fit_freeze_required': True}
BUILD_KEYS = {'input_bindings', 'prepared_contract_sha256', 'metadata_read_scope', 'runtime'}


def load_planner():
    path = Path(__file__).resolve().with_name('plan_native30_evaluation_schedule_v1.py')
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != PLANNER_SHA:
        raise ValueError('immutable seven-family planner SHA')
    spec = importlib.util.spec_from_file_location('_native30_pinned_schedule_only', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


b = load_planner()
require, binding, read, canonical, value_hash = b.require, b.binding, b.read, b.canonical, b.value_hash
FAMILIES = (*b.FAMILIES, 'BC')
COMBINATIONS = ['+'.join(c) for n in range(1, len(FAMILIES)+1) for c in itertools.combinations(FAMILIES,n)]


def catalogue_expression(node):
    """Evaluate a tiny call-free literal/comprehension expression, never a module."""
    allowed = (ast.Dict, ast.List, ast.Tuple, ast.Constant, ast.ListComp, ast.comprehension,
               ast.Name, ast.Load, ast.Store, ast.BinOp, ast.Add, ast.JoinedStr, ast.FormattedValue)
    require(all(type(part) in allowed for part in ast.walk(node)), 'unsupported catalogue AST node')
    require(all(not part.is_async for part in ast.walk(node) if isinstance(part, ast.comprehension)), 'async catalogue forbidden')
    try:
        return eval(compile(ast.Expression(node), '<pinned-family-catalogue-only>', 'eval'), {'__builtins__': {}}, {})
    except Exception as error:
        raise ValueError('invalid literal family catalogue') from error


def family_config_from_source(source):
    """Read only the exact FAMILIES assignment; all other module code stays inert."""
    tree = ast.parse(source)
    assignments = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        and any(isinstance(part, ast.Name) and part.id == 'FAMILIES'
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target]) for part in ast.walk(target))]
    require(len(assignments) == 1 and assignments[0] in tree.body and type(assignments[0]) is ast.Assign and
        len(assignments[0].targets) == 1 and isinstance(assignments[0].targets[0], ast.Name), 'one exact top-level FAMILIES assignment required')
    result = catalogue_expression(assignments[0].value)
    require(type(result) is dict and list(result) == list(b.FAMILIES) and
        all(type(v) is list and all(type(c) is str and c for c in v) for v in result.values()), 'source-defined family order/schema')
    columns = sum(result.values(), [])
    require(len(columns) == len(set(columns)) == 54, 'exact54 distinct original columns')
    require({k:len(v) for k,v in result.items()} == {'S':15,'D':3,'R':3,'P':6,'F':15,'H':6,'SC':6}, 'original family cardinalities')
    return result


def cell_counts(cells):
    require(all(c['status'] in ('eligible_metadata_cell','omitted') for c in cells), 'unknown schedule status')
    result = {role: {'intended_cells':sum(c['analysis_role']==role for c in cells),
        'eligible_cells':sum(c['analysis_role']==role and c['status']=='eligible_metadata_cell' for c in cells),
        'omitted_cells':sum(c['analysis_role']==role and c['status']=='omitted' for c in cells)} for role in ('primary','diagnostic')}
    result['total'] = {key:sum(result[role][key] for role in ('primary','diagnostic'))
                       for key in ('intended_cells','eligible_cells','omitted_cells')}
    return result


def derive_schedule(original, replay, old_family_config):
    """Pure metadata extension. Caller supplies already verified original evidence."""
    pure = {k:v for k,v in original.items() if k not in BUILD_KEYS}
    require(pure == replay and original['version'] == b.VERSION, 'original complete seven-family schedule replay mismatch')
    require(original['families'] == list(b.FAMILIES) and original['combinations'] == b.COMBINATIONS,
            'original family/combination order mismatch')
    require(list(old_family_config) == list(b.FAMILIES) and len(sum(old_family_config.values(),[])) == 54, 'original feature order/schema')
    require([c for c in COMBINATIONS if 'BC' not in c.split('+')] == original['combinations'] and
            len(set(COMBINATIONS)) == 2**len(FAMILIES)-1 == 255, 'exact old127 projection from new255 subsets')
    snapshots = {key:value_hash(getattr(b,key)) for key in ('FAMILIES','COMBINATIONS','CAPS','PRIMARY_MODE','DIAGNOSTIC_MODES')}
    experiments = [{'combination':combination,'schedule_uid':schedule['schedule_uid'],'status':schedule['status'],
        'feature_mode':mode,'analysis_role':'primary' if mode==b.PRIMARY_MODE else 'diagnostic'}
        for schedule in original['cap_schedules'] for combination in COMBINATIONS
        for mode in ((b.PRIMARY_MODE,*b.DIAGNOSTIC_MODES) if schedule['quantity']=='all' else (b.PRIMARY_MODE,))]
    old_cells = [e for e in experiments if 'BC' not in e['combination'].split('+')]
    new_cells = [e for e in experiments if 'BC' in e['combination'].split('+')]
    require(old_cells == original['combination_schedule_cells'], 'original full experiment projection changed')
    accounting = dict(original['accounting']); full = cell_counts(experiments); old = cell_counts(old_cells); added = cell_counts(new_cells)
    require(old['primary'] == accounting['primary'] and old['diagnostic'] == accounting['diagnostic'] and
        old['total']['eligible_cells'] == accounting['eligible_combination_cap_cells'] and
        old['total']['intended_cells'] == accounting['intended_combination_cap_cells'], 'old projection accounting replay')
    accounting.update(intended_combination_cap_cells=full['total']['intended_cells'],
        eligible_combination_cap_cells=full['total']['eligible_cells'], omitted_combination_cap_cells=full['total']['omitted_cells'],
        primary=full['primary'],diagnostic=full['diagnostic'])
    family_config = {**old_family_config,'BC':[BC_FEATURE]}
    feature_names = sum(family_config.values(),[])
    combination_features = {c:sum((family_config[f] for f in c.split('+')),[]) for c in COMBINATIONS}
    require(feature_names[:-1] == sum(old_family_config.values(),[]) and len(set(feature_names)) == 55, 'append one BC column only')
    require(all(combination_features[c] == sum((old_family_config[f] for f in c.split('+')),[]) for c in b.COMBINATIONS), 'old combination feature ordering changed')
    pairs = [{'reference_combination':c,'bc_combination':c+'+BC'} for c in b.COMBINATIONS]
    semantics = {**original['semantics'],
        'shared_candidate_rows':'all255 family subsets share exact original schedule references at every cap, without feature availability filtering',
        'diagnostic_modes':'median_only and missingness_only enumerate all255 subsets at all-cap only; exact primary schedule IDs/rows reused',
        'BC_reference':'127 predeclared X+BC versus nonempty X comparisons; BC alone has no empty-family fitted reference'}
    result = {**pure,'version':VERSION,'families':list(FAMILIES),'combinations':COMBINATIONS,
        'family_config':family_config,'feature_names':feature_names,'combination_feature_names':combination_features,
        'combination_schedule_cells':experiments,'accounting':accounting,'semantics':semantics,
        'original_seven_family_projection':{'combinations':b.COMBINATIONS,'accounting':old,
            'experiment_cells_sha256':value_hash(old_cells),'feature_names':sum(old_family_config.values(),[])},
        'additional_BC_projection':{'combination_count':sum('BC' in c.split('+') for c in COMBINATIONS),'accounting':added},
        'bc_matched_comparisons':pairs,'BC_standalone_reference':None,
        'original_protocol_semantics':original['semantics'],
        'status':'BC_catalogue_metadata_draft_not_measurement_or_fit_authorization',
        'BC_native30_measurement_authorized':False,'BC_predictive_evaluation_authorized':False}
    for key in ('fold_cells','cap_schedules','omitted_fold_cells','schedule_sha256','input_population','eligible_population',
                'protected_excluded_population','shared_groups','seed','global_hash_buckets','quantities'):
        require(result[key] == original[key], 'immutable fold/cap/population projection: '+key)
    require(snapshots == {key:value_hash(getattr(b,key)) for key in snapshots}, 'original planner globals mutated')
    return result


def verify_transfer_metadata(decision, audit, audit_entry, integration_entry):
    require(decision.get('version') == 'bc_native30_transfer_decision_v1' and
        decision.get('status') == 'external_measurement_gate_accepted_prospective_transfer_implementation_only' and
        decision.get('scope') == DECISION_SCOPE and decision.get('prospective_cohort_rows') == 3830 and
        decision.get('prospective_feature') == BC_FEATURE and decision.get('integration_draft_sha256') == integration_entry['sha256'] and
        decision.get('independent_audit') == {'path':audit_entry['path'],'sha256':audit_entry['sha256']}, 'implementation-only transfer decision')
    require(audit.get('version') == 'audit_bc_guitarset_reserved_admission_v2' and audit.get('passed') is True and
        audit.get('all_scientific_requirements_satisfied_after_replay') is True and audit.get('BC_admitted') is False and
        audit.get('classifier_fits') == 0 and audit.get('thresholds_changed') is False and
        audit['result_COMMIT']['sha256'] == decision['producer_commit_sha256'] and
        audit['correction_authority']['sha256'] == decision['correction_authority_sha256'] and
        audit['measurement_counts']['all'] == {'attempted':1440,'expected':1440,'failed_after_attempt':0,'skipped_before_attempt':0,'successful':1440},
        'completed external numerical/scientific gate metadata')


def build(cohort_path, cohort_sha, plan_path, screen_path, schedule_root, decision_path, audit_path, integration_path):
    require(b.hash_string(cohort_sha), 'mandatory cohort metadata SHA')
    here = Path(__file__).resolve().parent; schedule_root = b.path_checked(schedule_root)
    paths = {'cohort':(cohort_path,cohort_sha),'plan':(plan_path,b.PLAN_SHA),'screen':(screen_path,b.SCREEN_SHA),
        'original_schedule':(schedule_root/'schedule_draft.json',SCHEDULE_SHA),'original_commit':(schedule_root/'COMMIT.json',SCHEDULE_COMMIT_SHA),
        'transfer_decision':(decision_path,DECISION_SHA),'external_audit':(audit_path,AUDIT_SHA),'integration_draft':(integration_path,INTEGRATION_SHA)}
    bindings = {name:binding(path,expected) for name,(path,expected) in paths.items()}
    bindings.update({'planner':binding(here/'plan_native30_evaluation_schedule_v1.py',PLANNER_SHA),
        'planner_tests':binding(here/'test_plan_native30_evaluation_schedule_v1.py',PLANNER_TEST_SHA),
        'family_catalogue_code':binding(here/'evaluate_native30_v3.py',EVALUATOR_SHA),
        'code':binding(Path(__file__).resolve()),'tests':binding(here/('test_'+VERSION+'.py')),
        'runtime_executable':binding(Path(sys.executable).resolve())})
    original, commit = read(bindings['original_schedule']['path']), read(bindings['original_commit']['path'])
    require(commit == {'status':'committed_metadata_draft_not_evaluation_authorization','version':b.VERSION,
        'products':{'schedule_draft.json':bindings['original_schedule']},'draft_sha256':SCHEDULE_SHA,'classifier_fits':0,'fitting_authorized':False}, 'immutable original schedule COMMIT')
    require(set(original) >= BUILD_KEYS and original['prepared_contract_sha256'] == cohort_sha, 'original schedule/cohort join')
    historical = original['input_bindings']
    for name in ('cohort','plan','screen'):
        require(historical.get(bindings[name]['path']) == bindings[name], 'supplied metadata differs from original binding')
    for path,entry in historical.items():
        require(path == entry['path'] and binding(path) == entry, 'historical metadata/code binding mutation')
    require(binding(original['runtime']['executable']['path']) == original['runtime']['executable'], 'historical interpreter binding')
    decision, audit = read(bindings['transfer_decision']['path']),read(bindings['external_audit']['path'])
    verify_transfer_metadata(decision,audit,bindings['external_audit'],bindings['integration_draft'])
    contract,plan,screen = (read(bindings[key]['path']) for key in ('cohort','plan','screen'))
    for key,expected in (('plan',b.PLAN_SHA),('screen',b.SCREEN_SHA)):
        require(contract['bindings'].get(bindings[key]['path'],{}).get('sha256') == expected, 'cohort metadata source graph')
    rows = b.validate_prepared_contract(contract,plan,screen)
    replay = b.make_schedule(rows,screen)
    config = family_config_from_source(Path(bindings['family_catalogue_code']['path']).read_text())
    result = derive_schedule(original,replay,config)
    require(result['eligible_population']['rows'] == 3830 and result['protected_excluded_population']['rows'] == 0, 'exact original3830 population')
    require(result['accounting']['eligible_combination_cap_cells'] == 103530 and
        result['original_seven_family_projection']['accounting']['total']['eligible_cells'] == 51562 and
        result['additional_BC_projection']['accounting']['total']['eligible_cells'] == 51968, 'recomputed actual cell counts differ from prospective protocol')
    result.update(prepared_contract_sha256=cohort_sha,input_bindings=bindings,
        historical_original_schedule_provenance={key:original[key] for key in BUILD_KEYS},
        metadata_read_scope='Pinned JSON metadata, historical listed code/interpreter bytes, and restricted static FAMILIES expression only; no recursive audio/feature/product traversal',
        runtime={'python':platform.python_version(),'executable':bindings['runtime_executable'],
                 'implementation':'Python standard library only; no numerical or classifier modules imported'})
    verify_bindings(result)
    return result


def verify_bindings(result):
    for entry in result['input_bindings'].values(): require(binding(entry['path']) == entry,'metadata/code changed during derivation')
    historical = result['historical_original_schedule_provenance']
    for entry in historical['input_bindings'].values(): require(binding(entry['path']) == entry,'historical binding changed during derivation')
    require(binding(historical['runtime']['executable']['path']) == historical['runtime']['executable'],'historical runtime changed')


def preflight_summary(result):
    verify_bindings(result)
    return {'version':VERSION,'status':'metadata_only_preflight_passed_no_schedule_published',
        'families':result['families'],'combination_count':len(result['combinations']),'feature_count':len(result['feature_names']),
        'accounting':result['accounting'],'old_projection':result['original_seven_family_projection']['accounting'],
        'additional_BC':result['additional_BC_projection'],'schedule_sha256':result['schedule_sha256'],
        'prospective_payload_sha256':value_hash(result),'input_bindings_sha256':value_hash(result['input_bindings']),
        'audio_files_opened':0,'feature_files_opened':0,'classifier_fits':0,'predictions_computed':0,
        'schedule_published':False,'measurement_authorized':False,'fitting_authorized':False}


def publish_draft(result, output):
    """Available for a separately reviewed future draft publication; never authority."""
    output=b.path_checked(output); require(not output.exists() and output.parent.is_dir(),'new draft root only')
    verify_bindings(result); output.mkdir()
    path=output/'schedule_draft.json'
    with path.open('xb') as stream: stream.write(canonical(result))
    require(read(path)==result,'derived metadata JSON roundtrip'); verify_bindings(result)
    product=binding(path)
    commit={'version':VERSION,'status':'committed_BC_metadata_schedule_draft_not_measurement_or_evaluation_authorization',
        'products':{'schedule_draft.json':product},'draft_sha256':product['sha256'],
        'original_schedule':result['input_bindings']['original_schedule'],'original_commit':result['input_bindings']['original_commit'],
        'classifier_fits':0,'fitting_authorized':False,'measurement_authorized':False}
    with (output/'COMMIT.json').open('xb') as stream: stream.write(canonical(commit))
    return {'output':str(output),'draft_sha256':product['sha256'],'commit_sha256':b.sha(output/'COMMIT.json'),'classifier_fits':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cohort-contract','cohort-contract-sha256','plan','screen','original-schedule-root','transfer-decision','external-audit','integration-draft'):
        p.add_argument('--'+name,required=True)
    mode=p.add_mutually_exclusive_group(required=True); mode.add_argument('--preflight',action='store_true'); mode.add_argument('--output')
    x=p.parse_args(); result=build(x.cohort_contract,x.cohort_contract_sha256,x.plan,x.screen,x.original_schedule_root,x.transfer_decision,x.external_audit,x.integration_draft)
    print(canonical(preflight_summary(result) if x.preflight else publish_draft(result,x.output)).decode().strip())


if __name__=='__main__': main()
