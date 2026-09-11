"""Append the fixed BC scalar to a deeply verified, caller-pinned v3 package.

Reads committed JSON/NPZ evidence, never cohort audio, and never measures or fits.
The production interface has no reduced-cohort, pin-bypass, or latest-file mode.
Every verification replays original receipts, the BC gate/freeze/measurement
products, and the eight-family schedule. A new output root is always required.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib
from pathlib import Path

VERSION = 'prepare_native30_evaluation_inputs_bc_v1'
HERE = Path(__file__).resolve().parent
PINS = {
    'prepare_native30_evaluation_inputs_v3.py': 'db69144303079f379c2f3865250f0d4788fcdd03e8e5176306abbf9701b03031',
    'test_prepare_native30_evaluation_inputs_v3.py': '1cd9388dee09809db024748f6a21de00ac3bf67278f489232e54b59f67dcb826',
    'run_native30_bc_v2.py': '822f51f3e961bb4edf16389283a91b898393084f34d1a87da0956d7a92cffe0d',
    'test_run_native30_bc_v2.py': '4d24a5e8689eb4d7599669e560ab504eb29a34b8409e9471b6a304c48ff5c7dd',
    'plan_native30_evaluation_schedule_bc_v1.py': 'c432815b11e6dc315191f68f9722764b06be8e477b2d5db75308067ffe6764e5',
    'test_plan_native30_evaluation_schedule_bc_v1.py': 'be470106aabb8fd4864744b636d4057a9b1822bccf0a4261396f4c8be1bccf5d',
}


def module(name):
    path = HERE / (name + '.py')
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != PINS[path.name]:
        raise ValueError('immutable adapter dependency SHA: ' + name)
    loaded = importlib.import_module(name)
    if Path(loaded.__file__).resolve() != path:
        raise ValueError('dependency import origin: ' + name)
    return loaded


b = module('prepare_native30_evaluation_inputs_v3')
bc = module('run_native30_bc_v2')
s = module('plan_native30_evaluation_schedule_bc_v1')
require, binding, read, canonical, value_hash = b.require, b.binding, b.read_json, b.canonical, b.value_hash
STATUS = b.STATUS
SCOPE = {**b.SCOPE, 'BC_classifier_admitted': False, 'BC_diagnostics_are_predictors': False}
FAMILY_CONFIG = {**b.FAMILY_CONFIG, 'BC': [bc.FEATURE]}
FEATURE_NAMES = [*b.FEATURE_NAMES, bc.FEATURE]
LINEAGE_SCHEMA = [*b.LINEAGE_SCHEMA, 'parent_lineage_sha256', 'bc_receipt',
                  'bc_receipt_payload_sha256', 'bc_metadata', 'bc_arrays', 'bc_analysis_view_sha256']
REQUEST_NAMES = {'parent_commit': 'COMMIT.json', 'bc_freeze': None, 'bc_commit': 'COMMIT.json',
                 'schedule_draft': 'schedule_draft.json', 'schedule_commit': 'COMMIT.json'}
PRODUCT_NAMES = ('contract.json', 'metadata.json', 'features.json', 'lineage.json', 'bc_coverage.json')


def code_bindings():
    result = {name: binding(HERE / name, sha) for name, sha in PINS.items()}
    result.update(assembler=binding(Path(__file__).resolve()),
                  tests=binding(HERE / ('test_' + VERSION + '.py')))
    return result


def validate_request(request):
    require(type(request) is dict and set(request) == set(REQUEST_NAMES), 'exact five caller-pinned authorities required')
    for key, basename in REQUEST_NAMES.items():
        entry = request[key]
        require(type(entry) is dict and set(entry) == {'path', 'bytes', 'sha256'}
                and b.hash_string(entry['sha256']), 'caller binding schema/SHA: ' + key)
        require(basename is None or Path(entry['path']).name == basename, 'authority filename: ' + key)
        require(binding(entry['path'], entry['sha256']) == entry, 'caller binding changed: ' + key)
        require(value_hash(read(entry['path'])) == entry['sha256'], 'canonical authority: ' + key)
    require(Path(request['schedule_draft']['path']).parent == Path(request['schedule_commit']['path']).parent,
            'schedule authority roots differ')


def verify_bc(request):
    """Use unchanged producer validation, explicitly disabling waveform reads."""
    run = bc.validate_freeze(request['bc_freeze']['path'], request['bc_freeze']['sha256'])
    require(Path(request['bc_commit']['path']).parent == Path(run['prepared']['output_root']), 'BC output/freeze join')
    bc.check_inputs(run, audio=False)
    result = bc.verify_output(run, audio=False)
    require(result['commit'] == request['bc_commit'], 'BC caller COMMIT mismatch')
    commit = read(result['commit']['path'])
    require(commit == {'version': bc.VERSION, 'status': 'committed_native30_BC_measurements_not_admitted',
        'contract_sha256': value_hash(run), 'completed': 3830, 'products': commit.get('products'),
        'all_bound_inputs_and_products_end_rehashed': True, **bc.SCOPE}, 'exact BC terminal COMMIT scope/count')
    require(run['prepared']['expected_count'] == 3830 and run['prepared']['source_counts'] == bc.physical.SOURCES,
            'exact3830 BC prepared cohort')
    bc.check_inputs(run, audio=False)
    require(binding(request['bc_commit']['path']) == request['bc_commit'], 'BC COMMIT changed during verification')
    return run, result


def verify_schedule(request, parent):
    root = Path(request['schedule_draft']['path']).parent
    require({p.name for p in root.iterdir()} == {'schedule_draft.json', 'COMMIT.json'},
            'derived schedule exact root inventory')
    draft, commit = read(request['schedule_draft']['path']), read(request['schedule_commit']['path'])
    require(commit == {'version': s.VERSION,
        'status': 'committed_BC_metadata_schedule_draft_not_measurement_or_evaluation_authorization',
        'products': {'schedule_draft.json': request['schedule_draft']},
        'draft_sha256': request['schedule_draft']['sha256'],
        'original_schedule': parent['contract']['bindings']['schedule_draft'],
        'original_commit': parent['contract']['bindings']['schedule_commit'],
        'classifier_fits': 0, 'fitting_authorized': False, 'measurement_authorized': False}, 'derived schedule COMMIT')
    sources = parent['contract']['bindings']; inputs = draft['input_bindings']
    replay = s.build(sources['cohort_contract']['path'], sources['cohort_contract']['sha256'],
        sources['origin_plan']['path'], sources['screen']['path'],
        Path(sources['schedule_draft']['path']).parent, inputs['transfer_decision']['path'],
        inputs['external_audit']['path'], inputs['integration_draft']['path'])
    # A schedule may have been published by a different pinned interpreter. Its
    # recorded executable is rehashed, while pure metadata must replay exactly.
    s.verify_bindings(draft)
    require(set(inputs) == set(replay['input_bindings']), 'derived schedule binding inventory')
    require({k: v for k, v in inputs.items() if k != 'runtime_executable'} ==
            {k: v for k, v in replay['input_bindings'].items() if k != 'runtime_executable'}, 'derived schedule input graph')
    require(draft['runtime']['executable'] == inputs['runtime_executable'] and
            set(draft['runtime']) == set(replay['runtime']) and
            draft['runtime']['implementation'] == replay['runtime']['implementation'], 'derived schedule runtime evidence')
    require(canonical({k: v for k, v in draft.items() if k not in ('runtime', 'input_bindings')}) ==
            canonical({k: v for k, v in replay.items() if k not in ('runtime', 'input_bindings')}),
            'derived schedule full replay / original127 projection')
    require(draft['family_config'] == FAMILY_CONFIG and draft['feature_names'] == FEATURE_NAMES,
            'exact55 feature catalogue')
    return draft


def join_rows(parent, run, schedule):
    """Pure append/join boundary; callers must first deep-verify source packages."""
    metadata, features, lineage = (parent[k] for k in ('metadata', 'features', 'lineage'))
    rows = run['prepared']['rows']; ids = [r['id'] for r in metadata]
    require(ids == sorted(set(ids)) == [r['id'] for r in rows] == [r['id'] for r in features]
            == [r['id'] for r in lineage] == schedule['input_population']['ids']
            == schedule['eligible_population']['ids'], 'exact ordered ID roster across parent/BC/schedule')
    require(schedule['protected_excluded_population']['rows'] == 0, 'protected exclusions changed')
    output = Path(run['prepared']['output_root']); out_features, out_lineage = [], []
    for position, (meta, old, origin, row) in enumerate(zip(metadata, features, lineage, rows)):
        ident = row['id']; payload, entry, payload_sha = b.receipt(output, ident)
        require(all(meta[k] == row[k] for k in b.IDENTITY) and meta['duration_view_s'] == 30
                and origin['cohort_row_sha256'] == value_hash(row) and payload['row'] == row
                and payload['row_sha256'] == value_hash(row) and payload['id'] == ident
                and payload['contract_sha256'] == value_hash(run), 'BC row/hash/identity join: ' + ident)
        require(origin['source_input'] == row['input'] and meta['input_sha256'] == row['input']['sha256']
                and origin['waveform_float32_sha256'] == meta['waveform_float32_sha256'] == row['waveform_float32_sha256']
                and all(origin[k] == row[k] for k in ('origin_plan_row_sha256', 'screen_row_sha256', 'component_sha256'))
                and origin['schedule_input_position'] == position
                and payload['native_origin'] == run['prepared']['native_origins'][ident]
                and meta['native_sample_rate_hz'] == payload['native_origin']['sample_rate_hz'], 'BC input/PCM/origin lineage: ' + ident)
        require(payload.get('version') == bc.VERSION and payload.get('status') == 'measured_BC_not_classifier_admitted'
                and all(payload.get(k) == v for k, v in bc.SCOPE.items())
                and set(payload['features']) == {bc.FEATURE}, 'BC receipt scope/exact scalar schema')
        value = b.scalar(payload['features'][bc.FEATURE], bc.FEATURE)
        require(value is None or 0 <= value <= 1, 'BC squared bicoherence range')
        require(value == payload['scalar']['median_squared_bicoherence'] and payload['scalar']['pool_count'] == 2
                and payload['scalar']['discarded_tail_samples'] == 0, 'BC scalar support/reduction join')
        require(set(old) == {'id', *b.FEATURE_NAMES}, 'exact old54 schema')
        appended = {**copy.deepcopy(old), bc.FEATURE: value}
        require(canonical({k: v for k, v in appended.items() if k != bc.FEATURE}) == canonical(old),
                'old54 exact JSON numeric preservation')
        out_features.append(appended)
        out_lineage.append({**copy.deepcopy(origin), 'parent_lineage_sha256': value_hash(origin),
            'bc_receipt': entry, 'bc_receipt_payload_sha256': payload_sha,
            'bc_metadata': payload['metadata'], 'bc_arrays': payload['arrays'],
            'bc_analysis_view_sha256': value_hash(payload['analysis_view'])})
    return copy.deepcopy(metadata), out_features, out_lineage


def build(request):
    validate_request(request); codes = code_bindings()
    parent = b.verify_package(Path(request['parent_commit']['path']).parent, request['parent_commit']['sha256'])
    require(parent['contract']['expected_count'] == len(parent['metadata']) == 3830
            and parent['contract']['bindings']['assembler']['sha256'] == PINS['prepare_native30_evaluation_inputs_v3.py']
            and parent['contract']['bindings']['tests']['sha256'] == PINS['test_prepare_native30_evaluation_inputs_v3.py'],
            'actual pinned v3 exact3830 package required')
    run, proof = verify_bc(request)
    require(run['prepared']['request']['cohort_contract'] == parent['contract']['cohort_contract']
            and run['prepared']['request']['plan'] == parent['contract']['bindings']['origin_plan']
            and run['prepared']['request']['screen'] == parent['contract']['bindings']['screen'], 'BC/parent authority graph join')
    schedule = verify_schedule(request, parent)
    require(schedule['input_bindings']['transfer_decision'] == run['prepared']['request']['decision']
            and schedule['input_bindings']['external_audit'] == run['prepared']['request']['audit'], 'schedule/BC accepted gate join')
    metadata, features, lineage = join_rows(parent, run, schedule)
    require(dict(Counter(r['source_group'] for r in metadata)) == bc.physical.SOURCES
            and Counter(r['label'] for r in metadata) == {'0': 1664, '1': 2166}, 'original source/label counts')
    coverage = {'version': VERSION, 'source_coverage': proof['source_coverage'],
                'status': 'descriptive_BC_availability_not_predictors_not_admission', **SCOPE}
    contract = {'version': VERSION, 'status': STATUS, 'expected_count': 3830,
        'family_config': FAMILY_CONFIG, 'feature_names': FEATURE_NAMES, 'feature_count': 55,
        'metadata_schema': b.METADATA_SCHEMA, 'feature_schema': ['id', *FEATURE_NAMES],
        'lineage_schema': LINEAGE_SCHEMA, 'source_counts': parent['contract']['source_counts'],
        'scientific_missingness': parent['contract']['scientific_missingness'],
        'request': request, 'request_sha256': value_hash(request), 'code_bindings': codes,
        'parent_products': parent['products'], 'bc_proof': proof,
        'original54_features_sha256': parent['products']['features.json']['sha256'],
        'schedule_accounting': schedule['accounting'], 'schedule_sha256': schedule['schedule_sha256'],
        'cohort_contract': parent['contract']['cohort_contract'],
        'analysis_duration_s': 8, 'duration_view_s': 30, **SCOPE}
    validate_request(request); require(code_bindings() == codes, 'adapter code changed during assembly')
    return dict(zip(PRODUCT_NAMES, (contract, metadata, features, lineage, coverage)))


def publish(request, output):
    output = b.safe_path(output); validate_request(request)
    require(not output.exists() and output.parent.is_dir(), 'new package output directory required')
    # Reject any ancestor/descendant relationship to declared authorities.
    roots = [Path(request[k]['path']).parent for k in ('parent_commit', 'bc_commit', 'schedule_commit')]
    require(not any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots)
            and not any(Path(e['path']).is_relative_to(output) for e in request.values()), 'output overlaps immutable authority')
    values = build(request)
    parent_contract = read(values['contract.json']['parent_products']['contract.json']['path'])
    upstream = parent_contract['bindings']
    roots += [Path(upstream[key]['path']).parent for key in ('fhsc_contract', 'sdrp_contract', 'schedule_draft')]
    cohort = read(parent_contract['cohort_contract']['path'])
    roots += [Path(root) for root in cohort.get('upstream_inventories', {})]
    roots += [HERE]
    require(not any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots),
            'output overlaps transitive immutable source root')
    output.mkdir()
    for name, value in values.items(): b.write_new(output / name, value)
    require(canonical(build(request)) == canonical(values), 'source evidence changed during assembly')
    products = {name: binding(output / name) for name in PRODUCT_NAMES}
    commit = {'version': VERSION, 'status': STATUS, 'expected_count': 3830,
              'products': products, 'request_sha256': value_hash(request),
              'all_source_evidence_deep_verified_before_and_after': True,
              'audio_files_opened': 0, **SCOPE}
    b.write_new(output / 'COMMIT.json', commit)
    return {'commit': binding(output / 'COMMIT.json'), 'rows': 3830, 'feature_count': 55, 'status': STATUS, **SCOPE}


def verify_package(root, commit_sha):
    root = b.safe_path(root)
    require(b.hash_string(commit_sha), 'caller package COMMIT SHA required')
    require({p.name for p in root.iterdir()} == {*PRODUCT_NAMES, 'COMMIT.json'}, 'BC package exact root inventory')
    entry = binding(root / 'COMMIT.json', commit_sha); commit = read(entry['path'])
    require(value_hash(commit) == commit_sha, 'canonical BC package COMMIT')
    products = {name: binding(root / name) for name in PRODUCT_NAMES}
    contract = read(root / 'contract.json'); request = contract['request']
    require(commit == {'version': VERSION, 'status': STATUS, 'expected_count': 3830,
        'products': products, 'request_sha256': value_hash(request),
        'all_source_evidence_deep_verified_before_and_after': True, 'audio_files_opened': 0, **SCOPE},
        'BC package COMMIT/products/scope')
    expected = build(request)
    for name, value in expected.items():
        require((root / name).read_bytes() == canonical(value), 'BC package product does not replay: ' + name)
    require({name: binding(root / name) for name in PRODUCT_NAMES} == products
            and binding(root / 'COMMIT.json') == entry, 'BC package changed during verification')
    return {'contract': expected['contract.json'], 'metadata': expected['metadata.json'],
            'features': expected['features.json'], 'lineage': expected['lineage.json'],
            'coverage': expected['bc_coverage.json'], 'commit': entry, 'products': products}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'assemble', 'verify'), default='preflight')
    parser.add_argument('--request'); parser.add_argument('--request-sha256')
    parser.add_argument('--output'); parser.add_argument('--package'); parser.add_argument('--commit-sha256')
    args = parser.parse_args()
    if args.mode == 'verify':
        require(args.package and args.commit_sha256, 'package and caller COMMIT SHA required')
        proof = verify_package(args.package, args.commit_sha256)
        result = {'status': STATUS, 'commit': proof['commit'], 'rows': len(proof['metadata']), **SCOPE}
    else:
        require(args.request and b.hash_string(args.request_sha256), 'request and caller request SHA required')
        entry = binding(args.request, args.request_sha256); request = read(entry['path'])
        require(value_hash(request) == args.request_sha256, 'canonical request required')
        if args.mode == 'assemble':
            require(args.output, 'new output required'); result = publish(request, args.output)
        else:
            values = build(request)
            result = {'status': 'deep_evidence_preflight_no_publication', 'rows': 3830,
                      'feature_count': 55, 'prospective_payload_sha256': value_hash(values),
                      'audio_files_opened': 0, **SCOPE}
    print(canonical(result).decode().strip())


if __name__ == '__main__': main()
