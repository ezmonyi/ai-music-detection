"""Frozen CPU-only native30 S/D/R/P adapter; no inference/recovery/fitting.

Calls the unchanged September5 process(duration=30) only after exact physical
and successful inference receipt checks. Legacy scientific NaNs become JSON
null; processing failures never become scientific missingness or a COMMIT.
Default metadata preflight never opens cohort audio or neural products.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import uuid

VERSION = 'extract_native30_sdrp_v1'
FREEZE_VERSION = 'native30-sdrp-extraction-parent-freeze-v1'
FREEZE_STATUS = 'parent_frozen_for_native30_S_D_R_P_extraction'
INFERENCE_V2_SHA = '1aaf8f2498986fbffe23f818ae6db2d37f59303eb239362447fbf8d8f88b6274'
V1_SHA = '84dce45426596888b3a4e3b20e4ad1c61dc976645506f269c7ec7990b12f3933'
BASE_SHA = '165f28d4e553ee679772aa1ac47b31f9a485b94d3a8ac409f25595577abe395a'
EXTRACTOR_SHA = '98fc6caa8b54ed1370269db13fe8d49559d0b877b4b0d5a3a069c8409906c5fe'
CORE_SHA = '8b9745085c7518e84613b2ba499dbae775a57e4dcf95670c5e86a05ab524ff00'
BIAS_SHA = 'bcacfecac5ce69927dfed2a51ec21cc346f613a7874c519e419d22d35a97de5e'
PLAN_SHA = '94982cda45a4bae59371ec06895894227a39fdb84d8043045f009f6282e9e964'
SCOPE = {'classifier_fits': 0, 'cohort_admitted': False, 'neural_inference_performed': False,
         'recovery_applied': False, 'M_predictor': False}
PREFIXES = ('s16__', 's8__', 'd__', 'r__', 'p__')
EVIDENCE_KEYS = ('stage_receipts', 'logs', 'products', 'inputs', 'stage_receipts_sha256',
                 'logs_sha256', 'products_sha256', 'inputs_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_string(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def load_module(path, expected, name):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink(), 'noncanonical code path')
    require(hash_string(expected) and path.is_file() and not path.is_symlink()
            and hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'pinned code mismatch: ' + str(path))
    if name in sys.modules:
        module = sys.modules[name]
        require(Path(module.__file__).resolve() == path, 'unexpected module import path: ' + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


HERE = Path(__file__).resolve().parent
base = load_module(HERE / 'materialize_native30_new1695_v1.py', BASE_SHA, 'materialize_native30_new1695_v1')
physical = load_module(HERE / 'run_native30_inference_batches_v1.py', V1_SHA, 'run_native30_inference_batches_v1')


def load_extractor(root):
    require(sys.flags.optimize == 0, 'optimized Python not allowed with frozen legacy assertions')
    root = physical.safe_path(root)
    core = load_module(root / 'expanded_feature_definitions.py', CORE_SHA, 'expanded_feature_definitions')
    extractor = load_module(root / 'extract_expanded_four_family.py', EXTRACTOR_SHA, '_native30_sdrp_frozen_extractor')
    require(extractor.read_audio is core.read_audio and extractor.spectral_families is core.spectral_families,
            'extractor/core module linkage changed')
    return extractor


def runtime_snapshot():
    import numpy as np
    import scipy
    import soundfile as sf
    runtime = base.runtime_binding(SimpleNamespace(np=np, scipy=scipy, sf=sf))
    require((runtime['python'], runtime['numpy'], runtime['scipy'], runtime['soundfile']) ==
            ('3.11.15', '1.26.4', '1.17.1', '0.14.0'), 'frozen CPU numerical runtime versions')
    extra = {}
    for module in (np, scipy):
        root = Path(module.__file__).resolve().parent
        paths = list(root.rglob('*'))
        sibling = root.with_name(root.name + '.libs')
        if sibling.is_dir():
            paths += list(sibling.rglob('*'))
        for path in paths:
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'}:
                extra[str(path)] = base.file_binding(path)
    runtime.update(cpu_package_files=extra,
                   threads={key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
    return runtime


def mapped_rows(cohort_rows, plan, inference_freeze):
    planned = {r['id']: r for r in plan['rows']}
    require(len(planned) == len(plan['rows']), 'duplicate plan IDs')
    shards = {ident: s['index'] for s in inference_freeze['shards'] for ident in s['item_ids']}
    require(len(shards) == sum(len(s['item_ids']) for s in inference_freeze['shards']) == len(cohort_rows)
            and set(shards) == {r['id'] for r in cohort_rows}, 'inference shard/cohort identity mismatch')
    rows = []
    for row in cohort_rows:
        origin = planned[row['id']]
        require(base.value_hash(origin) == row['origin_plan_row_sha256']
                and all(row[k] == origin[k] for k in physical.IDENTITY), 'native plan row/identity SHA join')
        rate = origin['source_origin']['sample_rate_hz']
        require(type(rate) is int and rate > 0 and origin['source_origin']['channels'] == 2, 'native sample-rate/stereo evidence')
        mapped = {'item_id': row['id'], 'source_id': row['source_group'], 'label': row['label'],
                  'group_id': row['group_id'], 'standardized_path': row['input']['path'],
                  'native_sample_rate_hz': str(rate), 'duration': '30', 'audio_offset_s': '0'}
        rows.append({'cohort_row': row, 'extractor_row': mapped, 'native_sample_rate_hz': rate,
                     'beat_receipt_path': str(Path(inference_freeze['output_root']) / 'receipts' / f'beats_{shards[row["id"]]:03d}.json')})
    require(len(rows) == len({r['cohort_row']['id'] for r in rows}), 'duplicate extraction IDs')
    return sorted(rows, key=lambda r: r['cohort_row']['id'])


def prepare(freeze_path, freeze_sha):
    freeze_path = physical.safe_path(freeze_path)
    require(hash_string(freeze_sha) and base.digest(freeze_path) == freeze_sha, 'caller extraction freeze SHA mismatch')
    freeze = base.read_json(freeze_path)
    require(freeze.get('version') == FREEZE_VERSION and freeze.get('status') == FREEZE_STATUS
            and freeze.get('duration_s') == 30 and freeze.get('cpu_only') is True
            and freeze.get('feature_extraction_authorized') is True
            and all(freeze.get(k) == v for k, v in SCOPE.items()), 'separate extraction freeze scope/version')
    require(hash_string(INFERENCE_V2_SHA), 'final inference v2 pin not installed; draft cannot authorize extraction')
    keys = ('runner', 'tests', 'inference_runner', 'inference_parent_freeze', 'inference_completion',
            'cohort_contract', 'origin_plan', 'extractor', 'core', 'bias')
    bindings = {key: freeze[key] for key in keys}
    for entry in bindings.values():
        require(base.file_binding(physical.safe_path(entry['path'])) == entry, 'extraction frozen binding changed')
    expected = {'runner': (HERE / (VERSION + '.py'), None), 'tests': (HERE / ('test_' + VERSION + '.py'), None),
                'inference_runner': (HERE / 'run_native30_inference_batches_v2.py', INFERENCE_V2_SHA),
                'extractor': (Path(freeze['extractor']['path']), EXTRACTOR_SHA),
                'core': (Path(freeze['extractor']['path']).with_name('expanded_feature_definitions.py'), CORE_SHA),
                'bias': (Path(freeze['bias']['path']), BIAS_SHA), 'origin_plan': (Path(freeze['origin_plan']['path']), PLAN_SHA)}
    for key, (path, digest) in expected.items():
        require(bindings[key]['path'] == str(path) and (digest is None or bindings[key]['sha256'] == digest), 'frozen code/bias/plan pin: ' + key)
    inference = load_module(HERE / 'run_native30_inference_batches_v2.py', INFERENCE_V2_SHA, 'run_native30_inference_batches_v2')
    infer_freeze, cohort, _ = inference.validate_freeze(freeze['inference_parent_freeze']['path'], freeze['inference_parent_freeze']['sha256'])
    require(infer_freeze['cohort_contract'] == freeze['cohort_contract'] and infer_freeze['plan'] == freeze['origin_plan'], 'inference/cohort/plan freeze join')
    completion_path = Path(infer_freeze['output_root']) / 'completion.json'
    require(freeze['inference_completion']['path'] == str(completion_path), 'inference completion root')
    completion = base.read_json(completion_path)
    require(completion.get('status') == 'passed_native30_inference_not_feature_extraction'
            and completion.get('rows') == physical.EXPECTED['total'], 'complete inference required before extraction')
    rows = mapped_rows(cohort['rows'], base.read_json(freeze['origin_plan']['path']), infer_freeze)
    physical.validate_rows(cohort['rows'], physical.EXPECTED)
    require(freeze['bias'] == infer_freeze['bias'] and Path(freeze['extractor']['path']).parent == Path(infer_freeze['old_code_root']), 'unchanged inference/extraction calibration and code')
    extractor = load_extractor(Path(freeze['extractor']['path']).parent)
    runtime = runtime_snapshot()
    require(runtime == freeze['cpu_runtime'], 'parent-pinned CPU extraction runtime mismatch')
    output = physical.safe_path(freeze['output_root'])
    require(output.parent.is_dir() and not output.is_relative_to(Path(infer_freeze['output_root']))
            and not Path(infer_freeze['output_root']).is_relative_to(output)
            and not any(Path(v['path']).is_relative_to(output) for v in bindings.values()), 'unsafe extraction output overlap')
    bindings['extraction_freeze'] = base.file_binding(freeze_path)
    contract = {'version': VERSION, 'status': 'frozen_CPU_extraction_only', 'duration_s': 30,
                'expected_count': physical.EXPECTED['total'], 'rows': rows, 'output_root': str(output),
                'inference_root': infer_freeze['output_root'], 'bindings': bindings, 'cpu_runtime': runtime,
                'source_counts': physical.EXPECTED['sources'], 'primary_families': {'S': 's8__', 'D': 'd__', 'R': 'r__', 'P': 'p__'},
                'additional_spectral_audit': 's16__', 'numerics': 'unchanged_pinned_legacy_process_duration30',
                'missingness_encoding': 'JSON_null; legacy_feature_payload_hash_replays_null_as_NaN',
                'native_rate_policy': 'SHA-joined_origin_plan_source_rate_not_standardized_rate',
                'padding_policy': 'exact_physical_verification_before_legacy_loaders_and_all_input_hashes_after',
                'empty_beat_policy': 'successful_receipted_emitted_empty_file_only_no_recovery', **SCOPE}
    check_contract(contract)
    return contract, extractor


def check_contract(contract):
    base.recheck(contract['bindings'])
    runtime = contract.get('cpu_runtime')
    if runtime:
        require(runtime_snapshot() == runtime, 'CPU runtime changed')


def verify_completed_inference(contract):
    runner = load_module(HERE / 'run_native30_inference_batches_v2.py', INFERENCE_V2_SHA, 'run_native30_inference_batches_v2')
    freeze = contract['bindings']['inference_parent_freeze']
    proof = runner.verify_completion(freeze['path'], freeze['sha256'])
    require(proof['status'] == 'verified_complete_native30_inference' and proof['rows'] == contract['expected_count']
            and proof['completion'] == contract['bindings']['inference_completion'], 'inference completion binding mismatch')
    completion = base.read_json(proof['completion']['path'])
    evidence = {key: completion[key] for key in EVIDENCE_KEYS}
    require(base.value_hash(evidence) == proof['evidence_sha256'] and base.file_binding(proof['completion']['path']) == proof['completion'],
            'inference completion evidence changed after audit')
    return {'verification': proof, 'evidence': evidence}


def inspect_inputs(contract, mapped, audited):
    row, ident = mapped['cohort_row'], mapped['cohort_row']['id']
    root = Path(contract['inference_root'])
    evidence = audited['evidence']
    source = physical.inspect_audio(row['input']['path'], row, input_audio=True)
    require(evidence['inputs'].get(ident) == source, 'unverified extraction input')
    products = {**physical.verify_products('allinone', [row], root), **physical.verify_products('beats', [row], root)}
    require(all(evidence['products'].get(path) == record for path, record in products.items()), 'unverified/changed neural products')
    receipt_path = physical.safe_path(mapped['beat_receipt_path'])
    receipt_binding = base.file_binding(receipt_path)
    declared = evidence['stage_receipts'].get(str(receipt_path))
    require(declared and declared['binding'] == receipt_binding, 'successful beat receipt absent from completed evidence')
    envelope = base.read_json(receipt_path)
    receipt = envelope['payload']
    require(set(envelope) == {'payload', 'receipt_sha256'} and envelope['receipt_sha256'] == base.value_hash(receipt)
            and declared['payload_sha256'] == envelope['receipt_sha256'], 'beat stage receipt hash')
    require(receipt.get('status') == 'passed' and receipt.get('stage') == 'beats' and ident in receipt['item_ids']
            and receipt.get('ordinary_empty_beat_allowed_only_after_this_successful_command') is True
            and receipt.get('recovery_applied') is False, 'empty/ordinary beat requires successful nonrecovery inference')
    paths = {'source_audio_sha256': row['input']['sha256'],
             **{stem + '_sha256': products[str(root / 'demix/htdemucs' / ident / (stem + '.wav'))]['sha256'] for stem in physical.STEMS},
             'beats_sha256': products[str(root / 'beats' / (ident + '.beats'))]['sha256'],
             'structure_sha256': products[str(root / 'structure' / (ident + '.json'))]['sha256']}
    return {'input': source, 'neural_products': products, 'beat_stage_receipt': receipt_binding, 'extractor_input_hashes': paths}


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float):
        require(not math.isinf(value), 'infinite scientific measurement')
        return None if math.isnan(value) else value
    return value


def feature_names(extractor):
    return [prefix + name for prefix, names in (('s16__', extractor.S16_FEATURES), ('s8__', extractor.S8_FEATURES),
                                                ('d__', extractor.D_FEATURES), ('r__', extractor.R_FEATURES), ('p__', extractor.P_FEATURES)) for name in names]


def validate_result(result, mapped, contract_sha, bias_sha, inputs, extractor):
    expected = mapped['extractor_row']
    require(result.get('status') == result.get('feature_status') == 'complete' and result.get('errors') == '',
            'legacy processing failure: ' + str(result.get('errors')))
    require(all(result[k] == expected[k] for k in ('item_id', 'source_id', 'label', 'group_id'))
            and result['duration_sec'] == 30 and result['native_sample_rate_hz'] == mapped['native_sample_rate_hz']
            and result['bias_sha256'] == bias_sha and result['run_fingerprint'] == contract_sha, 'extraction identity/rate/contract/calibration mismatch')
    require(json.loads(result['input_hashes']) == inputs['extractor_input_hashes'], 'extractor read unverified/changed inputs')
    names = feature_names(extractor)
    require({k for k in result if k.startswith(PREFIXES)} == set(names), 'fixed S/D/R/P feature set changed')
    require(all(result[k] is None or type(result[k]) in (float, int) for k in names), 'malformed feature scalar')
    raw = {k: float('nan') if result[k] is None else result[k] for k in names}
    require(extractor.sha256_json(raw) == result['feature_payload_sha256'], 'legacy feature payload hash mismatch')
    rate = mapped['native_sample_rate_hz']
    require(result['s16_native_eligible'] == int(rate >= 40000) and result['s8_native_eligible'] == int(rate >= 16000), 'native eligibility changed')
    clean(result)  # Infinities fail, whereas scientific NaNs remain missing.


def recheck_item_inputs(inputs):
    entries = [inputs['input'], inputs['beat_stage_receipt'], *inputs['neural_products'].values()]
    for entry in entries:
        require(base.file_binding(entry['path']) == {k: entry[k] for k in ('path', 'bytes', 'sha256')}, 'input changed during extraction')


def verify_receipt(path, mapped, contract, audited, extractor):
    require(path.is_file() and not path.is_symlink(), 'regular extraction receipt required')
    envelope = base.read_json(path)
    require(set(envelope) == {'payload', 'receipt_sha256'} and envelope['receipt_sha256'] == base.value_hash(envelope['payload'])
            and base.digest(path) == base.value_hash(envelope), 'extraction receipt canonical payload hash')
    receipt = envelope['payload']
    sha = base.value_hash(contract)
    require(receipt['contract_sha256'] == sha and receipt['row'] == mapped and receipt['row_sha256'] == base.value_hash(mapped)
            and receipt['status'] == 'measured_native30_S_D_R_P_not_admitted' and all(receipt[k] == v for k, v in SCOPE.items()), 'extraction receipt lineage/scope')
    inputs = inspect_inputs(contract, mapped, audited)
    require(receipt['inputs'] == inputs, 'resumed extraction inputs changed')
    validate_result(receipt['legacy_result'], mapped, sha, contract['bindings']['bias']['sha256'], inputs, extractor)
    recheck_item_inputs(inputs)
    return receipt


def one(contract, mapped, audited, extractor, bias):
    ident, output = mapped['cohort_row']['id'], Path(contract['output_root'])
    item = output / 'items' / (ident + '.json')
    if item.exists() or item.is_symlink():
        return verify_receipt(item, mapped, contract, audited, extractor)
    require(not list((output / 'items').glob('.' + ident + '.json.*.tmp')), 'orphan extraction temporary retained')
    inputs = inspect_inputs(contract, mapped, audited)
    root, sha = Path(contract['inference_root']), base.value_hash(contract)
    args = SimpleNamespace(duration=30, demix_root=[root / 'demix'], beat_root=[root / 'beats'], structure_root=[root / 'structure'])
    result = extractor.process(mapped['extractor_row'], args, bias, contract['bindings']['bias']['sha256'], sha)
    validate_result(result, mapped, sha, contract['bindings']['bias']['sha256'], inputs, extractor)
    recheck_item_inputs(inputs)
    receipt = {'status': 'measured_native30_S_D_R_P_not_admitted', 'row': mapped, 'row_sha256': base.value_hash(mapped),
               'contract_sha256': sha, 'inputs': inputs, 'legacy_result': clean(result), **SCOPE}
    base.write_new(item, {'payload': receipt, 'receipt_sha256': base.value_hash(receipt)})
    return receipt


def inventory(output, rows, complete=False):
    import re
    ids = {r['cohort_row']['id'] for r in rows}
    permitted = {'writer.lock', 'contract.json', 'items', 'failures', 'runs', 'manifest.json', 'COMMIT.json'}
    require({p.name for p in output.iterdir()} <= permitted, 'unexpected extraction root product')
    for path in output.iterdir():
        require(not path.is_symlink() and (path.is_dir() if path.name in {'items', 'failures', 'runs'} else path.is_file()), 'unsafe extraction root entry')
    for name in ('items', 'failures', 'runs'):
        for path in (output / name).iterdir():
            require(path.is_file() and not path.is_symlink(), 'unsafe extraction product')
            if name == 'items':
                valid = path.name in {i + '.json' for i in ids}
            elif name == 'failures':
                match = re.fullmatch(r'(.+)\.[0-9a-f]{32}\.json', path.name)
                valid = match and match[1] in ids
            else:
                valid = re.fullmatch(r'[0-9a-f]{32}\.json', path.name)
            require(valid, 'unexpected ' + name + ' product')
    if complete:
        require({p.stem for p in (output / 'items').iterdir()} == ids, 'incomplete per-item extraction inventory')


def products(output):
    return {str(path.relative_to(output)): {k: v for k, v in base.file_binding(path).items() if k != 'path'}
            for path in sorted(output.rglob('*')) if path.is_file() and path.name not in {'writer.lock', 'COMMIT.json'}}


def commit_record(contract_sha, count, bound):
    return {'version': VERSION, 'status': 'committed_native30_S_D_R_P_not_admitted',
            'contract_sha256': contract_sha, 'completed': count, 'products': bound,
            'all_inputs_products_runtime_end_rehashed': True, **SCOPE}


def run_contract(contract, extractor, bias, audit_callback=verify_completed_inference):
    """Reduced contracts/callbacks are test-only Python API; never CLI options."""
    rows, output = contract['rows'], physical.safe_path(contract['output_root'])
    ids = [r['cohort_row']['id'] for r in rows]
    require(len(ids) == len(set(ids)) == contract['expected_count'] and ids == sorted(ids), 'exact deterministic extraction roster')
    lock = physical.safe_path(Path(contract['inference_root']) / 'writer.lock')
    require(lock.is_file(), 'existing inference writer lock required')
    with lock.open('rb') as inference_lock, base.writer_lock(output):
        fcntl.flock(inference_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        check_contract(contract)
        import numpy as np
        require(np.array_equal(bias, extractor.load_bias(Path(contract['bindings']['bias']['path']))),
                'supplied numerical bias differs from bound calibration')
        path, sha = output / 'contract.json', base.value_hash(contract)
        if path.exists():
            require(not path.is_symlink() and base.read_json(path) == contract and base.digest(path) == sha, 'resumed extraction contract conflict')
        else:
            require({p.name for p in output.iterdir()} == {'writer.lock'}, 'nonempty output without extraction contract')
            base.write_new(path, contract)
        for name in ('items', 'failures', 'runs'):
            (output / name).mkdir(exist_ok=True)
        inventory(output, rows)
        before = audit_callback(contract)
        commit = output / 'COMMIT.json'
        committed = commit.exists()
        if committed:
            recorded = base.read_json(commit)
            require(recorded == commit_record(sha, len(rows), products(output))
                    and base.digest(commit) == base.value_hash(recorded), 'existing extraction COMMIT changed')
        run_id, receipts, failures = uuid.uuid4().hex, [], []
        for mapped in rows:
            ident = mapped['cohort_row']['id']
            try:
                result = (verify_receipt(output / 'items' / (ident + '.json'), mapped, contract, before, extractor)
                          if committed else one(contract, mapped, before, extractor, bias))
                receipts.append(result)
            except Exception as exc:
                failure = {'id': ident, 'contract_sha256': sha, 'row_sha256': base.value_hash(mapped),
                           'exception': type(exc).__name__, 'message': str(exc), 'run_id': run_id}
                failures.append(failure)
                if not committed:
                    base.write_new(output / 'failures' / (ident + '.' + run_id + '.json'), failure)
            if (len(receipts) + len(failures)) % 25 == 0 or len(receipts) + len(failures) == len(rows):
                print(base.canonical({'event': 'native30_SDRP_progress', 'expected': len(rows), 'measured_or_verified': len(receipts), 'failed': len(failures)}).decode().strip(), flush=True)
        check_contract(contract)
        after = audit_callback(contract)
        require(after == before, 'inference completion graph changed during extraction')
        if committed:
            require(not failures and base.read_json(commit)['products'] == products(output), 'committed extraction verification failed')
            return {'status': 'verified_existing_COMMIT', 'completed': len(rows), 'commit_sha256': base.digest(commit)}
        summary = {'status': 'partial_no_COMMIT' if failures else 'all_items_processed_scientific_missingness_retained',
                   'contract_sha256': sha, 'completed': len(receipts), 'expected': len(rows), 'failed': len(failures),
                   'failures': failures, 'run_id': run_id, **SCOPE}
        base.write_new(output / 'runs' / (run_id + '.json'), summary)
        if failures:
            return summary
        receipts = [verify_receipt(output / 'items' / (r['cohort_row']['id'] + '.json'), r, contract, after, extractor) for r in rows]
        availability = {key: dict(Counter(str(r['legacy_result'][key]) for r in receipts)) for key in
                        ('s8_computed', 's8_native_eligible', 'd_eligible', 'r_eligible', 'p_eligible', 'beat_output_status')}
        manifest = {'version': VERSION, 'status': 'all_items_processed_scientific_missingness_retained',
                    'contract_sha256': sha, 'rows': len(rows), 'ids': ids, 'source_counts': dict(Counter(r['cohort_row']['source_group'] for r in rows)),
                    'availability': availability, 'inference_verification': after['verification'],
                    'items': {ident: base.file_binding(output / 'items' / (ident + '.json')) for ident in ids}, **SCOPE}
        path = output / 'manifest.json'
        if path.exists():
            require(base.read_json(path) == manifest, 'retained extraction manifest conflict')
        else:
            base.write_new(path, manifest)
        check_contract(contract)
        require(audit_callback(contract) == after, 'final inference audit changed')
        inventory(output, rows, complete=True)
        bound = products(output)
        require(products(output) == bound, 'extraction product changed during final rehash')
        base.write_new(commit, commit_record(sha, len(rows), bound))
        return {**summary, 'availability': availability, 'commit_sha256': base.digest(commit)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', required=True)
    parser.add_argument('--frozen-sha256', required=True)
    parser.add_argument('--mode', choices=('preflight', 'run'), default='preflight')
    args = parser.parse_args()
    contract, extractor = prepare(args.frozen, args.frozen_sha256)
    if args.mode == 'preflight':
        result = {'status': 'metadata_preflight_no_cohort_audio_or_neural_products_opened',
                  'rows': len(contract['rows']), 'contract_sha256': base.value_hash(contract), **SCOPE}
    else:
        bias = extractor.load_bias(Path(contract['bindings']['bias']['path']))
        result = run_contract(contract, extractor, bias)
    print(base.canonical(result).decode().strip())
    return 1 if result['status'] == 'partial_no_COMMIT' else 0


if __name__ == '__main__':
    raise SystemExit(main())
