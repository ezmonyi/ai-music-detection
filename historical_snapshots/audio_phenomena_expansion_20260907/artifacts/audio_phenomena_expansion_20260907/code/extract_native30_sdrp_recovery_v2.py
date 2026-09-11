"""Separately frozen CPU SDRP extraction from typed native30 recovery evidence.

The ordinary extractor and all scientific numerical code remain unchanged.
This adapter does not run neural inference, repair outputs, fit or admit rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import hashlib
import importlib
from pathlib import Path
from types import SimpleNamespace
import uuid

ORDINARY_SHA = 'a170b673c484ff24472e9d479c86e71c345b8d1f36a3ff899c48405dfe63cddc'
RECOVERY_SHA = '91f5fec0fb25b163f69d6b46e032a5f9b27fb7206f93ac7aa4afb4933ae0e6bf'
HERE = Path(__file__).resolve().parent
ordinary_path = HERE / 'extract_native30_sdrp_v1.py'
if ordinary_path.is_symlink() or hashlib.sha256(ordinary_path.read_bytes()).hexdigest() != ORDINARY_SHA:
    raise ValueError('ordinary extractor pin changed')
o = importlib.import_module('extract_native30_sdrp_v1')
if Path(o.__file__).resolve() != ordinary_path:
    raise ValueError('ordinary extractor origin changed')
base, physical, require = o.base, o.physical, o.require
VERSION = 'extract_native30_sdrp_recovery_v2'
FREEZE_VERSION = 'native30-sdrp-recovery-extraction-parent-freeze-v2'
FREEZE_STATUS = 'parent_frozen_for_native30_S_D_R_P_recovery_aware_extraction_v2'
SCOPE = {**o.SCOPE, 'recovery_evidence_consumed': True, 'recovery_provenance_is_predictor': False}
ITEM_STATUS = 'measured_native30_S_D_R_P_with_recovery_provenance_not_admitted'
COMMIT_STATUS = 'committed_native30_S_D_R_P_with_recovery_provenance_not_admitted'
EVIDENCE_KEYS = o.EVIDENCE_KEYS


def load_recovery():
    return o.load_module(HERE / 'continue_native30_empty_beats_v2.py', RECOVERY_SHA, 'continue_native30_empty_beats_v2')


def completion_epoch(completion, bindings, root):
    """Require the v2 correction authority, not merely the shared status string."""
    root = physical.safe_path(root); authority = bindings['recovery_parent_freeze']
    require(completion.get('version') == 'continue_native30_empty_beats_v2' and
            completion.get('previous_continuation_freeze') == bindings['previous_continuation_freeze'] and
            completion.get('torch_cuda_source') == bindings['torch_cuda_source'] and
            completion.get('environment_transition') == load_recovery().ENVIRONMENT_TRANSITION and
            completion.get('continuation_parent_freeze') == {'path': authority['path'], 'sha256': authority['sha256']},
            'exact v2 completion epoch authority required')
    require(completion.get('driver_execution') == base.file_binding(root / 'driver_run_v2.json') and
            completion.get('historical_driver_execution') == base.file_binding(root / 'driver_run.json'),
            'separate v2/historical driver execution bindings required')
    return {k: completion[k] for k in ('version', 'continuation_parent_freeze', 'previous_continuation_freeze',
                                     'torch_cuda_source', 'environment_transition', 'driver_execution', 'historical_driver_execution')}


def prepare(freeze_path, freeze_sha):
    freeze_path = physical.safe_path(freeze_path)
    require(o.hash_string(freeze_sha) and base.digest(freeze_path) == freeze_sha, 'caller extraction freeze SHA mismatch')
    freeze = base.read_json(freeze_path)
    require(freeze.get('version') == FREEZE_VERSION and freeze.get('status') == FREEZE_STATUS and
            freeze.get('duration_s') == 30 and freeze.get('cpu_only') is True and
            freeze.get('feature_extraction_authorized') is True and all(freeze.get(k) == v for k, v in SCOPE.items()),
            'separate recovery-aware extraction scope/version')
    keys = ('runner', 'tests', 'ordinary_extractor', 'recovery_runner', 'recovery_parent_freeze',
            'inference_runner', 'inference_parent_freeze', 'inference_completion', 'cohort_contract',
            'origin_plan', 'extractor', 'core', 'bias', 'previous_continuation_freeze', 'torch_cuda_source')
    bindings = {k: freeze[k] for k in keys}
    base.recheck(bindings)
    expected = {'runner': (HERE / (VERSION + '.py'), None), 'tests': (HERE / ('test_' + VERSION + '.py'), None),
                'ordinary_extractor': (ordinary_path, ORDINARY_SHA),
                'recovery_runner': (HERE / 'continue_native30_empty_beats_v2.py', RECOVERY_SHA),
                'inference_runner': (HERE / 'run_native30_inference_batches_v2.py', o.INFERENCE_V2_SHA),
                'extractor': (Path(freeze['extractor']['path']), o.EXTRACTOR_SHA),
                'core': (Path(freeze['extractor']['path']).with_name('expanded_feature_definitions.py'), o.CORE_SHA),
                'bias': (Path(freeze['bias']['path']), o.BIAS_SHA), 'origin_plan': (Path(freeze['origin_plan']['path']), o.PLAN_SHA)}
    for key, (path, sha) in expected.items():
        require(bindings[key]['path'] == str(path) and (sha is None or bindings[key]['sha256'] == sha), 'frozen backend/code pin: ' + key)
    recovery = load_recovery(); authority = bindings['recovery_parent_freeze']
    continuation, prior, cohort, _ = recovery.validate_freeze(authority['path'], authority['sha256'], audio=False)
    require(continuation['original_parent_freeze'] == bindings['inference_parent_freeze'] and
            continuation['v2_runner'] == bindings['inference_runner'] and prior['cohort_contract'] == bindings['cohort_contract'] and
            prior['plan'] == bindings['origin_plan'] and continuation['previous_continuation_freeze'] == bindings['previous_continuation_freeze'] and
            continuation['torch_cuda_source'] == bindings['torch_cuda_source'] and continuation['driver'] == bindings['recovery_runner'],
            'recovery/original/cohort/plan/v2 authority join')
    completion_path = Path(continuation['output_root']) / 'completion.json'
    require(bindings['inference_completion']['path'] == str(completion_path), 'derivative inference completion root')
    completion = base.read_json(completion_path)
    require(completion.get('status') == 'passed_native30_inference_with_frozen_recovery_not_feature_extraction' and
            completion.get('rows') == physical.EXPECTED['total'] and completion.get('original_v2_status') == 'partial_without_own_completion' and
            completion.get('original_parent_freeze') == bindings['inference_parent_freeze'] and
            completion.get('cohort_contract') == bindings['cohort_contract'] and
            completion.get('continuation_parent_freeze') == {'path': authority['path'], 'sha256': authority['sha256']},
            'typed derivative completion authority')
    epoch = completion_epoch(completion, bindings, continuation['output_root'])
    physical.validate_rows(cohort['rows'], physical.EXPECTED)
    rows = o.mapped_rows(cohort['rows'], base.read_json(bindings['origin_plan']['path']), prior)
    require(set(completion['beat_provenance']) == {r['cohort_row']['id'] for r in rows}, 'exact recovery provenance roster')
    for mapped in rows:
        provenance = completion['beat_provenance'][mapped['cohort_row']['id']]
        mapped['beat_receipt_path'] = provenance['receipt']['path']
        mapped['beat_provenance'] = provenance
    require(bindings['bias'] == prior['bias'] and Path(bindings['extractor']['path']).parent == Path(prior['old_code_root']),
            'unchanged frozen numerical code/calibration')
    extractor = o.load_extractor(Path(bindings['extractor']['path']).parent)
    runtime = o.runtime_snapshot(); require(runtime == freeze['cpu_runtime'], 'pinned CPU numerical runtime mismatch')
    output = physical.safe_path(freeze['output_root'])
    require(output.parent.is_dir() and not any(output.is_relative_to(Path(root)) or Path(root).is_relative_to(output)
            for root in (prior['output_root'], continuation['output_root'])) and
            not any(Path(entry['path']).is_relative_to(output) for entry in bindings.values()), 'extraction output overlap')
    bindings['extraction_freeze'] = base.file_binding(freeze_path)
    contract = {'version': VERSION, 'status': 'frozen_CPU_recovery_aware_extraction_only', 'duration_s': 30,
                'expected_count': physical.EXPECTED['total'], 'rows': rows, 'output_root': str(output),
                'inference_root': prior['output_root'], 'recovery_root': continuation['output_root'],
                'recovery_epoch': epoch,
                'bindings': bindings, 'cpu_runtime': runtime, 'source_counts': physical.EXPECTED['sources'],
                'primary_families': {'S': 's8__', 'D': 'd__', 'R': 'r__', 'P': 'p__'}, 'additional_spectral_audit': 's16__',
                'numerics': 'unchanged_pinned_legacy_process_duration30', 'missingness_encoding': 'JSON_null; legacy_feature_payload_hash_replays_null_as_NaN',
                'native_rate_policy': 'SHA-joined_origin_plan_source_rate_not_standardized_rate',
                'padding_policy': 'exact_physical_verification_before_legacy_loaders_and_all_input_hashes_after',
                'empty_beat_policy': 'ordinary_success_or_separately_verified_exact_serializer_recovery_only', **SCOPE}
    o.check_contract(contract)
    return contract, extractor


def verify_completed_inference(contract):
    authority = contract['bindings']['recovery_parent_freeze']
    proof = load_recovery().verify_completion(authority['path'], authority['sha256'])
    require(proof['status'] == 'verified_complete_native30_inference_with_frozen_recovery' and
            proof['rows'] == contract['expected_count'] and proof['completion'] == contract['bindings']['inference_completion'],
            'typed derivative inference verification')
    completion = base.read_json(proof['completion']['path']); evidence = {k: completion[k] for k in EVIDENCE_KEYS}
    epoch = completion_epoch(completion, contract['bindings'], contract['recovery_root'])
    require(epoch == contract['recovery_epoch'], 'frozen v2 correction epoch changed')
    require(base.value_hash(evidence) == proof['evidence_sha256'] and
            base.file_binding(proof['completion']['path']) == proof['completion'] and
            proof['beat_provenance'] == completion['beat_provenance'] and
            completion['original_parent_freeze'] == contract['bindings']['inference_parent_freeze'] and
            completion['cohort_contract'] == contract['bindings']['cohort_contract'], 'derivative audited evidence changed')
    require(completion['beat_provenance'] == {m['cohort_row']['id']: m['beat_provenance'] for m in contract['rows']},
            'frozen per-item recovery provenance changed')
    return {'verification': proof, 'evidence': evidence, 'beat_provenance': completion['beat_provenance'],
            'recovery_evidence': completion['recovery_evidence'], 'recovered_ids': completion['recovered_ids'], 'recovery_epoch': epoch}


def inspect_inputs(contract, mapped, audited):
    row, ident = mapped['cohort_row'], mapped['cohort_row']['id']; root = Path(contract['inference_root'])
    require(audited['recovery_epoch'] == contract['recovery_epoch'], 'audited v2 epoch changed before per-item extraction')
    provenance = audited['beat_provenance'].get(ident)
    require(provenance == mapped['beat_provenance'] and provenance['receipt']['path'] == mapped['beat_receipt_path'], 'item provenance binding')
    if provenance['kind'] == 'ordinary_v2':
        require(provenance['item_status'] == 'ordinary_v2_success' and ident not in audited['recovered_ids'], 'ordinary item provenance type')
        inputs = o.inspect_inputs(contract, mapped, audited)
        require(inputs['beat_stage_receipt'] == provenance['receipt'], 'ordinary item receipt identity')
        return {**inputs, 'beat_provenance': provenance}
    require(provenance['kind'] == 'recovered_serializer_v1', 'unrecognized recovery provenance type')
    evidence = audited['evidence']; source = physical.inspect_audio(row['input']['path'], row, input_audio=True)
    require(evidence['inputs'].get(ident) == source, 'unverified extraction input')
    products = {**physical.verify_products('allinone', [row], root), **physical.verify_products('beats', [row], root)}
    require(all(evidence['products'].get(path) == record for path, record in products.items()), 'unverified/changed neural products')
    path = physical.safe_path(mapped['beat_receipt_path']); binding = base.file_binding(path)
    declared = evidence['stage_receipts'].get(str(path)); envelope = base.read_json(path); payload = envelope['payload']
    require(binding == provenance['receipt'] and declared and declared['binding'] == binding and
            set(envelope) == {'payload', 'receipt_sha256'} and envelope['receipt_sha256'] == base.value_hash(payload) == declared['payload_sha256'],
            'recovered receipt binding/envelope')
    require(payload['status'] == 'recovered_serializer_v1' and payload['recovery_applied'] is True and payload['ordinary_success'] is False and
            ident in payload['item_ids'] and payload['raw_replay'] == provenance['raw_replay'], 'separately typed recovered receipt required')
    record = audited['recovery_evidence'].get(str(path))
    require(record and all(record[k] == payload[k] for k in ('request', 'raw_replay', 'replay_log')), 'recovered request/raw/log evidence')
    validation = payload.get('recovery_validation')
    authority = contract['bindings']['recovery_parent_freeze']
    require(isinstance(validation, dict) and validation.get('raw_authority_epoch') in ('v1_snapshot_bound', 'v2_recorded'),
            'v2 validation must retain historical versus recorded raw authority')
    require(validation == {'validator': contract['bindings']['recovery_runner'],
            'parent_freeze': {'path': authority['path'], 'sha256': authority['sha256']},
            'raw_authority_epoch': validation['raw_authority_epoch'],
            'environment_transition': contract['recovery_epoch']['environment_transition'],
            'torch_cuda_source': contract['bindings']['torch_cuda_source']} and
            audited['recovery_epoch'] == contract['recovery_epoch'], 'recovered receipt exact v2 validation authority')
    beat = products[str(root / 'beats' / (ident + '.beats'))]
    missing = ident in record['missing_ids']; require(missing == (ident in audited['recovered_ids']), 'recovered ID set mismatch')
    expected_status = 'recovered_empty_unavailable' if missing else 'preserved_cli_success'
    require(provenance['item_status'] == expected_status, 'preserved versus repaired provenance mismatch')
    if missing:
        require(beat['bytes'] == 0 and beat['status'] == 'empty_unavailable' and beat['beat_count'] == 0 and
                beat['sha256'] == hashlib.sha256(b'').hexdigest(), 'recovered missing item must remain genuinely empty')
    paths = {'source_audio_sha256': row['input']['sha256'],
             **{stem + '_sha256': products[str(root / 'demix/htdemucs' / ident / (stem + '.wav'))]['sha256'] for stem in physical.STEMS},
             'beats_sha256': beat['sha256'], 'structure_sha256': products[str(root / 'structure' / (ident + '.json'))]['sha256']}
    return {'input': source, 'neural_products': products, 'beat_stage_receipt': binding, 'extractor_input_hashes': paths,
            'beat_provenance': provenance, 'recovery_evidence': record, 'recovery_validation': validation}


def recheck_item_inputs(inputs):
    o.recheck_item_inputs(inputs)
    if 'recovery_evidence' in inputs:
        for key in ('request', 'raw_replay', 'replay_log'):
            entry = inputs['recovery_evidence'][key]
            require(base.file_binding(entry['path']) == entry, 'recovery evidence changed during extraction')


def verify_receipt(path, mapped, contract, audited, extractor):
    require(path.is_file() and not path.is_symlink(), 'regular extraction receipt required')
    envelope = base.read_json(path); receipt = envelope['payload']; sha = base.value_hash(contract)
    require(set(envelope) == {'payload', 'receipt_sha256'} and envelope['receipt_sha256'] == base.value_hash(receipt) and
            base.digest(path) == base.value_hash(envelope), 'canonical extraction receipt hash')
    require(receipt['contract_sha256'] == sha and receipt['row'] == mapped and receipt['row_sha256'] == base.value_hash(mapped) and
            receipt['status'] == ITEM_STATUS and all(receipt[k] == v for k, v in SCOPE.items()), 'recovery extraction receipt lineage/scope')
    inputs = inspect_inputs(contract, mapped, audited); require(receipt['inputs'] == inputs, 'resumed recovery extraction inputs changed')
    o.validate_result(receipt['legacy_result'], mapped, sha, contract['bindings']['bias']['sha256'], inputs, extractor)
    recheck_item_inputs(inputs)
    return receipt


def one(contract, mapped, audited, extractor, bias):
    ident = mapped['cohort_row']['id']; output = Path(contract['output_root']); path = output / 'items' / (ident + '.json')
    if path.exists() or path.is_symlink(): return verify_receipt(path, mapped, contract, audited, extractor)
    require(not list((output / 'items').glob('.' + ident + '.json.*.tmp')), 'orphan extraction temporary retained')
    inputs = inspect_inputs(contract, mapped, audited); root = Path(contract['inference_root']); sha = base.value_hash(contract)
    args = SimpleNamespace(duration=30, demix_root=[root / 'demix'], beat_root=[root / 'beats'], structure_root=[root / 'structure'])
    result = extractor.process(mapped['extractor_row'], args, bias, contract['bindings']['bias']['sha256'], sha)
    o.validate_result(result, mapped, sha, contract['bindings']['bias']['sha256'], inputs, extractor); recheck_item_inputs(inputs)
    receipt = {'status': ITEM_STATUS, 'row': mapped, 'row_sha256': base.value_hash(mapped), 'contract_sha256': sha,
               'inputs': inputs, 'legacy_result': o.clean(result), **SCOPE}
    base.write_new(path, {'payload': receipt, 'receipt_sha256': base.value_hash(receipt)})
    return receipt


def commit_record(sha, count, bound):
    return {'version': VERSION, 'status': COMMIT_STATUS, 'contract_sha256': sha, 'completed': count, 'products': bound,
            'all_inputs_products_runtime_end_rehashed': True, **SCOPE}


def run_contract(contract, extractor, bias, audit_callback=verify_completed_inference):
    """Reduced contracts/callbacks are synthetic-test API, never CLI options."""
    rows = contract['rows']; ids = [m['cohort_row']['id'] for m in rows]; output = physical.safe_path(contract['output_root'])
    require(len(ids) == len(set(ids)) == contract['expected_count'] and ids == sorted(ids), 'exact deterministic extraction roster')
    with ExitStack() as stack:
        for root in sorted({contract['inference_root'], contract['recovery_root']}):
            lock = physical.safe_path(Path(root) / 'writer.lock'); require(lock.is_file(), 'existing upstream writer lock required')
            stream = stack.enter_context(lock.open('rb')); fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        stack.enter_context(base.writer_lock(output)); o.check_contract(contract)
        import numpy as np
        require(np.array_equal(bias, extractor.load_bias(Path(contract['bindings']['bias']['path']))), 'supplied bias differs from calibration')
        path, sha = output / 'contract.json', base.value_hash(contract)
        if path.exists(): require(base.read_json(path) == contract and base.digest(path) == sha, 'resumed extraction contract conflict')
        else:
            require({p.name for p in output.iterdir()} == {'writer.lock'}, 'nonempty output without extraction contract')
            base.write_new(path, contract)
        for folder in ('items', 'failures', 'runs'): (output / folder).mkdir(exist_ok=True)
        o.inventory(output, rows); before = audit_callback(contract); commit = output / 'COMMIT.json'; committed = commit.exists()
        if committed:
            recorded = base.read_json(commit)
            require(recorded == commit_record(sha, len(rows), o.products(output)) and base.digest(commit) == base.value_hash(recorded), 'existing COMMIT changed')
        run_id, receipts, failures = uuid.uuid4().hex, [], []
        for mapped in rows:
            ident = mapped['cohort_row']['id']
            try:
                receipt = (verify_receipt(output / 'items' / (ident + '.json'), mapped, contract, before, extractor)
                           if committed else one(contract, mapped, before, extractor, bias))
                receipts.append(receipt)
            except Exception as exc:
                failure = {'id': ident, 'contract_sha256': sha, 'row_sha256': base.value_hash(mapped),
                           'exception': type(exc).__name__, 'message': str(exc), 'run_id': run_id}
                failures.append(failure)
                if not committed: base.write_new(output / 'failures' / (ident + '.' + run_id + '.json'), failure)
            if (len(receipts) + len(failures)) % 25 == 0 or len(receipts) + len(failures) == len(rows):
                print(base.canonical({'event': 'native30_recovery_SDRP_progress', 'expected': len(rows), 'measured_or_verified': len(receipts), 'failed': len(failures)}).decode().strip(), flush=True)
        o.check_contract(contract); after = audit_callback(contract); require(after == before, 'inference/recovery evidence changed during extraction')
        if committed:
            require(not failures and base.read_json(commit)['products'] == o.products(output), 'committed extraction verification failed')
            return {'status': 'verified_existing_COMMIT', 'completed': len(rows), 'commit_sha256': base.digest(commit)}
        summary = {'status': 'partial_no_COMMIT' if failures else 'all_items_processed_scientific_missingness_retained',
                   'contract_sha256': sha, 'completed': len(receipts), 'expected': len(rows), 'failed': len(failures),
                   'failures': failures, 'run_id': run_id, **SCOPE}
        base.write_new(output / 'runs' / (run_id + '.json'), summary)
        if failures: return summary
        receipts = [verify_receipt(output / 'items' / (m['cohort_row']['id'] + '.json'), m, contract, after, extractor) for m in rows]
        availability = {key: dict(Counter(str(r['legacy_result'][key]) for r in receipts)) for key in
                        ('s8_computed', 's8_native_eligible', 'd_eligible', 'r_eligible', 'p_eligible', 'beat_output_status')}
        manifest = {'version': VERSION, 'status': 'all_items_processed_scientific_missingness_retained', 'contract_sha256': sha,
                    'rows': len(rows), 'ids': ids, 'source_counts': dict(Counter(m['cohort_row']['source_group'] for m in rows)),
                    'availability': availability, 'inference_verification': after['verification'],
                    'beat_provenance': after['beat_provenance'], 'recovered_ids': after['recovered_ids'],
                    'recovery_epoch': after['recovery_epoch'],
                    'items': {i: base.file_binding(output / 'items' / (i + '.json')) for i in ids}, **SCOPE}
        path = output / 'manifest.json'
        if path.exists(): require(base.read_json(path) == manifest and base.digest(path) == base.value_hash(manifest), 'retained manifest changed')
        else: base.write_new(path, manifest)
        o.check_contract(contract); require(audit_callback(contract) == after, 'final recovery inference audit changed')
        o.inventory(output, rows, complete=True); bound = o.products(output); require(o.products(output) == bound, 'final extraction product mutation')
        base.write_new(commit, commit_record(sha, len(rows), bound))
        return {**summary, 'availability': availability, 'commit_sha256': base.digest(commit)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', required=True); parser.add_argument('--frozen-sha256', required=True)
    parser.add_argument('--mode', choices=('preflight', 'run'), default='preflight'); args = parser.parse_args()
    contract, extractor = prepare(args.frozen, args.frozen_sha256)
    if args.mode == 'preflight':
        result = {'status': 'metadata_preflight_no_cohort_audio_or_neural_products_opened', 'rows': len(contract['rows']),
                  'contract_sha256': base.value_hash(contract), **SCOPE}
    else: result = run_contract(contract, extractor, extractor.load_bias(Path(contract['bindings']['bias']['path'])))
    print(base.canonical(result).decode().strip()); return int(result['status'] == 'partial_no_COMMIT')


if __name__ == '__main__':
    raise SystemExit(main())
