#!/usr/bin/env python3
"""Build a nonauthorizing SDRP operational-extraction parent-freeze draft.

The builder is metadata-only: it never opens cohort audio or neural WAV products.
It requires a caller-pinned terminal operational completion and validates its
canonical metadata graph.  The later extraction runner performs the exhaustive
completion/product verification before it can extract any feature.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


VERSION = 'prepare_native30_sdrp_operational_draft_v1'
DRAFT_STATUS = 'draft_requires_parent_review_not_feature_extraction_authorized'
BACKEND_SHA = '5e99d91487ff3b2b621101ec06512261d2478fe870564fb44204e50b64964a98'
BACKEND_TESTS_SHA = 'de6626540e188da2f79d1e709cce1a03cee1a384abf9be42296d3880bb0b9ff5'
OPERATIONAL_SHA = '102875006494669f289e8501e66f37d276385f9f371a0de40815d37a718e4e11'
OPERATIONAL_FREEZE_SHA = 'e31fb8003e0a30cdb1fe20adb1adfc5d55800cef2b4789db7bca865b6f635d99'
EXPECTED_ROWS = 3830
HERE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hash_string(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def canonical(value):
    # This is an evidence protocol, not a presentation choice: operational
    # completion hashes were produced by run_native30_inference_batches_v1.b.
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path, expected=None):
    candidate = Path(path)
    require(candidate.is_absolute() and candidate.is_file() and not candidate.is_symlink(),
            'safe absolute non-symlink file required')
    path = candidate.resolve()
    result = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
    require(expected is None or result['sha256'] == expected, 'file SHA mismatch: ' + str(path))
    return result


def read_json(path, expected=None):
    entry = binding(path, expected)
    return json.loads(Path(entry['path']).read_text(encoding='utf-8')), entry


def write_json_new(path, value):
    path = Path(path)
    require(path.is_absolute() and not path.exists() and not path.is_symlink()
            and path.parent.is_dir() and not path.parent.is_symlink(), 'new safe draft path required')
    with path.open('xb') as stream:
        stream.write(canonical(value))


def load_backend():
    path = HERE / 'extract_native30_sdrp_operational_v1.py'
    binding(path, BACKEND_SHA)
    spec = importlib.util.spec_from_file_location('_sdrp_operational_draft_backend', path)
    require(spec is not None and spec.loader is not None, 'backend module loader')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path and binding(path, BACKEND_SHA)['sha256'] == BACKEND_SHA,
            'backend module origin changed')
    return module


def validate_terminal_metadata(backend, operational, prior, cohort, runtime,
                               completion_path, completion_sha):
    """Validate terminal JSON lineage without rehashing audio/product payloads."""
    require(hash_string(completion_sha), 'caller must supply exact completion SHA')
    completion_path = Path(completion_path)
    completion, completion_binding = read_json(completion_path, completion_sha)
    root = Path(operational['output_root'])
    require(completion_path.resolve() == root / 'completion.json', 'completion path/root')
    engine = backend.load_operational()
    authority = binding(operational['_path'], operational['_sha256'])
    require(completion.get('version') == engine.VERSION
            and completion.get('status') == engine.STATUS
            and completion.get('rows') == EXPECTED_ROWS == len(cohort['rows'])
            and completion.get('original_v2_status') == 'partial_without_own_completion'
            and completion.get('base_recovery_v2_status') == 'partial_preserved_without_own_completion'
            and completion.get('operational_parent_freeze') == {
                'path': authority['path'], 'sha256': authority['sha256']}
            and completion.get('base_freeze') == operational['base_freeze']
            and completion.get('base_driver') == operational['base_driver']
            and completion.get('placement') == engine.PLACEMENT
            and completion.get('runtime') == runtime
            and completion.get('cohort_contract') == prior['cohort_contract']
            and completion.get('original_parent_freeze') == operational['_base']['original_parent_freeze'],
            'nonterminal or wrong operational completion authority')
    final_audit = completion.get('final_audit')
    require(isinstance(final_audit, dict) and final_audit == binding(root / 'final_audit.json'),
            'terminal final-audit binding')
    audit, _ = read_json(final_audit['path'], final_audit['sha256'])
    require(audit == {key: value for key, value in completion.items() if key != 'final_audit'},
            'completion/final-audit canonical payload mismatch')
    base_keys = ('stage_receipts', 'logs', 'products', 'inputs')
    for key in base_keys:
        require(isinstance(completion.get(key), dict) and completion[key]
                and completion.get(key + '_sha256') == value_hash(completion[key]),
                'completion evidence map/hash: ' + key)
    ids = {row['id'] for row in cohort['rows']}
    provenance = completion.get('beat_provenance')
    require(isinstance(provenance, dict) and set(provenance) == ids
            and isinstance(completion.get('recovered_ids'), list)
            and completion['recovered_ids'] == sorted(completion['recovered_ids'])
            and set(completion['recovered_ids']) <= ids,
            'completion exact beat-provenance roster')
    executions = completion.get('stage_executions')
    recovery = completion.get('recovery_evidence')
    require(isinstance(executions, dict) and executions
            and completion.get('replacement_attempt_096') in executions.values()
            and isinstance(recovery, dict) and recovery,
            'terminal operational execution/recovery evidence')
    run = read_json(operational['original_run_contract']['path'],
                    operational['original_run_contract']['sha256'])[0]
    require(completion.get('source_graph') == run['source_graph_start'],
            'completion source graph declaration')
    epoch = backend.completion_epoch(completion, {
        'operational_parent_freeze': authority,
        'recovery_parent_freeze': operational['base_freeze'],
        'recovery_runner': operational['base_driver']}, root)
    return {'status': 'terminal_operational_completion_metadata_validated_without_audio_product_rehash',
            'completion': completion_binding, 'final_audit': final_audit,
            'evidence_sha256': value_hash({k: completion[k] for key in base_keys
                                           for k in (key, key + '_sha256')}),
            'beat_provenance_sha256': value_hash(provenance),
            'recovered_ids_sha256': value_hash(completion['recovered_ids']),
            'recovery_epoch': epoch, 'rows': len(ids),
            'audio_or_neural_product_bytes_opened': False}


def compose(backend, operational, prior, cohort, runtime, completion_proof,
            output_root, builder_tests, cpu_runtime=None):
    continuation = operational['_base']
    output = Path(output_root)
    require(output.is_absolute() and output.resolve() == output and not output.exists()
            and output.parent.is_dir() and not output.parent.is_symlink()
            and not output.is_symlink(), 'new canonical extraction output root required')
    roots = (Path(prior['output_root']), Path(continuation['output_root']),
             Path(operational['output_root']))
    require(all(not output.is_relative_to(root) and not root.is_relative_to(output) for root in roots),
            'extraction output overlaps inference/recovery evidence root')
    authority = binding(operational['_path'], operational['_sha256'])
    bindings = {
        'runner': binding(HERE / 'extract_native30_sdrp_operational_v1.py', BACKEND_SHA),
        'tests': binding(HERE / 'test_extract_native30_sdrp_operational_v1.py', BACKEND_TESTS_SHA),
        'ordinary_extractor': binding(backend.ordinary_path, backend.ORDINARY_SHA),
        'recovery_runner': operational['base_driver'],
        'recovery_parent_freeze': operational['base_freeze'],
        'inference_runner': continuation['v2_runner'],
        'inference_parent_freeze': continuation['original_parent_freeze'],
        'inference_completion': completion_proof['completion'],
        'cohort_contract': prior['cohort_contract'],
        'origin_plan': prior['plan'],
        'extractor': prior['extractor'],
        'core': binding(Path(prior['extractor']['path']).with_name('expanded_feature_definitions.py'),
                        backend.o.CORE_SHA),
        'bias': prior['bias'],
        'previous_continuation_freeze': continuation['previous_continuation_freeze'],
        'torch_cuda_source': continuation['torch_cuda_source'],
        'operational_runner': operational['driver'],
        'operational_parent_freeze': authority}
    for name, entry in bindings.items():
        require(entry == binding(entry['path']), 'authority binding changed: ' + name)
    cpu_runtime = backend.o.runtime_snapshot() if cpu_runtime is None else cpu_runtime
    return {'version': backend.FREEZE_VERSION, 'status': DRAFT_STATUS,
        'duration_s': 30, 'cpu_only': True, 'feature_extraction_authorized': False,
        **backend.SCOPE, **bindings, 'cpu_runtime': cpu_runtime, 'output_root': str(output),
        'terminal_completion_metadata_validation': completion_proof,
        'draft_builder': binding(Path(__file__).resolve()),
        'draft_builder_tests': binding(builder_tests),
        'authorization_boundary': {
            'this_draft_authorizes_extraction': False,
            'separate_parent_freeze_required': True,
            'authorized_freeze_status_if_separately_published': backend.FREEZE_STATUS,
            'authorized_freeze_must_set_feature_extraction_authorized': True,
            'actual_extraction_prepare_and_run_reverify_full_completion': True,
            'completion_SHA_is_caller_supplied_not_invented': True},
        'scope_note': 'metadata-only draft; no cohort audio/product bytes opened; no extraction or fitting'}


def build(operational_freeze, operational_freeze_sha, completion, completion_sha,
          output_root, builder_tests, *, backend=None, context_loader=None, cpu_runtime=None):
    require(operational_freeze_sha == OPERATIONAL_FREEZE_SHA,
            'only the reviewed operational parent freeze is eligible')
    backend = load_backend() if backend is None else backend
    engine = backend.load_operational()
    require(binding(operational_freeze, operational_freeze_sha)['sha256'] == OPERATIONAL_FREEZE_SHA
            and binding(engine.__file__, OPERATIONAL_SHA)['sha256'] == OPERATIONAL_SHA,
            'operational runner/freeze pin')
    loader = engine.setup_context if context_loader is None else context_loader
    context = loader(operational_freeze, operational_freeze_sha, audio=False)
    require(isinstance(context, tuple) and len(context) == 6, 'operational metadata context')
    operational, prior, _, cohort, _, runtime = context
    require(operational['_path'] == str(Path(operational_freeze).resolve())
            and operational['_sha256'] == operational_freeze_sha,
            'operational context authority')
    proof = validate_terminal_metadata(backend, operational, prior, cohort, runtime,
                                       completion, completion_sha)
    draft = compose(backend, operational, prior, cohort, runtime, proof,
                    output_root, builder_tests, cpu_runtime)
    require(binding(completion, completion_sha) == proof['completion']
            and binding(operational_freeze, operational_freeze_sha) == draft['operational_parent_freeze'],
            'terminal authorities changed during draft construction')
    return draft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operational-parent-freeze', required=True)
    parser.add_argument('--operational-parent-freeze-sha256', required=True)
    parser.add_argument('--completion', required=True)
    parser.add_argument('--completion-sha256', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--tests', required=True)
    parser.add_argument('--draft', required=True)
    parser.add_argument('--mode', choices=('preflight', 'publish'), default='preflight')
    args = parser.parse_args()
    draft = build(args.operational_parent_freeze, args.operational_parent_freeze_sha256,
                  args.completion, args.completion_sha256, args.output_root, args.tests)
    result = {'status': 'terminal_metadata_preflight_passed_draft_not_authorized',
              'rows': draft['terminal_completion_metadata_validation']['rows'],
              'completion': draft['inference_completion'],
              'output_root': draft['output_root'],
              'feature_extraction_authorized': False,
              'audio_or_neural_product_bytes_opened': False}
    if args.mode == 'publish':
        write_json_new(Path(args.draft), draft)
        result.update(status='nonauthorizing_operational_SDRP_parent_freeze_draft_published',
                      draft=binding(Path(args.draft)))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
