#!/usr/bin/env python3
"""Build a metadata-only, nonauthorizing reserved-BC parent-freeze candidate.

The candidate deliberately fails the producer's authorizing freeze gate.  A
parent must separately review, copy the validated fields, change the status and
scope, and bind that new file by SHA before any reserved audio may be opened.
This program never hashes or opens roster audio/annotation bytes and never
executes FFmpeg/FFprobe codecs; it only inspects their frozen metadata, hashes
the executable/toolchain files, and runs ``ldd`` to freeze the complete dynamic
library graph required by the independent auditor.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys


VERSION = 'prepare_bc_reserved_parent_draft_v1'
CANDIDATE_STATUS = 'nonauthorizing_reserved_parent_freeze_candidate_requires_parent_review'
PRODUCER_SHA = '1bc7a4f38184f011d381c1e1216a0f04c85fc207a23ff7011423ae2633c6e17d'
PRODUCER_TESTS_SHA = '9e65047619b42f937784e16be00be813a071dcb4a30fb3c9e5b955e4c30b0d3f'
DRAFT_SHA = '87803da916de04c35d226d7f9e41462d88ca52423ec1a935e723d327211b135e'
DRAFT_COMMIT_SHA = '5d0dcb01974cb08f52ca9d7898a38be4a727f5a6776dd8780ec544493c69f4ac'
EXPECTED = {'recordings': 90, 'float64': 630, 'precision': 270, 'codec': 540,
            'measurements': 1440, 'pools': 2880}
AUTHORIZED_SCOPE = {'reserved_audio_access_authorized': True,
                    'reserved_BC_measurement_authorized': True,
                    'codec_encoding_authorized': True,
                    'classifier_fits_authorized': False,
                    'model_scoring_authorized': False,
                    'BC_admission_authorized': False,
                    'threshold_changes_authorized': False}
NONAUTHORIZING_SCOPE = {key: False for key in AUTHORIZED_SCOPE}
HERE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hash_string(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_file(path):
    candidate = Path(path)
    require(candidate.is_absolute() and candidate.resolve() == candidate
            and candidate.is_file() and not candidate.is_symlink(),
            'safe canonical regular file required: ' + str(candidate))
    return candidate


def binding(path, expected=None):
    path = safe_file(path)
    result = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
    require(expected is None or result['sha256'] == expected,
            'binding SHA mismatch: ' + str(path))
    return result


def read_json(path, expected=None):
    entry = binding(path, expected)
    return json.loads(Path(entry['path']).read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))), entry


def write_json_new(path, value):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and not path.exists()
            and path.parent.is_dir() and not path.parent.is_symlink(),
            'new canonical candidate path required')
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n')


def load_producer():
    path = HERE / 'run_bc_guitarset_reserved_admission_v1.py'
    binding(path, PRODUCER_SHA)
    spec = importlib.util.spec_from_file_location('_bc_reserved_candidate_producer', path)
    require(spec is not None and spec.loader is not None, 'producer loader unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path and binding(path, PRODUCER_SHA)['sha256'] == PRODUCER_SHA,
            'producer origin changed')
    return module


def literal_constants(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'), filename=str(path))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def load_auditor(path, expected):
    path = Path(path)
    binding(path, expected)
    spec = importlib.util.spec_from_file_location('_bc_reserved_candidate_independent_auditor', path)
    require(spec is not None and spec.loader is not None, 'auditor loader unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path and binding(path, expected)['sha256'] == expected,
            'auditor origin changed')
    require(callable(getattr(module, 'independent_auditor_runtime', None)),
            'auditor runtime binder unavailable')
    return module


def validate_auditor(auditor_path, auditor_sha, tests_path, tests_sha,
                     draft_entry, draft_commit_entry, *, module=None):
    require(hash_string(auditor_sha) and hash_string(tests_sha),
            'caller must pin exact stable auditor and auditor-tests SHAs')
    auditor = binding(auditor_path, auditor_sha)
    tests = binding(tests_path, tests_sha)
    require(auditor['path'] != tests['path'], 'auditor and tests must be distinct')
    constants = literal_constants(auditor['path'])
    require(constants.get('PRODUCER_VERSION') == 'run_bc_guitarset_reserved_admission_v1'
            and constants.get('PRODUCER_SHA') == PRODUCER_SHA
            and constants.get('DRAFT_SHA') == draft_entry['sha256'] == DRAFT_SHA
            and constants.get('DRAFT_COMMIT_SHA') == draft_commit_entry['sha256'] == DRAFT_COMMIT_SHA
            and constants.get('EXPECTED') == EXPECTED,
            'auditor is not aligned to the frozen producer/draft/denominators')
    module = load_auditor(auditor['path'], auditor_sha) if module is None else module
    runtime = module.independent_auditor_runtime()
    require(isinstance(runtime, dict) and set(runtime) == {'scipy', 'modules'}
            and isinstance(runtime['scipy'], str) and runtime['scipy']
            and set(runtime['modules']) == {'scipy', 'scipy.signal._upfirdn',
                'scipy.signal._upfirdn_apply', 'scipy.special._ufuncs'}
            and all(binding(entry['path']) == entry for entry in runtime['modules'].values()),
            'independent auditor SciPy runtime binding')
    return ({'independent_auditor': auditor, 'independent_auditor_tests': tests,
             'auditor_version': constants.get('VERSION')}, module, runtime)


def validate_reserved_declarations(document):
    """Stat declared reserved files without opening or hashing their bytes."""
    root = Path(document['source_root'])
    require(root.is_absolute() and root.resolve() == root and root.is_dir() and not root.is_symlink(),
            'safe reserved source root metadata')
    roster = document.get('roster')
    require(isinstance(roster, list) and len(roster) == EXPECTED['recordings'],
            'reserved roster count')
    declarations = {}
    stats = {}
    for row in roster:
        item = row.get('item_id')
        require(isinstance(item, str) and item and row.get('split_role') == 'reserved',
                'reserved roster identity/role')
        values = {}
        facts = {}
        for key in ('source_audio', 'source_annotation'):
            entry = row.get(key)
            require(isinstance(entry, dict) and set(entry) == {'path', 'bytes', 'sha256'}
                    and isinstance(entry['bytes'], int) and entry['bytes'] > 0
                    and hash_string(entry['sha256']), 'reserved binding declaration: ' + str(item))
            path = Path(entry['path'])
            require(path.is_absolute() and path.resolve() == path and path.is_relative_to(root)
                    and not path.is_symlink(), 'reserved declaration path boundary')
            info = path.stat()
            require(stat.S_ISREG(info.st_mode) and info.st_size == entry['bytes'],
                    'reserved declaration stat/size mismatch')
            values[key] = dict(entry)
            facts[key] = {'path': str(path), 'bytes': info.st_size, 'device': info.st_dev,
                          'inode': info.st_ino, 'mtime_ns': info.st_mtime_ns}
        require(item not in declarations, 'duplicate reserved item ID')
        declarations[item], stats[item] = values, facts
    require(list(declarations) == sorted(declarations), 'reserved roster must be ordered')
    return {'declarations_sha256': value_hash(declarations), 'stat_snapshot': stats,
            'files_statted': 2 * len(declarations), 'reserved_files_opened_or_hashed': False}


def validate_codec_metadata(document, modules):
    """Validate only synthetic result metadata; never read its waveform products."""
    probe = modules['codec_probe']
    commit_entry = document['authorities']['synthetic_codec_COMMIT']
    results_entry = document['authorities']['synthetic_codec_results']
    commit, rebound_commit = read_json(commit_entry['path'], commit_entry['sha256'])
    results, rebound_results = read_json(results_entry['path'], results_entry['sha256'])
    require(rebound_commit == commit_entry and rebound_results == results_entry
            and Path(results_entry['path']) == Path(commit_entry['path']).parent / 'results.json',
            'synthetic codec metadata authority')
    product = commit.get('products', {}).get('results.json')
    require(product == {key: results_entry[key] for key in ('bytes', 'sha256')}
            and commit.get('version') == probe.VERSION and commit.get('status') == probe.STATUS
            and all(commit.get(key) == value for key, value in probe.SCOPE.items())
            and commit.get('results_sha256') == probe.value_hash(results)
            and results.get('version') == probe.VERSION
            and results.get('status') == 'synthetic_codec_roundtrip_characterized_not_admission'
            and all(results.get(key) == value for key, value in probe.SCOPE.items())
            and results.get('codec_recipes') == probe.CODECS == document['codec_recipes']
            and results.get('operation_order') == probe.OPERATION_ORDER
            and commit.get('toolchain_bindings_before') == commit.get('toolchain_bindings_end'),
            'synthetic codec recipe/toolchain metadata contract')
    current = probe.tool_bindings(results['toolchain'])
    require(current == commit['toolchain_bindings_end'], 'synthetic codec toolchain binding changed')
    return {'toolchain': results['toolchain'], 'codec_recipes': results['codec_recipes'],
            'synthetic_codec_COMMIT': commit_entry, 'synthetic_codec_results': results_entry,
            'validation': 'metadata_and_tool_files_only_synthetic_waveforms_not_opened'}


def linked_library_bindings(toolchain, *, run=subprocess.run):
    """Freeze the flat union used by the independent auditor for both tools."""
    paths = set()
    environment = {'LANG': 'C', 'LC_ALL': 'C', 'PATH': '/usr/bin:/bin'}
    for component in ('ffmpeg', 'ffprobe'):
        executable = toolchain[component]['path']
        require(binding(executable) == toolchain[component], 'codec executable changed: ' + component)
        completed = run(['/usr/bin/ldd', executable], capture_output=True, text=True,
                        env=environment, check=False)
        require(completed.returncode == 0 and 'not found' not in completed.stdout + completed.stderr,
                'ldd failed or dependency missing: ' + component)
        for line in completed.stdout.splitlines():
            import re
            for match in re.findall(r'(?:=>\s*)?(/[^\s]+)\s+\(', line):
                path = Path(match).resolve()
                require(path.is_absolute() and path.is_file() and not path.is_symlink(),
                        'ldd dependency is not a resolved regular file')
                paths.add(str(path))
    require(paths, 'empty FFmpeg/FFprobe dynamic library union')
    return {path: binding(path) for path in sorted(paths)}


def safe_output_root(document):
    output = Path(document['output_root'])
    source = Path(document['source_root'])
    require(output.is_absolute() and output.resolve() == output and not output.exists()
            and output.parent.is_dir() and not output.parent.is_symlink()
            and not output.is_symlink(), 'new canonical reserved result output required')
    require(not output.is_relative_to(source) and not source.is_relative_to(output),
            'reserved result/source overlap')
    return output


def build(draft_dir, draft_commit_sha, auditor_path, auditor_sha, auditor_tests,
          auditor_tests_sha, builder_tests, *, producer=None, modules=None,
          auditor_module=None, ldd_run=subprocess.run):
    require(draft_commit_sha == DRAFT_COMMIT_SHA, 'only committed reserved draft is eligible')
    producer = load_producer() if producer is None else producer
    require(binding(Path(producer.__file__).resolve(), PRODUCER_SHA)['sha256'] == PRODUCER_SHA,
            'frozen producer pin')
    producer_tests = binding(HERE / 'test_run_bc_guitarset_reserved_admission_v1.py',
                             PRODUCER_TESTS_SHA)
    draft_dir = Path(draft_dir)
    require(draft_dir.is_absolute() and draft_dir.resolve() == draft_dir
            and draft_dir.is_dir() and not draft_dir.is_symlink(), 'safe committed draft directory')
    draft_entry = binding(draft_dir / 'draft.json', DRAFT_SHA)
    draft_commit, draft_commit_entry = read_json(draft_dir / 'COMMIT.json', DRAFT_COMMIT_SHA)
    require(draft_commit.get('draft') == draft_entry
            and draft_commit.get('status') == 'committed_nonauthorizing_reserved_admission_draft'
            and draft_commit.get('reserved_audio_accessed') is False,
            'committed nonauthorizing reserved draft authority')
    modules = producer.load_frozen(HERE) if modules is None else modules
    document, verified_commit = producer.verify_draft(draft_dir, draft_commit_sha, modules)
    require(verified_commit == draft_commit_entry, 'producer draft verification binding')
    for entry in document['authorities'].values():
        require(binding(entry['path']) == entry, 'draft metadata authority changed')
    output = safe_output_root(document)
    source_before = validate_reserved_declarations(document)
    codec_before = validate_codec_metadata(document, modules)
    runtime_before = modules['codec'].runtime_snapshot()
    auditor, auditor_module, auditor_runtime = validate_auditor(
        auditor_path, auditor_sha, auditor_tests, auditor_tests_sha,
        draft_entry, draft_commit_entry, module=auditor_module)
    libraries_before = linked_library_bindings(codec_before['toolchain'], run=ldd_run)
    candidate = {'version': producer.FREEZE_VERSION, 'status': CANDIDATE_STATUS,
        'draft_COMMIT': draft_commit_entry, 'draft': draft_entry,
        'output_root': str(output), 'authorized_scope': NONAUTHORIZING_SCOPE,
        'expected_measurements': EXPECTED, 'storage_estimate': producer.storage_estimate(),
        'runtime': runtime_before, 'toolchain': codec_before['toolchain'],
        'independent_auditor_runtime': auditor_runtime,
        'codec_recipes': codec_before['codec_recipes'],
        'runner': binding(Path(producer.__file__).resolve(), PRODUCER_SHA),
        'runner_tests': producer_tests, **auditor,
        'linked_library_bindings': libraries_before,
        'draft_builder': binding(Path(__file__).resolve()),
        'draft_builder_tests': binding(builder_tests),
        'metadata_validation': {
            'reserved_roster_sha256': document['reserved_roster_sha256'],
            'reserved_source_declarations_sha256': source_before['declarations_sha256'],
            'reserved_source_files_statted': source_before['files_statted'],
            'reserved_source_bytes_opened_or_hashed': False,
            'synthetic_codec_waveform_products_opened': False,
            'codec_or_feature_commands_executed': False,
            'ldd_executed_for_dependency_inventory_only': True,
            'codec_metadata': codec_before},
        'authorization_boundary': {
            'this_candidate_authorizes_reserved_audio_access': False,
            'this_candidate_authorizes_reserved_BC_measurement': False,
            'this_candidate_authorizes_codec_encoding': False,
            'separate_parent_authorizing_freeze_required': True,
            'required_authorized_status': 'authorized_reserved_measurement_not_admission',
            'required_authorized_scope': AUTHORIZED_SCOPE,
            'BC_admission_remains_unauthorized_after_measurement': True}}
    document_end, commit_end = producer.verify_draft(draft_dir, draft_commit_sha, modules)
    source_end = validate_reserved_declarations(document_end)
    codec_end = validate_codec_metadata(document_end, modules)
    libraries_end = linked_library_bindings(codec_end['toolchain'], run=ldd_run)
    require(document_end == document and commit_end == draft_commit_entry
            and source_end == source_before and codec_end == codec_before
            and libraries_end == libraries_before
            and modules['codec'].runtime_snapshot() == runtime_before
            and auditor_module.independent_auditor_runtime() == auditor_runtime
            and binding(auditor_path, auditor_sha) == candidate['independent_auditor']
            and binding(auditor_tests, auditor_tests_sha) == candidate['independent_auditor_tests']
            and binding(Path(__file__).resolve()) == candidate['draft_builder']
            and binding(builder_tests) == candidate['draft_builder_tests'],
            'metadata/code/runtime/toolchain graph changed during candidate construction')
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft-dir', required=True)
    parser.add_argument('--draft-commit-sha256', required=True)
    parser.add_argument('--independent-auditor', required=True)
    parser.add_argument('--independent-auditor-sha256', required=True)
    parser.add_argument('--independent-auditor-tests', required=True)
    parser.add_argument('--independent-auditor-tests-sha256', required=True)
    parser.add_argument('--builder-tests', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--mode', choices=('preflight', 'publish'), default='preflight')
    args = parser.parse_args()
    candidate = build(args.draft_dir, args.draft_commit_sha256,
        args.independent_auditor, args.independent_auditor_sha256,
        args.independent_auditor_tests, args.independent_auditor_tests_sha256,
        args.builder_tests)
    result = {'status': 'metadata_preflight_passed_reserved_parent_candidate_not_authorized',
              'reserved_rows': EXPECTED['recordings'], 'output_root': candidate['output_root'],
              'authorized_scope': NONAUTHORIZING_SCOPE,
              'reserved_source_bytes_opened_or_hashed': False,
              'codec_or_feature_commands_executed': False}
    if args.mode == 'publish':
        write_json_new(Path(args.candidate), candidate)
        result.update(status='nonauthorizing_reserved_parent_freeze_candidate_published',
                      candidate=binding(Path(args.candidate)))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
