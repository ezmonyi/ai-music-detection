#!/usr/bin/env python3
"""Build a non-authorizing draft of the native30 v2 inference parent freeze.

The output is a wrapper whose embedded ``candidate`` must be independently
reviewed and published by the parent.  The wrapper itself is deliberately not
accepted by the inference runner.  No cohort audio, GPU, or inference is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
from pathlib import Path


VERSION = 'prepare_native30_inference_freeze_v2'
STATUS = 'draft_only_not_parent_frozen_not_authorized'
INFERENCE_V2_SHA = '1aaf8f2498986fbffe23f818ae6db2d37f59303eb239362447fbf8d8f88b6274'
RUNTIME_ROOT = '/mnt/nfs-code/users/yi/dynamics_rhythm_external_benchmark_20260903'
RUNTIME_CONTRACT = '/mnt/nfs-data/users/yi/source_diversity_expansion_20260905/results/inference_runtime_contract.json'
CHECKPOINT_MANIFEST = '/mnt/nfs-data/users/yi/source_diversity_expansion_20260905/results/inference_checkpoint_manifest.json'
OLD_CODE_ROOT = '/mnt/nfs-code/users/yi/source_diversity_expansion_20260905/code'
BIAS = '/mnt/nfs-code/users/yi/demucs_bias_corrected_1000_20260901/demucs_frequency_bias.npz'


def _load_inference():
    path = Path(__file__).resolve().with_name('run_native30_inference_batches_v2.py')
    if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != INFERENCE_V2_SHA:
        raise RuntimeError('pinned native30 inference v2 changed')
    module = importlib.import_module('run_native30_inference_batches_v2')
    if Path(module.__file__).resolve() != path:
        raise RuntimeError('native30 inference v2 import path changed')
    return module


r = _load_inference()
b = r.b


def parse_gpus(value):
    try:
        result = [int(item) for item in value.split(',')]
    except ValueError as exc:
        raise ValueError('GPU lists must be comma-separated nonnegative integers') from exc
    b.require(result and all(item >= 0 for item in result) and len(result) == len(set(result)),
              'GPU lists must be nonempty, unique and nonnegative')
    return result


def find_binding(bindings, digest, name):
    matches = [entry for entry in bindings.values() if entry.get('sha256') == digest]
    b.require(len(matches) == 1, 'cohort must contain exactly one ' + name + ' binding')
    b._binding_matches(matches[0], audio=False)
    return matches[0]


def make_shards(rows, shard_size):
    b.require(type(shard_size) is int and shard_size > 0, 'positive shard size required')
    shards = []
    for index, start in enumerate(range(0, len(rows), shard_size)):
        ids = [row['id'] for row in rows[start:start + shard_size]]
        shards.append({'index': index, 'rows': len(ids), 'item_ids': ids,
                       'item_ids_sha256': b.value_hash(ids)})
    b.validate_shards(shards, rows)
    return shards


def build_candidate(cohort_path, cohort_sha, inference_output_root, aio_gpus, beat_gpus, shard_size,
                    *, runtime_root=RUNTIME_ROOT, runtime_contract=RUNTIME_CONTRACT,
                    checkpoint_manifest=CHECKPOINT_MANIFEST, old_code_root=OLD_CODE_ROOT, bias=BIAS,
                    expected=b.EXPECTED, helper_loader=b.load_cohort_helper,
                    plan_sha=b.PLAN_SHA, screen_sha=b.SCREEN_SHA,
                    runtime_contract_sha=b.RUNTIME_CONTRACT_SHA,
                    checkpoint_manifest_sha=b.CHECKPOINT_MANIFEST_SHA, bias_sha=b.BIAS_SHA):
    cohort_path = b.safe_path(cohort_path)
    b.require(b.hash_string(cohort_sha) and b.digest(cohort_path) == cohort_sha, 'prepared cohort caller SHA mismatch')
    cohort = b.read_json(cohort_path)
    b.require(b.value_hash(cohort) == cohort_sha and cohort.get('version') == 'run_native30_fhsc_cohort_v1'
              and cohort.get('status') == 'frozen_before_any_measurement_audio_reads'
              and cohort.get('expected_count') == expected['total'], 'prepared cohort schema/count/canonical hash')
    b.validate_rows(cohort['rows'], expected)
    helper_entry = b.binding(Path(__file__).resolve().with_name('run_native30_fhsc_cohort_v1.py'))
    b.require(cohort['bindings'].get(helper_entry['path']) == helper_entry, 'cohort helper absent from prepared graph')
    helper = helper_loader(helper_entry)
    # Inference needs the helper's source-graph replay below, not its unrelated
    # F/H/SC numerical-runtime check.  This draft therefore stays metadata-only.
    plan = find_binding(cohort['bindings'], plan_sha, 'origin plan')
    screen = find_binding(cohort['bindings'], screen_sha, 'parent review screen')
    commits = {}
    for origin in ('new', 'prior'):
        declared = cohort['upstream_commits'][origin]
        entry = cohort['bindings'].get(declared['path'])
        b.require(entry and entry['sha256'] == declared['sha256'], 'cohort upstream COMMIT binding: ' + origin)
        commits[origin] = entry
    runtime = b.safe_path(runtime_root)
    output = b.safe_path(inference_output_root)
    b.require(output.parent.is_dir() and not output.exists()
              and output != Path(cohort['output_root'])
              and all(not output.is_relative_to(Path(entry['path']).parent) for entry in commits.values()),
              'draft inference output must be a new, non-overlapping path under an existing parent')
    b.require(aio_gpus and beat_gpus and len(aio_gpus) == len(set(aio_gpus))
              and len(beat_gpus) == len(set(beat_gpus)) and not set(aio_gpus) & set(beat_gpus)
              and all(type(item) is int and item >= 0 for item in aio_gpus + beat_gpus),
              'All-In-One and Beat This GPU sets must be nonempty, unique, nonnegative and disjoint')
    candidate = {'version': r.FREEZE_VERSION, 'status': r.FREEZE_STATUS,
                 'classifier_fits': 0, 'cohort_admitted': False, 'feature_extraction_authorized': False,
                 'recovery_applied': False, 'duration_s': 30, 'input_format': cohort['input_format'],
                 'cohort_contract': b.binding(cohort_path), 'cohort_helper': helper_entry,
                 'plan': plan, 'screen': screen, 'upstream_commits': commits,
                 'runner': b.binding(Path(r.__file__).resolve()),
                 'tests': b.binding(Path(r.__file__).resolve().with_name('test_' + r.VERSION + '.py')),
                 'v1_dependency': b.binding(Path(b.__file__).resolve()),
                 'runtime_root': str(runtime),
                 'runtime_contract': b.binding(b.safe_path(runtime_contract)),
                 'checkpoint_manifest': b.binding(b.safe_path(checkpoint_manifest)),
                 'old_code_root': str(b.safe_path(old_code_root)),
                 'bias': b.binding(b.safe_path(bias)), 'output_root': str(output),
                 'aio_gpus': aio_gpus, 'beat_gpus': beat_gpus,
                 'shards': make_shards(cohort['rows'], shard_size),
                 'child_environment': r.child_environment_policy(runtime), 'torch_hub_dir': r.TORCH_HUB}
    b.require(candidate['runtime_contract']['sha256'] == runtime_contract_sha
              and candidate['checkpoint_manifest']['sha256'] == checkpoint_manifest_sha
              and candidate['bias']['sha256'] == bias_sha, 'runtime/checkpoint/bias fixed SHA mismatch')
    b.verify_source_graph(candidate, cohort, helper, expected, plan_sha=plan_sha, screen_sha=screen_sha)
    b.validate_shards(candidate['shards'], cohort['rows'])
    return candidate


def make_draft(candidate):
    return {'version': VERSION, 'status': STATUS, 'candidate': candidate,
            'candidate_sha256': b.value_hash(candidate),
            'candidate_rows': sum(shard['rows'] for shard in candidate['shards']),
            'candidate_shards': len(candidate['shards']),
            'generator': b.binding(Path(__file__).resolve()),
            'tests': b.binding(Path(__file__).resolve().with_name('test_' + VERSION + '.py')),
            'parent_review_required': True, 'production_authorized': False,
            'audio_files_opened': 0, 'gpu_commands_run': 0, 'inference_runs': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--cohort-sha256', required=True)
    parser.add_argument('--inference-output-root', type=Path, required=True)
    parser.add_argument('--aio-gpus', required=True)
    parser.add_argument('--beat-gpus', required=True)
    parser.add_argument('--shard-size', type=int, default=24)
    parser.add_argument('--draft-output', type=Path, required=True)
    args = parser.parse_args()
    candidate = build_candidate(args.cohort, args.cohort_sha256, args.inference_output_root,
                                parse_gpus(args.aio_gpus), parse_gpus(args.beat_gpus), args.shard_size)
    draft = make_draft(candidate)
    output = b.safe_path(args.draft_output)
    b.require(output.parent.is_dir() and not output.exists() and not output.is_symlink(), 'new draft output required')
    b.write_new(output, draft)
    print(b.canonical({'status': STATUS, 'draft': b.binding(output),
                       'candidate_sha256': draft['candidate_sha256'],
                       'rows': draft['candidate_rows'], 'shards': draft['candidate_shards']}).decode().strip())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
