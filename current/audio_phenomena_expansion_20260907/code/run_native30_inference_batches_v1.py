#!/usr/bin/env python3
"""Receipt-locked native30 All-In-One and Beat This inference.

Preflight is the default and opens no cohort audio.  ``run`` is unreachable
without a caller-supplied SHA of a separate parent freeze.  The freeze binds a
prepared cohort contract, its graph authority, runtime, output, GPUs and exact
shards.  This module performs no feature extraction or classifier fitting.

Known Beat This serializer failures are retained as failed, unreceipted shards;
recovery is deliberately outside this v1 runner.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid

import numpy as np
import soundfile as sf


VERSION = 'run_native30_inference_batches_v1'
FREEZE_VERSION = 'native30-inference-parent-freeze-v1'
FREEZE_STATUS = 'parent_frozen_for_native30_neural_inference'
COHORT_HELPER_SHA = 'dc26465459df133b89d6023bb1c732e91ebab5b63ff02143280b50c383e7b3cc'
PLAN_SHA = '94982cda45a4bae59371ec06895894227a39fdb84d8043045f009f6282e9e964'
SCREEN_SHA = 'ee21f8a1c28fa6a847f2fc2893f9acf41f30baabee72682c0ac69a80ed38ec87'
RUNTIME_CONTRACT_SHA = 'c6581c737fdb584ff763cc88752dc98864d9ea583a3e4cf9dd552a42ce791268'
CHECKPOINT_MANIFEST_SHA = 'ec65f4d774d21340b0cf40cd91ab526a92f9545bb5314c440a8db9d6ab618ad4'
BIAS_SHA = 'bcacfecac5ce69927dfed2a51ec21cc346f613a7874c519e419d22d35a97de5e'
OLD_CODE = {
    'extract_expanded_four_family.py': '98fc6caa8b54ed1370269db13fe8d49559d0b877b4b0d5a3a069c8409906c5fe',
    'expanded_feature_definitions.py': '8b9745085c7518e84613b2ba499dbae775a57e4dcf95670c5e86a05ab524ff00',
}
EXTRA_RUNTIME_CODE = {
    'repos/all-in-one-infer/src/allin1_infer/spectrogram.py': '5c9683567867bee0a83e91b0e4c9fba0af74d67a76466a98487a0dccd310dbc7',
    'repos/all-in-one-infer/src/allin1_infer/analyze.py': '8f2d4271ee9464c987341e63f40c9eeeb0f5bd8183713f06f6998a8c3499df52',
    'repos/beat_this/beat_this/preprocessing.py': '5aaeefa811c2fd1340c8a45233af4c2e814720b65c2124e59e78538fed03b3bc',
    'repos/beat_this/beat_this/inference.py': 'ccccff4665399b931ebf181b73d264a6ad64fc1fe8acb4d3b950df825c47ce2d',
    'repos/beat_this/beat_this/utils.py': '779d6102ea28bdf94c8227157790191f5abbbb4564d0c36ce5ce0859e067cfab',
    'repos/beat_this/beat_this/model/postprocessor.py': '431d6aeef41a8e9e9e63ee2577e895a5e214e0dbbcd2d505458cdfec02a611ee',
}
LAUNCHERS = {
    'venv/bin/all-in-one-infer': '62169b6b57c0abf2db30fe8c098c20118c94dcd807b7ae5239ffe709cbf8e3e4',
    'venv/bin/beat_this': '3a198c6a1c4e973e1098868c23f317e5d2b21148efd310adcc87b949d357622e',
}
VERSIONS = {'numpy': '1.26.4', 'scipy': '1.17.1', 'librosa': '0.11.0',
            'soundfile': '0.14.0', 'demucs-infer': '4.2.2',
            'all-in-one-infer': '3.1.0', 'beat_this': '1.1.0',
            'torch': '2.8.0+cu128', 'torchaudio': '2.8.0+cu128'}
IDENTITY = ('id', 'source_group', 'label', 'role', 'group_id', 'component_id')
STEMS = ('bass', 'drums', 'other', 'vocals')
AUDIO_SUFFIXES = {'.wav', '.wave', '.flac', '.mp3', '.ogg', '.opus', '.m4a',
                  '.aac', '.aif', '.aiff', '.wma', '.mp4', '.webm'}
EXPECTED = {'total': 3830, 'new': 1656, 'prior': 2174, 'excluded': 39,
            'human': 1664, 'ai': 2166,
            'sources': {'ACE-Step': 400, 'FMA': 354, 'HeartMuLa': 366,
                        'MTG-Jamendo': 500, 'Mureka_v9': 500, 'Suno': 400,
                        'Udio': 500, 'human_maestro_v3': 300,
                        'human_medleydb': 168, 'human_moisesdb': 239,
                        'human_saraga_hindustani_v1': 103}}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def hash_string(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def safe_path(value):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts and path.resolve() == path,
            'absolute unredirected path required: ' + str(path))
    return path


def read_json(path):
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def binding(path, *, hash_file=True):
    path = safe_path(path)
    require(path.is_file() and not path.is_symlink(), 'regular file required: ' + str(path))
    result = {'path': str(path), 'bytes': path.stat().st_size}
    if hash_file:
        result['sha256'] = digest(path)
    return result


def write_new(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def load_cohort_helper(record):
    path = safe_path(record['path'])
    expected = record['sha256']
    require(path == Path(__file__).resolve().with_name('run_native30_fhsc_cohort_v1.py'),
            'cohort helper must be reviewed sibling')
    require(expected == COHORT_HELPER_SHA and binding(path) == record, 'cohort helper pin mismatch')
    module = importlib.import_module('run_native30_fhsc_cohort_v1')
    require(Path(module.__file__).resolve() == path, 'cohort helper import path mismatch')
    return module


def _binding_matches(entry, *, audio):
    require(set(entry) == {'path', 'bytes', 'sha256'} and hash_string(entry['sha256'])
            and type(entry['bytes']) is int and entry['bytes'] >= 0, 'malformed binding')
    path = safe_path(entry['path'])
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == entry['bytes'],
            'bound file missing/size changed: ' + str(path))
    if audio or path.suffix.lower() not in AUDIO_SUFFIXES:
        require(digest(path) == entry['sha256'], 'bound file hash changed: ' + str(path))


def verify_source_graph(freeze, cohort, helper, expected=EXPECTED, *, plan_sha=PLAN_SHA, screen_sha=SCREEN_SHA):
    """Replay the helper's source graph without importing its F/H/SC runtime."""
    bindings = {}
    for key, fixed in (('plan', plan_sha), ('screen', screen_sha)):
        entry = freeze[key]
        require(entry['sha256'] == fixed, key + ' fixed SHA mismatch')
        _binding_matches(entry, audio=False)
        helper.add_binding(bindings, entry)
    new_entry, prior_entry = freeze['upstream_commits']['new'], freeze['upstream_commits']['prior']
    for entry in (new_entry, prior_entry):
        _binding_matches(entry, audio=False)
    new_root, prior_root = Path(new_entry['path']).parent, Path(prior_entry['path']).parent
    nc, new = helper.bind_commit(new_root, new_entry['sha256'], 'new', bindings)
    pc, prior = helper.bind_commit(prior_root, prior_entry['sha256'], 'prior', bindings)
    require(nc['eligible'] == expected['new'] and nc['excluded_short'] == expected['excluded']
            and pc['completed'] == expected['prior'], 'source COMMIT count mismatch')
    upstream = helper.base.read_json(new_root / 'upstream_bindings.json')
    helper.collect_upstream(upstream, bindings)
    original = Path(new['original_root'])
    helper.bind_receipts(original, new, bindings, 'new')
    helper.bind_receipts(prior_root, prior, bindings, 'prior')
    rows = helper.reconcile(read_json(freeze['plan']['path']), read_json(freeze['screen']['path']),
                            new, prior, expected={**helper.EXPECTED, **expected})
    require(rows == cohort['rows'], 'cohort rows do not replay from source graph')
    declared = cohort['bindings']
    for path, entry in bindings.items():
        require(declared.get(path) == entry, 'cohort omitted/changed source-graph binding: ' + path)
    require(cohort['upstream_commits'] == {
        'new': {'path': new_entry['path'], 'sha256': new_entry['sha256']},
        'prior': {'path': prior_entry['path'], 'sha256': prior_entry['sha256']},
    }, 'cohort upstream COMMIT declaration mismatch')
    inventories = cohort.get('upstream_inventories', {})
    require(set(inventories) == {str(new_root), str(prior_root), str(original)}, 'cohort upstream inventory roots')
    for root in (new_root, prior_root, original):
        require(inventories[str(root)] == sorted(helper.tree_files(root, ('writer.lock',))),
                'cohort upstream inventory changed: ' + str(root))
    require(all(str(original / name) in cohort['bindings'] for name in inventories[str(original)]),
            'unbound original partial product')
    return rows


def validate_rows(rows, expected=EXPECTED):
    require(isinstance(rows, list) and len(rows) == expected['total'], 'native30 exact row count')
    ids = [r['id'] for r in rows]
    require(ids == sorted(ids) and len(ids) == len(set(ids)), 'rows must be unique and ID-sorted')
    for row in rows:
        require(set(IDENTITY) <= set(row) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', row['id']), 'row identity')
        require(row['role'] == 'development' and row['label'] in {'0', '1'} and row['group_id'], 'row role/label/group')
        require(row['origin_family'] in {'new', 'prior'} and all(hash_string(row[k]) for k in
                ('origin_plan_row_sha256', 'screen_row_sha256', 'component_sha256', 'producer_receipt_sha256',
                 'waveform_float32_sha256')), 'row provenance hashes')
        require(set(row['input']) == {'path', 'bytes', 'sha256'}, 'row input binding schema')
        require(Path(row['input']['path']).name == row['id'] + '.wav', 'canonical input filename/ID mismatch')
        _binding_matches(row['input'], audio=False)
    origins = Counter(r['origin_family'] for r in rows)
    labels = Counter(r['label'] for r in rows)
    require({key: origins[key] for key in ('new', 'prior')} == {'new': expected['new'], 'prior': expected['prior']}, 'origin counts')
    require({key: labels[key] for key in ('0', '1')} == {'0': expected['human'], '1': expected['ai']}, 'label counts')
    require(dict(Counter(r['source_group'] for r in rows)) == expected['sources'], 'source counts')


def validate_shards(shards, rows):
    require(isinstance(shards, list) and shards, 'nonempty frozen shards required')
    ids = [r['id'] for r in rows]
    flattened = []
    for index, shard in enumerate(shards):
        require(shard.get('index') == index and isinstance(shard.get('item_ids'), list) and shard['item_ids'], 'contiguous shard indices')
        require(shard.get('item_ids_sha256') == value_hash(shard['item_ids']), 'shard ID hash mismatch')
        require(shard.get('rows') == len(shard['item_ids']), 'shard row count mismatch')
        flattened.extend(shard['item_ids'])
    require(flattened == ids, 'shards omit/duplicate/reorder cohort rows')


def validate_freeze(path, expected_sha, *, expected=EXPECTED, helper_loader=load_cohort_helper,
                    plan_sha=PLAN_SHA, screen_sha=SCREEN_SHA):
    path = safe_path(path)
    require(hash_string(expected_sha) and digest(path) == expected_sha, 'caller parent-freeze SHA mismatch')
    freeze = read_json(path)
    require(freeze.get('version') == FREEZE_VERSION and freeze.get('status') == FREEZE_STATUS, 'parent freeze schema/status')
    require(freeze.get('classifier_fits') == 0 and freeze.get('cohort_admitted') is False
            and freeze.get('feature_extraction_authorized') is False, 'freeze scope')
    require(freeze.get('duration_s') == 30 and freeze.get('input_format') ==
            {'format': 'WAV', 'subtype': 'FLOAT', 'sample_rate_hz': 44100, 'channels': 2, 'frames': 1323000}, 'frozen input format')
    for key in ('cohort_contract', 'cohort_helper', 'plan', 'screen'):
        _binding_matches(freeze[key], audio=False)
    require(freeze['runner'] == binding(Path(__file__).resolve())
            and freeze['tests'] == binding(Path(__file__).resolve().with_name('test_' + VERSION + '.py')), 'runner/test not parent-pinned')
    cohort_entry = freeze['cohort_contract']
    cohort = read_json(cohort_entry['path'])
    require(value_hash(cohort) == cohort_entry['sha256'], 'cohort contract must be canonical/hash bound')
    require(cohort.get('version') == 'run_native30_fhsc_cohort_v1'
            and cohort.get('status') == 'frozen_before_any_measurement_audio_reads'
            and cohort.get('expected_count') == expected['total'], 'prepared cohort contract schema/count')
    validate_rows(cohort['rows'], expected)
    require(cohort.get('input_format') == freeze['input_format'], 'cohort/freeze input format mismatch')
    require(cohort.get('source_counts') == expected['sources'] and cohort.get('human') == expected['human']
            and cohort.get('ai') == expected['ai'], 'cohort accounting')
    for row in cohort['rows']:
        require(cohort['bindings'].get(row['input']['path']) == row['input'], 'input absent from cohort binding graph')
    helper = helper_loader(freeze['cohort_helper'])
    verify_source_graph(freeze, cohort, helper, expected, plan_sha=plan_sha, screen_sha=screen_sha)
    validate_shards(freeze['shards'], cohort['rows'])
    output = safe_path(freeze['output_root'])
    require(output.parent.is_dir() and all(not output.is_relative_to(Path(x)) for x in
            (Path(freeze['upstream_commits']['new']['path']).parent,
             Path(freeze['upstream_commits']['prior']['path']).parent))
            and output != Path(cohort['output_root']), 'unsafe inference output overlap')
    require(not output.exists() or (output.is_dir() and not output.is_symlink()), 'unsafe inference output root')
    aio, beats = freeze['aio_gpus'], freeze['beat_gpus']
    require(aio and beats and len(aio) == len(set(aio)) and len(beats) == len(set(beats))
            and not (set(aio) & set(beats)) and all(type(x) is int and x >= 0 for x in aio + beats), 'disjoint GPU sets')
    return freeze, cohort


def require_hash(path, expected):
    require(digest(path) == expected, 'hash mismatch: ' + str(path))
    return expected


def verify_runtime(freeze):
    runtime = safe_path(freeze['runtime_root'])
    require(Path(sys.executable).resolve() == (runtime / 'venv/bin/python').resolve(), 'must use frozen runtime Python')
    runtime_contract = safe_path(freeze['runtime_contract']['path'])
    checkpoints_path = safe_path(freeze['checkpoint_manifest']['path'])
    old_code = safe_path(freeze['old_code_root'])
    bias = safe_path(freeze['bias']['path'])
    require(freeze['runtime_contract'] == binding(runtime_contract) and freeze['runtime_contract']['sha256'] == RUNTIME_CONTRACT_SHA,
            'runtime contract binding')
    require(freeze['checkpoint_manifest'] == binding(checkpoints_path) and freeze['checkpoint_manifest']['sha256'] == CHECKPOINT_MANIFEST_SHA,
            'checkpoint manifest binding')
    require(freeze['bias'] == binding(bias) and freeze['bias']['sha256'] == BIAS_SHA, 'bias binding')
    contract, manifest = read_json(runtime_contract), read_json(checkpoints_path)
    require(contract['checkpoint_manifest_sha256'] == CHECKPOINT_MANIFEST_SHA
            and contract['all_in_one']['model'] == 'harmonix-all'
            and contract['all_in_one']['demucs_model'] == 'htdemucs'
            and contract['all_in_one']['demucs_overlap'] == 0.25
            and contract['all_in_one']['demucs_fp16'] is False
            and contract['all_in_one']['demucs_shifts'] == 1
            and contract['beat_this']['dbn'] is False and contract['beat_this']['float16'] is True
            and contract['beat_this']['checkpoint_sha256'] == '8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331',
            'frozen inference recipe changed')
    checked = {}
    for row in manifest['checkpoints']:
        path = safe_path(row['path']); checked[str(path)] = require_hash(path, row['sha256'])
    for package, relative, key in (('allin1_infer', 'cli.py', 'allin1_cli_sha256'),
                                   ('allin1_infer', 'stems.py', 'allin1_stems_sha256'),
                                   ('demucs_infer', 'apply.py', 'demucs_apply_sha256')):
        spec = importlib.util.find_spec(package)
        require(spec and spec.origin, 'missing inference package: ' + package)
        path = Path(spec.origin).parent / relative
        checked[str(path)] = require_hash(path, contract['all_in_one'][key])
    for relative, expected in {**EXTRA_RUNTIME_CODE, **LAUNCHERS}.items():
        path = runtime / relative; checked[str(path)] = require_hash(path, expected)
    for name, expected in OLD_CODE.items():
        path = old_code / name; checked[str(path)] = require_hash(path, expected)
    checked[str(bias)] = require_hash(bias, BIAS_SHA)
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    require(versions == VERSIONS and sys.version_info[:3] == (3, 11, 15), 'inference runtime version mismatch')
    beat_repo = runtime / 'repos/beat_this'
    commit = subprocess.check_output(['git', '-C', str(beat_repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(beat_repo), 'status', '--porcelain'], text=True).strip()
    require(commit == contract['beat_this']['repository_commit'] == 'b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c'
            and not dirty, 'Beat This checkout revision/cleanliness')
    return {'checked_sha256': checked, 'versions': versions, 'beat_this_commit': commit,
            'input_decode': 'All-In-One SoundFile float32; Beat This decoded float64 then mono/soxr/float32',
            'stem_interpretation': 'established on-disk Demucs stems are PCM16; FLOAT is required only for cohort inputs'}


def inspect_audio(path, expected=None, *, input_audio=False, expected_subtype=None):
    path = safe_path(path); before = path.stat(); pcm_hash = hashlib.sha256()
    with sf.SoundFile(path) as stream:
        require((stream.format, stream.samplerate, stream.channels, stream.frames) == ('WAV', 44100, 2, 1323000),
                'exact native30 WAV/rate/channel/frame mismatch: ' + str(path))
        if input_audio:
            require(stream.subtype == 'FLOAT', 'cohort input must be WAV FLOAT')
        if expected_subtype is not None:
            require(stream.subtype == expected_subtype, 'audio subtype mismatch: ' + str(path))
        subtype = stream.subtype
        finite = True; frames = 0
        for block in stream.blocks(blocksize=65536, dtype='float32', always_2d=True):
            finite = finite and bool(np.isfinite(block).all()); frames += len(block)
            if input_audio:
                pcm_hash.update(np.ascontiguousarray(block, dtype='<f4').tobytes())
    require(finite and frames == 1323000, 'nonfinite/short audio: ' + str(path))
    result = binding(path)
    require((before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns), 'audio changed during inspection')
    if expected:
        require(result == expected['input'], 'input file binding changed')
        require(pcm_hash.hexdigest() == expected['waveform_float32_sha256'], 'input FLOAT PCM hash changed')
    return {**result, 'format': 'WAV', 'subtype': subtype, 'sample_rate_hz': 44100,
            'channels': 2, 'frames': frames, 'finite': True,
            **({'waveform_float32_sha256': pcm_hash.hexdigest()} if input_audio else {})}


def inspect_beats(path):
    path = safe_path(path)
    if path.stat().st_size == 0:
        return {**binding(path), 'status': 'empty_unavailable', 'beat_count': 0}
    data = np.loadtxt(path, ndmin=2)
    require(data.ndim == 2 and data.shape[1] == 2 and np.isfinite(data).all(), 'malformed beat data')
    require(not (np.diff(data[:, 0]) <= 0).any() and not (data[:, 0] < 0).any()
            and not (data[:, 0] > 30).any(), 'beat times unordered/outside native30')
    require(not (data[:, 1] < 1).any() and np.equal(data[:, 1], np.round(data[:, 1])).all(), 'invalid beat positions')
    return {**binding(path), 'status': 'nonempty', 'beat_count': len(data)}


def inspect_structure(path, audio_path):
    path = safe_path(path); payload = read_json(path)
    require(payload.get('path') == str(audio_path), 'structure input provenance mismatch')
    segments = payload.get('segments', [])
    require(isinstance(segments, list) and segments, 'empty structure output')
    previous = 0.0
    for segment in segments:
        start, end = float(segment['start']), float(segment['end'])
        require(np.isfinite([start, end]).all() and start == previous and end > start and end <= 30,
                'noncontiguous/out-of-context structure')
        previous = end
    require(abs(previous - 30) <= 1e-8, 'structure does not span native30')
    return {**binding(path), 'segment_count': len(segments), 'input_path': payload['path']}


def command(stage, inputs, runtime, output):
    if stage == 'allinone':
        return [str(runtime / 'venv/bin/all-in-one-infer'), *inputs, '-o', str(output / 'structure'),
                '-m', 'harmonix-all', '-d', 'cuda', '-k', '--demix-dir', str(output / 'demix'),
                '--spec-dir', str(output / 'spec')]
    require(stage == 'beats', 'unknown stage')
    return [str(runtime / 'venv/bin/beat_this'), *inputs, '-o', str(output / 'beats'), '--model',
            str(runtime / 'checkpoints/hub/checkpoints/beat_this-final0.ckpt'), '--no-dbn', '--gpu', '0',
            '--float16', '--skip-existing']


def product_paths(stage, ident, output):
    if stage == 'beats':
        return [output / 'beats' / (ident + '.beats')]
    return [*(output / 'demix/htdemucs' / ident / (stem + '.wav') for stem in STEMS),
            output / 'structure' / (ident + '.json'), output / 'spec' / (ident + '.npy')]


def verify_products(stage, rows, output):
    products = {}
    for row in rows:
        ident = row['id']
        if stage == 'beats':
            evidence = inspect_beats(output / 'beats' / (ident + '.beats'))
            products[evidence['path']] = evidence
        else:
            for stem in STEMS:
                evidence = inspect_audio(output / 'demix/htdemucs' / ident / (stem + '.wav'), expected_subtype='PCM_16')
                products[evidence['path']] = evidence
            evidence = inspect_structure(output / 'structure' / (ident + '.json'), row['input']['path'])
            products[evidence['path']] = evidence
            path = output / 'spec' / (ident + '.npy')
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            require(array.shape == (4, 3000, 81) and array.dtype == np.float32 and np.isfinite(array).all(),
                    'invalid native30 All-In-One spectrogram: ' + ident)
            products[str(path)] = {**binding(path), 'shape': [4, 3000, 81], 'dtype': 'float32', 'finite': True}
    return products


def require_idle_5090(gpu):
    data = subprocess.check_output(['nvidia-smi', '--id=' + str(gpu), '--query-gpu=name,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    name, memory, utilization = [x.strip() for x in data.strip().split(',')]
    require('RTX 5090' in name and int(memory) <= 1024 and int(utilization) <= 5,
            'refusing missing/busy RTX 5090: ' + data.strip())


def assignments(shards, gpus):
    return [(gpu, shards[index::len(gpus)]) for index, gpu in enumerate(gpus)]


def validate_receipt(receipt, run_sha, stage, index, ids):
    require(receipt.get('status') == 'passed' and receipt.get('run_contract_sha256') == run_sha
            and receipt.get('stage') == stage and receipt.get('shard_index') == index
            and receipt.get('item_ids') == ids, 'resume receipt identity mismatch')
    for path, expected in receipt['outputs'].items():
        require(binding(path) == {k: expected[k] for k in ('path', 'bytes', 'sha256')}, 'resume output changed')


def validate_output_inventory(output, rows, shards, *, completion=False):
    ids = {r['id'] for r in rows}
    roots = {'writer.lock', 'run_contract.json', 'structure', 'demix', 'spec', 'beats', 'receipts', 'logs'}
    if completion:
        roots.add('completion.json')
    require({p.name for p in output.iterdir()} == roots, 'inference root inventory mismatch')
    for path in output.rglob('*'):
        require(not path.is_symlink(), 'inference symlink forbidden')
    require({p.name for p in (output / 'structure').iterdir()} == {x + '.json' for x in ids}, 'structure inventory')
    require({p.name for p in (output / 'spec').iterdir()} == {x + '.npy' for x in ids}, 'spectrogram inventory')
    require({p.name for p in (output / 'beats').iterdir()} == {x + '.beats' for x in ids}, 'beat inventory')
    demix = output / 'demix'
    require({p.name for p in demix.iterdir()} == {'htdemucs'} and
            {p.name for p in (demix / 'htdemucs').iterdir()} == ids, 'demix item inventory')
    for ident in ids:
        require({p.name for p in (demix / 'htdemucs' / ident).iterdir()} == {x + '.wav' for x in STEMS}, 'stem inventory')
    receipt_names = {f'{stage}_{shard["index"]:03d}.json' for stage in ('allinone', 'beats') for shard in shards}
    log_names = {f'{stage}_{shard["index"]:03d}.log' for stage in ('allinone', 'beats') for shard in shards}
    require({p.name for p in (output / 'receipts').iterdir()} == receipt_names, 'receipt inventory')
    require({p.name for p in (output / 'logs').iterdir()} == log_names, 'log inventory')


def execute(freeze_path, freeze_sha, freeze, cohort, runtime_checks):
    output, runtime = safe_path(freeze['output_root']), safe_path(freeze['runtime_root'])
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / 'writer.lock'
    require(not lock_path.is_symlink(), 'writer lock symlink forbidden')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ('structure', 'demix', 'spec', 'beats', 'receipts', 'logs'):
            (output / name).mkdir(exist_ok=True)
            require((output / name).is_dir() and not (output / name).is_symlink(), 'unsafe inference directory: ' + name)
        rows = cohort['rows']; by_id = {r['id']: r for r in rows}
        for row in rows:
            inspect_audio(row['input']['path'], row, input_audio=True)
        run_contract = {'version': VERSION, 'status': 'parent_frozen_execution',
                        'parent_freeze': {'path': str(freeze_path), 'sha256': freeze_sha},
                        'cohort_contract': freeze['cohort_contract'], 'rows': len(rows),
                        'row_ids_sha256': value_hash([r['id'] for r in rows]), 'shards': freeze['shards'],
                        'aio_gpus': freeze['aio_gpus'], 'beat_gpus': freeze['beat_gpus'],
                        'output_root': str(output), 'runtime_root': str(runtime), 'runtime_checks': runtime_checks,
                        'commands': {stage: command(stage, ['<frozen shard inputs>'], runtime, output)
                                     for stage in ('allinone', 'beats')},
                        'resume_policy': 'verify completed receipts; retain and fail on unreceipted partial products',
                        'beat_serializer_recovery': 'not_in_v1; logged failure and partial products retained',
                        'classifier_fits': 0, 'cohort_admitted': False, 'feature_extraction_authorized': False}
        run_path = output / 'run_contract.json'
        if run_path.exists():
            require(read_json(run_path) == run_contract, 'existing run contract conflict')
        else:
            require({p.name for p in output.iterdir()} == {'writer.lock', 'structure', 'demix', 'spec', 'beats', 'receipts', 'logs'},
                    'nonempty inference root before run contract')
            write_new(run_path, run_contract)
        run_sha = digest(run_path)
        stopped = threading.Event()

        def worker(stage, gpu, assigned):
            for shard in assigned:
                if stopped.is_set():
                    return
                index, ids = shard['index'], shard['item_ids']
                selected = [by_id[x] for x in ids]
                receipt_path = output / 'receipts' / f'{stage}_{index:03d}.json'
                try:
                    require(digest(freeze_path) == freeze_sha, 'parent freeze changed during inference')
                    if receipt_path.exists():
                        validate_receipt(read_json(receipt_path), run_sha, stage, index, ids)
                        continue
                    paths = [p for ident in ids for p in product_paths(stage, ident, output)]
                    require(not any(p.exists() or p.is_symlink() for p in paths), 'unreceipted partial stage products')
                    before = {r['id']: inspect_audio(r['input']['path'], r, input_audio=True) for r in selected}
                    require_idle_5090(gpu)  # immediately before the shard subprocess
                    cmd = command(stage, [r['input']['path'] for r in selected], runtime, output)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
                    log_path = output / 'logs' / f'{stage}_{index:03d}.log'
                    require(not log_path.exists() and not log_path.is_symlink(), 'unreceipted stage log retained')
                    started = now()
                    with log_path.open('x') as log:
                        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    products = verify_products(stage, selected, output)
                    after = {r['id']: inspect_audio(r['input']['path'], r, input_audio=True) for r in selected}
                    require(before == after, 'cohort input changed during inference')
                    payload = {'status': 'passed', 'stage': stage, 'shard_index': index, 'item_ids': ids,
                               'run_contract_sha256': run_sha, 'gpu': gpu, 'command': cmd, 'started_utc': started,
                               'completed_utc': now(), 'inputs': before, 'outputs': products,
                               'log': binding(log_path), 'ordinary_empty_beat_allowed_only_after_this_successful_command': True,
                               'recovery_applied': False}
                    write_new(receipt_path, payload)
                except Exception:
                    stopped.set()
                    raise

        work = [(stage, gpu, assigned) for stage, gpus in (('allinone', freeze['aio_gpus']), ('beats', freeze['beat_gpus']))
                for gpu, assigned in assignments(freeze['shards'], gpus)]
        with ThreadPoolExecutor(max_workers=len(work)) as pool:
            futures = [pool.submit(worker, *job) for job in work]
            for future in futures:
                future.result()
        for row in rows:
            inspect_audio(row['input']['path'], row, input_audio=True)
        require(digest(freeze_path) == freeze_sha, 'parent freeze changed during inference')
        receipts = list((output / 'receipts').glob('*.json'))
        require(len(receipts) == 2 * len(freeze['shards']), 'incomplete/extra stage receipt count')
        validate_output_inventory(output, rows, freeze['shards'], completion=(output / 'completion.json').exists())
        completion = {'status': 'passed_native30_inference_not_feature_extraction', 'rows': len(rows),
                      'run_contract_sha256': run_sha, 'completed_stage_shards': len(receipts),
                      'all_inputs_end_rehashed': True, 'classifier_fits': 0, 'cohort_admitted': False}
        completion_path = output / 'completion.json'
        if completion_path.exists():
            require(read_json(completion_path) == completion, 'existing completion conflict')
        else:
            write_new(completion_path, completion)
        validate_output_inventory(output, rows, freeze['shards'], completion=True)
        return {**completion, 'completion_sha256': digest(output / 'completion.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--frozen-sha256', required=True)
    parser.add_argument('--mode', choices=('preflight', 'run'), default='preflight')
    args = parser.parse_args()
    freeze, cohort = validate_freeze(args.frozen, args.frozen_sha256)
    runtime_checks = verify_runtime(freeze)
    if args.mode == 'preflight':
        result = {'status': 'metadata_preflight_only_no_audio_reads_or_writes', 'parent_freeze_sha256': args.frozen_sha256,
                  'rows': len(cohort['rows']), 'shards': len(freeze['shards']), 'runtime_checks': runtime_checks,
                  'classifier_fits': 0, 'cohort_admitted': False, 'feature_extraction_authorized': False}
    else:
        result = execute(args.frozen.resolve(), args.frozen_sha256, freeze, cohort, runtime_checks)
    print(canonical(result).decode().strip(), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
