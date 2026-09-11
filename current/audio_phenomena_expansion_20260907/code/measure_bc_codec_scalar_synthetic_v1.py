#!/usr/bin/env python3
"""Measure frozen BC extractor/scalar on the committed synthetic codec package.

This runner accepts no arbitrary audio paths.  It reads exactly the four
original and eight decoded 4 s FLOAT WAVs committed by
probe_bc_codec_roundtrip_v1, converts float32 to float64 without changing sample
values, and applies the unchanged bicoherence_audio_v1 and
bicoherence_scalar_v1.  It does not crop, pad, align, normalize, select a grid
cell, set a threshold, or authorize BC admission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import uuid

import numpy as np


VERSION = 'measure_bc_codec_scalar_synthetic_v1'
STATUS = 'committed_synthetic_BC_codec_scalar_measurements_not_admission'
SOURCE_COMMIT_SHA = '9e07622b5c45179e5086d66f2c6697c4a76bbe7e3b5b39c82ecaea0cb936909b'
PINS = {
    'probe_bc_codec_roundtrip_v1.py': 'dbf1f548356046568329574ecbd57ddf99aab3164d666e0fa0b02b563d724696',
    'bicoherence_audio_v1.py': 'e59fd575add722ca10280f5e19b64d1abe6a1587dac6b0a790fb9d57401f64a1',
    'bicoherence_scalar_v1.py': 'b648653830e182dcbaaf1fbf3518f48810fd4b63beaa1136cba97ea538100f85',
    'bicoherence_primitive_v2.py': '9cedc47b0ee42775c996bbf3e9c8eb237aa1acde4a051634b077fd72fd7614d5',
}
SIGNALS = ('silence', 'impulse', 'tone_997hz', 'multitone_500_750_1250hz')
CODECS = ('mp3_128k', 'opus_96k')
SCOPE = {'synthetic_codec_package_only': True, 'development_audio_read': False,
         'reserved_audio_read': False, 'waveform_crop': False, 'waveform_padding': False,
         'waveform_alignment': False, 'gain_changed': False, 'normalization': False,
         'grid_selection': False, 'thresholds_chosen': False, 'BC_admitted': False,
         'classifier_fits': 0, 'model_scoring': False, 'causality_inferred': False,
         'music_codec_invariance_established': False}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path, expected=None):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink(),
            'absolute regular file required: ' + str(path))
    result = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
    require(expected is None or result['sha256'] == expected, 'bound file SHA mismatch: ' + str(path))
    return result


def write_new(path, payload):
    path = Path(path); temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    write_new(path, canonical(value))


def save_npz(path, arrays):
    path = Path(path); temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def modules():
    here = Path(__file__).resolve().parent
    loaded = {}
    for filename, expected in PINS.items():
        binding(here / filename, expected)
        name = Path(filename).stem
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve() == here / filename, 'module origin changed: ' + name)
        loaded[name] = module
    require(loaded['bicoherence_audio_v1'].VERSION == 'bicoherence_audio_v1'
            and loaded['bicoherence_scalar_v1'].VERSION == 'bicoherence_scalar_v1'
            and loaded['bicoherence_primitive_v2'].VERSION == 'bicoherence_primitive_v2',
            'BC module version changed')
    return loaded


def validate_source(root, commit_sha=SOURCE_COMMIT_SHA):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir() and not root.is_symlink(),
            'safe codec source root required')
    require(commit_sha == SOURCE_COMMIT_SHA, 'exact committed codec source required')
    loaded = modules(); codec = loaded['probe_bc_codec_roundtrip_v1']
    proof = codec.verify_result(root, commit_sha)
    results = proof['results']
    require(results.get('sample_rate_hz') == 16000 and results.get('channels') == 1
            and results.get('input_subtype') == results.get('decoded_subtype') == 'FLOAT'
            and results.get('samples_per_fixture') == 64000
            and results.get('synthetic_only') is True and results.get('BC_measured') is False,
            'codec source scope/format changed')
    rows = {(row['signal'], row['codec']): row for row in results['rows']}
    require(list(rows) == [(signal, codec_name) for signal in SIGNALS for codec_name in CODECS],
            'codec source row order/roster')
    entries = []
    for signal in SIGNALS:
        originals = [rows[(signal, codec_name)]['input'] for codec_name in CODECS]
        require(originals[0] == originals[1]
                and originals[0]['path'] == str(root / 'synthetic' / (signal + '.wav')),
                'original fixture binding mismatch')
        entries.append({'id': 'original__' + signal, 'kind': 'original', 'signal': signal,
                        'codec': None, 'input': originals[0], 'source_codec_row_sha256': None})
        for codec_name in CODECS:
            row = rows[(signal, codec_name)]
            require(row['decoded']['path'] == str(root / 'decoded' / (signal + '__' + codec_name + '.wav')),
                    'decoded fixture binding mismatch')
            entries.append({'id': 'decoded__' + signal + '__' + codec_name, 'kind': 'decoded',
                            'signal': signal, 'codec': codec_name, 'input': row['decoded'],
                            'source_codec_row_sha256': value_hash(row)})
    require(len(entries) == 12 and len({entry['id'] for entry in entries}) == 12, 'exact 12 measurement inputs')
    return loaded, proof, entries


def array_hash(array):
    array = np.asarray(array)
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def arrays_manifest(arrays):
    return {name: {'dtype': array.dtype.str, 'shape': list(array.shape),
                   'c_order_bytes_sha256': array_hash(array)}
            for name, array in sorted(arrays.items())}


def measure_entry(entry, output, codec, extractor, scalar):
    input_entry = binding(entry['input']['path'])
    require(input_entry == entry['input'], 'measurement input binding changed')
    float32 = codec.read_float_wav(input_entry['path'])
    require(float32.dtype == np.float32 and float32.shape == (64000,)
            and np.isfinite(float32).all(), 'exact 16k four-second FLOAT waveform required')
    float64 = float32.astype(np.float64)
    float32_sha = hashlib.sha256(float32.astype('<f4', copy=False).tobytes()).hexdigest()
    float64_sha = hashlib.sha256(float64.astype('<f8', copy=False).tobytes()).hexdigest()
    base = {**entry, 'input': input_entry, 'sample_rate_hz': 16000, 'samples': 64000,
            'input_float32_pcm_sha256': float32_sha, 'extractor_float64_pcm_sha256': float64_sha,
            'float32_to_float64_value_preserving': bool(np.array_equal(float64.astype(np.float32), float32)),
            **SCOPE}
    try:
        extracted = extractor.extract(float64, 16000)
        crop = {'start_sample': 0, 'stop_sample_exclusive': 64000,
                'source_resampled_samples': 64000}
        reduced = scalar.reduce_metadata(extracted['metadata'], crop=crop)
        array_path = output / 'arrays' / (entry['id'] + '.npz')
        save_npz(array_path, extracted['arrays'])
        record = {**base, 'status': 'measured_one_exact_pool_not_admitted',
                  'raw_metadata': extracted['metadata'], 'scalar': reduced,
                  'arrays': binding(array_path), 'arrays_manifest': arrays_manifest(extracted['arrays']),
                  'processing_failure': None}
    except Exception as error:
        record = {**base, 'status': 'processing_failure_retained_not_scientific_missingness',
                  'raw_metadata': None, 'scalar': None, 'arrays': None, 'arrays_manifest': None,
                  'processing_failure': {'type': type(error).__name__, 'message': str(error)}}
    write_json(output / 'measurements' / (entry['id'] + '.json'), record)
    return record


def compare_pair(original, decoded):
    base = {'signal': original['signal'], 'codec': decoded['codec'],
            'original_id': original['id'], 'decoded_id': decoded['id']}
    if original['processing_failure'] or decoded['processing_failure']:
        return {**base, 'status': 'processing_failure_retained',
                'original_eligibility_mask': None, 'decoded_eligibility_mask': None,
                'mask_agreement': None, 'transition': None,
                'original_scalar': None, 'decoded_scalar': None,
                'scalar_delta_decoded_minus_original': None}
    first, second = original['scalar'], decoded['scalar']
    first_mask, second_mask = first['eligibility_mask'], second['eligibility_mask']
    require(len(first_mask) == len(second_mask) == 1, 'one exact pool required')
    if first_mask[0] and second_mask[0]:
        transition = 'eligible_to_eligible'
    elif not first_mask[0] and not second_mask[0]:
        transition = 'missing_to_missing'
    elif first_mask[0]:
        transition = 'eligible_to_missing'
    else:
        transition = 'missing_to_eligible'
    a, b = first['median_squared_bicoherence'], second['median_squared_bicoherence']
    delta = b - a if a is not None and b is not None else None
    require(delta is None or math.isfinite(delta), 'nonfinite scalar delta')
    return {**base, 'status': 'compared_without_threshold_or_admission',
            'original_eligibility_mask': first_mask, 'decoded_eligibility_mask': second_mask,
            'mask_agreement': first_mask == second_mask, 'transition': transition,
            'original_scalar': a, 'decoded_scalar': b,
            'scalar_delta_decoded_minus_original': delta}


def products(root):
    return {path.relative_to(root).as_posix(): {'bytes': path.stat().st_size, 'sha256': digest(path)}
            for path in sorted(root.rglob('*')) if path.is_file() and path.name != 'COMMIT.json'}


def run(source_root, output):
    source_root, output = Path(source_root), Path(output)
    require(output.is_absolute() and output.resolve() == output and not output.exists()
            and output.parent.is_dir() and not (output.is_relative_to(source_root)
            or source_root.is_relative_to(output)), 'fresh nonoverlapping output required')
    loaded, source_before, entries = validate_source(source_root)
    output.mkdir(); (output / 'measurements').mkdir(); (output / 'arrays').mkdir()
    records = [measure_entry(entry, output, loaded['probe_bc_codec_roundtrip_v1'],
                             loaded['bicoherence_audio_v1'], loaded['bicoherence_scalar_v1'])
               for entry in entries]
    indexed = {record['id']: record for record in records}
    comparisons = [compare_pair(indexed['original__' + signal],
                                indexed['decoded__' + signal + '__' + codec_name])
                   for signal in SIGNALS for codec_name in CODECS]
    summary = {'version': VERSION, 'status': 'synthetic_BC_codec_scalar_measurements_complete_not_admission',
               'source_codec_COMMIT': source_before['commit'], 'measurement_count': len(records),
               'processing_failure_count': sum(record['processing_failure'] is not None for record in records),
               'scientific_null_count': sum(record['scalar'] is not None
                                            and record['scalar']['median_squared_bicoherence'] is None
                                            for record in records),
               'measurement_receipts': {record['id']: binding(output / 'measurements' / (record['id'] + '.json'))
                                        for record in records},
               'comparisons': comparisons,
               'interpretation_limits': ['stationary harmonic multitone can have squared bicoherence near 1 without nonlinear causation',
                                         'single-pool synthetic results do not establish music or codec invariance'],
               **SCOPE}
    write_json(output / 'summary.json', summary)
    source_end = loaded['probe_bc_codec_roundtrip_v1'].verify_result(source_root, SOURCE_COMMIT_SHA)
    require(source_end['commit'] == source_before['commit']
            and source_end['products'] == source_before['products'], 'codec source changed during BC measurement')
    here = Path(__file__).resolve().parent
    bindings = {'runner': binding(Path(__file__).resolve()),
                'tests': binding(here / ('test_' + VERSION + '.py')),
                'source_codec_COMMIT': source_end['commit'],
                **{Path(name).stem: binding(here / name, sha) for name, sha in PINS.items()}}
    inventory = products(output)
    commit = {'version': VERSION, 'status': STATUS, 'products': inventory,
              'products_sha256': value_hash(inventory), 'summary_sha256': value_hash(summary),
              'bindings': bindings, 'source_products_sha256': value_hash(source_end['products']),
              'runtime': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__},
              'measurement_count': len(records), 'comparison_count': len(comparisons),
              'processing_failure_count': summary['processing_failure_count'], **SCOPE}
    write_json(output / 'COMMIT.json', commit)
    return verify_result(output, digest(output / 'COMMIT.json'))


def verify_npz(path, manifest, expected_arrays):
    require(binding(path) is not None, 'array product missing')
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(expected_arrays) == set(manifest), 'array inventory changed')
        for name, expected in expected_arrays.items():
            actual = saved[name]
            require(actual.dtype == expected.dtype and actual.shape == expected.shape
                    and np.array_equal(actual, expected, equal_nan=True)
                    and manifest[name] == {'dtype': expected.dtype.str, 'shape': list(expected.shape),
                                           'c_order_bytes_sha256': array_hash(expected)},
                    'array contents changed: ' + name)


def verify_result(root, commit_sha):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir() and not root.is_symlink(),
            'safe result root required')
    commit_entry = binding(root / 'COMMIT.json', commit_sha); commit = json.loads(Path(commit_entry['path']).read_text())
    require(value_hash(commit) == commit_sha and commit.get('version') == VERSION
            and commit.get('status') == STATUS and all(commit.get(key) == value for key, value in SCOPE.items()),
            'result COMMIT schema/scope')
    require(commit.get('runtime') == {'python': sys.version, 'platform': platform.platform(),
                                      'numpy': np.__version__}, 'measurement runtime changed')
    inventory = products(root)
    require(commit.get('products') == inventory and commit.get('products_sha256') == value_hash(inventory),
            'result product inventory/hash changed')
    for entry in commit['bindings'].values():
        require(binding(entry['path']) == entry, 'bound authority changed')
    source_root = Path(commit['bindings']['source_codec_COMMIT']['path']).parent
    loaded, source, entries = validate_source(source_root)
    require(value_hash(source['products']) == commit.get('source_products_sha256'), 'source product graph changed')
    summary = json.loads((root / 'summary.json').read_text())
    limits = ['stationary harmonic multitone can have squared bicoherence near 1 without nonlinear causation',
              'single-pool synthetic results do not establish music or codec invariance']
    require(value_hash(summary) == commit.get('summary_sha256')
            and summary.get('status') == 'synthetic_BC_codec_scalar_measurements_complete_not_admission'
            and summary.get('source_codec_COMMIT') == source['commit']
            and summary.get('interpretation_limits') == limits
            and all(summary.get(k) == v for k, v in SCOPE.items()),
            'summary hash/scope')
    records = []
    for entry in entries:
        path = root / 'measurements' / (entry['id'] + '.json')
        record = json.loads(path.read_text())
        require(summary['measurement_receipts'][entry['id']] == binding(path)
                and all(record.get(key) == value for key, value in SCOPE.items())
                and all(record.get(key) == value for key, value in entry.items()),
                'measurement receipt lineage/scope')
        float32 = loaded['probe_bc_codec_roundtrip_v1'].read_float_wav(entry['input']['path'])
        samples = float32.astype(np.float64)
        require(record.get('sample_rate_hz') == 16000 and record.get('samples') == 64000
                and record.get('input_float32_pcm_sha256') == hashlib.sha256(
                    float32.astype('<f4', copy=False).tobytes()).hexdigest()
                and record.get('extractor_float64_pcm_sha256') == hashlib.sha256(
                    samples.astype('<f8', copy=False).tobytes()).hexdigest()
                and record.get('float32_to_float64_value_preserving') is True,
                'measurement waveform conversion lineage')
        try:
            extracted = loaded['bicoherence_audio_v1'].extract(samples, 16000)
            reduced = loaded['bicoherence_scalar_v1'].reduce_metadata(
                extracted['metadata'], crop={'start_sample': 0, 'stop_sample_exclusive': 64000,
                                             'source_resampled_samples': 64000})
        except Exception as error:
            require(record['status'] == 'processing_failure_retained_not_scientific_missingness'
                    and record['processing_failure'] == {'type': type(error).__name__, 'message': str(error)}
                    and record['raw_metadata'] is record['scalar'] is record['arrays'] is record['arrays_manifest'] is None
                    and not (root / 'arrays' / (entry['id'] + '.npz')).exists(),
                    'processing failure replay changed')
        else:
            require(record['status'] == 'measured_one_exact_pool_not_admitted'
                    and record['processing_failure'] is None
                    and record['raw_metadata'] == extracted['metadata'] and record['scalar'] == reduced
                    and record['arrays'] == binding(root / 'arrays' / (entry['id'] + '.npz')),
                    'measurement replay changed')
            verify_npz(record['arrays']['path'], record['arrays_manifest'], extracted['arrays'])
        records.append(record)
    indexed = {record['id']: record for record in records}
    comparisons = [compare_pair(indexed['original__' + signal],
                                indexed['decoded__' + signal + '__' + codec_name])
                   for signal in SIGNALS for codec_name in CODECS]
    require(summary['comparisons'] == comparisons and commit.get('measurement_count') == len(records) == 12
            and commit.get('comparison_count') == len(comparisons) == 8
            and set(summary.get('measurement_receipts', {})) == {record['id'] for record in records}
            and summary.get('processing_failure_count') == sum(record['processing_failure'] is not None for record in records)
            and summary.get('scientific_null_count') == sum(record['scalar'] is not None
                and record['scalar']['median_squared_bicoherence'] is None for record in records)
            and commit.get('processing_failure_count') == summary['processing_failure_count'],
            'comparison/count replay changed')
    return {'commit': commit_entry, 'summary': summary, 'measurements': records, 'products': inventory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codec-root', required=True)
    parser.add_argument('--codec-commit-sha256', default=SOURCE_COMMIT_SHA)
    parser.add_argument('--mode', choices=('preflight', 'run'), default='preflight')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    _, proof, entries = validate_source(Path(args.codec_root), args.codec_commit_sha256)
    if args.mode == 'preflight':
        result = {'version': VERSION, 'status': 'synthetic_source_preflight_no_BC_measured',
                  'source_codec_COMMIT': proof['commit'], 'measurement_inputs': len(entries), **SCOPE}
    else:
        require(args.output is not None, '--output required for run')
        verified = run(Path(args.codec_root), args.output.resolve())
        result = {'version': VERSION, 'status': STATUS, 'commit': verified['commit'],
                  'measurement_count': len(verified['measurements']), **SCOPE}
    print(canonical(result).decode().strip())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
