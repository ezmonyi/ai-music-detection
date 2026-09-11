#!/usr/bin/env python3
"""Prospective GuitarSet codec development runner.

Draft/preflight reads accepted JSON metadata only. Audio and array products are
never opened before an exact, separately supplied parent-freeze receipt is
verified. The run is development-only and never admits BC or chooses a gate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import traceback

import numpy as np


VERSION = 'run_bc_guitarset_codec_development_v1'
DRAFT_VERSION = 'bc_guitarset_codec_development_draft_v1'
FREEZE_VERSION = 'bc_guitarset_codec_development_parent_freeze_v1'
SOURCE_COMMIT_SHA = 'cd11c3fb80755260b3908c542f8b8c9c218320c931c4987215936e11b73d639c'
AUDIT_SHA = '61a926a92c764d6852608f9829395382e6593e56544d1434ec0f28af509ffc93'
CODEC_COMMIT_SHA = '9e07622b5c45179e5086d66f2c6697c4a76bbe7e3b5b39c82ecaea0cb936909b'
PINS = {
    'source_producer': 'dcb8bfd41b89e3ab59bd241214ab64e36dbb7a5264409898bf3e423a03de8d6e',
    'source_auditor': '843dbda8922f942c038e05d9397f4314006dea5f77d44de1676efcd8add63e60',
    'bicoherence_audio_v1': 'e59fd575add722ca10280f5e19b64d1abe6a1587dac6b0a790fb9d57401f64a1',
    'bicoherence_scalar_v1': 'b648653830e182dcbaaf1fbf3518f48810fd4b63beaa1136cba97ea538100f85',
    'bicoherence_primitive_v2': '9cedc47b0ee42775c996bbf3e9c8eb237aa1acde4a051634b077fd72fd7614d5',
    'codec_probe': 'dbf1f548356046568329574ecbd57ddf99aab3164d666e0fa0b02b563d724696',
}
RATE = 16000
SAMPLES = 128000
POOLS = 2
DEVELOPMENT_RECORDINGS = 90
SELECTED_CONDITIONS = ('baseline', 'closed_minus6db', 'independent_minus6db')
ALL_SOURCE_CONDITIONS = ('baseline', 'common_gain', 'polarity', 'closed_minus6db',
                         'independent_minus6db', 'closed_0db', 'independent_0db')
REPRESENTATIONS = ('accepted_float64', 'float32_control', 'mp3_128k', 'opus_96k')
CODECS = ('mp3_128k', 'opus_96k')
SCOPE = {
    'development_recordings': 90, 'reserved_recordings_read': False,
    'unused_recordings_read': False, 'reserved_BC_measured': False,
    'unused_BC_measured': False, 'classifier_fits': 0, 'model_scoring': False,
    'grid_selection': False, 'thresholds_chosen': False, 'BC_admitted': False,
    'automatic_alignment': False, 'automatic_trim': False,
    'automatic_padding': False, 'gain_changed': False, 'normalization': False,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def value_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_file(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'safe regular file required')
    return path


def binding(path, expected=None):
    path = safe_file(path)
    result = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
    require(expected is None or result['sha256'] == expected, 'SHA-256 mismatch: ' + str(path))
    return result


def declared_product(root, relative, product):
    """Bind an accepted product without reading its bytes (draft boundary)."""
    require(set(product) == {'bytes', 'sha256'} and isinstance(product['bytes'], int)
            and len(product['sha256']) == 64, 'invalid accepted product binding')
    path = root / relative
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and path.stat().st_size == product['bytes'], 'accepted product stat mismatch: ' + relative)
    return {'path': str(path), **product}


def read_json(path, expected=None):
    entry = binding(path, expected)
    return json.loads(Path(entry['path']).read_text(encoding='utf-8')), entry


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def write_json(path, value):
    write_new(path, canonical(value) + b'\n')


def module_from(path, name, expected):
    path = safe_file(path)
    binding(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'module spec unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == path.resolve(), 'module origin changed')
    return module


def load_frozen(code_root):
    code_root = Path(code_root).resolve()
    modules = {}
    for name in ('bicoherence_primitive_v2', 'bicoherence_audio_v1', 'bicoherence_scalar_v1'):
        modules[name] = module_from(code_root / (name + '.py'), '_codec_dev_' + name, PINS[name])
        sys.modules[name] = modules[name]
    modules['codec_probe'] = module_from(code_root / 'probe_bc_codec_roundtrip_v1.py',
                                         '_codec_dev_probe', PINS['codec_probe'])
    return modules


def expected_source_names(item_ids):
    names = {'frozen_draft.json', 'parent_freeze_receipt.json', 'storage_estimate.json',
             'summary.json', 'verification.json'}
    for item_id in item_ids:
        names |= {f'{item_id}/construction.json', f'{item_id}/construction.npz',
                  f'{item_id}/summary.json'}
        for condition in ALL_SOURCE_CONDITIONS:
            names |= {f'{item_id}/{condition}.wav', f'{item_id}/{condition}.json',
                      f'{item_id}/{condition}.npz'}
    return names


def reduction_signature(reduction):
    require(reduction['input_samples'] == SAMPLES and reduction['pool_count'] == POOLS
            and reduction['discarded_tail_samples'] == 0
            and len(reduction['eligibility_mask']) == POOLS, 'source scalar support mismatch')
    return reduction


def build_roster(summary, products, raw_root, scalar, *, expected_count=DEVELOPMENT_RECORDINGS,
                 metadata_loader=None):
    """Construct only selected development rows; WAV/NPZ are statted, never read."""
    require(summary.get('version') == 'bicoherence_guitarset_development_pilot_v1'
            and summary.get('recording_denominator') == expected_count, 'accepted summary identity/count')
    records = summary.get('per_recording')
    require(isinstance(records, list) and len(records) == expected_count, 'development record count')
    ids = [row.get('item_id') for row in records]
    require(ids == sorted(ids) and len(set(ids)) == expected_count, 'development IDs order/uniqueness')
    require(set(products) == expected_source_names(ids), 'full frozen seven-condition inventory mismatch')
    load = metadata_loader or (lambda p, sha: read_json(p, sha)[0])
    roster = []
    for row in records:
        require(row.get('split_role') == 'development' and row.get('performance') in ('comp', 'solo')
                and set(row.get('conditions', {})) == set(ALL_SOURCE_CONDITIONS),
                'development identity/source conditions mismatch')
        identity = {key: row[key] for key in
                    ('item_id', 'player_id', 'score_id', 'performance', 'style_from_score_prefix',
                     'split_role')}
        for condition in SELECTED_CONDITIONS:
            prefix = f'{row["item_id"]}/{condition}'
            wav = declared_product(raw_root, prefix + '.wav', products[prefix + '.wav'])
            arrays = declared_product(raw_root, prefix + '.npz', products[prefix + '.npz'])
            meta_entry = declared_product(raw_root, prefix + '.json', products[prefix + '.json'])
            metadata = load(Path(meta_entry['path']), meta_entry['sha256'])
            reduced = reduction_signature(scalar.reduce_metadata(metadata, crop={
                'start_sample': 0, 'stop_sample_exclusive': SAMPLES,
                'source_resampled_samples': SAMPLES}))
            roster.append(identity | {'condition': condition, 'source_float64_wav': wav,
                          'source_metadata': meta_entry, 'source_arrays': arrays,
                          'accepted_float64_reduction': reduced})
    require(len(roster) == expected_count * len(SELECTED_CONDITIONS), 'selected condition denominator')
    return roster


def runtime_snapshot():
    import soundfile as sf
    return {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
            'soundfile': sf.__version__, 'libsndfile': sf.__libsndfile_version__,
            'executable': binding(Path(sys.executable).resolve()),
            'numpy_module': binding(Path(np.__file__).resolve()),
            'soundfile_module': binding(Path(sf.__file__).resolve())}


def validate_authorities(raw_root, rc_root, codec_root, output_root,
                         *, expected_count=DEVELOPMENT_RECORDINGS, metadata_loader=None):
    raw_root, rc_root, codec_root, output_root = map(lambda p: Path(p).resolve(),
                                                     (raw_root, rc_root, codec_root, output_root))
    require(raw_root.is_dir() and rc_root.is_dir() and codec_root.is_dir(), 'authority root missing')
    require(not output_root.exists() and all(output_root != p and p not in output_root.parents
            and output_root not in p.parents for p in (raw_root, rc_root, codec_root)),
            'new exclusive non-overlapping output required')
    modules = load_frozen(rc_root / 'code')
    source_commit, source_commit_binding = read_json(raw_root / 'COMMIT.json', SOURCE_COMMIT_SHA)
    mirror_commit, mirror_binding = read_json(rc_root / 'audit/guitarset_bc_producer_COMMIT_v1.json',
                                              SOURCE_COMMIT_SHA)
    require(source_commit == mirror_commit and source_commit.get('status') == 'committed'
            and len(source_commit.get('products', {})) == 2165, 'accepted source COMMIT mismatch')
    summary, summary_binding = read_json(raw_root / 'summary.json',
                                         source_commit['products']['summary.json']['sha256'])
    mirror_summary, mirror_summary_binding = read_json(rc_root / 'audit/guitarset_bc_summary_v1.json',
                                                       source_commit['products']['summary.json']['sha256'])
    require(summary == mirror_summary, 'accepted summary mirror mismatch')
    audit, audit_binding = read_json(rc_root / 'audit/guitarset_bc_independent_replay_v1.json', AUDIT_SHA)
    require(audit.get('passed') is True and audit.get('result_commit_sha256') == SOURCE_COMMIT_SHA
            and audit.get('recordings') == 90 and audit.get('reserved_or_unused_decoded_or_measured') is False
            and audit.get('classifier_fits') == 0, 'accepted independent audit scope mismatch')
    codec_proof = modules['codec_probe'].verify_result(codec_root, CODEC_COMMIT_SHA)
    roster = build_roster(summary, source_commit['products'], raw_root,
                          modules['bicoherence_scalar_v1'], expected_count=expected_count,
                          metadata_loader=metadata_loader)
    bindings = {
        'accepted_raw_COMMIT': source_commit_binding, 'accepted_COMMIT_mirror': mirror_binding,
        'accepted_summary': summary_binding, 'accepted_summary_mirror': mirror_summary_binding,
        'accepted_independent_audit': audit_binding,
        'source_producer': binding(rc_root / 'code/bicoherence_guitarset_pilot_v1.py', PINS['source_producer']),
        'source_auditor': binding(rc_root / 'code/audit_bicoherence_guitarset_pilot_v1.py', PINS['source_auditor']),
        'bicoherence_audio_v1': binding(rc_root / 'code/bicoherence_audio_v1.py', PINS['bicoherence_audio_v1']),
        'bicoherence_scalar_v1': binding(rc_root / 'code/bicoherence_scalar_v1.py', PINS['bicoherence_scalar_v1']),
        'bicoherence_primitive_v2': binding(rc_root / 'code/bicoherence_primitive_v2.py', PINS['bicoherence_primitive_v2']),
        'codec_probe': binding(rc_root / 'code/probe_bc_codec_roundtrip_v1.py', PINS['codec_probe']),
        'synthetic_codec_COMMIT': codec_proof['commit'],
    }
    return {'raw_root': str(raw_root), 'rc_root': str(rc_root), 'codec_root': str(codec_root),
            'output_root': str(output_root), 'source_commit': source_commit,
            'source_products_sha256': value_hash(source_commit['products']), 'roster': roster,
            'bindings': bindings, 'codec_results': codec_proof['results'],
            'runtime': runtime_snapshot(), 'modules': modules}


def draft_document(authorities, runner, tests, protocol):
    bindings = dict(authorities['bindings'])
    bindings.update(runner=binding(runner), tests=binding(tests), protocol=binding(protocol))
    return {
        'version': DRAFT_VERSION,
        'status': 'frozen_development_codec_design_not_authorized_for_execution_or_admission',
        **SCOPE, 'metadata_only_development_draft': True, 'development_audio_opened': False,
        'synthetic_codec_package_replayed': True,
        'selected_conditions': list(SELECTED_CONDITIONS),
        'preserved_source_conditions': list(ALL_SOURCE_CONDITIONS),
        'representations': list(REPRESENTATIONS),
        'denominators': {'recordings': 90, 'selected_source_conditions': 270,
                         'precision_controls': 270, 'codec_encodes': 540,
                         'codec_decodes': 540, 'expected_new_BC_measurements': 810},
        'operation_order': [
            'verify accepted full seven-condition source inventory and independent audit',
            'open only the 270 accepted FLOAT64 development WAVs after parent freeze',
            'retain accepted FLOAT64 scalar metadata; cast every sample once to FLOAT32',
            'save and measure the exact FLOAT32 precision control without gain or normalization',
            'encode that complete FLOAT32 control with each unchanged frozen codec recipe',
            'decode to 16000 Hz mono FLOAT32 and require exactly 128000 samples',
            'measure unchanged fixed-target BC without trim, pad, alignment, or gain',
            'separate FLOAT32-minus-FLOAT64 precision effects from decoded-minus-FLOAT32 codec effects'],
        'policy': {'sample_rate_hz': RATE, 'samples': SAMPLES, 'pools': POOLS,
                   'target_frequency_hz': [500, 750, 1250],
                   'codec_recipes': authorities['codec_results']['codec_recipes'],
                   'primary_scalar': 'median of eligible fixed-target pool squared-bicoherence values',
                   'condition_contrasts': ['closed_minus6db median minus independent_minus6db median',
                                           'mean of paired-pool closed_minus6db minus independent_minus6db'],
                   'summaries': ['all recordings', 'equal score', 'equal player secondary', 'comp', 'solo'],
                   'failure_policy': 'retain failures distinctly; never convert to scientific missingness',
                   'length_policy': 'require exactly 128000 decoded samples; never trim or pad'},
        'raw_root': authorities['raw_root'], 'codec_root': authorities['codec_root'],
        'output_root': authorities['output_root'], 'bindings': bindings,
        'runtime': authorities['runtime'], 'toolchain': authorities['codec_results']['toolchain'],
        'accepted_source_products': authorities['source_commit']['products'],
        'accepted_source_products_sha256': authorities['source_products_sha256'],
        'rows': authorities['roster'], 'parent_freeze_required': True,
    }


def publish_draft(authorities, draft_dir, runner, tests, protocol):
    draft_dir = Path(draft_dir).resolve()
    require(not draft_dir.exists(), 'new draft directory required')
    require(all(draft_dir != Path(authorities[k]) and Path(authorities[k]) not in draft_dir.parents
                and draft_dir not in Path(authorities[k]).parents
                for k in ('raw_root', 'codec_root', 'output_root')), 'draft/source/output overlap')
    document = draft_document(authorities, runner, tests, protocol)
    draft_dir.mkdir(parents=True, exist_ok=False)
    write_json(draft_dir / 'draft.json', document)
    draft_entry = binding(draft_dir / 'draft.json')
    commit = {'version': DRAFT_VERSION, 'status': 'committed_metadata_only_draft_not_authorized',
              'draft': draft_entry, 'source_COMMIT_sha256': SOURCE_COMMIT_SHA,
              'independent_audit_sha256': AUDIT_SHA, 'codec_COMMIT_sha256': CODEC_COMMIT_SHA,
              **SCOPE}
    write_json(draft_dir / 'COMMIT.json', commit)
    return {'draft': draft_entry, 'commit': binding(draft_dir / 'COMMIT.json')}


def verify_draft(draft_dir, commit_sha):
    draft_dir = Path(draft_dir).resolve()
    commit, commit_entry = read_json(draft_dir / 'COMMIT.json', commit_sha)
    document, draft_entry = read_json(draft_dir / 'draft.json', commit['draft']['sha256'])
    require(commit['draft'] == draft_entry and commit.get('version') == DRAFT_VERSION
            and commit.get('status') == 'committed_metadata_only_draft_not_authorized'
            and document.get('version') == DRAFT_VERSION and document.get('parent_freeze_required') is True
            and all(document.get(k) == v and commit.get(k) == v for k, v in SCOPE.items()), 'draft schema/scope')
    require(len(document['rows']) == 270 and len(document['accepted_source_products']) == 2165
            and value_hash(document['accepted_source_products']) == document['accepted_source_products_sha256'],
            'draft denominators/inventory')
    for entry in document['bindings'].values():
        require(binding(entry['path']) == entry, 'draft authority changed')
    return document, commit_entry


def verify_freeze(draft_dir, draft_commit_sha, freeze_path, freeze_sha):
    document, draft_commit = verify_draft(draft_dir, draft_commit_sha)
    freeze, freeze_entry = read_json(freeze_path, freeze_sha)
    expected = {'version': FREEZE_VERSION,
                'status': 'authorized_development_codec_measurement_not_admission',
                'draft_COMMIT': draft_commit, 'draft': binding(Path(draft_dir) / 'draft.json'),
                'output_root': document['output_root'], **SCOPE}
    require(freeze == expected, 'parent freeze schema/authority mismatch')
    return document, freeze, freeze_entry


def wav_float32_bytes(samples):
    samples = np.asarray(samples)
    require(samples.dtype == np.float32 and samples.ndim == 1 and samples.shape == (SAMPLES,)
            and np.isfinite(samples).all(), 'exact finite FLOAT32 control required')
    payload = samples.astype('<f4', copy=False).tobytes()
    fmt = struct.pack('<HHIIHH', 3, 1, RATE, RATE * 4, 4, 32)
    fact = struct.pack('<I', SAMPLES)
    body = b'fmt ' + struct.pack('<I', len(fmt)) + fmt + b'fact' + struct.pack('<I', 4) + fact
    body += b'data' + struct.pack('<I', len(payload)) + payload
    return b'RIFF' + struct.pack('<I', len(body) + 4) + b'WAVE' + body


def save_npz(path, arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    write_new(path, stream.getvalue())


def measure(samples32, extractor, scalar, prefix):
    samples32 = np.asarray(samples32)
    require(samples32.dtype == np.float32 and samples32.shape == (SAMPLES,)
            and np.isfinite(samples32).all(), 'measurement FLOAT32 support')
    samples64 = samples32.astype(np.float64)
    extracted = extractor.extract(samples64, RATE)
    reduced = reduction_signature(scalar.reduce_metadata(extracted['metadata'], crop={
        'start_sample': 0, 'stop_sample_exclusive': SAMPLES, 'source_resampled_samples': SAMPLES}))
    write_json(str(prefix) + '.metadata.json', extracted['metadata'])
    save_npz(str(prefix) + '.arrays.npz', extracted['arrays'])
    return {'status': 'measured_not_admitted', 'input_float32_pcm_sha256': hashlib.sha256(
                samples32.astype('<f4', copy=False).tobytes()).hexdigest(),
            'extractor_float64_pcm_sha256': hashlib.sha256(
                samples64.astype('<f8', copy=False).tobytes()).hexdigest(),
            'metadata': binding(str(prefix) + '.metadata.json'),
            'arrays': binding(str(prefix) + '.arrays.npz'), 'scalar': reduced,
            'processing_failure': None}


def rehash_source_products(document):
    root = Path(document['raw_root'])
    actual = {}
    for relative, expected in document['accepted_source_products'].items():
        entry = binding(root / relative)
        require({key: entry[key] for key in ('bytes', 'sha256')} == expected,
                'accepted full source product changed: ' + relative)
        actual[relative] = expected
    require(value_hash(actual) == document['accepted_source_products_sha256'],
            'accepted full source graph changed')
    return document['accepted_source_products_sha256']


def scalar_comparison(first, second, direction):
    """Compare two retained reductions without changing nulls or masks."""
    require(direction in ('float32_minus_float64', 'decoded_minus_float32'), 'comparison direction')
    if first is None or second is None:
        return {'direction': direction, 'status': 'processing_failure_not_scientific_missingness',
                'pool_denominator': POOLS, 'mask_transition_counts': None, 'scalar_delta': None}
    a, b = first['eligibility_mask'], second['eligibility_mask']
    require(len(a) == len(b) == POOLS, 'comparison pool support')
    transitions = {'eligible_to_eligible': 0, 'missing_to_missing': 0,
                   'missing_to_eligible': 0, 'eligible_to_missing': 0}
    for left, right in zip(a, b, strict=True):
        transitions[('eligible' if left else 'missing') + '_to_' +
                    ('eligible' if right else 'missing')] += 1
    x, y = first['median_squared_bicoherence'], second['median_squared_bicoherence']
    return {'direction': direction, 'status': 'compared_not_thresholded',
            'pool_denominator': POOLS, 'first_mask': a, 'second_mask': b,
            'mask_transition_counts': transitions,
            'first_scalar': x, 'second_scalar': y,
            'scalar_delta': y - x if x is not None and y is not None else None}


def condition_contrast(closed, independent):
    if closed is None or independent is None:
        return {'status': 'processing_failure_not_scientific_missingness',
                'operational_median_difference': None, 'paired_pool_mean_difference': None,
                'paired_pool_count': 0, 'pool_denominator': POOLS}
    require(closed['pool_count'] == independent['pool_count'] == POOLS, 'contrast pool support')
    a, b = closed['median_squared_bicoherence'], independent['median_squared_bicoherence']
    differences = [x['squared_bicoherence'] - y['squared_bicoherence']
                   if x['eligible'] and y['eligible'] else None
                   for x, y in zip(closed['pools'], independent['pools'], strict=True)]
    finite = [x for x in differences if x is not None]
    return {'status': 'compared_not_thresholded',
            'operational_median_difference': a - b if a is not None and b is not None else None,
            'paired_pool_mean_difference': math.fsum(finite) / len(finite) if finite else None,
            'paired_pool_count': len(finite), 'pool_denominator': POOLS,
            'paired_pool_differences': differences}


def equal_group(rows, key, metric):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row[metric])
    records = []
    for identity, values in sorted(groups.items()):
        finite = [v for v in values if v is not None]
        records.append({key: identity, 'recording_denominator': len(values),
                        'covered_recordings': len(finite),
                        'mean': math.fsum(finite) / len(finite) if finite else None})
    finite = [r['mean'] for r in records if r['mean'] is not None]
    return {'group_denominator': len(records), 'covered_groups': len(finite), 'rows': records,
            'equal_group_mean': math.fsum(finite) / len(finite) if finite else None}


def summarize_contrasts(rows):
    result = {'recording_denominator': len(rows),
              'processing_failure_recordings': sum(r['status'].startswith('processing_failure') for r in rows),
              'metrics': {}}
    for metric in ('operational_median_difference', 'paired_pool_mean_difference'):
        finite = [r[metric] for r in rows if r[metric] is not None]
        result['metrics'][metric] = {
            'covered_recordings': len(finite),
            'recording_mean': math.fsum(finite) / len(finite) if finite else None,
            'equal_score': equal_group(rows, 'score_id', metric),
            'equal_player_secondary': equal_group(rows, 'player_id', metric),
            'performance_strata': {label: summarize_metric(
                [r for r in rows if r['performance'] == label], metric) for label in ('comp', 'solo')}}
    result['paired_pool_denominator'] = sum(r['pool_denominator'] for r in rows)
    result['paired_covered_pools'] = sum(r['paired_pool_count'] for r in rows)
    return result


def summarize_metric(rows, metric):
    values = [r[metric] for r in rows if r[metric] is not None]
    return {'recording_denominator': len(rows), 'covered_recordings': len(values),
            'mean': math.fsum(values) / len(values) if values else None}


def quantile_linear(values, probability):
    """NumPy-compatible linear quantile: h=(n-1)q with linear interpolation."""
    require(values and 0 <= probability <= 1, 'quantile support/probability')
    ordered = sorted(values)
    h = (len(ordered) - 1) * probability
    low, high = math.floor(h), math.ceil(h)
    return ordered[low] + (h - low) * (ordered[high] - ordered[low])


def delta_summary(comparisons):
    successful = [c for c in comparisons if c['status'] == 'compared_not_thresholded']
    values = [c['scalar_delta'] for c in successful if c['scalar_delta'] is not None]
    absolute = [abs(value) for value in values]
    return {'comparison_denominator': len(comparisons),
            'successful_comparisons': len(successful),
            'processing_failure_comparisons': len(comparisons) - len(successful),
            'scalar_pair_covered': len(values),
            'scientific_null_pairs': len(successful) - len(values),
            'signed_mean_delta': math.fsum(values) / len(values) if values else None,
            'median_absolute_delta': float(statistics.median(absolute)) if absolute else None,
            'p95_absolute_delta': quantile_linear(absolute, .95) if absolute else None,
            'p95_quantile_method': 'linear interpolation at h=(n-1)*0.95'}


def coverage_summary(reductions):
    valid = [value for value in reductions if value is not None]
    return {'recording_denominator': len(reductions), 'successful_measurements': len(valid),
            'processing_failures_or_skips': len(reductions) - len(valid),
            'pool_denominator': len(reductions) * POOLS,
            'eligible_pools': sum(value['eligible_pool_count'] for value in valid),
            'scientifically_missing_pools': sum(value['missing_pool_count'] for value in valid),
            'unmeasured_pools_due_to_failure_or_skip': (len(reductions) - len(valid)) * POOLS,
            'covered_recording_scalars': sum(value['median_squared_bicoherence'] is not None
                                              for value in valid),
            'scientifically_null_recording_scalars': sum(value['median_squared_bicoherence'] is None
                                                           for value in valid)}


def measurement_counts(receipts):
    statuses = [r['measurement_status'][name] for r in receipts
                for name in ('float32_control', *CODECS)]
    result = {'expected': len(receipts) * 3,
              'attempted': sum(s['attempted'] for s in statuses),
              'successful': sum(s['status'] == 'success' for s in statuses),
              'failed_after_attempt': sum(s['attempted'] and s['status'] != 'success' for s in statuses),
              'skipped_before_attempt': sum(not s['attempted'] for s in statuses)}
    require(result['expected'] == result['successful'] + result['failed_after_attempt']
            + result['skipped_before_attempt'], 'measurement outcome accounting')
    return result


def assemble_summary(receipts):
    require(len(receipts) == 270, 'complete selected-condition receipt denominator')
    index = {(r['item_id'], r['condition']): r for r in receipts}
    ids = sorted({r['item_id'] for r in receipts})
    require(len(ids) == 90 and len(index) == 270, 'complete recording-condition identities')
    contrasts = {}
    for representation in REPRESENTATIONS:
        rows = []
        for item_id in ids:
            closed = index[(item_id, 'closed_minus6db')]
            independent = index[(item_id, 'independent_minus6db')]
            compared = condition_contrast(closed['representations'].get(representation),
                                          independent['representations'].get(representation))
            rows.append({key: closed[key] for key in
                         ('item_id', 'player_id', 'score_id', 'performance')} | compared)
        contrasts[representation] = summarize_contrasts(rows) | {'per_recording': rows}
    precision = [r['comparisons']['float32_minus_float64'] for r in receipts]
    codecs = [r['comparisons']['decoded_minus_float32'][codec] for r in receipts for codec in CODECS]
    baseline = [r for r in receipts if r['condition'] == 'baseline']
    counts = measurement_counts(receipts)
    status = ('development_codec_measurements_complete_not_admission'
              if counts['successful'] == counts['expected'] else
              'development_codec_measurements_committed_with_retained_failures_not_admission')
    delta_distributions = {
        'precision_float32_minus_float64': {
            'all_conditions': delta_summary(precision),
            'by_condition': {condition: delta_summary([r['comparisons']['float32_minus_float64']
                for r in receipts if r['condition'] == condition]) for condition in SELECTED_CONDITIONS}},
        'codec_decoded_minus_float32': {codec: {
            'all_conditions': delta_summary([r['comparisons']['decoded_minus_float32'][codec]
                                             for r in receipts]),
            'by_condition': {condition: delta_summary([
                r['comparisons']['decoded_minus_float32'][codec]
                for r in receipts if r['condition'] == condition]) for condition in SELECTED_CONDITIONS}}
            for codec in CODECS}}
    return {'version': VERSION, 'status': status,
            **SCOPE, 'denominators': {'recordings': 90, 'selected_conditions': 270,
                'precision_comparisons': len(precision), 'codec_comparisons': len(codecs)},
            'new_BC_measurements': counts,
            'processing_failures': sum(c['status'].startswith('processing_failure')
                                       for c in precision + codecs),
            'baseline': {'recording_denominator': 90,
                         'representation_coverage': {representation: coverage_summary([
                             r['representations'].get(representation) for r in baseline])
                             for representation in REPRESENTATIONS},
                         'precision_pool_transition_counts': transition_totals(
                             [r['comparisons']['float32_minus_float64'] for r in baseline]),
                         'codec_pool_transition_counts': {codec: transition_totals([
                             r['comparisons']['decoded_minus_float32'][codec] for r in baseline])
                             for codec in CODECS}},
            'scalar_delta_distributions': delta_distributions,
            'condition_contrasts': contrasts,
            'independent_numerical_replay_required': True,
            'interpretation': ['codec effects are decoded FLOAT32 minus precision-control FLOAT32',
                'precision effects are FLOAT32 control minus accepted FLOAT64',
                'missingness and processing failures are distinct',
                'development results cannot authorize a threshold or BC admission']}


def transition_totals(comparisons):
    totals = {'eligible_to_eligible': 0, 'missing_to_missing': 0,
              'missing_to_eligible': 0, 'eligible_to_missing': 0,
              'comparison_denominator': len(comparisons), 'processing_failures': 0}
    for comparison in comparisons:
        transitions = comparison.get('mask_transition_counts')
        if transitions is None:
            totals['processing_failures'] += 1
        else:
            for key, value in transitions.items():
                totals[key] += value
    return totals


def source_float64(path):
    import soundfile as sf
    info = sf.info(str(path))
    require(info.format == 'WAV' and info.subtype == 'DOUBLE' and info.samplerate == RATE
            and info.channels == 1 and info.frames == SAMPLES, 'accepted FLOAT64 WAV format/support')
    samples, rate = sf.read(str(path), dtype='float64', always_2d=True)
    require(rate == RATE and samples.shape == (SAMPLES, 1) and np.isfinite(samples).all(),
            'accepted FLOAT64 WAV decode/support')
    return samples[:, 0]


def command(codec, command, log):
    completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=codec.command_environment(), check=False)
    receipt = {'command': list(map(str, command)), 'environment': codec.command_environment(),
               'returncode': completed.returncode,
               'stdout_utf8': completed.stdout.decode('utf-8', errors='strict'),
               'stderr_utf8': completed.stderr.decode('utf-8', errors='strict')}
    write_json(log, receipt)
    require(completed.returncode == 0, 'codec command failed')
    return receipt


def process_row(row, destination, modules, recipes):
    destination.mkdir(parents=True, exist_ok=False)
    result = {key: row[key] for key in ('item_id', 'player_id', 'score_id', 'performance',
                                        'style_from_score_prefix', 'split_role', 'condition')}
    result.update(source_float64_wav=row['source_float64_wav'], source_metadata=row['source_metadata'],
                  source_arrays=row['source_arrays'], representations={
                      'accepted_float64': row['accepted_float64_reduction']}, comparisons={},
                  artifacts={}, measurement_status={})
    control_measurement = None
    control_attempted = False
    try:
        require(binding(row['source_float64_wav']['path']) == row['source_float64_wav'],
                'selected source WAV changed')
        original = source_float64(row['source_float64_wav']['path'])
        control = original.astype(np.float32)
        result['source_float64_pcm_sha256'] = hashlib.sha256(
            original.astype('<f8', copy=False).tobytes()).hexdigest()
        result['float64_to_float32_operation'] = 'NumPy value cast once; no gain, normalization, clipping, trim, or pad'
        write_new(destination / 'float32_control.wav', wav_float32_bytes(control))
        require(np.array_equal(modules['codec_probe'].read_float_wav(
            destination / 'float32_control.wav'), control), 'FLOAT32 control round trip')
        control_attempted = True
        control_measurement = measure(control, modules['bicoherence_audio_v1'],
                                      modules['bicoherence_scalar_v1'], destination / 'float32_control')
        result['artifacts']['float32_control'] = {
            'wav': binding(destination / 'float32_control.wav'), 'measurement': control_measurement}
        result['measurement_status']['float32_control'] = {'attempted': True, 'status': 'success'}
        result['representations']['float32_control'] = control_measurement['scalar']
        result['comparisons']['float32_minus_float64'] = scalar_comparison(
            row['accepted_float64_reduction'], control_measurement['scalar'], 'float32_minus_float64')
    except Exception as error:
        result['representations']['float32_control'] = None
        result['comparisons']['float32_minus_float64'] = scalar_comparison(
            row['accepted_float64_reduction'], None, 'float32_minus_float64')
        failure_path = destination / 'float32_control.failure.json'
        write_json(failure_path, {
            'status': 'processing_failure_not_scientific_missingness',
            'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        result['artifacts']['float32_control'] = {'failure': binding(failure_path)}
        if (destination / 'float32_control.wav').exists():
            result['artifacts']['float32_control']['wav'] = binding(destination / 'float32_control.wav')
        result['measurement_status']['float32_control'] = {
            'attempted': control_attempted,
            'status': 'measurement_failure' if control_attempted else 'skipped_pipeline_failure'}
    result['comparisons']['decoded_minus_float32'] = {}
    for name in CODECS:
        prefix = destination / name
        codec_attempted = False
        try:
            require(control_measurement is not None, 'FLOAT32 control unavailable')
            encoded = Path(str(prefix) + recipes[name]['extension'])
            decoded = Path(str(prefix) + '.decoded.wav')
            encode_log = Path(str(prefix) + '.encode.log.json')
            decode_log = Path(str(prefix) + '.decode.log.json')
            command(modules['codec_probe'], modules['codec_probe'].encode_command(
                recipes['_ffmpeg'], destination / 'float32_control.wav', encoded, recipes[name]),
                    encode_log)
            command(modules['codec_probe'], modules['codec_probe'].decode_command(
                recipes['_ffmpeg'], encoded, decoded), decode_log)
            decoded_samples = modules['codec_probe'].read_float_wav(decoded)
            require(decoded_samples.dtype == np.float32 and decoded_samples.shape == (SAMPLES,),
                    'decoded length must be exactly 128000; no trim or padding')
            codec_attempted = True
            measured = measure(decoded_samples, modules['bicoherence_audio_v1'],
                               modules['bicoherence_scalar_v1'], prefix)
            result['artifacts'][name] = {'recipe': recipes[name], 'encoded': binding(encoded),
                'decoded_wav': binding(decoded), 'encode_log': binding(encode_log),
                'decode_log': binding(decode_log), 'measurement': measured}
            result['measurement_status'][name] = {'attempted': True, 'status': 'success'}
            result['representations'][name] = measured['scalar']
            result['comparisons']['decoded_minus_float32'][name] = scalar_comparison(
                control_measurement['scalar'], measured['scalar'], 'decoded_minus_float32')
        except Exception as error:
            result['representations'][name] = None
            result['comparisons']['decoded_minus_float32'][name] = scalar_comparison(
                control_measurement['scalar'] if control_measurement else None, None,
                'decoded_minus_float32')
            failure_path = Path(str(prefix) + ('.failure.json' if control_measurement else '.skipped.json'))
            failure_status = ('processing_failure_not_scientific_missingness' if control_measurement else
                              'skipped_due_to_float32_control_failure_not_scientific_missingness')
            write_json(failure_path, {'status': failure_status,
                'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
            result['artifacts'][name] = {'failure_or_skip': binding(failure_path)}
            for label, candidate in (
                    ('encoded', Path(str(prefix) + recipes[name]['extension'])),
                    ('decoded_wav', Path(str(prefix) + '.decoded.wav')),
                    ('encode_log', Path(str(prefix) + '.encode.log.json')),
                    ('decode_log', Path(str(prefix) + '.decode.log.json')),
                    ('metadata', Path(str(prefix) + '.metadata.json')),
                    ('arrays', Path(str(prefix) + '.arrays.npz'))):
                if candidate.exists():
                    result['artifacts'][name][label] = binding(candidate)
            result['measurement_status'][name] = {'attempted': codec_attempted,
                'status': 'measurement_failure' if codec_attempted else
                          'skipped_pipeline_failure' if control_measurement else
                          'skipped_dependency_failure'}
    write_json(destination / 'receipt.json', result)
    return result


def products(root):
    return {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('COMMIT.json', 'FAILED.json')}


def verify_result(output, commit_sha):
    output = Path(output).resolve()
    require(output.is_dir() and not output.is_symlink(), 'safe result root required')
    commit, commit_entry = read_json(output / 'COMMIT.json', commit_sha)
    require(commit.get('version') == VERSION and commit.get('status') in (
        'committed_development_codec_measurements_complete_not_admission',
        'committed_development_codec_measurements_with_retained_failures_not_admission')
        and all(commit.get(k) == v for k, v in SCOPE.items()), 'result COMMIT schema/scope')
    inventory = products(output)
    require(commit.get('products') == inventory and commit.get('products_sha256') == value_hash(inventory),
            'result product graph changed')
    summary, summary_entry = read_json(output / 'summary.json')
    require(value_hash(summary) == commit.get('summary_sha256')
            and summary.get('new_BC_measurements') == commit.get('measurement_counts')
            and all(summary.get(k) == v for k, v in SCOPE.items()), 'result summary changed')
    draft_path = Path(commit['draft_COMMIT']['path']).parent
    document, draft_entry = verify_draft(draft_path, commit['draft_COMMIT']['sha256'])
    freeze_path = Path(commit['parent_freeze']['path'])
    verified_document, freeze, freeze_entry = verify_freeze(
        draft_path, commit['draft_COMMIT']['sha256'], freeze_path,
        commit['parent_freeze']['sha256'])
    require(document == verified_document and freeze_entry == commit['parent_freeze']
            and binding(output / 'frozen_draft.json') == inventory['frozen_draft.json'] | {
                'path': str(output / 'frozen_draft.json')}
            and json.loads((output / 'frozen_draft.json').read_text()) == document
            and json.loads((output / 'parent_freeze.json').read_text()) == freeze,
            'copied execution authorities changed')
    modules = load_frozen(Path(document['bindings']['codec_probe']['path']).parent)
    codec_proof = modules['codec_probe'].verify_result(document['codec_root'], CODEC_COMMIT_SHA)
    require(codec_proof['results']['toolchain'] == document['toolchain']
            and runtime_snapshot() == document['runtime']
            and rehash_source_products(document) == document['accepted_source_products_sha256'],
            'runtime/toolchain/source graph changed')
    receipts = []
    for row in document['rows']:
        receipt_path = output / 'items' / row['item_id'] / row['condition'] / 'receipt.json'
        receipt = json.loads(receipt_path.read_text())
        require({key: receipt[key] for key in ('item_id', 'player_id', 'score_id', 'performance',
                'style_from_score_prefix', 'split_role', 'condition')} == {key: row[key] for key in
                ('item_id', 'player_id', 'score_id', 'performance', 'style_from_score_prefix',
                 'split_role', 'condition')},
                'result receipt identity/order')
        receipts.append(receipt)
    require(assemble_summary(receipts) == summary, 'result summary reconstruction changed')
    return {'commit': commit_entry, 'summary': summary_entry, 'products': inventory,
            'measurement_counts': summary['new_BC_measurements']}


def run(draft_dir, draft_commit_sha, freeze_path, freeze_sha):
    document, freeze, freeze_entry = verify_freeze(draft_dir, draft_commit_sha, freeze_path, freeze_sha)
    output = Path(document['output_root'])
    require(not output.exists(), 'new output required')
    modules = load_frozen(Path(document['bindings']['codec_probe']['path']).parent)
    codec_proof = modules['codec_probe'].verify_result(document['codec_root'], CODEC_COMMIT_SHA)
    require(codec_proof['results']['toolchain'] == document['toolchain'], 'codec toolchain changed')
    require(runtime_snapshot() == document['runtime'], 'numerical runtime changed')
    for entry in document['bindings'].values():
        require(binding(entry['path']) == entry, 'bound authority changed before run')
    source_graph_start = rehash_source_products(document)
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    recipes = dict(codec_proof['results']['codec_recipes'])
    recipes['_ffmpeg'] = codec_proof['results']['toolchain']['ffmpeg']['path']
    try:
        write_json(output / 'frozen_draft.json', document)
        write_json(output / 'parent_freeze.json', freeze)
        for index, row in enumerate(document['rows']):
            destination = output / 'items' / row['item_id'] / row['condition']
            receipts.append(process_row(row, destination, modules, recipes))
            print(f'BC codec development {index + 1}/270 {row["item_id"]} {row["condition"]}',
                  file=sys.stderr, flush=True)
        summary = assemble_summary(receipts)
        write_json(output / 'summary.json', summary)
        end_document, end_freeze, end_entry = verify_freeze(
            draft_dir, draft_commit_sha, freeze_path, freeze_sha)
        codec_end = modules['codec_probe'].verify_result(document['codec_root'], CODEC_COMMIT_SHA)
        require(end_document == document and end_freeze == freeze and end_entry == freeze_entry
                and runtime_snapshot() == document['runtime']
                and codec_end == codec_proof
                and codec_end['results']['toolchain'] == document['toolchain'],
                'authority/runtime/toolchain changed during run')
        require(rehash_source_products(document) == source_graph_start, 'full source graph changed during run')
        inventory = products(output)
        commit_status = ('committed_development_codec_measurements_complete_not_admission'
                         if summary['new_BC_measurements']['successful'] == 810 else
                         'committed_development_codec_measurements_with_retained_failures_not_admission')
        commit = {'version': VERSION, 'status': commit_status,
                  **SCOPE, 'draft_COMMIT': binding(Path(draft_dir) / 'COMMIT.json'),
                  'parent_freeze': freeze_entry, 'source_COMMIT_sha256': SOURCE_COMMIT_SHA,
                  'independent_audit_sha256': AUDIT_SHA, 'codec_COMMIT_sha256': CODEC_COMMIT_SHA,
                  'measurement_counts': summary['new_BC_measurements'], 'precision_comparison_count': 270,
                  'codec_comparison_count': 540, 'summary_sha256': value_hash(summary),
                  'independent_numerical_replay_passed': False,
                  'products': inventory, 'products_sha256': value_hash(inventory)}
        write_json(output / 'COMMIT.json', commit)
        return {'commit': binding(output / 'COMMIT.json'), 'summary': summary}
    except BaseException as error:
        if not (output / 'FAILED.json').exists():
            write_json(output / 'FAILED.json', {'status': 'failed_no_COMMIT_outputs_preserved',
                       'type': type(error).__name__, 'message': str(error),
                       'completed_selected_conditions': len(receipts), 'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('preflight', 'draft', 'run'), default='preflight')
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--rc-root', required=True)
    parser.add_argument('--codec-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--draft-dir')
    parser.add_argument('--draft-commit-sha256')
    parser.add_argument('--parent-freeze')
    parser.add_argument('--parent-freeze-sha256')
    parser.add_argument('--tests')
    parser.add_argument('--protocol')
    args = parser.parse_args()
    if args.mode == 'run':
        require(all((args.draft_dir, args.draft_commit_sha256, args.parent_freeze,
                     args.parent_freeze_sha256)), 'run requires exact draft and separate parent freeze')
        result = run(args.draft_dir, args.draft_commit_sha256,
                     args.parent_freeze, args.parent_freeze_sha256)
    else:
        authorities = validate_authorities(args.raw_root, args.rc_root, args.codec_root, args.output)
        if args.mode == 'preflight':
            result = {'status': 'metadata_preflight_passed_no_development_audio_opened_no_execution_authority',
                      'development_recordings': 90, 'selected_conditions': 270,
                      'full_source_products_bound': 2165,
                      'synthetic_codec_package_replayed': True,
                      'output': args.output, **SCOPE}
        else:
            require(all((args.draft_dir, args.tests, args.protocol)),
                    'draft mode requires draft-dir, tests, and protocol')
            result = publish_draft(authorities, args.draft_dir, Path(__file__).resolve(),
                                   Path(args.tests).resolve(), Path(args.protocol).resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
