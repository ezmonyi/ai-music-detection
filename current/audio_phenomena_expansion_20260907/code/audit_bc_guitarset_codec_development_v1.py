#!/usr/bin/env python3
"""Independent fixed-target numerical audit for GuitarSet codec development.

This module imports no producer, extractor, primitive, scalar, or reducer code.
It only accepts a terminal COMMIT supplied by exact SHA-256. It reads development
products only and never encodes audio, selects thresholds, or admits a feature.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import struct
import sys

import numpy as np


VERSION = 'audit_bc_guitarset_codec_development_v1'
PRODUCER_SHA = '2c607b1175694b6966b3ce5200c6e306f20c40dd00de8f0229e6236387cc013e'
DRAFT_SHA = '24d0544582d0992441b856bc00594f52082a97fb6dfce2780e6825bca29aa8f7'
DRAFT_COMMIT_SHA = '2f53c482e7934d23c120ab20343f3e66f377efd3626a8466608944e4f4ad30ed'
FREEZE_SHA = '9405118a484d5578b73caff964538571920ae786ea9329e8aa846408ef85e53b'
SOURCE_COMMIT_SHA = 'cd11c3fb80755260b3908c542f8b8c9c218320c931c4987215936e11b73d639c'
CODEC_COMMIT_SHA = '9e07622b5c45179e5086d66f2c6697c4a76bbe7e3b5b39c82ecaea0cb936909b'
AUDIT_SHA = '61a926a92c764d6852608f9829395382e6593e56544d1434ec0f28af509ffc93'
RATE = 16000
SAMPLES = 128000
POOL_SAMPLES = 64000
POOLS = 2
N_FFT = 1024
HOP = 256
FRAMES = 247
TARGET = (32, 48, 80)
CONDITIONS = ('baseline', 'closed_minus6db', 'independent_minus6db')
CODECS = ('mp3_128k', 'opus_96k')
REPRESENTATIONS = ('accepted_float64', 'float32_control', 'mp3_128k', 'opus_96k')
SCOPE = {
    'development_recordings': 90, 'reserved_recordings_read': False,
    'unused_recordings_read': False, 'reserved_BC_measured': False,
    'unused_BC_measured': False, 'classifier_fits': 0, 'model_scoring': False,
    'grid_selection': False, 'thresholds_chosen': False, 'BC_admitted': False,
    'automatic_alignment': False, 'automatic_trim': False,
    'automatic_padding': False, 'gain_changed': False, 'normalization': False,
}
FLOAT_GUID = struct.pack('<IHH8s', 3, 0, 0x0010, b'\x80\x00\x00\xaa\x00\x38\x9b\x71')


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


def binding(path, expected=None):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'safe regular file required')
    result = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
    require(expected is None or result['sha256'] == expected, 'SHA mismatch: ' + str(path))
    return result


def read_json(path, expected=None):
    entry = binding(path, expected)
    return json.loads(Path(entry['path']).read_text(encoding='utf-8')), entry


def close(actual, expected, message, *, atol=2e-12, rtol=2e-11):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True), message)


def compare(actual, expected, message='value'):
    """Exact schemas and discrete values; tight tolerance for finite numerics."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), message + ' keys')
        for key in expected:
            compare(actual[key], expected[key], message + '.' + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), message + ' length')
        for index, value in enumerate(expected):
            compare(actual[index], value, f'{message}[{index}]')
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool), message + ' numeric')
        close(actual, expected, message)
    else:
        require(actual == expected, message)


def read_ieee_float_wav(path, *, bits):
    """Independent RIFF reader; accepts IEEE_FLOAT or validated EXTENSIBLE float."""
    data = Path(path).read_bytes()
    require(data[:4] == b'RIFF' and data[8:12] == b'WAVE'
            and struct.unpack_from('<I', data, 4)[0] + 8 == len(data), 'canonical RIFF/WAVE size')
    offset, fmt, payloads, fact = 12, None, [], None
    while offset + 8 <= len(data):
        chunk, size = data[offset:offset + 4], struct.unpack_from('<I', data, offset + 4)[0]
        start, stop = offset + 8, offset + 8 + size
        require(stop <= len(data), 'RIFF chunk outside file')
        body = data[start:stop]
        if chunk == b'fmt ':
            require(fmt is None and len(body) >= 16, 'single valid fmt chunk')
            tag, channels, rate, byte_rate, align, width = struct.unpack_from('<HHIIHH', body)
            if tag == 0xfffe:
                require(len(body) >= 40 and struct.unpack_from('<H', body, 16)[0] >= 22
                        and struct.unpack_from('<H', body, 18)[0] == bits
                        and body[24:40] == FLOAT_GUID, 'invalid extensible FLOAT subtype')
            else:
                require(tag == 3, 'IEEE FLOAT WAV required')
            require(channels == 1 and rate == RATE and width == bits
                    and align == bits // 8 and byte_rate == RATE * align,
                    'WAV rate/channel/width mismatch')
            fmt = {'tag': tag, 'channels': channels, 'rate': rate, 'bits': width}
        elif chunk == b'fact':
            require(len(body) >= 4, 'invalid fact chunk')
            fact = struct.unpack_from('<I', body)[0]
        elif chunk == b'data':
            payloads.append(body)
        offset = stop + (size & 1)
    require(offset == len(data) and fmt is not None and len(payloads) == 1, 'RIFF chunk layout')
    payload = payloads[0]
    require(len(payload) == SAMPLES * bits // 8, 'WAV exact 128000-sample length; no trim/pad')
    require(fact in (None, SAMPLES), 'WAV fact sample count')
    values = np.frombuffer(payload, dtype='<f4' if bits == 32 else '<f8').copy()
    require(values.shape == (SAMPLES,) and np.isfinite(values).all(), 'finite WAV samples')
    return values, {'bits': bits, 'format_tag': fmt['tag'], 'fact_samples': fact,
                    'file_sha256': hashlib.sha256(data).hexdigest(),
                    'pcm_sha256': hashlib.sha256(payload).hexdigest()}


def target_pool(samples):
    samples = np.asarray(samples)
    require(samples.dtype == np.float64 and samples.shape == (POOL_SAMPLES,)
            and np.isfinite(samples).all(), 'one finite float64 pool')
    window = .5 - .5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    offsets = np.arange(FRAMES, dtype=np.int64) * HOP
    frames = np.array([samples[start:start + N_FFT] for start in offsets])
    means = np.mean(frames, axis=1)
    spectra = np.array([np.fft.rfft((frame - mean) * window) / np.sum(window)
                        for frame, mean in zip(frames, means, strict=True)])
    require(spectra.shape == (FRAMES, 513) and np.isfinite(spectra).all(), 'finite STFT')
    energy = np.sum(np.abs(spectra) ** 2, axis=0)
    total = float(np.sum(energy[1:]))
    zero = not np.any(samples)
    pool_status = 'zero_amplitude' if zero else 'zero_non_dc_energy' if total == 0 else 'ok'
    fractions = energy / total if total else np.full(513, np.nan)
    columns = spectra[:, TARGET]
    scales = np.max(np.maximum(np.abs(columns.real), np.abs(columns.imag)), axis=0)
    primitive_status, raw_score = 'ok', None
    if np.any(scales == 0):
        primitive_status = 'missing_triad_energy' if np.any(spectra) else 'zero_energy'
    else:
        normalized = columns / scales
        product = normalized[:, 0] * normalized[:, 1]
        third = normalized[:, 2]
        product_energy = float(np.sum(np.abs(product) ** 2))
        third_energy = float(np.sum(np.abs(third) ** 2))
        if product_energy == 0 or third_energy == 0:
            primitive_status = 'missing_triad_product_energy'
        else:
            triple = complex(np.sum(product * np.conjugate(third)))
            score = (abs(triple) / math.sqrt(product_energy) / math.sqrt(third_energy)) ** 2
            require(math.isfinite(score) and score <= 1 + 64 * np.finfo(float).eps,
                    'fixed-target Cauchy-Schwarz bound')
            raw_score = min(1.0, score)
    floor = bool(total and np.all(fractions[list(TARGET)] >= 1e-6))
    eligible = floor and primitive_status == 'ok'
    status = 'ok' if eligible else primitive_status if primitive_status != 'ok' else 'below_energy_fraction_floor'
    return {'pool_status': pool_status, 'total_non_dc_coefficient_energy': total,
            'target_energy_fractions': fractions[list(TARGET)].tolist() if total else [None] * 3,
            'energy_floor_passed': floor, 'primitive_status': primitive_status,
            'primitive_squared_bicoherence': raw_score, 'target_status': status,
            'eligible': eligible, 'squared_bicoherence': raw_score if eligible else None,
            'window': window, 'offsets': offsets, 'means': means, 'spectra': spectra,
            'energy': energy, 'fractions': fractions}


def reduce_waveform(samples, construction_status=None):
    samples = np.asarray(samples)
    require(samples.dtype == np.float64 and samples.shape == (SAMPLES,)
            and np.isfinite(samples).all(), 'exact full float64 measurement input')
    pools, records = [], []
    for index in range(POOLS):
        calculated = target_pool(samples[index * POOL_SAMPLES:(index + 1) * POOL_SAMPLES])
        pools.append(calculated)
        records.append({'pool_index': index, 'start_sample': index * POOL_SAMPLES,
            'stop_sample_exclusive': (index + 1) * POOL_SAMPLES,
            'source_start_sample': index * POOL_SAMPLES,
            'source_stop_sample_exclusive': (index + 1) * POOL_SAMPLES,
            'pool_status': calculated['pool_status'], 'target_status': calculated['target_status'],
            'eligible': calculated['eligible'], 'squared_bicoherence': calculated['squared_bicoherence']})
    values = [pool['squared_bicoherence'] for pool in pools if pool['eligible']]
    reduction = {'version': 'bicoherence_scalar_v1', 'target_frequency_bins': list(TARGET),
        'target_frequency_hz': [500, 750, 1250],
        'crop': {'start_sample': 0, 'stop_sample_exclusive': SAMPLES,
                 'source_resampled_samples': SAMPLES},
        'input_samples': SAMPLES, 'pool_count': POOLS, 'analyzed_samples': SAMPLES,
        'discarded_tail_samples': 0, 'eligible_pool_count': len(values),
        'missing_pool_count': POOLS - len(values),
        'eligibility_mask': [pool['eligible'] for pool in pools],
        'status': 'ok' if values else 'no_eligible_target_pools',
        'median_squared_bicoherence': float(statistics.median(values)) if values else None,
        'pools': records, 'construction_status': construction_status,
        'null_calibrated': False, 'significance_inferred': False,
        'external_validation_passed': False, 'classifier_admitted': False}
    return reduction, pools


def verify_measurement(samples32, prefix, reported_reduction):
    samples32 = np.asarray(samples32)
    require(samples32.dtype == np.float32 and samples32.shape == (SAMPLES,), 'FLOAT32 measurement')
    samples64 = samples32.astype(np.float64)
    metadata, metadata_binding = read_json(str(prefix) + '.metadata.json')
    reduction, pools = reduce_waveform(samples64, metadata.get('construction_status'))
    compare(reported_reduction, reduction, 'reported scalar reduction')
    constants = {'version': 'bicoherence_audio_v1', 'primitive_version': 'bicoherence_primitive_v2',
        'input_samples': SAMPLES, 'sample_rate_hz': RATE, 'pool_samples': POOL_SAMPLES,
        'pool_count': POOLS, 'discarded_tail_samples': 0, 'n_fft': N_FFT,
        'hop_samples': HOP, 'frames_per_pool': FRAMES, 'energy_fraction_min_inclusive': 1e-6}
    for key, expected in constants.items():
        require(metadata.get(key) == expected, 'raw metadata constant: ' + key)
    require(len(metadata.get('pools', [])) == POOLS, 'raw metadata pool count')
    arrays_path = Path(str(prefix) + '.arrays.npz')
    arrays_binding = binding(arrays_path)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        required = {'window', 'frame_offset_samples', 'frame_means', 'spectra',
                    'bin_coefficient_energy', 'total_non_dc_coefficient_energy',
                    'bin_energy_fraction', 'frequency_bins', 'energy_floor_mask',
                    'primitive_defined_mask', 'eligible_mask', 'squared_bicoherence',
                    'primitive_squared_bicoherence'}
        require(required <= set(arrays.files), 'raw array keys')
        frequency_bins = arrays['frequency_bins']
        matches = np.flatnonzero(np.all(frequency_bins == np.asarray(TARGET), axis=1))
        require(matches.shape == (1,), 'unique fixed target array cell')
        cell_index = int(matches[0])
        close(arrays['window'], pools[0]['window'], 'periodic Hann', atol=0, rtol=0)
        close(arrays['frame_offset_samples'], pools[0]['offsets'], 'frame offsets', atol=0, rtol=0)
        for index, calculated in enumerate(pools):
            raw_pool = metadata['pools'][index]
            require(raw_pool['pool_index'] == index and raw_pool['coefficient_rows'] == FRAMES
                    and raw_pool['status'] == calculated['pool_status'], 'raw pool identity/status')
            cells = raw_pool['cells']
            require(isinstance(cells, list) and len(cells) == 228, 'full raw cell metadata retained')
            matches = [cell for cell in cells if cell['frequency_bins'] == list(TARGET)]
            require(len(matches) == 1, 'unique raw fixed target cell')
            cell = matches[0]
            expected_cell = {'status': calculated['target_status'],
                'energy_fractions': calculated['target_energy_fractions'],
                'energy_floor_passed': calculated['energy_floor_passed'],
                'eligible': calculated['eligible'],
                'squared_bicoherence': calculated['squared_bicoherence']}
            for key, expected in expected_cell.items():
                compare(cell[key], expected, 'raw fixed target.' + key)
            require(cell['primitive']['status'] == calculated['primitive_status'], 'primitive status')
            compare(cell['primitive']['squared_bicoherence'],
                    calculated['primitive_squared_bicoherence'], 'primitive score')
            close(arrays['frame_means'][index], calculated['means'], 'frame means')
            close(arrays['spectra'][index, :, list(TARGET)],
                  calculated['spectra'][:, list(TARGET)].T, 'fixed-bin STFT')
            close(arrays['bin_coefficient_energy'][index, list(TARGET)],
                  calculated['energy'][list(TARGET)], 'fixed-bin energy')
            close(arrays['total_non_dc_coefficient_energy'][index],
                  calculated['total_non_dc_coefficient_energy'], 'non-DC energy')
            expected_fractions = np.array([np.nan if x is None else x
                                           for x in calculated['target_energy_fractions']])
            close(arrays['bin_energy_fraction'][index, list(TARGET)], expected_fractions,
                  'fixed-bin fractions')
            require(bool(arrays['energy_floor_mask'][index, cell_index]) ==
                    calculated['energy_floor_passed'], 'array energy mask')
            require(bool(arrays['eligible_mask'][index, cell_index]) == calculated['eligible'],
                    'array eligible mask')
            require(bool(arrays['primitive_defined_mask'][index, cell_index]) ==
                    (calculated['primitive_status'] == 'ok'
                     and calculated['primitive_squared_bicoherence'] is not None),
                    'array primitive-defined mask')
            expected = calculated['squared_bicoherence']
            observed = arrays['squared_bicoherence'][index, cell_index]
            require((np.isnan(observed) if expected is None else
                     np.isclose(observed, expected, atol=2e-12, rtol=2e-11)), 'array target score')
            raw_expected = calculated['primitive_squared_bicoherence']
            raw_observed = arrays['primitive_squared_bicoherence'][index, cell_index]
            require((np.isnan(raw_observed) if raw_expected is None else
                     np.isclose(raw_observed, raw_expected, atol=2e-12, rtol=2e-11)),
                    'array primitive target score')
    expected_measurement = {'status': 'measured_not_admitted',
        'input_float32_pcm_sha256': hashlib.sha256(samples32.astype('<f4', copy=False).tobytes()).hexdigest(),
        'extractor_float64_pcm_sha256': hashlib.sha256(samples64.astype('<f8', copy=False).tobytes()).hexdigest(),
        'metadata': metadata_binding, 'arrays': arrays_binding, 'scalar': reduction,
        'processing_failure': None}
    return reduction, expected_measurement


def compare_scalars(first, second, direction):
    if first is None or second is None:
        return {'direction': direction, 'status': 'processing_failure_not_scientific_missingness',
                'pool_denominator': POOLS, 'mask_transition_counts': None, 'scalar_delta': None}
    transitions = {'eligible_to_eligible': 0, 'missing_to_missing': 0,
                   'missing_to_eligible': 0, 'eligible_to_missing': 0}
    for left, right in zip(first['eligibility_mask'], second['eligibility_mask'], strict=True):
        transitions[('eligible' if left else 'missing') + '_to_' +
                    ('eligible' if right else 'missing')] += 1
    x, y = first['median_squared_bicoherence'], second['median_squared_bicoherence']
    return {'direction': direction, 'status': 'compared_not_thresholded',
            'pool_denominator': POOLS, 'first_mask': first['eligibility_mask'],
            'second_mask': second['eligibility_mask'], 'mask_transition_counts': transitions,
            'first_scalar': x, 'second_scalar': y,
            'scalar_delta': y - x if x is not None and y is not None else None}


def condition_difference(closed, independent):
    if closed is None or independent is None:
        return {'status': 'processing_failure_not_scientific_missingness',
                'operational_median_difference': None, 'paired_pool_mean_difference': None,
                'paired_pool_count': 0, 'pool_denominator': POOLS}
    differences = [a['squared_bicoherence'] - b['squared_bicoherence']
                   if a['eligible'] and b['eligible'] else None
                   for a, b in zip(closed['pools'], independent['pools'], strict=True)]
    finite = [value for value in differences if value is not None]
    a, b = closed['median_squared_bicoherence'], independent['median_squared_bicoherence']
    return {'status': 'compared_not_thresholded',
        'operational_median_difference': a - b if a is not None and b is not None else None,
        'paired_pool_mean_difference': math.fsum(finite) / len(finite) if finite else None,
        'paired_pool_count': len(finite), 'pool_denominator': POOLS,
        'paired_pool_differences': differences}


def linear_quantile(values, probability=.95):
    ordered = sorted(values)
    require(ordered, 'quantile values')
    h = (len(ordered) - 1) * probability
    low, high = math.floor(h), math.ceil(h)
    return ordered[low] + (h - low) * (ordered[high] - ordered[low])


def delta_stats(comparisons):
    successful = [row for row in comparisons if row['status'] == 'compared_not_thresholded']
    values = [row['scalar_delta'] for row in successful if row['scalar_delta'] is not None]
    absolute = [abs(value) for value in values]
    return {'comparison_denominator': len(comparisons), 'successful_comparisons': len(successful),
        'processing_failure_comparisons': len(comparisons) - len(successful),
        'scalar_pair_covered': len(values), 'scientific_null_pairs': len(successful) - len(values),
        'signed_mean_delta': math.fsum(values) / len(values) if values else None,
        'median_absolute_delta': float(statistics.median(absolute)) if absolute else None,
        'p95_absolute_delta': linear_quantile(absolute) if absolute else None,
        'p95_quantile_method': 'linear interpolation at h=(n-1)*0.95'}


def transition_totals(comparisons):
    result = {'eligible_to_eligible': 0, 'missing_to_missing': 0,
              'missing_to_eligible': 0, 'eligible_to_missing': 0,
              'comparison_denominator': len(comparisons), 'processing_failures': 0}
    for comparison in comparisons:
        if comparison['mask_transition_counts'] is None:
            result['processing_failures'] += 1
        else:
            for key, value in comparison['mask_transition_counts'].items():
                result[key] += value
    return result


def coverage(reductions):
    valid = [value for value in reductions if value is not None]
    return {'recording_denominator': len(reductions), 'successful_measurements': len(valid),
        'processing_failures_or_skips': len(reductions) - len(valid),
        'pool_denominator': len(reductions) * POOLS,
        'eligible_pools': sum(value['eligible_pool_count'] for value in valid),
        'scientifically_missing_pools': sum(value['missing_pool_count'] for value in valid),
        'unmeasured_pools_due_to_failure_or_skip': (len(reductions) - len(valid)) * POOLS,
        'covered_recording_scalars': sum(value['median_squared_bicoherence'] is not None for value in valid),
        'scientifically_null_recording_scalars': sum(value['median_squared_bicoherence'] is None
                                                      for value in valid)}


def simple_metric(rows, metric):
    values = [row[metric] for row in rows if row[metric] is not None]
    return {'recording_denominator': len(rows), 'covered_recordings': len(values),
            'mean': math.fsum(values) / len(values) if values else None}


def equal_group(rows, key, metric):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row[metric])
    records = []
    for identity, values in sorted(groups.items()):
        finite = [value for value in values if value is not None]
        records.append({key: identity, 'recording_denominator': len(values),
            'covered_recordings': len(finite),
            'mean': math.fsum(finite) / len(finite) if finite else None})
    finite = [row['mean'] for row in records if row['mean'] is not None]
    return {'group_denominator': len(records), 'covered_groups': len(finite), 'rows': records,
            'equal_group_mean': math.fsum(finite) / len(finite) if finite else None}


def contrast_summary(rows):
    result = {'recording_denominator': len(rows),
        'processing_failure_recordings': sum(row['status'].startswith('processing_failure') for row in rows),
        'metrics': {}, 'paired_pool_denominator': sum(row['pool_denominator'] for row in rows),
        'paired_covered_pools': sum(row['paired_pool_count'] for row in rows)}
    for metric in ('operational_median_difference', 'paired_pool_mean_difference'):
        finite = [row[metric] for row in rows if row[metric] is not None]
        result['metrics'][metric] = {'covered_recordings': len(finite),
            'recording_mean': math.fsum(finite) / len(finite) if finite else None,
            'equal_score': equal_group(rows, 'score_id', metric),
            'equal_player_secondary': equal_group(rows, 'player_id', metric),
            'performance_strata': {label: simple_metric(
                [row for row in rows if row['performance'] == label], metric)
                for label in ('comp', 'solo')}}
    result['per_recording'] = rows
    return result


def outcome_counts(receipts):
    statuses = [row['measurement_status'][name] for row in receipts
                for name in ('float32_control', *CODECS)]
    for row in statuses:
        require(type(row.get('attempted')) is bool and row.get('status') in
                ('success', 'measurement_failure', 'skipped_pipeline_failure',
                 'skipped_dependency_failure'), 'measurement status schema')
        require((row['status'] == 'success') <= row['attempted']
                and (row['status'] == 'measurement_failure') <= row['attempted']
                and (row['status'].startswith('skipped_')) <= (not row['attempted']),
                'measurement attempted/status consistency')
    result = {'expected': len(receipts) * 3, 'attempted': sum(row['attempted'] for row in statuses),
        'successful': sum(row['status'] == 'success' for row in statuses),
        'failed_after_attempt': sum(row['attempted'] and row['status'] != 'success' for row in statuses),
        'skipped_before_attempt': sum(not row['attempted'] for row in statuses)}
    require(result['expected'] == result['successful'] + result['failed_after_attempt']
            + result['skipped_before_attempt'], 'measurement outcome accounting')
    return result


def reconstruct_summary(receipts):
    require(len(receipts) == 270, '270 selected-condition receipts')
    index = {(row['item_id'], row['condition']): row for row in receipts}
    ids = sorted({row['item_id'] for row in receipts})
    require(len(ids) == 90 and len(index) == 270, '90 x 3 identity denominator')
    precision = [row['comparisons']['float32_minus_float64'] for row in receipts]
    codec_comparisons = [row['comparisons']['decoded_minus_float32'][codec]
                         for row in receipts for codec in CODECS]
    baseline = [row for row in receipts if row['condition'] == 'baseline']
    counts = outcome_counts(receipts)
    status = ('development_codec_measurements_complete_not_admission'
              if counts['successful'] == counts['expected'] else
              'development_codec_measurements_committed_with_retained_failures_not_admission')
    contrasts = {}
    for representation in REPRESENTATIONS:
        rows = []
        for item_id in ids:
            closed = index[(item_id, 'closed_minus6db')]
            independent = index[(item_id, 'independent_minus6db')]
            result = condition_difference(closed['representations'][representation],
                                          independent['representations'][representation])
            rows.append({key: closed[key] for key in ('item_id', 'player_id', 'score_id', 'performance')} | result)
        contrasts[representation] = contrast_summary(rows)
    delta_distributions = {'precision_float32_minus_float64': {
        'all_conditions': delta_stats(precision),
        'by_condition': {condition: delta_stats([row['comparisons']['float32_minus_float64']
            for row in receipts if row['condition'] == condition]) for condition in CONDITIONS}},
        'codec_decoded_minus_float32': {codec: {
            'all_conditions': delta_stats([row['comparisons']['decoded_minus_float32'][codec]
                                           for row in receipts]),
            'by_condition': {condition: delta_stats([row['comparisons']['decoded_minus_float32'][codec]
                for row in receipts if row['condition'] == condition]) for condition in CONDITIONS}}
            for codec in CODECS}}
    return {'version': 'run_bc_guitarset_codec_development_v1', 'status': status, **SCOPE,
        'denominators': {'recordings': 90, 'selected_conditions': 270,
                         'precision_comparisons': 270, 'codec_comparisons': 540},
        'new_BC_measurements': counts,
        'processing_failures': sum(row['status'].startswith('processing_failure')
                                   for row in precision + codec_comparisons),
        'baseline': {'recording_denominator': 90,
            'representation_coverage': {representation: coverage([
                row['representations'][representation] for row in baseline])
                for representation in REPRESENTATIONS},
            'precision_pool_transition_counts': transition_totals([
                row['comparisons']['float32_minus_float64'] for row in baseline]),
            'codec_pool_transition_counts': {codec: transition_totals([
                row['comparisons']['decoded_minus_float32'][codec] for row in baseline])
                for codec in CODECS}},
        'scalar_delta_distributions': delta_distributions, 'condition_contrasts': contrasts,
        'independent_numerical_replay_required': True,
        'interpretation': ['codec effects are decoded FLOAT32 minus precision-control FLOAT32',
            'precision effects are FLOAT32 control minus accepted FLOAT64',
            'missingness and processing failures are distinct',
            'development results cannot authorize a threshold or BC admission']}


def verify_command_log(path, expected):
    receipt, entry = read_json(path)
    require(receipt.get('command') == expected and receipt.get('returncode') == 0
            and receipt.get('environment') == {'LANG': 'C', 'LC_ALL': 'C', 'PATH': '/usr/bin:/bin'},
            'codec command provenance')
    return entry


def check_file_bindings(value):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= set(value):
            require(binding(value['path']) == {key: value[key] for key in ('path', 'bytes', 'sha256')},
                    'retained artifact binding changed')
        for child in value.values():
            check_file_bindings(child)
    elif isinstance(value, list):
        for child in value:
            check_file_bindings(child)


def update_scalar_error(maxima, reported, calculated):
    if reported is not None and calculated is not None:
        a, b = reported['median_squared_bicoherence'], calculated['median_squared_bicoherence']
        if a is not None and b is not None:
            maxima['fixed_target_scalar_absolute_error'] = max(
                maxima['fixed_target_scalar_absolute_error'], abs(a - b))


def expected_codec_commands(draft, row, output, codec):
    recipe = draft['policy']['codec_recipes'][codec]
    prefix = output / 'items' / row['item_id'] / row['condition']
    control = prefix / 'float32_control.wav'
    encoded = Path(str(prefix / codec) + recipe['extension'])
    decoded = Path(str(prefix / codec) + '.decoded.wav')
    ffmpeg = draft['toolchain']['ffmpeg']['path']
    encode = [ffmpeg, '-hide_banner', '-nostdin', '-n', '-loglevel', 'info', '-i', str(control),
        '-map', '0:a:0', '-vn', '-sn', '-dn', '-ac', '1', '-ar', '16000', '-threads', '1',
        '-c:a', recipe['encoder'], '-b:a', recipe['bitrate'], *recipe['encoder_options'],
        '-map_metadata', '-1', str(encoded)]
    decode = [ffmpeg, '-hide_banner', '-nostdin', '-n', '-loglevel', 'info', '-i', str(encoded),
        '-map', '0:a:0', '-vn', '-sn', '-dn', '-ac', '1', '-ar', '16000', '-threads', '1',
        '-c:a', 'pcm_f32le', '-map_metadata', '-1', str(decoded)]
    return encode, decode, encoded, decoded


def validate_terminal(output, commit_sha, draft_dir, freeze_path):
    output = Path(output).resolve()
    require(output.is_dir() and not output.is_symlink() and not (output / 'FAILED.json').exists(),
            'terminal committed output without FAILED marker required')
    commit, commit_entry = read_json(output / 'COMMIT.json', commit_sha)
    require(commit.get('version') == 'run_bc_guitarset_codec_development_v1'
            and commit.get('status') in ('committed_development_codec_measurements_complete_not_admission',
                'committed_development_codec_measurements_with_retained_failures_not_admission')
            and all(commit.get(key) == value for key, value in SCOPE.items()), 'producer COMMIT scope')
    require(commit.get('source_COMMIT_sha256') == SOURCE_COMMIT_SHA
            and commit.get('independent_audit_sha256') == AUDIT_SHA
            and commit.get('codec_COMMIT_sha256') == CODEC_COMMIT_SHA
            and commit.get('independent_numerical_replay_passed') is False, 'upstream/audit boundary')
    inventory = {path.relative_to(output).as_posix(): {'bytes': path.stat().st_size,
                 'sha256': digest(path)} for path in sorted(output.rglob('*'))
                 if path.is_file() and path.name not in ('COMMIT.json', 'FAILED.json')}
    require(commit.get('products') == inventory and commit.get('products_sha256') == value_hash(inventory),
            'complete result inventory')
    draft_dir, freeze_path = Path(draft_dir).resolve(), Path(freeze_path).resolve()
    draft, draft_entry = read_json(draft_dir / 'draft.json', DRAFT_SHA)
    draft_commit, draft_commit_entry = read_json(draft_dir / 'COMMIT.json', DRAFT_COMMIT_SHA)
    freeze, freeze_entry = read_json(freeze_path, FREEZE_SHA)
    require(draft_commit['draft'] == draft_entry and commit['draft_COMMIT'] == draft_commit_entry
            and commit['parent_freeze'] == freeze_entry and freeze['draft_COMMIT'] == draft_commit_entry
            and freeze['draft'] == draft_entry and freeze['output_root'] == str(output), 'draft/freeze chain')
    require(draft['bindings']['runner']['sha256'] == PRODUCER_SHA
            and draft['output_root'] == str(output) and len(draft['rows']) == 270
            and {row['split_role'] for row in draft['rows']} == {'development'}, 'frozen producer/roster')
    return commit, commit_entry, draft, freeze, inventory


def audit(output, commit_sha, draft_dir, freeze_path, receipt_path, audit_code_sha,
          tests_path, tests_sha):
    output, receipt_path = Path(output).resolve(), Path(receipt_path).resolve()
    require(not receipt_path.exists() and not receipt_path.is_relative_to(output), 'new external audit receipt')
    self_binding = binding(Path(__file__).resolve(), audit_code_sha)
    tests_binding = binding(Path(tests_path).resolve(), tests_sha)
    commit, commit_entry, draft, freeze, inventory = validate_terminal(
        output, commit_sha, draft_dir, freeze_path)
    receipts, maxima = [], {'float32_cast_absolute_error': 0.0,
        'fixed_target_scalar_absolute_error': 0.0}
    for index, row in enumerate(draft['rows']):
        require(row['split_role'] == 'development' and row['condition'] in CONDITIONS,
                'development-only selected row')
        require(binding(row['source_float64_wav']['path']) == row['source_float64_wav']
                and binding(row['source_metadata']['path']) == row['source_metadata']
                and binding(row['source_arrays']['path']) == row['source_arrays'],
                'accepted source row binding changed')
        source, _ = read_ieee_float_wav(row['source_float64_wav']['path'], bits=64)
        source_reduction, _ = reduce_waveform(
            source, row['accepted_float64_reduction']['construction_status'])
        compare(row['accepted_float64_reduction'], source_reduction, 'accepted FLOAT64 reduction')
        update_scalar_error(maxima, row['accepted_float64_reduction'], source_reduction)
        item_root = output / 'items' / row['item_id'] / row['condition']
        reported, _ = read_json(item_root / 'receipt.json')
        identity = ('item_id', 'player_id', 'score_id', 'performance', 'style_from_score_prefix',
                    'split_role', 'condition')
        require({key: reported[key] for key in identity} == {key: row[key] for key in identity},
                'item receipt identity')
        check_file_bindings(reported['artifacts'])
        reconstructed = dict(reported)
        reconstructed['representations'] = {'accepted_float64': source_reduction}
        reconstructed['comparisons'] = {}
        control_status = reported['measurement_status']['float32_control']
        control_reduction = None
        if control_status == {'attempted': True, 'status': 'success'}:
            control, control_info = read_ieee_float_wav(item_root / 'float32_control.wav', bits=32)
            expected_control = source.astype(np.float32)
            require(np.array_equal(control, expected_control), 'exact FLOAT64 to FLOAT32 cast')
            maxima['float32_cast_absolute_error'] = max(maxima['float32_cast_absolute_error'],
                float(np.max(np.abs(control.astype(np.float64) - expected_control.astype(np.float64)))))
            control_reduction, control_products = verify_measurement(
                control, item_root / 'float32_control', reported['representations']['float32_control'])
            update_scalar_error(maxima, reported['representations']['float32_control'], control_reduction)
            require(reported['artifacts']['float32_control']['wav'] == binding(item_root / 'float32_control.wav'),
                    'control WAV provenance')
            compare(reported['artifacts']['float32_control']['measurement'], control_products,
                    'control measurement provenance')
            require(reported['source_float64_pcm_sha256'] == hashlib.sha256(
                source.astype('<f8', copy=False).tobytes()).hexdigest(), 'source PCM lineage')
        else:
            require(control_status['status'] in ('measurement_failure', 'skipped_pipeline_failure')
                    and reported['representations']['float32_control'] is None, 'control failure state')
            failure = reported['artifacts']['float32_control'].get('failure')
            require(isinstance(failure, dict), 'control retained failure binding')
            failure_receipt, failure_entry = read_json(failure['path'])
            require(failure == failure_entry
                    and failure_receipt.get('status') == 'processing_failure_not_scientific_missingness',
                    'control retained failure status')
        reconstructed['representations']['float32_control'] = control_reduction
        reconstructed['comparisons']['float32_minus_float64'] = compare_scalars(
            source_reduction, control_reduction, 'float32_minus_float64')
        reconstructed['comparisons']['decoded_minus_float32'] = {}
        for codec in CODECS:
            status = reported['measurement_status'][codec]
            codec_reduction = None
            encode, decode, encoded, decoded = expected_codec_commands(draft, row, output, codec)
            if status == {'attempted': True, 'status': 'success'}:
                require(control_reduction is not None, 'codec success requires control')
                encode_log = verify_command_log(Path(str(item_root / codec) + '.encode.log.json'), encode)
                decode_log = verify_command_log(Path(str(item_root / codec) + '.decode.log.json'), decode)
                encoded_entry = binding(encoded)
                decoded_samples, _ = read_ieee_float_wav(decoded, bits=32)
                codec_reduction, codec_products = verify_measurement(
                    decoded_samples, item_root / codec, reported['representations'][codec])
                update_scalar_error(maxima, reported['representations'][codec], codec_reduction)
                artifact = reported['artifacts'][codec]
                require(artifact['recipe'] == draft['policy']['codec_recipes'][codec]
                        and artifact['encoded'] == encoded_entry
                        and artifact['decoded_wav'] == binding(decoded)
                        and artifact['encode_log'] == encode_log and artifact['decode_log'] == decode_log,
                        'codec artifact provenance')
                compare(artifact['measurement'], codec_products, 'codec measurement provenance')
            else:
                require(status['status'] in ('measurement_failure', 'skipped_pipeline_failure',
                    'skipped_dependency_failure') and reported['representations'][codec] is None,
                    'codec failure/skip state')
                failure = reported['artifacts'][codec].get('failure_or_skip')
                require(isinstance(failure, dict), 'codec retained failure/skip binding')
                failure_receipt, failure_entry = read_json(failure['path'])
                expected_failure = ('skipped_due_to_float32_control_failure_not_scientific_missingness'
                    if status['status'] == 'skipped_dependency_failure' else
                    'processing_failure_not_scientific_missingness')
                require(failure == failure_entry and failure_receipt.get('status') == expected_failure,
                        'codec retained failure/skip status')
                if status['status'] == 'skipped_dependency_failure':
                    require(not encoded.exists() and not decoded.exists()
                            and not Path(str(item_root / codec) + '.encode.log.json').exists()
                            and not Path(str(item_root / codec) + '.decode.log.json').exists(),
                            'dependency skip created codec product/log')
            reconstructed['representations'][codec] = codec_reduction
            reconstructed['comparisons']['decoded_minus_float32'][codec] = compare_scalars(
                control_reduction, codec_reduction, 'decoded_minus_float32')
        compare(reported['representations'], reconstructed['representations'], 'independent representations')
        compare(reported['comparisons'], reconstructed['comparisons'], 'independent comparisons')
        receipts.append(reconstructed)
        print(f'BC codec independent audit {index + 1}/270 {row["item_id"]} {row["condition"]}',
              file=sys.stderr, flush=True)
    reconstructed_summary = reconstruct_summary(receipts)
    reported_summary, summary_entry = read_json(output / 'summary.json')
    compare(reported_summary, reconstructed_summary, 'independent summary reconstruction')
    counts = outcome_counts(receipts)
    require(commit['measurement_counts'] == counts == reported_summary['new_BC_measurements']
            and commit['summary_sha256'] == value_hash(reported_summary),
            'COMMIT measurement counts/summary')
    expected_commit_status = ('committed_development_codec_measurements_complete_not_admission'
        if counts['successful'] == counts['expected'] else
        'committed_development_codec_measurements_with_retained_failures_not_admission')
    require(commit['status'] == expected_commit_status, 'COMMIT success/failure qualification')
    end_inventory = {path.relative_to(output).as_posix(): {'bytes': path.stat().st_size,
                     'sha256': digest(path)} for path in sorted(output.rglob('*'))
                     if path.is_file() and path.name not in ('COMMIT.json', 'FAILED.json')}
    end_commit = binding(output / 'COMMIT.json', commit_sha)
    require(end_commit == commit_entry and end_inventory == inventory
            and value_hash(end_inventory) == commit['products_sha256'],
            'result product graph/COMMIT changed during audit')
    for row in draft['rows']:
        require(binding(row['source_float64_wav']['path']) == row['source_float64_wav']
                and binding(row['source_metadata']['path']) == row['source_metadata']
                and binding(row['source_arrays']['path']) == row['source_arrays'],
                'accepted source binding changed during audit')
    require(binding(Path(__file__).resolve(), audit_code_sha) == self_binding
            and binding(Path(tests_path).resolve(), tests_sha) == tests_binding
            and binding(Path(draft_dir).resolve() / 'draft.json', DRAFT_SHA)['sha256'] == DRAFT_SHA
            and binding(Path(freeze_path).resolve(), FREEZE_SHA)['sha256'] == FREEZE_SHA,
            'auditor/tests/draft/freeze changed during audit')
    receipt = {'version': VERSION, 'status': 'passed_independent_fixed_target_numerical_replay_not_admission',
        'passed': True, **SCOPE, 'result_COMMIT': commit_entry,
        'draft': binding(Path(draft_dir).resolve() / 'draft.json', DRAFT_SHA),
        'parent_freeze': binding(Path(freeze_path).resolve(), FREEZE_SHA),
        'audit_code': self_binding, 'audit_tests': tests_binding,
        'runtime': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__},
        'recordings': 90, 'selected_conditions': 270, 'codec_pairs': 540,
        'measurement_counts': counts, 'accepted_float64_pools_recomputed': 270 * POOLS,
        'successful_new_measurement_pools_recomputed': counts['successful'] * POOLS,
        'total_fixed_target_pools_recomputed': (270 + counts['successful']) * POOLS,
        'summary': summary_entry, 'result_product_count': len(inventory), 'maximum_errors': maxima,
        'producer_extractor_primitive_scalar_reducer_imported': False,
        'full_grid_numerically_replayed': False,
        'fixed_target_bins': list(TARGET), 'reserved_or_unused_audio_read': False,
        'interpretation': 'development codec integrity only; no threshold, gate, causality, or admission'}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open('xb') as stream:
        stream.write(canonical(receipt) + b'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', required=True)
    parser.add_argument('--result-commit-sha256', required=True)
    parser.add_argument('--draft-dir', required=True)
    parser.add_argument('--parent-freeze', required=True)
    parser.add_argument('--output-receipt', required=True)
    parser.add_argument('--audit-code-sha256', required=True)
    parser.add_argument('--audit-tests', required=True)
    parser.add_argument('--audit-tests-sha256', required=True)
    args = parser.parse_args()
    result = audit(args.result, args.result_commit_sha256, args.draft_dir, args.parent_freeze,
                   args.output_receipt, args.audit_code_sha256, args.audit_tests,
                   args.audit_tests_sha256)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
