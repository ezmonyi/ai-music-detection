#!/usr/bin/env python3
"""Independent reserved BC replay; no producer or frozen numerical imports.

Executing this auditor requires a terminal result COMMIT and an authorizing
parent freeze pinned by SHA. Merely importing/testing this file accesses no
reserved audio. Audit success is not BC admission and performs no classifier fit.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import statistics
import struct
import subprocess
import sys

import numpy as np
import scipy
import scipy.signal._upfirdn as scipy_upfirdn
import scipy.signal._upfirdn_apply as scipy_upfirdn_kernel
import scipy.special._ufuncs as scipy_special_ufuncs
from scipy.signal import upfirdn
from scipy.special import i0

VERSION = 'audit_bc_guitarset_reserved_admission_v1'
INDEPENDENT_CODEC_AUDITOR_SHA = 'd5d308e1dbc87365c04edc36bfc15e5377c5dd89dbe5d8db1da937fd79f99425'
DRAFT_SHA = '87803da916de04c35d226d7f9e41462d88ca52423ec1a935e723d327211b135e'
DRAFT_COMMIT_SHA = '5d0dcb01974cb08f52ca9d7898a38be4a727f5a6776dd8780ec544493c69f4ac'
SOURCE_COMMIT_SHA = 'aa33f0dfc55e3bbc6f48e618d1203c29e258608c309412d333f06c6e671b73dc'
ROSTER_SHA = '11c59e4984a127b0f99d6282fab65c4d34554b13c40e6473daafb14f4b3fb58b'
ORIGINAL_SHA = 'dcb8bfd41b89e3ab59bd241214ab64e36dbb7a5264409898bf3e423a03de8d6e'
CODEC_PRODUCER_SHA = '2c607b1175694b6966b3ce5200c6e306f20c40dd00de8f0229e6236387cc013e'
PRODUCER_VERSION = 'run_bc_guitarset_reserved_admission_v1'
PRODUCER_SHA = '87ee9ecd86856b001a7cd824905d9ee1aa22ebc058c78ebbb1d7bdce2b139819'
RATE, SAMPLES, POOL, FRAMES = 16000, 128000, 64000, 247
CONDITIONS = ('baseline', 'common_gain', 'polarity', 'closed_minus6db', 'independent_minus6db', 'closed_0db', 'independent_0db')
SELECTED = ('baseline', 'closed_minus6db', 'independent_minus6db')
CODECS = ('mp3_128k', 'opus_96k')
IDENTITY = ('item_id', 'player_id', 'score_id', 'performance', 'style_from_score_prefix', 'split_role')
EXPECTED = {'recordings': 90, 'float64': 630, 'precision': 270, 'codec': 540, 'measurements': 1440, 'pools': 2880}
PRODUCER_SCOPE = {'reserved_recordings': 90, 'development_audio_read': False, 'unused_audio_read': False,
                  'classifier_fits': 0, 'model_scoring': False, 'BC_admitted': False, 'independent_numerical_replay_passed': False}
FORBIDDEN = ('run_bc_guitarset_reserved_admission_v1', 'bicoherence_audio_v1', 'bicoherence_primitive_v2',
             'bicoherence_scalar_v1', 'bicoherence_guitarset_pilot_v1', 'run_bc_guitarset_codec_development_v1',
             'draft_bicoherence_guitarset_pilot_v2')


def load_independent_codec_auditor():
    path = Path(__file__).resolve().with_name('audit_bc_guitarset_codec_development_v1.py')
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != INDEPENDENT_CODEC_AUDITOR_SHA:
        raise ValueError('accepted independent codec auditor pin')
    spec = importlib.util.spec_from_file_location('_reserved_independent_codec_audit', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


a = load_independent_codec_auditor()
require, canonical, digest, binding, read_json, value_hash = a.require, a.canonical, a.digest, a.binding, a.read_json, a.value_hash
GRID = np.asarray([(x, y, x + y) for x in range(8, 193, 8) for y in range(x, 193, 8) if x + y <= 256], dtype=np.int64)
TARGET_INDEX = next(i for i, x in enumerate(GRID.tolist()) if x == [32, 48, 80])


def forbid_producer_imports():
    require(not any(key in sys.modules or any(Path(str(getattr(v, '__file__', ''))).stem == key
        for v in list(sys.modules.values()) if v is not None) for key in FORBIDDEN), 'forbidden producer/numerical module imported')


def exact_array(left, right, message):
    left, right = np.asarray(left), np.asarray(right)
    require(left.dtype == right.dtype and left.shape == right.shape and
            left.tobytes(order='C') == right.tobytes(order='C'), message)


def native_pcm16(path, row):
    """Decode the protocol's PCM16 RIFF independently of libsndfile."""
    require(binding(path) == row['source_audio'], 'native source binding')
    data = Path(path).read_bytes(); require(data[:4] == b'RIFF' and data[8:12] == b'WAVE' and
        len(data) == struct.unpack_from('<I', data, 4)[0] + 8, 'native RIFF header/length')
    offset, fmt, payload = 12, None, None
    while offset < len(data):
        require(offset + 8 <= len(data), 'native truncated chunk')
        name, size = data[offset:offset + 4], struct.unpack_from('<I', data, offset + 4)[0]
        stop = offset + 8 + size; require(stop <= len(data), 'native chunk outside RIFF')
        body = data[offset + 8:stop]
        if name == b'fmt ':
            require(fmt is None and size >= 16, 'native single format')
            fmt = struct.unpack_from('<HHIIHH', body)
            require(fmt == (1, 1, 44100, 88200, 2, 16), 'native mono44.1k PCM16 only')
        elif name == b'data':
            require(payload is None and size % 2 == 0, 'native single PCM data'); payload = body
        offset = stop + (size & 1)
    require(offset == len(data) and fmt is not None and payload is not None, 'native chunk layout')
    x = np.frombuffer(payload, dtype='<i2').astype(np.float64) / 32768.0
    facts = row['historical_native_decode']
    require(facts['format'] == 'WAV' and facts['subtype'] == 'PCM_16' and facts['sample_rate_hz'] == 44100 and
        facts['channels'] == 1 and len(x) == facts['decoded_frames'] == facts['header_frames'] and
        hashlib.sha256(x.astype('<f8').tobytes()).hexdigest() == row['historical_native_float64_pcm_sha256'] == facts['decoded_pcm_sha256'],
        'native decode facts/PCM hash')
    return x


def native_resample(x):
    """Independent FIR design/padding/indexing; upfirdn is the convolution kernel.

    Never calls resample_poly, firwin, get_window or the frozen construction code.
    The coefficients use normalized sinc times symmetric Kaiser5 with unit DC
    response, then the rational gain160. Filter origin and output count implement
    whole-record zero-extension, not crop-then-resample.
    """
    require(isinstance(x, np.ndarray) and x.dtype == np.float64 and x.ndim == 1 and len(x) > 0 and np.isfinite(x).all(), 'native finite FLOAT64')
    half, up, down = 4410, 160, 441
    index = np.arange(-half, half + 1, dtype=np.float64)
    cutoff = 1.0 / down
    taps = cutoff * np.sinc(cutoff * index)
    taps *= i0(5.0 * np.sqrt(1.0 - (index / half) ** 2)) / i0(5.0)
    taps /= np.sum(taps); taps *= up
    prepad = down - half % down; remove = (half + prepad) // down
    expected = (len(x) * up + down - 1) // down
    padded = np.r_[np.zeros(prepad, dtype=np.float64), taps]
    y = upfirdn(padded, x, up=up, down=down, mode='constant', cval=0.0)[remove:remove + expected]
    require(len(y) == expected and np.isfinite(y).all(), 'independent whole-record resampling support')
    return y


def independent_crop(native, row):
    full = native_resample(native); start = (len(full) - SAMPLES) // 2
    require(start >= 0, 'short source, no padding'); crop = full[start:start + SAMPLES].copy()
    plan = row['prospective_preprocessing']
    require(plan['native_frames'] == len(native) and plan['native_sample_rate_hz'] == 44100 and
        plan['expected_resampled_frames'] == len(full) and plan['crop_start_sample'] == start and
        plan['crop_stop_sample_exclusive'] == start + SAMPLES and plan['padding_samples'] == 0 and
        plan['up'] == 160 and plan['down'] == 441 and plan['window'] == ['kaiser', 5.0] and
        plan['padtype'] == 'constant' and plan['cval'] == 0.0 and plan['resample_scope'] == 'entire_native_recording_before_crop', 'crop plan arithmetic')
    return crop


def construct(crop, item_id):
    require(crop.dtype == np.float64 and crop.shape == (SAMPLES,) and np.isfinite(crop).all(), 'construction crop')
    seed = int.from_bytes(hashlib.sha256(('BC-GuitarSet-injection-20260907|' + item_id).encode()).digest()[:8], 'big')
    knots = np.random.Generator(np.random.PCG64(seed)).uniform(-np.pi, np.pi, size=(3, 65))
    unwrapped = np.unwrap(knots, axis=1)
    knot_times, time = np.arange(65, dtype=np.float64) / 8, np.arange(SAMPLES, dtype=np.float64) / RATE
    phases = np.asarray([np.interp(time, knot_times, row) for row in unwrapped])
    closed_phase = np.asarray([phases[0], phases[1], phases[0] + phases[1]])
    carriers = 2 * np.pi * np.asarray([500., 750., 1250.])[:, None] * time
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        prototypes = [0.2 * np.cos(carriers + phi).sum(axis=0) for phi in (closed_phase, phases)]
        rms = [float(np.sqrt(np.mean(wave ** 2))) for wave in prototypes]
        require(all(v > 0 for v in rms), 'prototype RMS')
        background = 0.25 * crop; bg_rms = float(np.sqrt(np.mean(background ** 2)))
        require(not np.any((crop != 0) & (background == 0)) and not (bg_rms == 0 and np.any(background)), 'unsupported background arithmetic')
        units = [wave / level for wave, level in zip(prototypes, rms, strict=True)]
        waves = {'baseline': background, 'common_gain': 0.1 * background, 'polarity': -background}
        arrays = {'standardized_crop': crop, 'background': background, 'sample_times_seconds': time,
            'phase_knot_times_seconds': knot_times, 'phase_knots_wrapped': knots, 'phase_knots_unwrapped': unwrapped,
            'independent_phase_trajectories': phases, 'closed_phase_trajectories': closed_phase,
            'closed_prototype': prototypes[0], 'independent_prototype': prototypes[1],
            'closed_unit_rms': units[0], 'independent_unit_rms': units[1], 'prototype_rms': np.asarray(rms), 'background_rms': np.asarray(bg_rms)}
        for label, db in (('minus6db', -6), ('0db', 0)):
            for kind, unit in zip(('closed', 'independent'), units, strict=True):
                injection = bg_rms * 10 ** (db / 20) * unit
                arrays[kind + '_' + label + '_injection'] = injection; waves[kind + '_' + label] = background + injection
    require(tuple(waves) == CONDITIONS and all(np.isfinite(v).all() for v in waves.values()), 'constructed conditions')
    meta = {'item_id': item_id, 'seed': seed, 'bit_generator': 'PCG64', 'status': 'ok' if bg_rms > 0 else 'unsupported_zero_background_rms',
        'background_rms': bg_rms, 'prototype_rms': dict(zip(('closed', 'independent'), rms)),
        'prototype_normalization_factors': {kind: 1 / v for kind, v in zip(('closed', 'independent'), rms)},
        'injected_full_record_rms_matched': bg_rms > 0, 'framewise_power_matched': False, 'mixtures_clipped_or_normalized': False,
        'zero_rms_interpretation': 'zero derivatives retained; relative-level intervention unsupported'}
    return waves, arrays, meta


def primitive(coefficients, bins):
    """Independent amplitude-weighted normalized triple-product calculation."""
    columns = coefficients[:, bins]; scales = np.maximum(np.abs(columns.real).max(axis=0), np.abs(columns.imag).max(axis=0))
    result = {'version': 'bicoherence_primitive_v2', 'realizations': len(coefficients), 'frequency_bins': list(map(int, bins)),
        'status': 'ok', 'squared_bicoherence': None, 'biphase_radians': None, 'biphase_status': 'missing_triad_energy',
        'biphase_interpretation': 'descriptive_only_no_significance', 'coefficient_scales': scales.tolist(),
        'normalized_sums': None, 'raw_sums': None, 'raw_sum_encoding': 'value = mantissa * 2**exponent2',
        'external_validation_passed': False, 'classifier_admitted': False}
    if np.any(scales == 0):
        result['status'] = 'missing_triad_energy' if np.any(coefficients) else 'zero_energy'; return result
    normalized = np.empty(columns.shape, np.complex128)
    normalized.real, normalized.imag = columns.real / scales, columns.imag / scales
    u, v = normalized[:, 0] * normalized[:, 1], normalized[:, 2]
    e1, e2, z = float(np.sum(np.abs(u) ** 2)), float(np.sum(np.abs(v) ** 2)), complex(np.sum(u * v.conjugate()))
    sums = dict(product_energy_sum=e1, sum_frequency_energy_sum=e2, triple_sum_real=z.real, triple_sum_imag=z.imag)
    def scaled(value, factors):
        mantissa, exponent = math.frexp(value)
        if mantissa == 0: return {'mantissa': 0.0, 'exponent2': 0}
        for factor in factors:
            m, e = math.frexp(float(factor)); mantissa, correction = math.frexp(mantissa * m); exponent += e + correction
        return {'mantissa': mantissa, 'exponent2': exponent}
    x, y, zscale = scales
    result['normalized_sums'] = sums
    result['raw_sums'] = {key: scaled(sums[key], factors) for key, factors in {
        'product_energy_sum': (x, x, y, y), 'sum_frequency_energy_sum': (zscale, zscale),
        'triple_sum_real': (x, y, zscale), 'triple_sum_imag': (x, y, zscale)}.items()}
    if e1 == 0 or e2 == 0: result['status'] = 'missing_triad_product_energy'; return result
    score = (abs(z) / math.sqrt(e1) / math.sqrt(e2)) ** 2
    require(math.isfinite(score) and 0 <= score <= 1 + 64 * np.finfo(float).eps, 'primitive Cauchy-Schwarz bound')
    result['squared_bicoherence'] = min(1., score)
    result['biphase_status'] = 'undefined_zero_resultant' if z == 0 else 'descriptive_only_no_significance'
    if z != 0: result['biphase_radians'] = math.atan2(z.imag, z.real)
    return result


def descriptor(metadata):
    return {'pools': [{'pool_index': i, 'pool_status': pool['status'], 'target': pool['cells'][TARGET_INDEX],
        'grid_cell_count': len(pool['cells']), 'eligible_cell_count': sum(c['eligible'] for c in pool['cells']),
        'grid_eligibility': [c['eligible'] for c in pool['cells']], 'grid_status': [c['status'] for c in pool['cells']],
        'grid_squared_bicoherence': [c['squared_bicoherence'] for c in pool['cells']],
        'grid_raw_squared_bicoherence': [c['primitive']['squared_bicoherence'] for c in pool['cells']]} for i, pool in enumerate(metadata['pools'])]}


def validate_array_schema(arrays):
    specs = {'window': ((1024,), np.float64), 'frequency_bins': ((228, 3), np.int64), 'frequency_hz': ((228, 3), np.float64),
        'pool_start_samples': ((2,), np.int64), 'frame_offset_samples': ((247,), np.int64), 'frame_start_samples': ((2, 247), np.int64),
        'frame_means': ((2, 247), np.float64), 'spectra': ((2, 247, 513), np.complex128),
        'bin_coefficient_energy': ((2, 513), np.float64), 'total_non_dc_coefficient_energy': ((2,), np.float64),
        'bin_energy_fraction': ((2, 513), np.float64),
        **{key: ((2, 228), np.bool_) for key in ('energy_floor_mask', 'primitive_defined_mask', 'eligible_mask')},
        **{key: ((2, 228), np.float64) for key in ('squared_bicoherence', 'biphase_radians', 'primitive_squared_bicoherence', 'primitive_biphase_radians')}}
    require(set(arrays.files) == set(specs), 'exact numerical array set')
    for key, (shape, dtype) in specs.items():
        value = arrays[key]; require(value.shape == shape and value.dtype == np.dtype(dtype), 'NPZ dtype/shape: ' + key)
        if value.dtype.kind in 'fc':
            missing_allowed = key in ('bin_energy_fraction', 'squared_bicoherence', 'biphase_radians', 'primitive_squared_bicoherence', 'primitive_biphase_radians')
            require(not np.isinf(value).any() and (missing_allowed or np.isfinite(value).all()), 'NPZ nonfinite: ' + key)
    exact_array(arrays['frequency_bins'], GRID, 'fixed228cell grid')
    exact_array(arrays['frequency_hz'], GRID.astype(np.float64) * RATE / 1024, 'grid frequency Hz')
    starts, offsets = np.arange(2, dtype=np.int64) * POOL, np.arange(FRAMES, dtype=np.int64) * 256
    exact_array(arrays['pool_start_samples'], starts, 'pool start samples')
    exact_array(arrays['frame_offset_samples'], offsets, 'frame offset samples')
    exact_array(arrays['frame_start_samples'], starts[:, None] + offsets[None, :], 'absolute frame starts')


def verify_numeric(samples, prefix, reported, *, full_grid=False, construction_status=None):
    """Replay target for every representation and every nuisance grid cell."""
    require(samples.dtype == np.float64 and samples.shape == (SAMPLES,), 'measurement exact FLOAT64 input')
    meta, meta_entry = read_json(str(prefix) + '.metadata.json'); arrays_entry = binding(str(prefix) + '.arrays.npz')
    reduction, pools = a.reduce_waveform(samples, construction_status)
    a.compare(reported, reduction, 'independent scalar')
    constants = {'version': 'bicoherence_audio_v1', 'primitive_version': 'bicoherence_primitive_v2', 'input_samples': SAMPLES,
        'sample_rate_hz': RATE, 'pool_samples': POOL, 'pool_count': 2, 'analyzed_samples': SAMPLES, 'discarded_tail_samples': 0,
        'n_fft': 1024, 'hop_samples': 256, 'frames_per_pool': FRAMES, 'energy_fraction_min_inclusive': 1e-6,
        'grid_cell_count': 228, 'null_calibrated': False, 'significance_inferred': False, 'external_validation_passed': False,
        'classifier_admitted': False, 'frames_overlap_and_are_not_asserted_independent': True}
    require(all(meta.get(k) == v for k, v in constants.items()) and len(meta['pools']) == 2 and
            meta.get('construction_status') == construction_status, 'measurement metadata constants')
    with np.load(arrays_entry['path'], allow_pickle=False) as arrays:
        validate_array_schema(arrays)
        exact_array(arrays['window'], pools[0]['window'], 'periodic Hann')
        for p, calculated in enumerate(pools):
            raw = meta['pools'][p]; require(raw['pool_index'] == p and raw['start_sample'] == p * POOL and
                raw['stop_sample_exclusive'] == (p + 1) * POOL and raw['coefficient_rows'] == FRAMES and
                raw['status'] == calculated['pool_status'] and [c['frequency_bins'] for c in raw['cells']] == GRID.tolist(), 'pool/grid identity')
            a.close(arrays['frame_means'][p], calculated['means'], 'frame means')
            a.close(arrays['spectra'][p], calculated['spectra'], 'all STFT coefficients')
            a.close(arrays['bin_coefficient_energy'][p], calculated['energy'], 'bin energy')
            a.close(arrays['total_non_dc_coefficient_energy'][p], calculated['total_non_dc_coefficient_energy'], 'nonDC denominator')
            a.close(arrays['bin_energy_fraction'][p], calculated['fractions'], 'bin fractions')
            for c in (range(len(GRID)) if full_grid else (TARGET_INDEX,)):
                bins = GRID[c]; prim = primitive(calculated['spectra'], bins); total = calculated['total_non_dc_coefficient_energy']
                fractions = calculated['fractions'][bins].tolist() if total else [None] * 3
                floor = bool(total and all(x >= 1e-6 for x in fractions)); defined = prim['status'] == 'ok' and prim['squared_bicoherence'] is not None
                eligible = floor and defined
                cell = {'frequency_bins': bins.tolist(), 'status': 'ok' if eligible else prim['status'] if not defined else 'below_energy_fraction_floor',
                    'energy_fractions': fractions, 'energy_floor_passed': floor, 'eligible': eligible,
                    'squared_bicoherence': prim['squared_bicoherence'] if eligible else None,
                    'biphase_radians': prim['biphase_radians'] if eligible else None, 'primitive': prim}
                a.compare(raw['cells'][c], cell, 'independent grid/target cell')
                for key, value in (('energy_floor_mask', floor), ('primitive_defined_mask', defined), ('eligible_mask', eligible)):
                    require(arrays[key].dtype == np.bool_ and bool(arrays[key][p, c]) == value, key)
                for key, value in (('squared_bicoherence', cell['squared_bicoherence']), ('biphase_radians', cell['biphase_radians']),
                                   ('primitive_squared_bicoherence', prim['squared_bicoherence']), ('primitive_biphase_radians', prim['biphase_radians'])):
                    observed = arrays[key][p, c]
                    require(np.isnan(observed) if value is None else np.isclose(observed, value, atol=2e-12, rtol=2e-11), key)
                # Independently calculated target/nuisance values feed all aggregate arithmetic.
                raw['cells'][c] = cell
    return reduction, descriptor(meta), meta_entry, arrays_entry


def outcome(record):
    require(isinstance(record, dict) and set(record) == {'attempted', 'status'} and type(record['attempted']) is bool and
            record['status'] in ('success', 'measurement_failure', 'skipped_pipeline_failure', 'skipped_dependency_failure'), 'attempt schema')
    require(record['attempted'] == (record['status'] in ('success', 'measurement_failure')), 'attempt/failure distinction')
    return record['status'] == 'success'


def failure(entry, expected_status):
    value, actual = read_json(entry['path']); require(actual == entry and value.get('status') == expected_status and
        all(isinstance(value.get(key), str) for key in ('type', 'message', 'traceback')), 'retained processing failure evidence')
    return value


def codec_commands(prefix, ffmpeg, recipe):
    encoded = Path(str(prefix) + recipe['extension']); decoded = Path(str(prefix) + '.decoded.wav')
    common = [ffmpeg, '-hide_banner', '-nostdin', '-n', '-loglevel', 'info']
    stream = ['-map', '0:a:0', '-vn', '-sn', '-dn', '-ac', '1', '-ar', '16000', '-threads', '1']
    encode = common + ['-i', str(prefix.parent / 'float32_control.wav')] + stream + [
        '-c:a', recipe['encoder'], '-b:a', recipe['bitrate'], *recipe['encoder_options'], '-map_metadata', '-1', str(encoded)]
    decode = common + ['-i', str(encoded)] + stream + ['-c:a', 'pcm_f32le', '-map_metadata', '-1', str(decoded)]
    return encode, decode, encoded, decoded


def codec_log(path, command, *, successful):
    value, entry = read_json(path)
    require(set(value) == {'command', 'environment', 'returncode', 'stdout_utf8', 'stderr_utf8'} and value['command'] == command and
        value['environment'] == {'LANG': 'C', 'LC_ALL': 'C', 'PATH': '/usr/bin:/bin'} and type(value['returncode']) is int and
        (not successful or value['returncode'] == 0) and isinstance(value['stdout_utf8'], str) and isinstance(value['stderr_utf8'], str), 'codec command/log evidence')
    return entry


def measurement32(samples, prefix, reduction):
    reduced, _, metadata, arrays = verify_numeric(samples.astype(np.float64), prefix, reduction)
    return reduced, {'status': 'measured_not_admitted',
        'input_float32_pcm_sha256': hashlib.sha256(samples.astype('<f4').tobytes()).hexdigest(),
        'extractor_float64_pcm_sha256': hashlib.sha256(samples.astype('<f8').tobytes()).hexdigest(),
        'metadata': metadata, 'arrays': arrays, 'scalar': reduced, 'processing_failure': None}


def verify_codec(row, condition, item, wave, directory, freeze):
    reported, _ = read_json(directory / 'receipt.json')
    require({key: reported[key] for key in IDENTITY} == {key: row[key] for key in IDENTITY} and
            reported['condition'] == condition, 'codec receipt identity')
    require(set(reported['measurement_status']) == {'float32_control', *CODECS} and
        set(reported['representations']) == {'accepted_float64', 'float32_control', *CODECS}, 'codec exact planned representations')
    a.check_file_bindings(reported['artifacts'])
    original = item['conditions'][condition]
    if original['scalar'] is None:
        require(reported['source_float64_wav'] is None and reported['source_metadata'] == original['failure'] and
            reported['source_arrays'] == original['failure'] and reported['artifacts'] == {'dependency_failure': original['failure']} and
            all(status == {'attempted': False, 'status': 'skipped_dependency_failure'} for status in reported['measurement_status'].values()) and
            all(value is None for value in reported['representations'].values()) and
            set(p.name for p in directory.iterdir()) == {'receipt.json'}, 'construction/float64 dependency skip graph')
    else:
        require(reported['source_float64_wav'] == original['waveform'] and reported['source_metadata'] == original['metadata'] and
                reported['source_arrays'] == original['arrays'], 'codec FLOAT64 parent products')
    expected_reps = {'accepted_float64': original['scalar']}; control = None
    state = reported['measurement_status']['float32_control']; success = outcome(state)
    if success:
        require(original['scalar'] is not None and wave is not None, 'precision success missing parent')
        require(not (directory / 'float32_control.failure.json').exists(), 'successful precision has retained unaccounted failure')
        control, _ = a.read_ieee_float_wav(directory / 'float32_control.wav', bits=32)
        exact_array(control, wave.astype(np.float32), 'exact FLOAT64-to-FLOAT32 cast including sign bits')
        require(reported.get('source_float64_pcm_sha256') == hashlib.sha256(wave.astype('<f8').tobytes()).hexdigest() and
            reported.get('float64_to_float32_operation') == 'NumPy value cast once; no gain, normalization, clipping, trim, or pad', 'precision operation provenance')
        reduced, product = measurement32(control, directory / 'float32_control', reported['representations']['float32_control'])
        a.compare(reported['artifacts']['float32_control'], {'wav': binding(directory / 'float32_control.wav'), 'measurement': product}, 'precision product graph')
    else:
        reduced = None
        if original['scalar'] is not None:
            require(state['status'] in ('measurement_failure', 'skipped_pipeline_failure'), 'precision failure class')
            require(reported['artifacts']['float32_control']['failure'] == binding(directory / 'float32_control.failure.json'), 'precision failure exact path')
            failure(reported['artifacts']['float32_control']['failure'], 'processing_failure_not_scientific_missingness')
            if (directory / 'float32_control.wav').exists():
                partial, _ = a.read_ieee_float_wav(directory / 'float32_control.wav', bits=32)
                exact_array(partial, wave.astype(np.float32), 'failed precision retained cast')
    expected_reps['float32_control'] = reduced
    for name in CODECS:
        state = reported['measurement_status'][name]; success = outcome(state); prefix = directory / name
        encode, decode, encoded, decoded = codec_commands(prefix, freeze['toolchain']['ffmpeg']['path'], freeze['codec_recipes'][name])
        if success:
            require(reduced is not None, 'codec success requires measured precision control')
            require(not Path(str(prefix) + '.failure.json').exists() and not Path(str(prefix) + '.skipped.json').exists(), 'successful codec has retained unaccounted failure')
            encode_log = codec_log(str(prefix) + '.encode.log.json', encode, successful=True)
            decode_log = codec_log(str(prefix) + '.decode.log.json', decode, successful=True)
            samples, _ = a.read_ieee_float_wav(decoded, bits=32)
            scalar, products = measurement32(samples, prefix, reported['representations'][name])
            a.compare(reported['artifacts'][name], {'recipe': freeze['codec_recipes'][name], 'encoded': binding(encoded),
                'decoded_wav': binding(decoded), 'encode_log': encode_log, 'decode_log': decode_log, 'measurement': products}, 'codec full product graph')
        else:
            scalar = None
            if original['scalar'] is not None:
                expected = 'skipped_due_to_float32_control_failure_not_scientific_missingness' if reduced is None else 'processing_failure_not_scientific_missingness'
                require(reported['artifacts'][name]['failure_or_skip'] == binding(str(prefix) + ('.skipped.json' if reduced is None else '.failure.json')), 'codec failure/skip exact path')
                failure(reported['artifacts'][name]['failure_or_skip'], expected)
                require((state['status'] == 'skipped_dependency_failure') == (reduced is None), 'codec dependency failure semantics')
                if reduced is None:
                    require(not any(p.exists() for p in (encoded, decoded, Path(str(prefix) + '.encode.log.json'), Path(str(prefix) + '.decode.log.json'))), 'dependency skip cannot produce codec artifacts')
                else:
                    for suffix, command in (('.encode.log.json', encode), ('.decode.log.json', decode)):
                        if Path(str(prefix) + suffix).exists(): codec_log(str(prefix) + suffix, command, successful=False)
            require(reported['representations'][name] is None, 'processing failure cannot become scientific null reduction')
        expected_reps[name] = scalar
    comparisons = {'float32_minus_float64': a.compare_scalars(expected_reps['accepted_float64'], reduced, 'float32_minus_float64'),
        'decoded_minus_float32': {name: a.compare_scalars(reduced, expected_reps[name], 'decoded_minus_float32') for name in CODECS}}
    a.compare(reported['representations'], expected_reps, 'independent codec representations')
    a.compare(reported['comparisons'], comparisons, 'independent codec differences')
    return {**reported, 'representations': expected_reps, 'comparisons': comparisons}


def verify_item(row, directory, freeze):
    item, _ = read_json(directory / 'receipt.json')
    require({key: item[key] for key in IDENTITY} == {key: row[key] for key in IDENTITY} and row['split_role'] == 'reserved' and
        item['source_audio'] == row['source_audio'] and set(item['conditions']) == set(item['condition_status']) == set(CONDITIONS) and
        set(item['codec_receipts']) == set(SELECTED), 'reserved item exact identity/condition sets')
    a.check_file_bindings(item)
    native = native_pcm16(row['source_audio']['path'], row); crop = independent_crop(native, row)
    waves, expected_arrays, construction = construct(crop, row['item_id'])
    if item['construction']['status'] == 'success':
        require(not (directory / 'construction.failure.json').exists(), 'successful construction has retained failure')
        meta, meta_entry = read_json(directory / 'construction.json')
        require(item['construction'] == {'status': 'success', 'metadata': meta_entry, 'arrays': binding(directory / 'construction.arrays.npz')}, 'construction binding graph')
        with np.load(directory / 'construction.arrays.npz', allow_pickle=False) as arrays:
            require(set(arrays.files) == set(expected_arrays), 'construction arrays exact set')
            for key, value in expected_arrays.items(): exact_array(arrays[key], value, 'bit-exact independent construction.' + key)
        for key, value in construction.items(): a.compare(meta[key], value, 'construction.' + key)
        require(meta['prospective_preprocessing'] == row['prospective_preprocessing'] and meta['native_pcm_sha256'] == row['historical_native_float64_pcm_sha256'], 'construction native/crop lineage')
        actual = meta['actual_preprocessing']; plan = row['prospective_preprocessing']
        require(actual['native_frames'] == len(native) and actual['native_sample_rate_hz'] == 44100 and actual['target_sample_rate_hz'] == RATE and
            actual['resampled_frames'] == plan['expected_resampled_frames'] and actual['crop_start'] == plan['crop_start_sample'] and
            actual['crop_stop_exclusive'] == plan['crop_stop_sample_exclusive'] and actual['crop_frames'] == SAMPLES and
            actual['up'] == 160 and actual['down'] == 441 and actual['window'] == ['kaiser', 5.0] and actual['padtype'] == 'constant' and
            actual['cval'] == 0.0 and actual['resample_scope'] == 'entire_native_recording_before_crop' and
            actual['crop_pcm_sha256'] == hashlib.sha256(crop.astype('<f8').tobytes()).hexdigest(), 'actual independent preprocessing')
    else:
        require(item['construction']['status'] == 'processing_failure', 'construction failure status')
        require(item['construction']['failure'] == binding(directory / 'construction.failure.json'), 'construction failure exact path')
        failure(item['construction']['failure'], 'processing_failure_not_scientific_missingness')
        require(all(v == {'attempted': False, 'status': 'skipped_dependency_failure'} for v in item['condition_status'].values()), 'construction failure must skip all seven conditions')
    conditions = {}
    for condition in CONDITIONS:
        measured = item['conditions'][condition]; state = item['condition_status'][condition]; success = outcome(state)
        prefix = directory / 'float64' / condition
        if success:
            require(item['construction']['status'] == 'success' and measured['status'] == 'measured_reserved_not_admitted', 'FLOAT64 success construction/status')
            require(not Path(str(prefix) + '.failure.json').exists(), 'successful FLOAT64 has retained unaccounted failure')
            samples, _ = a.read_ieee_float_wav(str(prefix) + '.wav', bits=64)
            exact_array(samples, waves[condition], 'independent FLOAT64 waveform')
            # All seven full grids are replayed because retained original descriptors include all cells;
            # only baseline/gain/polarity grid values participate in nuisance requirements.
            scalar, desc, metadata, arrays = verify_numeric(samples, prefix, measured['scalar'], full_grid=True, construction_status=construction['status'])
            expected = {'status': 'measured_reserved_not_admitted', 'waveform': binding(str(prefix) + '.wav'),
                        'metadata': metadata, 'arrays': arrays, 'scalar': scalar, 'descriptor': desc}
            a.compare(measured, expected, 'FLOAT64 measured receipt'); conditions[condition] = expected
        else:
            expected_status = 'processing_failure_not_scientific_missingness' if state['attempted'] else 'skipped_due_to_construction_failure_not_scientific_missingness'
            require(measured['failure'] == binding(str(prefix) + '.failure.json'), 'FLOAT64 failure exact path')
            failure(measured['failure'], expected_status)
            require(measured['status'] == 'failure_or_skip' and measured['scalar'] is None and measured['descriptor'] is None and
                    state['attempted'] == (item['construction']['status'] == 'success'), 'FLOAT64 failure/null distinction')
            conditions[condition] = measured
    rebuilt = {**item, 'conditions': conditions}
    rebuilt['_codec_values'] = []
    for condition in SELECTED:
        directory32 = directory / 'codec' / condition
        require(item['codec_receipts'][condition] == binding(directory32 / 'receipt.json'), 'selected codec receipt binding')
        rebuilt['_codec_values'].append(verify_codec(row, condition, rebuilt, waves[condition], directory32, freeze))
    return rebuilt


def counts(items):
    groups = {'float64': [i['condition_status'][c] for i in items for c in CONDITIONS],
        'float32_precision_control': [r['measurement_status']['float32_control'] for i in items for r in i['_codec_values']],
        'codec_decoded': [r['measurement_status'][c] for i in items for r in i['_codec_values'] for c in CODECS]}
    result = {}
    for group, records in groups.items():
        for record in records: outcome(record)
        result[group] = {'expected': len(records), 'attempted': sum(x['attempted'] for x in records),
            'successful': sum(x['status'] == 'success' for x in records), 'failed_after_attempt': sum(x['status'] == 'measurement_failure' for x in records),
            'skipped_before_attempt': sum(not x['attempted'] for x in records)}
        require(result[group]['expected'] == result[group]['successful'] + result[group]['failed_after_attempt'] + result[group]['skipped_before_attempt'], 'planned denominator accounting')
    result['all'] = {key: sum(record[key] for record in result.values()) for key in next(iter(result.values()))}
    return result


def nuisance(first, second):
    records = []
    for left, right in zip(first['pools'], second['pools'], strict=True):
        masks = list(zip(left['grid_eligibility'], right['grid_eligibility'], strict=True))
        values = list(zip(left['grid_squared_bicoherence'], right['grid_squared_bicoherence'], strict=True))
        raw = list(zip(left['grid_raw_squared_bicoherence'], right['grid_raw_squared_bicoherence'], strict=True))
        diffs = [abs(x - y) for x, y in values if x is not None and y is not None]
        rawdiffs = [abs(x - y) for x, y in raw if x is not None and y is not None]
        ta, tb = left['target'], right['target']
        records.append({'pool_index': left['pool_index'], 'grid_denominator': len(masks),
            'grid_eligibility_agreement_count': sum(x == y for x, y in masks), 'missing_to_finite_count': sum(not x and y for x, y in masks),
            'finite_to_missing_count': sum(x and not y for x, y in masks), 'common_finite_cell_count': len(diffs),
            'maximum_finite_b2_difference': max(diffs) if diffs else None, 'primary_difference_scope': 'reported_energy_masked_b2_common_eligible_cells',
            'raw_common_finite_cell_count': len(rawdiffs), 'raw_maximum_finite_b2_difference': max(rawdiffs) if rawdiffs else None,
            'raw_missing_to_finite_count': sum(x is None and y is not None for x, y in raw), 'raw_finite_to_missing_count': sum(x is not None and y is None for x, y in raw),
            'target_eligibility_agreement': ta['eligible'] == tb['eligible'], 'target_missing_to_finite': not ta['eligible'] and tb['eligible'],
            'target_finite_to_missing': ta['eligible'] and not tb['eligible'],
            'target_difference': tb['squared_bicoherence'] - ta['squared_bicoherence'] if ta['eligible'] and tb['eligible'] else None})
    return records


def historical_pair_summary(rows):
    values = [r['difference'] for r in rows if r['difference'] is not None]
    groups = {}
    for key in ('score_id', 'player_id'):
        grouped = defaultdict(list)
        for row in rows: grouped[row[key]].append(row)
        records = []
        for identity, subgroup in sorted(grouped.items()):
            v = [r['difference'] for r in subgroup if r['difference'] is not None]
            records.append({key: identity, 'recording_denominator': len(subgroup), 'covered_recordings': len(v),
                'pool_denominator': sum(r['pool_denominator'] for r in subgroup), 'paired_covered_pools': sum(r['covered_pools'] for r in subgroup),
                'mean_difference': statistics.fmean(v) if v else None})
        v = [r['mean_difference'] for r in records if r['mean_difference'] is not None]
        groups[key] = {'group_denominator': len(records), 'covered_groups': len(v), 'equal_group_mean_difference': statistics.fmean(v) if v else None, 'rows': records}
    return {'recording_denominator': len(rows), 'covered_recordings': len(values), 'pool_denominator': sum(r['pool_denominator'] for r in rows),
        'paired_covered_pools': sum(r['covered_pools'] for r in rows), 'positive_recordings': sum(v > 0 for v in values),
        'zero_recordings': sum(v == 0 for v in values), 'negative_recordings': sum(v < 0 for v in values),
        'equal_score': groups['score_id'], 'equal_player_secondary': groups['player_id'], 'per_recording': rows}


def original_summary(items):
    levels = {}
    for level in ('minus6db', '0db'):
        rows = [{key: item[key] for key in ('item_id', 'player_id', 'score_id', 'performance')} |
            a.condition_difference(item['conditions']['closed_' + level]['scalar'], item['conditions']['independent_' + level]['scalar']) for item in items]
        levels[level] = a.contrast_summary(rows)
    complete = all(item['conditions'][c]['descriptor'] is not None for item in items for c in CONDITIONS)
    historical = None
    if complete:
        records = [{key: item[key] for key in ('item_id', 'player_id', 'score_id', 'performance', 'style_from_score_prefix')} |
            {'conditions': {c: item['conditions'][c]['descriptor'] for c in CONDITIONS},
             'nuisance': {c: nuisance(item['conditions']['baseline']['descriptor'], item['conditions'][c]['descriptor']) for c in ('common_gain', 'polarity')}} for item in items]
        baseline = [p for r in records for p in r['conditions']['baseline']['pools']]
        historical = {'version': PRODUCER_VERSION, 'dataset_role': 'reserved', 'reused_numerical_implementation': {
            'version': 'bicoherence_guitarset_development_pilot_v1', 'sha256': ORIGINAL_SHA, 'development_policy_not_represented_as_reserved_scope': True},
            'recording_denominator': len(items), 'independent_replay_status': 'required_before_accepting_values',
            'baseline': {'target_pool_denominator': len(baseline), 'target_covered_pools': sum(p['target']['eligible'] for p in baseline),
                'grid_cell_denominator': sum(p['grid_cell_count'] for p in baseline), 'grid_eligible_cells': sum(p['eligible_cell_count'] for p in baseline)},
            'levels': {}, 'nuisance': {}, 'per_recording': records}
        for level in ('minus6db', '0db'):
            paired = []
            for record in records:
                poolrows = []
                for cp, ip in zip(record['conditions']['closed_' + level]['pools'], record['conditions']['independent_' + level]['pools'], strict=True):
                    c, i = cp['target'], ip['target']
                    poolrows.append({'pool_index': cp['pool_index'], 'closed_eligible': c['eligible'], 'independent_eligible': i['eligible'],
                        'closed_status': c['status'], 'independent_status': i['status'], 'closed_b2': c['squared_bicoherence'],
                        'independent_b2': i['squared_bicoherence'], 'difference': c['squared_bicoherence'] - i['squared_bicoherence'] if c['eligible'] and i['eligible'] else None})
                v = [p['difference'] for p in poolrows if p['difference'] is not None]
                paired.append({key: record[key] for key in ('item_id', 'player_id', 'score_id', 'performance', 'style_from_score_prefix')} |
                    {'pool_denominator': len(poolrows), 'covered_pools': len(v), 'difference': statistics.fmean(v) if v else None, 'paired_pools': poolrows})
            historical['levels'][level] = historical_pair_summary(paired) | {'performance_strata': {
                label: historical_pair_summary([r for r in paired if r['performance'] == label]) for label in ('comp', 'solo')}}
        for condition in ('common_gain', 'polarity'):
            checks = [p for r in records for p in r['nuisance'][condition]]
            maxima = [p['maximum_finite_b2_difference'] for p in checks if p['maximum_finite_b2_difference'] is not None]
            rawmax = [p['raw_maximum_finite_b2_difference'] for p in checks if p['raw_maximum_finite_b2_difference'] is not None]
            historical['nuisance'][condition] = {key: sum(p[key] for p in checks) for key in (
                'grid_denominator', 'grid_eligibility_agreement_count', 'missing_to_finite_count', 'finite_to_missing_count',
                'common_finite_cell_count', 'target_eligibility_agreement', 'target_missing_to_finite', 'target_finite_to_missing',
                'raw_common_finite_cell_count', 'raw_missing_to_finite_count', 'raw_finite_to_missing_count')}
            historical['nuisance'][condition].update(pool_denominator=len(checks), maximum_finite_b2_difference=max(maxima) if maxima else None,
                primary_difference_scope='reported_energy_masked_b2_common_eligible_cells', raw_maximum_finite_b2_difference=max(rawmax) if rawmax else None)
    return {'baseline_coverage': a.coverage([i['conditions']['baseline']['scalar'] for i in items]), 'levels': levels,
            'full_grid_historical_summary': historical, 'full_grid_summary_available': complete}


def strata_coverage(rows, getter):
    groups = defaultdict(list)
    for row in rows: groups[(row['player_id'], row['performance'])].append(getter(row))
    return [{'player_id': p, 'performance': perf, 'pool_denominator': 2 * len(values),
        'eligible_pools': sum(v['eligible_pool_count'] for v in values if v is not None),
        'eligible_fraction': sum(v['eligible_pool_count'] for v in values if v is not None) / (2 * len(values))}
        for (p, perf), values in sorted(groups.items())]


def margin_checks(items, original, codec, count, document):
    checks = []
    def add(name, observed, relation, threshold, passed):
        checks.append({'name': name, 'observed': observed, 'relation': relation, 'threshold': threshold, 'passed': bool(passed)})
    def sensitivity(prefix, summary):
        add(prefix + '.paired_pool_support', summary['paired_covered_pools'], '>=', 171, summary['paired_covered_pools'] >= 171)
        for metric in ('operational_median_difference', 'paired_pool_mean_difference'):
            record = summary['metrics'][metric]; values = [row[metric] for row in summary['per_recording'] if row[metric] is not None]
            add(prefix + '.' + metric + '.covered_recordings', len(values), '>=', 86, len(values) >= 86)
            equal = record['equal_score']['equal_group_mean']
            add(prefix + '.' + metric + '.equal_score', equal, '>=', .20, equal is not None and equal >= .20)
            positive = sum(v > 0 for v in values)
            add(prefix + '.' + metric + '.positive_recordings', positive, '>=', 72, positive >= 72)
            players = [r['mean'] for r in record['equal_player_secondary']['rows']]
            add(prefix + '.' + metric + '.all_players_positive', players, 'all >', 0, len(players) == 3 and all(v is not None and v > 0 for v in players))
            performances = [record['performance_strata'][key]['mean'] for key in ('comp', 'solo')]
            add(prefix + '.' + metric + '.both_performances_positive', performances, 'all >', 0, all(v is not None and v > 0 for v in performances))
    baseline = original['baseline_coverage']
    add('original.baseline.eligible_pools', baseline['eligible_pools'], '>=', 144, baseline['eligible_pools'] >= 144)
    add('original.baseline.covered_recording_scalars', baseline['covered_recording_scalars'], '>=', 81, baseline['covered_recording_scalars'] >= 81)
    strata = strata_coverage(items, lambda i: i['conditions']['baseline']['scalar'])
    add('original.baseline.every_player_performance_fraction', strata, 'all >=', .8, len(strata) == 6 and all(r['eligible_fraction'] >= .8 for r in strata))
    for level in ('minus6db', '0db'): sensitivity('original.' + level, original['levels'][level])
    historical = original['full_grid_historical_summary']
    for condition in ('common_gain', 'polarity'):
        result = historical['nuisance'][condition] if historical else None
        add('original.' + condition + '.full_grid_and_target_mask_exact', result, 'exact', True, result is not None and
            result['grid_eligibility_agreement_count'] == result['grid_denominator'] and result['target_eligibility_agreement'] == result['pool_denominator'])
        maximum = result['maximum_finite_b2_difference'] if result else None
        add('original.' + condition + '.maximum_common_finite_b2_error', maximum, '<=', 1e-10, maximum is not None and maximum <= 1e-10)
    baseline_receipts = [r for item in items for r in item['_codec_values'] if r['condition'] == 'baseline']
    for name in CODECS:
        cover = codec['baseline']['representation_coverage'][name]
        add(name + '.baseline.eligible_pools', cover['eligible_pools'], '>=', 144, cover['eligible_pools'] >= 144)
        add(name + '.baseline.covered_recording_scalars', cover['covered_recording_scalars'], '>=', 81, cover['covered_recording_scalars'] >= 81)
        strata = strata_coverage(baseline_receipts, lambda row: row['representations'][name])
        add(name + '.baseline.every_player_performance_fraction', strata, 'all >=', .8, len(strata) == 6 and all(r['eligible_fraction'] >= .8 for r in strata))
        trans = codec['baseline']['codec_pool_transition_counts'][name]
        agreement = (trans['eligible_to_eligible'] + trans['missing_to_missing']) / (2 * trans['comparison_denominator'])
        add(name + '.baseline.mask_agreement_fraction', agreement, '>=', .95, agreement >= .95 and trans['processing_failures'] == 0)
        for condition in SELECTED:
            delta = codec['scalar_delta_distributions']['codec_decoded_minus_float32'][name]['by_condition'][condition]
            for label, key, threshold in (('median_absolute_scalar_error', 'median_absolute_delta', .02), ('p95_absolute_scalar_error', 'p95_absolute_delta', .10)):
                value = delta[key]; add(name + '.' + condition + '.' + label, value, '<=', threshold, value is not None and value <= threshold)
        sensitivity(name + '.minus6db', codec['condition_contrasts'][name])
    add('processing.all_measurements_successful', count['all'], 'successful ==', 1440, count['all']['successful'] == 1440)
    return {'status': 'producer_checks_only_not_authoritative_not_admitted', 'margins_source_sha256': value_hash(document['margins']),
        'check_count': len(checks), 'passed_checks': sum(c['passed'] for c in checks), 'all_producer_checks_passed': all(c['passed'] for c in checks),
        'independent_replay_required': True, 'BC_admitted': False, 'checks': checks}


def reconstruct_summary(items, document):
    require(len(items) == 90 and [i['item_id'] for i in items] == [r['item_id'] for r in document['roster']], 'reserved roster order')
    raw = a.reconstruct_summary([r for i in items for r in i['_codec_values']])
    codec = {'version': PRODUCER_VERSION, 'dataset_role': 'reserved',
        'status': 'reserved_precision_and_codec_measurements_complete_not_admitted' if raw['new_BC_measurements']['successful'] == 810 else
                  'reserved_precision_and_codec_measurements_with_retained_failures_not_admitted',
        'reused_numerical_implementation': {'version': 'run_bc_guitarset_codec_development_v1', 'sha256': CODEC_PRODUCER_SHA, 'development_scope_fields_removed': True},
        **{key: raw[key] for key in ('denominators', 'new_BC_measurements', 'processing_failures', 'baseline', 'scalar_delta_distributions', 'condition_contrasts')}}
    count = counts(items); require(count['all']['expected'] == 1440, 'all planned measurement denominator')
    original = original_summary(items)
    return {'version': PRODUCER_VERSION, 'status': 'reserved_measurements_complete_pending_independent_replay_not_admitted' if count['all']['successful'] == 1440 else
        'reserved_measurements_with_retained_failures_pending_replay_not_admitted', **PRODUCER_SCOPE,
        'measurement_counts': count, 'expected_denominators': document['expected_denominators'], 'original_float64': original,
        'precision_and_codec': codec, 'margins': document['margins'], 'producer_margin_checks': margin_checks(items, original, codec, count, document),
        'producer_gate_status': 'descriptive_checks_only_not_authoritative_independent_replay_required',
        'independent_replay_required_before_gate': True, 'final_admission_decision': None,
        'per_recording': [{k: v for k, v in item.items() if k != '_codec_values'} for item in items]}


def file_inventory(root):
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), 'safe existing result root')
    result = {}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(), 'result symlink forbidden')
        if path.is_dir(): continue
        require(path.is_file(), 'regular products only')
        if path == root / 'COMMIT.json': continue
        require(path != root / 'FAILED.json', 'FAILED result cannot be audited as terminal COMMIT')
        entry = binding(path); result[path.relative_to(root).as_posix()] = {k: entry[k] for k in ('bytes', 'sha256')}
    return result


def allowed_products(document):
    allowed = {'frozen_draft.json', 'parent_freeze.json', 'storage_estimate.json', 'summary.json'}
    for row in document['roster']:
        prefix = 'items/' + row['item_id'] + '/'
        allowed.update(prefix + name for name in ('receipt.json', 'construction.json', 'construction.arrays.npz', 'construction.failure.json'))
        allowed.update(prefix + 'float64/' + c + ext for c in CONDITIONS for ext in ('.wav', '.metadata.json', '.arrays.npz', '.failure.json'))
        for condition in SELECTED:
            nested = prefix + 'codec/' + condition + '/'
            allowed.add(nested + 'receipt.json')
            allowed.update(nested + 'float32_control' + ext for ext in ('.wav', '.metadata.json', '.arrays.npz', '.failure.json'))
            for codec in CODECS:
                allowed.update(nested + codec + ext for ext in (document['codec_recipes'][codec]['extension'], '.decoded.wav', '.encode.log.json',
                    '.decode.log.json', '.metadata.json', '.arrays.npz', '.failure.json', '.skipped.json'))
    return allowed


def linked_libraries(freeze):
    """Resolve the actual whole ldd dependency map; never equate two codec libs with all libraries."""
    paths = set()
    for component in ('ffmpeg', 'ffprobe'):
        executable = freeze['toolchain'][component]['path']
        result = subprocess.run(['/usr/bin/ldd', executable], capture_output=True, text=True,
                                env={'LANG': 'C', 'LC_ALL': 'C', 'PATH': '/usr/bin:/bin'}, check=True)
        require('not found' not in result.stdout + result.stderr, 'missing linked library')
        paths.update(str(Path(match).resolve()) for line in result.stdout.splitlines()
                     for match in re.findall(r'(?:=>\s*)?(/[^\s]+)\s+\(', line))
    require(paths, 'empty dynamic library graph')
    actual = {path: binding(path) for path in sorted(paths)}
    require(freeze.get('linked_library_bindings') == actual, 'complete ldd-resolved library map differs from parent freeze')
    return actual


def independent_auditor_runtime(freeze=None):
    """Bind the actual SciPy source/kernel dependencies used by native replay."""
    actual = {'scipy': scipy.__version__, 'modules': {name: binding(Path(module.__file__).resolve())
        for name, module in (('scipy', scipy), ('scipy.signal._upfirdn', scipy_upfirdn),
            ('scipy.signal._upfirdn_apply', scipy_upfirdn_kernel), ('scipy.special._ufuncs', scipy_special_ufuncs))}}
    if freeze is not None:
        require(freeze.get('independent_auditor_runtime') == actual, 'independent auditor SciPy runtime binding')
    return actual


def source_bindings(document):
    result = {}
    for row in document['roster']:
        require(row['split_role'] == 'reserved', 'source role boundary')
        entries = {}
        for key in ('source_audio', 'source_annotation'):
            expected = row[key]; require(Path(expected['path']).is_relative_to(Path(document['source_root'])), 'source root escape')
            entries[key] = binding(expected['path']); require(entries[key] == expected, 'native/annotation byte mutation')
        result[row['item_id']] = entries
    return result


def validate_authorities(root, commit_sha, draft_dir, freeze_path, freeze_sha, self_sha, tests_path, tests_sha):
    forbid_producer_imports()
    own = binding(Path(__file__).resolve(), self_sha); tests = binding(tests_path, tests_sha)
    document, draft = read_json(draft_dir / 'draft.json', DRAFT_SHA)
    draft_commit, draft_commit_entry = read_json(draft_dir / 'COMMIT.json', DRAFT_COMMIT_SHA)
    require(draft_commit['draft'] == draft and document['reserved_roster_sha256'] == ROSTER_SHA and len(document['roster']) == 90 and
        document['conditions'] == {'float64': list(CONDITIONS), 'float32_precision_control': list(SELECTED), 'codec_decoded': list(SELECTED), 'codecs': list(CODECS)}, 'immutable reserved draft/roster')
    freeze, freeze_entry = read_json(freeze_path, freeze_sha)
    require(freeze.get('version') == 'bc_guitarset_reserved_admission_parent_freeze_v1' and freeze.get('status') == 'authorized_reserved_measurement_not_admission' and
        freeze.get('draft_COMMIT') == draft_commit_entry and freeze.get('draft') == draft and freeze.get('output_root') == document['output_root'] == str(root) and
        freeze.get('expected_measurements') == EXPECTED and freeze.get('runner', {}).get('sha256') == PRODUCER_SHA and
        freeze.get('independent_auditor') == own and freeze.get('codec_recipes') == document['codec_recipes'] and
        freeze.get('authorized_scope') == {'reserved_audio_access_authorized': True, 'reserved_BC_measurement_authorized': True,
            'codec_encoding_authorized': True, 'classifier_fits_authorized': False, 'model_scoring_authorized': False,
            'BC_admission_authorized': False, 'threshold_changes_authorized': False}, 'parent reserved audit/measurement authority')
    for entry in document['authorities'].values(): require(binding(entry['path']) == entry, 'historical metadata/code authority changed')
    a.check_file_bindings(freeze)
    require(freeze['runtime']['python'] == sys.version and freeze['runtime']['numpy'] == np.__version__ and
        freeze['runtime']['executable'] == binding(Path(sys.executable).resolve()), 'auditor pinned numerical runtime')
    library_graph = linked_libraries(freeze)
    independent_runtime = independent_auditor_runtime(freeze)
    commit, commit_entry = read_json(root / 'COMMIT.json', commit_sha)
    require(commit.get('version') == PRODUCER_VERSION and commit.get('status') in (
        'committed_reserved_measurements_complete_pending_independent_replay_not_admitted',
        'committed_reserved_measurements_with_retained_failures_pending_replay_not_admitted') and
        all(commit.get(k) == v for k, v in PRODUCER_SCOPE.items()) and commit.get('draft_COMMIT') == draft_commit_entry and
        commit.get('parent_freeze') == freeze_entry and commit.get('independent_auditor') == own and commit.get('expected_measurements') == EXPECTED,
        'terminal measurement COMMIT authority/scope')
    inventory = file_inventory(root)
    require(set(inventory) <= allowed_products(document) and commit['products'] == inventory and commit['products_sha256'] == value_hash(inventory), 'complete bounded product inventory')
    require(read_json(root / 'frozen_draft.json')[0] == document and read_json(root / 'parent_freeze.json')[0] == freeze,
            'output authority copies differ')
    return document, freeze, commit, inventory, {'audit_code': own, 'audit_tests': tests, 'draft': draft,
        'draft_COMMIT': draft_commit_entry, 'parent_freeze': freeze_entry, 'result_COMMIT': commit_entry, 'linked_library_bindings': library_graph,
        'independent_auditor_runtime': independent_runtime,
        'independent_codec_auditor': binding(Path(a.__file__).resolve(), INDEPENDENT_CODEC_AUDITOR_SHA)}


def audit(root, commit_sha, draft_dir, freeze_path, freeze_sha, receipt_path, self_sha, tests_path, tests_sha):
    root, draft_dir, freeze_path, receipt_path, tests_path = map(Path, (root, draft_dir, freeze_path, receipt_path, tests_path))
    require(root.is_absolute() and not root.is_symlink() and receipt_path.is_absolute() and not receipt_path.exists() and
        not receipt_path.is_relative_to(root) and not receipt_path.is_relative_to(draft_dir), 'new separate auditor output')
    document, freeze, commit, inventory, upstream = validate_authorities(root, commit_sha, draft_dir, freeze_path, freeze_sha, self_sha, tests_path, tests_sha)
    require(not receipt_path.is_relative_to(Path(document['source_root'])), 'audit output overlaps reserved source')
    sources = source_bindings(document); require(value_hash(sources) == commit['source_graph_sha256'], 'COMMIT source graph')
    items = []
    for index, row in enumerate(document['roster']):
        items.append(verify_item(row, root / 'items' / row['item_id'], freeze))
        print(f'BC reserved independent replay {index + 1}/90 {row["item_id"]}', file=sys.stderr, flush=True)
    calculated = reconstruct_summary(items, document); reported, summary_entry = read_json(root / 'summary.json')
    a.compare(reported, calculated, 'independent full reserved summary and margin arithmetic')
    require(commit['measurement_counts'] == calculated['measurement_counts'] and commit['summary_sha256'] == value_hash(reported), 'COMMIT measurement/summary join')
    complete = calculated['measurement_counts']['all']['successful'] == 1440 and all(i['construction']['status'] == 'success' for i in items)
    require((commit['status'] == 'committed_reserved_measurements_complete_pending_independent_replay_not_admitted') == complete, 'qualified complete/failure status')
    end = validate_authorities(root, commit_sha, draft_dir, freeze_path, freeze_sha, self_sha, tests_path, tests_sha)
    require(end == (document, freeze, commit, inventory, upstream) and source_bindings(document) == sources, 'source/product/code/library graph changed during audit')
    forbid_producer_imports()
    count = calculated['measurement_counts']; successful = count['all']['successful']
    receipt = {'version': VERSION, 'status': 'passed_independent_reserved_numerical_and_accountability_replay_not_admitted', 'passed': True,
        'reserved_audio_read': True, 'development_audio_read': False, 'unused_audio_read': False, 'classifier_fits': 0, 'model_scoring': False,
        'BC_admitted': False, 'thresholds_changed': False, 'final_admission_decision': None, 'separate_decision_required': True,
        **upstream, 'summary': summary_entry, 'measurement_counts': count, 'expected_denominators': document['expected_denominators'],
        'planned_fixed_target_measurements': 1440, 'planned_fixed_target_pools': 2880,
        'successful_fixed_target_pools_recomputed': successful * 2, 'full_float64_grid_pools_recomputed': count['float64']['successful'] * 2,
        'full_grid_cells': 228, 'native_decode_replay': 'independent_RIFF_PCM16_exact_FLOAT64_hash',
        'resampling_replay': 'independent_Kaiser5_FIR_and_centering_with_upfirdn_convolution_kernel',
        'construction_and_float32_cast_tolerance': 'bit_exact_dtype_shape_and_bytes_no_tolerance',
        'BC_numerical_comparison_tolerance': {'absolute': 2e-12, 'relative': 2e-11, 'masks_and_nulls': 'exact'},
        'independently_reconstructed_margin_checks': calculated['producer_margin_checks'],
        'all_scientific_requirements_satisfied_after_replay': complete and calculated['producer_margin_checks']['all_producer_checks_passed'],
        'processing_failures_do_not_reduce_denominators': True, 'producer_extractor_primitive_scalar_construction_reducer_imported': False,
        'source_graph_sha256': value_hash(sources), 'result_products_sha256': value_hash(inventory), 'result_product_count': len(inventory),
        'runtime': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()},
        'scope': 'independent numerical and retained-failure audit only; admission requires separately bound decision'}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open('xb') as stream: stream.write(canonical(receipt) + b'\n')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('result', 'result-commit-sha256', 'draft-dir', 'parent-freeze', 'parent-freeze-sha256',
                 'output-receipt', 'audit-code-sha256', 'audit-tests', 'audit-tests-sha256'): p.add_argument('--' + name, required=True)
    x = p.parse_args()
    result = audit(x.result, x.result_commit_sha256, x.draft_dir, x.parent_freeze, x.parent_freeze_sha256,
                   x.output_receipt, x.audit_code_sha256, x.audit_tests, x.audit_tests_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
