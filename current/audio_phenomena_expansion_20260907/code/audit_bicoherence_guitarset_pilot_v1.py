"""Independent GuitarSet development replay; never imports a measurement producer.

Audits committed bytes, crossed selection, native decode/resampling/crop, all
injections, full STFT/cells and pooled/grouped summaries. NumPy/SciPy/SoundFile
are shared numerical libraries, not independent numerical implementations.
No sensitivity gate, classifier, causal baseline label or significance follows.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import zipfile

import numpy as np
import scipy
from scipy.io import wavfile
from scipy.signal import resample_poly
import soundfile as sf

SOURCE_COMMIT_SHA = 'aa33f0dfc55e3bbc6f48e618d1203c29e258608c309412d333f06c6e671b73dc'
DRAFT_SHA = '7632e91ddb3b2c25e0ebebf9b10357b7c1ddfaa96d4e79c603b28f5ef4ee97eb'
DESIGN_SHA = '3626b35d6d6de1b445e42fbb0c9bca33a3bee213995584b4d857eba26af15f08'
EXTRACTOR_SHA = 'e59fd575add722ca10280f5e19b64d1abe6a1587dac6b0a790fb9d57401f64a1'
PRIMITIVE_SHA = '9cedc47b0ee42775c996bbf3e9c8eb237aa1acde4a051634b077fd72fd7614d5'
VERSION = 'bicoherence_guitarset_development_pilot_v1'
ENCODING = 'SHA-256 of decoded samples in frame-major/channel-minor C order; each sample is IEEE-754 float64 encoded explicitly little-endian (<f8); no resampling, mixing, normalization, or feature extraction'
CONDITIONS = ('baseline', 'common_gain', 'polarity', 'closed_minus6db',
              'independent_minus6db', 'closed_0db', 'independent_0db')
POLICY = {'conditions': list(CONDITIONS), 'target_bins': [32,48,80],
          'seed_prefix': 'BC-GuitarSet-injection-20260907|', 'phase_knots': 65,
          'knot_spacing_seconds': .125, 'sample_rate_hz': 16000, 'samples': 128000,
          'background_gain': .25, 'common_gain_factor': .1, 'relative_injection_db': [-6,0],
          'prototype_tone_amplitude': .2, 'framewise_power_matched': False,
          'mixture_normalization_or_clipping': False,
          'aggregation': 'paired pools mean within recording; mean within score; equal covered scores',
          'secondary_aggregation': 'mean within player; equal covered players',
          'performance_strata': ['comp','solo'], 'development_recordings': 90,
          'pools_per_recording_condition': 2, 'cells_per_pool': 228,
          'numeric_admission_threshold': None, 'null_calibrated': False,
          'significance_inferred': False, 'external_gate_passed': False,
          'classifier_admitted': False, 'classifier_fits': 0,
          'reserved_or_unused_waveforms_measured': False, 'baseline_causal_label': None}
RTOL, ATOL = 2e-12, 2e-14

def require(ok, label):
    if not ok:
        raise ValueError(label)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key: ' + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=pairs)


def compare(actual, expected, label='value', rtol=RTOL, atol=ATOL):
    """Compare complete schemas; missing/false/zero and array shape stay distinct."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), label + ': keys')
        for key in expected:
            compare(actual[key], expected[key], label + '.' + key, rtol, atol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ': list')
        for index, value in enumerate(expected):
            compare(actual[index], value, f'{label}[{index}]', rtol, atol)
    elif isinstance(expected, np.ndarray):
        require(isinstance(actual, np.ndarray) and actual.shape == expected.shape
                and actual.dtype == expected.dtype, label + ': array schema')
        if rtol==0 and atol==0:
            require(actual.tobytes(order='C')==expected.tobytes(order='C'),label+': exact C-order array bytes')
        else:
            require(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True), label + ': array values')
    elif isinstance(expected, float):
        require(type(actual) in (float, int) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), label + ': float')
    else:
        require(type(actual) is type(expected) and actual == expected, label + ': exact')


def child(root, name):
    root, rel = Path(root).resolve(strict=True), Path(name)
    require(not rel.is_absolute() and '..' not in rel.parts and bool(rel.parts), 'unsafe path')
    require(not any(root.joinpath(*rel.parts[:i]).is_symlink() for i in range(1, len(rel.parts)+1)), 'symlink')
    path = root / rel
    require(path.is_file() and path.resolve().is_relative_to(root), 'nonlocal product')
    return path


def product(path):
    return {'bytes': Path(path).stat().st_size, 'sha256': sha(path)}


def inventory(root, digest):
    root = Path(root).resolve(strict=True)
    marker = child(root, 'COMMIT.json')
    require(len(digest) == 64 and sha(marker) == digest, 'trusted COMMIT hash')
    commit = read_json(marker)
    require(commit.get('status') == 'committed' and isinstance(commit.get('products'), dict), 'COMMIT schema')
    paths = list(root.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'inventory symlink')
    require({str(p.relative_to(root)) for p in paths if p.is_file()} == set(commit['products']) | {'COMMIT.json'}, 'exact inventory')
    for name, binding in commit['products'].items():
        compare(product(child(root, name)), binding, name, 0, 0)
    return commit


def reconstruct(audio, note):
    require(audio.dtype == np.float64 and audio.shape == (128000,) and np.isfinite(audio).all(), 'construction input')
    seed = int.from_bytes(hashlib.sha256(('BC-GuitarSet-injection-20260907|'+note).encode()).digest()[:8], 'big')
    knots = np.random.Generator(np.random.PCG64(seed)).uniform(-np.pi, np.pi, (3, 65))
    unwrapped = np.unwrap(knots, axis=1)
    kt, t = np.arange(65, dtype=np.float64)/8, np.arange(128000, dtype=np.float64)/16000
    phases = np.array([np.interp(t, kt, p) for p in unwrapped])
    closed_phases = np.array([phases[0], phases[1], phases[0]+phases[1]])
    carriers = 2*np.pi*np.array([500., 750., 1250.])[:, None]*t
    c, i = [.2*np.cos(carriers+p).sum(axis=0) for p in (closed_phases, phases)]
    cr, ir = [float(np.sqrt(np.mean(v*v))) for v in (c, i)]
    require(cr > 0 and ir > 0, 'positive prototype RMS')
    b = audio*.25
    require(not np.any((audio!=0)&(b==0)), 'unsupported background gain underflow')
    rms = float(np.sqrt(np.mean(b*b)))
    require(math.isfinite(rms) and not (rms == 0 and np.any(b)), 'unsupported background RMS')
    cu, iu = c/cr, i/ir
    waves = {'baseline': b, 'common_gain': b*.1, 'polarity': -b}
    arrays = {'standardized_crop': audio, 'background': b, 'sample_times_seconds': t,
              'phase_knot_times_seconds': kt, 'phase_knots_wrapped': knots,
              'phase_knots_unwrapped': unwrapped, 'independent_phase_trajectories': phases,
              'closed_phase_trajectories': closed_phases, 'closed_prototype': c,
              'independent_prototype': i, 'closed_unit_rms': cu, 'independent_unit_rms': iu,
              'prototype_rms': np.array([cr, ir]), 'background_rms': np.asarray(rms)}
    for level, db in [('minus6db', -6), ('0db', 0)]:
        for kind, unit in [('closed', cu), ('independent', iu)]:
            name = kind+'_'+level
            arrays[name+'_injection'] = rms*10**(db/20)*unit
            waves[name] = b+arrays[name+'_injection']
    status = 'ok' if rms else 'unsupported_zero_background_rms'
    meta = {'item_id': note, 'seed': seed, 'bit_generator': 'PCG64', 'status': status,
            'background_rms': rms, 'prototype_rms': {'closed': cr, 'independent': ir},
            'prototype_normalization_factors': {'closed': 1/cr, 'independent': 1/ir},
            'injected_full_record_rms_matched': rms > 0, 'framewise_power_matched': False, 'mixtures_clipped_or_normalized': False,
            'zero_rms_interpretation': 'zero derivatives retained; relative-level intervention unsupported'}
    return waves, arrays, meta


def sums_for(columns):
    product_column = columns[:, 0]*columns[:, 1]
    third = columns[:, 2]
    require(not np.any((columns[:,0]!=0)&(columns[:,1]!=0)&(product_column==0)), 'unsupported product underflow')
    terms=product_column*np.conjugate(third)
    require(not np.any((product_column!=0)&(third!=0)&(terms==0)), 'unsupported triple underflow')
    for values in (product_column,third):
        magnitude=np.abs(values)
        require(not np.any((magnitude!=0)&(magnitude*magnitude==0)), 'unsupported energy underflow')
    triple = np.sum(terms)
    return {'product_energy_sum': float(np.sum(np.abs(product_column)**2)),
            'sum_frequency_energy_sum': float(np.sum(np.abs(third)**2)),
            'triple_sum_real': float(triple.real), 'triple_sum_imag': float(triple.imag)}


def binary(value, factors):
    # Longdouble yields raw sums without reproducing the producer's incremental
    # mantissa normalization algorithm. The current audio amplitudes are bounded.
    raw = np.longdouble(value)
    for factor in factors:
        raw *= np.longdouble(factor)
    if raw == 0:
        return {'mantissa': 0.0, 'exponent2': 0}
    m, e = np.frexp(raw)
    return {'mantissa': float(m), 'exponent2': int(e)}


def independent_primitive(spectra, triad):
    require(spectra.ndim==2 and len(spectra)>=2 and np.isfinite(spectra).all(), 'finite coefficient realizations')
    columns = spectra[:, triad]
    scales = np.max(np.maximum(abs(columns.real), abs(columns.imag)), axis=0)
    result = {'version': 'bicoherence_primitive_v2', 'realizations': len(spectra),
              'frequency_bins': list(map(int, triad)), 'status': 'ok', 'squared_bicoherence': None,
              'biphase_radians': None, 'biphase_status': 'missing_triad_energy',
              'biphase_interpretation': 'descriptive_only_no_significance',
              'coefficient_scales': scales.tolist(), 'normalized_sums': None, 'raw_sums': None,
              'raw_sum_encoding': 'value = mantissa * 2**exponent2',
              'external_validation_passed': False, 'classifier_admitted': False}
    if np.any(scales == 0):
        result['status'] = 'missing_triad_energy' if np.any(spectra) else 'zero_energy'
        return result
    normalized = np.empty_like(columns)
    normalized.real, normalized.imag = columns.real/scales, columns.imag/scales
    require(not np.any((columns!=0)&(normalized==0)), 'unsupported normalized coefficient underflow')
    sums = sums_for(normalized)
    result['normalized_sums'] = sums
    a, b, c = map(float, scales)
    factors = {'product_energy_sum': (a,a,b,b), 'sum_frequency_energy_sum': (c,c),
               'triple_sum_real': (a,b,c), 'triple_sum_imag': (a,b,c)}
    result['raw_sums'] = {k: binary(v, factors[k]) for k,v in sums.items()}
    direct = sums_for(columns)
    # Independently check raw sums in original units, including cancellation.
    natural = math.sqrt(direct['product_energy_sum'])*math.sqrt(direct['sum_frequency_energy_sum'])
    for key, encoded in result['raw_sums'].items():
        raw = math.ldexp(encoded['mantissa'], encoded['exponent2'])
        tolerance = 128*np.finfo(float).eps*(natural if key.startswith('triple') else abs(direct[key]))
        compare(raw, direct[key], 'direct raw '+key, 2e-12, tolerance)
    pe, te = sums['product_energy_sum'], sums['sum_frequency_energy_sum']
    if pe == 0 or te == 0:
        result['status'] = 'missing_triad_product_energy'
        return result
    triple = complex(sums['triple_sum_real'], sums['triple_sum_imag'])
    magnitude = abs(triple)/math.sqrt(pe)/math.sqrt(te)
    score = magnitude*magnitude
    require(not (triple!=0 and score==0), 'unsupported nonzero squared coherence underflow')
    require(math.isfinite(score) and score <= 1+64*np.finfo(float).eps, 'Cauchy Schwarz')
    result['squared_bicoherence'] = min(1., score)
    if triple == 0:
        result['biphase_status'] = 'undefined_zero_resultant'
    else:
        result['biphase_status'] = 'descriptive_only_no_significance'
        result['biphase_radians'] = math.atan2(triple.imag, triple.real)
    return result


def replay_pool(audio, construction_status):
    grid = np.asarray([(a,b,a+b) for a in range(8,193,8) for b in range(a,193,8) if a+b<=256], dtype=np.int64)
    window = .5-.5*np.cos(2*np.pi*np.arange(1024)/1024)
    offsets = np.arange(247, dtype=np.int64)*256
    frames = np.array([audio[start:start+1024] for start in offsets])
    means = np.mean(frames, axis=1)
    spectra = np.array([np.fft.rfft((frame-mean)*window)/np.sum(window) for frame,mean in zip(frames,means)])
    require(np.isfinite(spectra).all(),'finite full STFT')
    require(not np.any((np.abs(spectra)!=0)&(np.abs(spectra)**2==0)), 'unsupported STFT energy underflow')
    energy = np.sum(np.abs(spectra)**2, axis=0)
    total = float(np.sum(energy[1:]))
    fractions = energy/total if total else np.full(513, np.nan)
    floor = np.all(fractions[grid] >= 1e-6, axis=1)
    cells = []
    for index, triad in enumerate(grid):
        p = independent_primitive(spectra, triad)
        defined = p['status'] == 'ok' and p['squared_bicoherence'] is not None
        eligible = bool(floor[index] and defined)
        cells.append({'frequency_bins': triad.tolist(), 'status': 'ok' if eligible else
                      p['status'] if not defined else 'below_energy_fraction_floor',
                      'energy_fractions': fractions[triad].tolist() if total else [None]*3,
                      'energy_floor_passed': bool(floor[index]), 'eligible': eligible,
                      'squared_bicoherence': p['squared_bicoherence'] if eligible else None,
                      'biphase_radians': p['biphase_radians'] if eligible else None, 'primitive': p})
    def cell_array(key, primitive=False):
        return np.array([[np.nan if (v := (c['primitive'] if primitive else c)[key]) is None else v for c in cells]], dtype=np.float64)
    eligibility = np.array([[c['eligible'] for c in cells]], dtype=bool)
    arrays = {'window': window, 'frequency_bins': grid, 'frequency_hz': grid.astype(np.float64)*16000/1024,
              'pool_start_samples': np.array([0], dtype=np.int64), 'frame_offset_samples': offsets,
              'frame_start_samples': offsets[None,:], 'frame_means': means[None,:], 'spectra': spectra[None,:,:],
              'bin_coefficient_energy': energy[None,:], 'total_non_dc_coefficient_energy': np.array([total]),
              'bin_energy_fraction': fractions[None,:], 'energy_floor_mask': floor[None,:],
              'primitive_defined_mask': np.array([[c['primitive']['status']=='ok' and c['primitive']['squared_bicoherence'] is not None for c in cells]], dtype=bool),
              'eligible_mask': eligibility, 'squared_bicoherence': cell_array('squared_bicoherence'),
              'biphase_radians': cell_array('biphase_radians'),
              'primitive_squared_bicoherence': cell_array('squared_bicoherence', True),
              'primitive_biphase_radians': cell_array('biphase_radians', True)}
    zero = not np.any(audio)
    pool = {'pool_index': 0, 'start_sample': 0, 'stop_sample_exclusive': 64000,
            'status': 'zero_amplitude' if zero else 'zero_non_dc_energy' if total == 0 else 'ok',
            'zero_amplitude': zero, 'coefficient_rows': 247, 'independent_realization_count': None,
            'total_non_dc_coefficient_energy': total, 'eligible_cell_count': int(eligibility.sum()), 'cells': cells}
    metadata = {'version': 'bicoherence_audio_v1', 'primitive_version': 'bicoherence_primitive_v2',
                'status': 'ok', 'input_samples': 64000, 'sample_rate_hz': 16000, 'input_zero_amplitude': zero,
                'pool_samples': 64000, 'pool_duration_seconds': 4., 'pool_count': 1, 'analyzed_samples': 64000,
                'discarded_tail_samples': 0, 'tail_start_sample': 64000, 'tail_policy': 'discard_no_padding',
                'n_fft': 1024, 'hop_samples': 256, 'frames_per_pool': 247, 'window': 'periodic_hann',
                'window_sum': float(window.sum()), 'transform': 'rfft((frame - frame.mean()) * window) / window.sum()',
                'pool_boundary_policy': 'no_frame_crosses_pool_boundary',
                'energy': 'sum_frames(abs(coefficient)**2); no one-sided doubling',
                'fraction_denominator_bins_inclusive': [1,512], 'energy_fraction_min_inclusive': 1e-6,
                'grid_cell_count': 228, 'grid': 'positive parents 8..192 step 8; f1<=f2; f1+f2<=256',
                'biphase_convention': 'atan2(imag,real) in [-pi,pi]; descriptive_only',
                'array_missing_encoding': 'NaN; JSON metadata missing values are null',
                'independent_realization_count': None, 'frames_overlap_and_are_not_asserted_independent': True,
                'amplitude_weighted_not_phase_only': True, 'linear_fixed_phase_tones_can_have_high_bicoherence': True,
                'null_calibrated': False, 'significance_inferred': False, 'external_validation_passed': False,
                'classifier_admitted': False, 'pools': [pool], 'construction_status': construction_status}
    return arrays, metadata


def npz_check(path, expected, exact=False):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == set(expected) and len(data.files) == len(expected), 'NPZ complete keys')
        for key, value in expected.items():
            compare(data[key], value, str(path)+'.'+key, 0 if exact else RTOL, 0 if exact else ATOL)


def full_binding(path):
    p = Path(path)
    require(p.is_absolute() and p.is_file() and not p.is_symlink(), 'local regular bound file')
    return {'path': str(p.resolve(strict=True)), **product(p)}


def check_bound_tree(tree):
    """Rehash every explicit file binding, including scoped overlap evidence."""
    if isinstance(tree, dict):
        if {'path','bytes','sha256'} <= tree.keys():
            compare(full_binding(tree['path']), {k:tree[k] for k in ('path','bytes','sha256')}, 'bound file', 0, 0)
        for value in tree.values():
            check_bound_tree(value)
    elif isinstance(tree,list):
        for value in tree:
            check_bound_tree(value)


def runtime_snapshot(producer=False):
    require(np.__version__=='1.26.4' and scipy.__version__=='1.17.1' and sf.__version__=='0.14.0',
            'requires NumPy 1.26.4, SciPy 1.17.1, SoundFile 0.14.0')
    names = {'numpy':'numpy', 'numpy_core':'numpy.core._multiarray_umath', 'pcg64':'numpy.random._pcg64',
             'scipy':'scipy', 'soundfile':'soundfile', 'signaltools':'scipy.signal._signaltools',
             'fir_filter_design':'scipy.signal._fir_filter_design', 'upfirdn':'scipy.signal._upfirdn',
             'upfirdn_apply':'scipy.signal._upfirdn_apply'}
    exe = Path(sys.executable).absolute()
    soundfile_binary=Path(sf._full_path).resolve(strict=True)
    require(str(soundfile_binary) in repr(sf._snd),'loaded libsndfile image binding')
    result = {'python':platform.python_version(), 'platform':platform.platform(),
        'numpy':np.__version__, 'scipy':scipy.__version__, 'soundfile':sf.__version__,
        'libsndfile':sf.__libsndfile_version__, 'libsndfile_binary':full_binding(soundfile_binary),
        'modules':{key:full_binding(Path(importlib.import_module(name).__file__).resolve()) for key,name in names.items()},
        'executable':{'invocation_path':str(exe), 'invocation_is_symlink':exe.is_symlink(),
            'invocation_differs_from_resolved':exe != exe.resolve(), 'resolved_binary':full_binding(exe.resolve())}}
    if producer:
        result['producer_modules']={name:full_binding(Path(importlib.import_module(name).__file__).resolve())
            for name in ('scipy.io.wavfile','numpy.fft._pocketfft','numpy.fft._pocketfft_internal')}
    return result


def canonical_pcm(samples):
    return hashlib.sha256(np.asarray(samples,dtype='<f8',order='C').tobytes(order='C')).hexdigest()


def standardize(native, rate):
    require(type(rate) is int and rate>0 and native.dtype==np.float64 and native.ndim==1
            and native.size>0 and np.isfinite(native).all(), 'native mono finite float64')
    divisor=math.gcd(rate,16000)
    up,down=16000//divisor,rate//divisor
    changed=rate!=16000
    entire=resample_poly(native,up,down,window=('kaiser',5.0),padtype='constant',cval=0.) if changed else native
    require(entire.dtype==np.float64 and np.isfinite(entire).all(), 'finite full resampling')
    require(len(entire)==(len(native)*up+down-1)//down, 'resampled length')
    require(len(entire)>=128000, 'short source: no padding or substitution')
    start=(len(entire)-128000)//2
    crop=entire[start:start+128000].copy()
    provenance={'bc_extracted':False, 'crop_center_rule':'floor((resampled_frames - 128000)/2)',
        'crop_frames':128000, 'crop_pcm_encoding':ENCODING, 'crop_pcm_sha256':canonical_pcm(crop),
        'crop_start':start, 'crop_stop_exclusive':start+128000, 'crop_zero_amplitude':not np.any(crop),
        'cval':0., 'derived_audio_saved':False, 'down':down, 'measurement_support':'not_measured',
        'native_16khz_policy':'unchanged_array', 'native_frames':len(native), 'native_sample_rate_hz':rate,
        'padtype':'constant', 'resample_scope':'entire_native_recording_before_crop',
        'resampled_frames':len(entire), 'resampling_applied':changed, 'target_sample_rate_hz':16000,
        'up':up, 'window':['kaiser',5.]}
    return crop,provenance


def decode_development(source, row, prepared):
    require(row['split_role']=='development', 'reserved/unused decode forbidden')
    require(prepared['item_id']==row['item_id'], 'preprocessing identity')
    original=row['source_record']
    path=child(source,original['audio']['materialized_path'])
    compare(full_binding(path),prepared['original_audio'],'source audio binding',0,0)
    compare(full_binding(child(source,original['annotation']['materialized_path'])),
            prepared['original_annotation'],'annotation binding',0,0)
    expected=original['audio']['decoded']
    chunks=[]
    with sf.SoundFile(path,'r') as stream:
        require(stream.channels==1 and stream.samplerate==expected['sample_rate_hz']
                and stream.frames==expected['header_frames'] and stream.format==expected['format']
                and stream.subtype==expected['subtype'], 'native audio header')
        calls=0
        while True:
            block=stream.read(65536,dtype='float64',always_2d=True)
            calls+=1
            require(block.dtype==np.float64 and block.ndim==2 and block.shape[1]==1
                    and np.isfinite(block).all(), 'native finite full decode')
            if len(block)==0:
                break
            chunks.append(block[:,0])
    native=np.concatenate(chunks) if chunks else np.empty(0,dtype=np.float64)
    decoded={'channels':1, 'decode_block_frames':65536, 'decoded_frames':len(native),
        'decoded_pcm_canonical_encoding':ENCODING, 'decoded_pcm_sha256':canonical_pcm(native),
        'duration_seconds':len(native)/expected['sample_rate_hz'], 'empty_eof_observed':True,
        'float64_samples_checked_finite':len(native), 'format':expected['format'], 'header_frames':len(native),
        'materialized_file':product(path), 'nonfinite_samples':0, 'read_calls_including_empty_eof':calls,
        'sample_rate_hz':expected['sample_rate_hz'], 'subtype':expected['subtype']}
    compare(decoded,expected,'native full decode',0,0)
    require(prepared['source_pcm_sha256']==decoded['decoded_pcm_sha256'], 'frozen PCM hash')
    crop,preprocessing=standardize(native,expected['sample_rate_hz'])
    compare(preprocessing,prepared['preprocessing'],'full resample / center crop',0,0)
    return crop


def crossed_split(source_rows):
    require(len(source_rows)==360 and len({r['item_id'] for r in source_rows})==360, '360 unique source rows')
    players={r['player_id'] for r in source_rows}
    scores={r['score_id'] for r in source_rows}
    require(len(players)==6 and len(scores)==30, 'six players / 30 scores')
    require({(r['player_id'],r['score_id'],r['performance']) for r in source_rows}
            == {(p,s,f) for p in players for s in scores for f in ('comp','solo')}, 'complete crossed performance pairs')
    order=lambda values,prefix:sorted(values,key=lambda v:(hashlib.sha256((prefix+v).encode()).hexdigest(),v))
    players=order(players,'BC-GuitarSet-player-20260907|')
    scores=order(scores,'BC-GuitarSet-score-20260907|')
    rows=[]
    for original in sorted(source_rows,key=lambda r:r['item_id']):
        p,s,f=original['player_id'],original['score_id'],original['performance']
        require(original['item_id']==p+'_'+s+'_'+f, 'source identifier components')
        style=re.match(r'^([A-Za-z]+)',s)
        require(style is not None, 'score style prefix')
        role='development' if p in players[:3] and s in scores[:15] else (
            'reserved' if p in players[3:] and s in scores[15:] else 'unused')
        rows.append({'item_id':original['item_id'], 'player_id':p, 'score_id':s, 'performance':f,
            'source_record':original, 'split_role':role, 'style_from_score_prefix':style.group(1)})
    pcm=lambda role:{r['source_record']['audio']['decoded']['decoded_pcm_sha256'] for r in rows if r['split_role']==role}
    require(not pcm('development') & pcm('reserved'), 'development/reserved PCM collision')
    counts=dict(Counter(r['split_role'] for r in rows))
    require(counts=={'development':90,'reserved':90,'unused':180}, 'crossed role counts')
    return {'ordered_player_ids':players, 'ordered_score_ids':scores,
        'development_player_ids':players[:3], 'reserved_player_ids':players[3:],
        'development_score_ids':scores[:15], 'reserved_score_ids':scores[15:], 'role_counts':counts,
        'canonical_pcm_disjoint_development_reserved':True, 'independent_performers_claimed':False,
        'style_recording_counts_by_role':{role:dict(sorted(Counter(r['style_from_score_prefix'] for r in rows
             if r['split_role']==role).items())) for role in ('development','reserved','unused')}, 'rows':rows}


def verify_raw_archives(source, rows, summary):
    """Byte/CRC-only archive replay. No reserved or unused audio decode occurs."""
    acquisition=read_json(source/'source/acquisition_COMMIT.json')
    raw=Path(summary['source']['folder']).resolve(strict=True)
    require(sha(source/'source/acquisition_COMMIT.json')==summary['source']['acquisition_commit_sha256'], 'acquisition binding')
    compare(inventory(raw,summary['source']['acquisition_commit_sha256']),acquisition,'raw acquisition COMMIT',0,0)
    compare(product(source/'source/upstream_metadata.json'),summary['source']['copied_official_metadata'],'official metadata',0,0)
    known={'annotation.zip':(39132574,'b39b78e63d3446f2e54ddb7a54df9b10','8daa02e6417ccca1685feb44b135e95928ad7037e5032ecb326b5791856fda99'),
           'audio_mono-mic.zip':(656927981,'275966d6610ac34999b58426beb119c3','237cdc58353d25c3c9683f4565a0f1cf2db30a9051abca545a919f8f1296dc28')}
    for archive,(size,md5,digest) in known.items():
        path=child(raw,archive)
        compare(product(path),{'bytes':size,'sha256':digest},'official archive SHA',0,0)
        with path.open('rb') as stream:
            require(hashlib.file_digest(stream,'md5').hexdigest()==md5,'official archive MD5')
        compare(summary['source']['archives'][archive],{'bytes':size,'md5':md5,'sha256':digest},'archive receipt',0,0)
        section='annotation' if archive=='annotation.zip' else 'audio'
        members={r[section]['archive_member']:r[section] for r in rows}
        require(len(members)==360,'360 unique archive members')
        with zipfile.ZipFile(path) as package:
            infos=package.infolist()
            require(len(infos)==360 and {i.filename for i in infos}==set(members),'exact ZIP member names')
            for info in infos:
                require(not info.is_dir() and not info.flag_bits & 1,'regular unencrypted member')
                digest=hashlib.sha256(); count=0
                with package.open(info) as stream:
                    while data:=stream.read(1024*1024):
                        digest.update(data);count+=len(data)
                expected=members[info.filename]
                require(count==info.file_size==expected['materialized']['bytes'], 'ZIP full member length')
                require(digest.hexdigest()==expected['archive_member_sha256']==expected['materialized']['sha256'], 'ZIP full member SHA/CRC')
    return raw


def verify_source(document):
    frozen=document['source']
    source=Path(frozen['root']).resolve(strict=True)
    require(frozen['commit']['sha256']==SOURCE_COMMIT_SHA,'immutable validated source')
    compare(full_binding(source/'COMMIT.json'),frozen['commit'],'source COMMIT binding',0,0)
    commit=inventory(source,SOURCE_COMMIT_SHA)
    compare(commit['products'],frozen['products'],'source products',0,0)
    require(len(commit['products'])==725,'source product count')
    compare(commit['classifier_fits'],0,'source classifier fits')
    compare(commit['external_measurement_gate_passed'],False,'source gate')
    compare(full_binding(document['bindings']['source_validator']['path']),frozen['source_validator'],'source validator',0,0)
    rows=[json.loads(line) for line in (source/'source_manifest.jsonl').read_text().splitlines() if line.strip()]
    expected_names={'archive_member_inventory.json','source/acquisition_COMMIT.json','source/upstream_metadata.json',
                    'source_manifest.jsonl','validation_summary.json'}
    for row in rows:
        item=row['item_id']
        require(row['role']=='external_measurement_control_only' and row['class_label'] is None
                and row['annotation_is_ground_truth_for_nonlinear_coupling'] is False,'source scope')
        for section,name in [('audio',f'audio/{item}_mic.wav'),('annotation',f'annotations/{item}.jams')]:
            r=row[section]
            require(r['materialized_path']==name,'source relative item path')
            expected_names.add(name)
            compare(commit['products'][name],r['materialized'],'source row file',0,0)
            require(r['archive_member_sha256']==r['materialized']['sha256'],'archive/materialized identity')
    require(set(commit['products'])==expected_names,'source full audio/annotation pairing inventory')
    summary=read_json(source/'validation_summary.json')
    require(summary['audio']['all_files_fully_decoded_float64_finite_to_empty_eof'] is True
            and summary['counts']['exact_audio_annotation_pairs']==360
            and summary['claims']['audio_annotations_are_exact_identity_matched'] is True,'source prior physical acceptance')
    require(summary['source']['version']=='1.1.0','source version')
    raw=verify_raw_archives(source,rows,summary)
    split=crossed_split(rows)
    compare(document['split'],split,'independent crossed split',0,0)
    return source,raw,split


def verify_frozen_inputs(draft_path,freeze_path,freeze_sha,result_dir):
    require(len(freeze_sha)==64 and sha(freeze_path)==freeze_sha,'explicit trusted parent freeze')
    require(sha(draft_path)==DRAFT_SHA,'immutable reviewed draft')
    freeze,document=read_json(freeze_path),read_json(draft_path)
    require(set(freeze)=={'schema','status','draft','draft_commit','producer_bindings','policy','planned_output_dir','source_commit_sha256'},'freeze keys')
    require(freeze['schema']=='bc_guitarset_development_parent_freeze_v1'
            and freeze['status']=='authorized_development_only','freeze authorization scope')
    compare(freeze['policy'],POLICY,'frozen policy',0,0)
    require(freeze['planned_output_dir']==str(result_dir),'frozen output location')
    require(freeze['source_commit_sha256']==SOURCE_COMMIT_SHA,'frozen source identity')
    compare(full_binding(draft_path),freeze['draft'],'frozen draft',0,0)
    compare(full_binding(draft_path.parent/'COMMIT.json'),freeze['draft_commit'],'draft COMMIT',0,0)
    draft_commit=inventory(draft_path.parent,freeze['draft_commit']['sha256'])
    compare(draft_commit,{'status':'committed','kind':'guitarset_bc_partition_preprocessing_draft_only_v2',
        'products':{'draft.json':product(draft_path)},'bc_extracted':False,
        'source_materialization_commit_sha256':SOURCE_COMMIT_SHA},'draft commit schema',0,0)
    check_bound_tree(freeze['producer_bindings'])
    check_bound_tree(document['bindings'])
    require(document['bindings']['design']['sha256']==DESIGN_SHA,'design immutable hash')
    require(freeze['producer_bindings']['extractor']['sha256']==EXTRACTOR_SHA
            and freeze['producer_bindings']['primitive']['sha256']==PRIMITIVE_SHA,'extractor immutable hashes')
    compare(document['bindings']['runtime'],runtime_snapshot(),'draft runtime',0,0)
    compare(freeze['producer_bindings']['runtime'],runtime_snapshot(True),'producer runtime',0,0)
    require(document['version']=='draft_bicoherence_guitarset_pilot_v2'
            and document['stage']=='partition_preprocessing_draft_only_requires_parent_review','draft version/stage')
    for key in ('bc_extracted','injections_constructed','derived_audio_saved','reserved_or_unused_decoded_by_this_tool','external_gate_passed'):
        compare(document[key],False,'draft scope '+key)
    compare(document['classifier_fits'],0,'draft fits')
    source,raw,split=verify_source(document)
    overlap=document['overlap']
    check_bound_tree(overlap)
    compare(read_json(overlap['acceptance']['path']),overlap['evidence'],'accepted overlap evidence',0,0)
    evidence=overlap['evidence']
    require(evidence['status']=='accepted_for_development_split_only' and evidence['global_non_overlap_proven'] is False
            and evidence['unresolved_matches']==[] and evidence['source_materialization_commit_sha256']==SOURCE_COMMIT_SHA
            and evidence['source_manifest_sha256']==sha(source/'source_manifest.jsonl'),'scoped overlap acceptance')
    require(evidence['independent_review']['accepted'] is True
            and evidence['independent_review']['reviewed_report_sha256']==evidence['report']['sha256'],'overlap review identity binding')
    rows=[r for r in split['rows'] if r['split_role']=='development']
    prepared=document['development_preprocessing']
    require(len(prepared)==90 and [r['item_id'] for r in rows]==[r['item_id'] for r in prepared],'ordered 90 preprocessing rows')
    return document,freeze,source,raw,rows,prepared


def replay_features(audio,construction_status):
    require(audio.dtype==np.float64 and audio.shape==(128000,) and np.isfinite(audio).all(),'two-pool waveform')
    outputs=[replay_pool(audio[k*64000:(k+1)*64000],construction_status) for k in range(2)]
    fixed={'window','frequency_bins','frequency_hz','frame_offset_samples'}
    arrays={key:value if key in fixed else np.concatenate([o[0][key] for o in outputs],axis=0)
            for key,value in outputs[0][0].items()}
    arrays['pool_start_samples']=np.array([0,64000],dtype=np.int64)
    arrays['frame_start_samples']=arrays['frame_offset_samples'][None,:]+arrays['pool_start_samples'][:,None]
    metadata=outputs[0][1]
    metadata.update(input_samples=128000,input_zero_amplitude=not np.any(audio),pool_count=2,
                    analyzed_samples=128000,tail_start_sample=128000)
    pools=[]
    for k,(_,meta) in enumerate(outputs):
        p=meta['pools'][0]
        p.update(pool_index=k,start_sample=k*64000,stop_sample_exclusive=(k+1)*64000)
        pools.append(p)
    metadata['pools']=pools
    return arrays,metadata


def array_json_agreement(path,metadata):
    with np.load(path,allow_pickle=False) as arrays:
        for p,pool in enumerate(metadata['pools']):
            compare(float(arrays['total_non_dc_coefficient_energy'][p]),pool['total_non_dc_coefficient_energy'],'stored total energy',0,0)
            for index,cell in enumerate(pool['cells']):
                triad=cell['frequency_bins']
                compare(arrays['frequency_bins'][index],np.array(triad,dtype=np.int64),'stored grid',0,0)
                fractions=np.array([np.nan if v is None else v for v in cell['energy_fractions']],dtype=np.float64)
                compare(arrays['bin_energy_fraction'][p,triad],fractions,'stored energy fractions',0,0)
                for key,value in {'energy_floor_mask':cell['energy_floor_passed'],'eligible_mask':cell['eligible'],
                    'primitive_defined_mask':cell['primitive']['status']=='ok' and cell['primitive']['squared_bicoherence'] is not None}.items():
                    compare(bool(arrays[key][p,index]),value,'stored mask',0,0)
                for key in ('squared_bicoherence','biphase_radians'):
                    for raw in (False,True):
                        expected=(cell['primitive'] if raw else cell)[key]
                        actual=float(arrays[('primitive_' if raw else '')+key][p,index])
                        if expected is None:
                            require(math.isnan(actual),'stored missingness, not zero')
                        else:
                            compare(actual,expected,'stored raw/masked value',0,0)


def descriptor(metadata):
    return {'pools':[{'pool_index':p['pool_index'],'pool_status':p['status'],
        'target':next(c for c in p['cells'] if c['frequency_bins']==[32,48,80]),
        'grid_cell_count':len(p['cells']),'eligible_cell_count':sum(c['eligible'] for c in p['cells']),
        'grid_eligibility':[c['eligible'] for c in p['cells']], 'grid_status':[c['status'] for c in p['cells']],
        'grid_squared_bicoherence':[c['squared_bicoherence'] for c in p['cells']],
        'grid_raw_squared_bicoherence':[c['primitive']['squared_bicoherence'] for c in p['cells']]}
        for p in metadata['pools']]}


def nuisance(reference,other):
    require(len(reference['pools'])==len(other['pools'])==2,'nuisance pool count')
    result=[]
    for first,second in zip(reference['pools'],other['pools'],strict=True):
        require(first['pool_index']==second['pool_index'],'nuisance pool identities')
        states=Counter(zip(first['grid_eligibility'],second['grid_eligibility'],strict=True))
        observed=[abs(x-y) for x,y in zip(first['grid_squared_bicoherence'],second['grid_squared_bicoherence'],strict=True)
                  if x is not None and y is not None]
        raw=list(zip(first['grid_raw_squared_bicoherence'],second['grid_raw_squared_bicoherence'],strict=True))
        raw_observed=[abs(x-y) for x,y in raw if x is not None and y is not None]
        a,b=first['target'],second['target']
        result.append({'pool_index':first['pool_index'], 'grid_denominator':len(first['grid_eligibility']),
            'grid_eligibility_agreement_count':states[False,False]+states[True,True],
            'missing_to_finite_count':states[False,True], 'finite_to_missing_count':states[True,False],
            'common_finite_cell_count':len(observed), 'maximum_finite_b2_difference':max(observed) if observed else None,
            'primary_difference_scope':'reported_energy_masked_b2_common_eligible_cells',
            'raw_common_finite_cell_count':len(raw_observed),
            'raw_maximum_finite_b2_difference':max(raw_observed) if raw_observed else None,
            'raw_missing_to_finite_count':sum(x is None and y is not None for x,y in raw),
            'raw_finite_to_missing_count':sum(x is not None and y is None for x,y in raw),
            'target_eligibility_agreement':a['eligible']==b['eligible'],
            'target_missing_to_finite':not a['eligible'] and b['eligible'],
            'target_finite_to_missing':a['eligible'] and not b['eligible'],
            'target_difference':b['squared_bicoherence']-a['squared_bicoherence'] if a['eligible'] and b['eligible'] else None})
    return result


def mean_available(values):
    present=[v for v in values if v is not None]
    return math.fsum(present)/len(present) if present else None


def grouped(rows,key):
    groups=[]
    for identity in sorted({row[key] for row in rows}):
        subset=[row for row in rows if row[key]==identity]
        groups.append({key:identity,'recording_denominator':len(subset),
            'covered_recordings':sum(row['difference'] is not None for row in subset),
            'pool_denominator':sum(row['pool_denominator'] for row in subset),
            'paired_covered_pools':sum(row['covered_pools'] for row in subset),
            'mean_difference':mean_available([row['difference'] for row in subset])})
    return {'group_denominator':len(groups), 'covered_groups':sum(g['mean_difference'] is not None for g in groups),
        'equal_group_mean_difference':mean_available([g['mean_difference'] for g in groups]), 'rows':groups}


def paired(rows):
    values=[row['difference'] for row in rows if row['difference'] is not None]
    return {'recording_denominator':len(rows),'covered_recordings':len(values),
        'pool_denominator':sum(r['pool_denominator'] for r in rows),'paired_covered_pools':sum(r['covered_pools'] for r in rows),
        'positive_recordings':sum(v>0 for v in values), 'zero_recordings':sum(v==0 for v in values),
        'negative_recordings':sum(v<0 for v in values),'equal_score':grouped(rows,'score_id'),
        'equal_player_secondary':grouped(rows,'player_id'), 'per_recording':rows}


def summarize(records):
    baseline=[pool for r in records for pool in r['conditions']['baseline']['pools']]
    result={'version':VERSION,'policy':POLICY,'recording_denominator':len(records),
        'independent_replay_status':'required_before_accepting_values',
        'baseline':{'target_pool_denominator':len(baseline),'target_covered_pools':sum(p['target']['eligible'] for p in baseline),
            'grid_cell_denominator':sum(p['grid_cell_count'] for p in baseline),'grid_eligible_cells':sum(p['eligible_cell_count'] for p in baseline)},
        'levels':{},'nuisance':{},'per_recording':records}
    for level in ('minus6db','0db'):
        rows=[]
        for record in records:
            pairs=[]
            for c,i in zip(record['conditions']['closed_'+level]['pools'],record['conditions']['independent_'+level]['pools'],strict=True):
                require(c['pool_index']==i['pool_index'],'paired target pool identity')
                a,b=c['target'],i['target']
                pairs.append({'pool_index':c['pool_index'],'closed_eligible':a['eligible'],'independent_eligible':b['eligible'],
                    'closed_status':a['status'],'independent_status':b['status'],'closed_b2':a['squared_bicoherence'],
                    'independent_b2':b['squared_bicoherence'],
                    'difference':a['squared_bicoherence']-b['squared_bicoherence'] if a['eligible'] and b['eligible'] else None})
            rows.append({key:record[key] for key in ('item_id','player_id','score_id','performance','style_from_score_prefix')} |
                {'pool_denominator':len(pairs),'covered_pools':sum(p['difference'] is not None for p in pairs),
                 'difference':mean_available([p['difference'] for p in pairs]),'paired_pools':pairs})
        result['levels'][level]={**paired(rows),'performance_strata':{
            label:paired([r for r in rows if r['performance']==label]) for label in ('comp','solo')}}
    for name in ('common_gain','polarity'):
        checks=[p for r in records for p in r['nuisance'][name]]
        counts={key:sum(p[key] for p in checks) for key in ('grid_denominator','grid_eligibility_agreement_count',
            'missing_to_finite_count','finite_to_missing_count','common_finite_cell_count','target_eligibility_agreement',
            'target_missing_to_finite','target_finite_to_missing','raw_common_finite_cell_count',
            'raw_missing_to_finite_count','raw_finite_to_missing_count')}
        maxima=[p['maximum_finite_b2_difference'] for p in checks if p['maximum_finite_b2_difference'] is not None]
        raw=[p['raw_maximum_finite_b2_difference'] for p in checks if p['raw_maximum_finite_b2_difference'] is not None]
        result['nuisance'][name]={**counts,'pool_denominator':len(checks),
            'maximum_finite_b2_difference':max(maxima) if maxima else None,
            'primary_difference_scope':'reported_energy_masked_b2_common_eligible_cells',
            'raw_maximum_finite_b2_difference':max(raw) if raw else None}
    return result


def expected_product_names(rows):
    names={'frozen_draft.json','parent_freeze_receipt.json','storage_estimate.json','summary.json','verification.json'}
    for row in rows:
        item=row['item_id']
        require(Path(item).name==item and item not in ('','.', '..'),'unsafe item product identity')
        names.update(item+'/'+name for name in ('construction.json','construction.npz','summary.json'))
        names.update(item+'/'+name+'.'+ext for name in CONDITIONS for ext in ('wav','npz','json'))
    return names


def audit(result_dir,draft_path,freeze_receipt,freeze_sha256,commit_sha256,output):
    auditor_start=full_binding(Path(__file__).resolve())
    tests_path=Path(__file__).with_name('test_audit_bicoherence_guitarset_pilot_v1.py').resolve()
    tests_start=full_binding(tests_path)
    # Resolve destinations only after rejecting symlinks in each path component.
    supplied=[Path(p).absolute() for p in (result_dir,draft_path,freeze_receipt,output)]
    for path in supplied:
        require(not any(parent.is_symlink() for parent in [path,*path.parents]),'symlink audit argument')
    root,draft_path,freeze_path=[p.resolve(strict=True) for p in supplied[:3]]
    output=supplied[3].resolve()
    require(not output.exists() and output.parent.is_dir(),'new exclusive receipt with existing parent')
    for protected in (root,draft_path.parent):
        require(not output.is_relative_to(protected),'audit receipt outside scientific inputs')
    commit=inventory(root,commit_sha256)
    document,freeze,source,raw,rows,prepared=verify_frozen_inputs(draft_path,freeze_path,freeze_sha256,root)
    require(not output.is_relative_to(source) and not output.is_relative_to(raw),'receipt outside source archives')
    compare({k:v for k,v in commit.items() if k!='products'}, {'status':'committed','kind':'BC_GuitarSet_development_only_v1',
        'parent_freeze_receipt_sha256':freeze_sha256,'draft_sha256':DRAFT_SHA,'source_commit_sha256':SOURCE_COMMIT_SHA,
        'external_gate_passed':False,'classifier_admitted':False,'independent_replay_passed':False},'result COMMIT',0,0)
    require(set(commit['products'])==expected_product_names(rows),'complete 90-record/630-condition inventory')
    require(sha(root/'frozen_draft.json')==DRAFT_SHA,'frozen draft copied bytes')
    compare(read_json(root/'parent_freeze_receipt.json'),freeze,'parent freeze copied values',0,0)
    estimate=read_json(root/'storage_estimate.json')
    expected_bytes=90*(7*(2*247*513*16+128000*8+4*1024**2)+32*128000*8+1024**2)
    require(estimate['recordings']==90 and estimate['estimated_product_bytes']==expected_bytes
            and estimate['reserve_bytes']==1024**3 and estimate['required_free_bytes']==expected_bytes+1024**3
            and estimate['filesystem_free_bytes_before']>=estimate['required_free_bytes'],'storage before run')
    records=[]
    errors={'waveform_maximum_absolute_error':0.,'stft_maximum_absolute_error':0.,'b2_maximum_absolute_error':0.}
    for index,(row,preparation) in enumerate(zip(rows,prepared,strict=True)):
        crop=decode_development(source,row,preparation)
        waves,controls,construction=reconstruct(crop,row['item_id'])
        construction.update(frozen_preprocessing=preparation,standardized_crop_saved=True,
            source_pcm_sha256=row['source_record']['audio']['decoded']['decoded_pcm_sha256'])
        directory=root/row['item_id']
        npz_check(directory/'construction.npz',controls,exact=True)
        compare(read_json(directory/'construction.json'),construction,'full construction metadata',0,0)
        record={key:row[key] for key in ('item_id','player_id','score_id','performance','style_from_score_prefix','split_role')}
        record.update(source_record=row['source_record'],construction_status=construction['status'],conditions={})
        for name in CONDITIONS:
            rate,samples=wavfile.read(directory/(name+'.wav'))
            require(rate==16000,'float64 derivative WAV rate')
            compare(samples,waves[name],'full float64 derivative reconstruction',0,0)
            arrays,metadata=replay_features(samples,construction['status'])
            npz_check(directory/(name+'.npz'),arrays)
            saved=read_json(directory/(name+'.json'))
            compare(saved,metadata,row['item_id']+'.'+name)
            array_json_agreement(directory/(name+'.npz'),saved)
            with np.load(directory/(name+'.npz'),allow_pickle=False) as stored:
                errors['stft_maximum_absolute_error']=max(errors['stft_maximum_absolute_error'],float(np.max(abs(stored['spectra']-arrays['spectra']))))
                finite=np.isfinite(arrays['primitive_squared_bicoherence'])
                if np.any(finite):
                    errors['b2_maximum_absolute_error']=max(errors['b2_maximum_absolute_error'],float(np.max(abs(
                        stored['primitive_squared_bicoherence'][finite]-arrays['primitive_squared_bicoherence'][finite]))))
            for actual_pool,expected_pool in zip(saved['pools'],metadata['pools'],strict=True):
                for actual_cell,expected_cell in zip(actual_pool['cells'],expected_pool['cells'],strict=True):
                    a,e=actual_cell['primitive'],expected_cell['primitive']
                    compare(a['coefficient_scales'],e['coefficient_scales'],'relative coefficient scales',RTOL,0)
                    if e['raw_sums'] is not None:
                        compare(a['raw_sums'],e['raw_sums'],'binary raw sums',RTOL,0)
                        for key in ('product_energy_sum','sum_frequency_energy_sum'):
                            compare(a['normalized_sums'][key],e['normalized_sums'][key],'relative normalized energy sums',RTOL,0)
            record['conditions'][name]=descriptor(metadata)
        record['nuisance']={name:nuisance(record['conditions']['baseline'],record['conditions'][name]) for name in ('common_gain','polarity')}
        compare(read_json(directory/'summary.json'),record,'record summary')
        records.append(record)
        print(f'Independent GuitarSet replay {index+1}/90: {row["item_id"]}',file=sys.stderr,flush=True)
    compare(read_json(root/'summary.json'),summarize(records),'all paired/grouped/stratified aggregates')
    compare(read_json(root/'verification.json'),{'frozen_start_end_equal':True,'recordings':90,
        'derivative_float64_wavs':630,'condition_pools':1260,'cell_records':287280,
        'derivative_float64_wavs_exact_roundtrip':True,'reserved_or_unused_decoded_or_measured':False,
        'classifier_fits':0,'independent_replay_status':'required_before_accepting_values'},'verification',0,0)
    inventory(root,commit_sha256)
    end=verify_frozen_inputs(draft_path,freeze_path,freeze_sha256,root)
    compare(end[0],document,'draft stable',0,0)
    compare(end[1],freeze,'freeze stable',0,0)
    compare(full_binding(Path(__file__).resolve()),auditor_start,'auditor unchanged during replay',0,0)
    compare(full_binding(tests_path),tests_start,'auditor tests unchanged during replay',0,0)
    receipt={'passed':True,'scope':'independent_source_raw_bytes_crossed_split_native_decode_resample_crop_construction_full_STFT_all_cells_and_aggregates',
        'result_dir':str(root),'result_commit_sha256':commit_sha256,'draft_sha256':DRAFT_SHA,
        'parent_freeze_receipt_sha256':freeze_sha256,'source_commit_sha256':SOURCE_COMMIT_SHA,
        'audit_code_sha256':auditor_start['sha256'],'audit_tests_sha256':tests_start['sha256'],
        'audit_code_start_end_equal':True,'audit_tests_start_end_equal':True,
        'recordings':90,'development_players':3,'development_scores':15,'reserved_recordings_excluded':90,
        'unused_recordings_excluded':180,'derivative_float64_wavs':630,'condition_pools':1260,'cells_checked':287280,
        'stft_complex_coefficients_checked':1260*247*513,'native_decodes':90,'reserved_or_unused_decoded_or_measured':False,
        'construction_and_WAV_bit_exact':True,'float_rtol':RTOL,'float_atol':ATOL,'maximum_errors':errors,
        'producer_extractor_primitive_drafting_imported':False,'numerical_libraries_independent':False,
        'shared_library_limitation':'independent algorithm implementation shares NumPy FFT/PCG64, SciPy resampling/WAV and SoundFile/libsndfile',
        'overlap_scope':'prior accepted manifest/identifier/exact-hash screening only; global non-overlap not proven',
        'overlapping_frames_independent':False,'crossed_groups_independent':False,
        'biphase_interpretation':'descriptive_only_no_significance','baseline_causal_label':None,
        'positive_response_interpretation':'sensitivity to fixed known injections on selected backgrounds only',
        'external_gate_passed':False,'classifier_admitted':False,'classifier_fits':0,'null_calibrated':False,
        'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'soundfile':sf.__version__}
    with output.open('x') as stream:
        json.dump(receipt,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('result-dir','draft','freeze-receipt','freeze-sha256','commit-sha256','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    print(json.dumps(audit(args.result_dir,args.draft,args.freeze_receipt,args.freeze_sha256,args.commit_sha256,args.output),sort_keys=True))


if __name__=='__main__':
    main()
