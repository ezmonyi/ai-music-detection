"""Independent local BC NSynth replay. Imports no producer or extractor code.

Acceptance requires byte inventory, construction, full waveform STFT, every cell
and summaries. This receipt cannot admit a classifier or establish baseline cause.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import wave

import numpy as np
import scipy
from scipy.io import wavfile

IMMUTABLE = {
    'protocol': '5c2cb09dbc8f79cb44e42ca89fbf1d625f7ccd3e29a16703d8ed11c814f15368',
    'roster': '020c806a5785176cfbffd26a0d0c6ec6fa1f28bc8a0a1c1bb004ce7a0ec7f222',
    'extractor': 'e59fd575add722ca10280f5e19b64d1abe6a1587dac6b0a790fb9d57401f64a1',
    'primitive': '9cedc47b0ee42775c996bbf3e9c8eb237aa1acde4a051634b077fd72fd7614d5',
}
SOURCE_COMMIT = '85f70e63a033e1509d3c2fbad6c4561c3f1fce2511117756af64f78b9c39e330'
ARCHIVE_SHA = '0f9ba5d62beba9ec4612f918d19f5e87a681822f1c566124f05fe8b27a51934c'
CONDITIONS = ('baseline', 'common_gain', 'polarity', 'closed_minus6db',
              'independent_minus6db', 'closed_0db', 'independent_0db')
POLICY = {'conditions': list(CONDITIONS), 'target_bins': [32, 48, 80],
          'seed_prefix': 'BC-injection-20260907|', 'phase_knots': 33,
          'sample_rate': 16000, 'samples': 64000, 'background_gain': .25,
          'relative_injection_db': [-6, 0], 'prototype_tone_amplitude': .2,
          'aggregation': 'mean within instrument then equal weight covered instruments',
          'framewise_power_matched': False, 'numeric_admission_threshold': None,
          'external_gate_passed': False, 'classifier_admitted': False,
          'baseline_causal_label': None, 'reserved_waveforms_measured': False}
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


def verify_bindings(bindings):
    require(set(bindings) == {'runner', 'runner_tests', 'extractor', 'primitive',
                             'extractor_tests', 'primitive_tests', 'protocol', 'roster', 'runtime'}, 'binding schema')
    def file_binding(binding):
        path = Path(binding['path'])
        require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'local binding path')
        compare({'path': str(path.resolve()), **product(path)}, binding, 'binding', 0, 0)
    for key, binding in bindings.items():
        if key != 'runtime':
            file_binding(binding)
    for key, digest in IMMUTABLE.items():
        require(bindings[key]['sha256'] == digest, 'immutable ' + key)
    runtime = bindings['runtime']
    require(sys.version_info[:2] == (3, 12) and np.__version__ == '2.4.4'
            and scipy.__version__ == '1.17.1', 'auditor numerical runtime')
    modules = {'numpy': np, 'scipy': scipy, 'scipy.io.wavfile': wavfile, 'wave': wave,
               'numpy.fft._pocketfft': np.fft._pocketfft,
               'numpy.fft._pocketfft_umath': np.fft._pocketfft.pfu,
               'numpy._core._multiarray_umath': np._core._multiarray_umath}
    actual = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
              'platform': platform.platform(), 'executable': {'path': str(Path(sys.executable).resolve()), **product(sys.executable)},
              'modules': {k: {'path': str(Path(v.__file__).resolve()), **product(v.__file__)} for k, v in modules.items()}}
    compare(runtime, actual, 'runtime', 0, 0)
    file_binding(runtime['executable'])
    for binding in runtime['modules'].values():
        file_binding(binding)


def selection_from_source(source, frozen_source):
    commit = inventory(source, SOURCE_COMMIT)
    compare(frozen_source, {'source_dir': str(source), 'products': {
        **commit['products'], 'COMMIT.json': product(source/'COMMIT.json')}}, 'frozen source', 0, 0)
    require(commit['products']['nsynth-test.jsonwav.tar.gz']['sha256'] == ARCHIVE_SHA, 'archive identity')
    receipt = read_json(source/'acquisition_receipt.json')
    for key, value in {'decoded_all': True, 'notes': 4096, 'instruments': 53,
                       'features_extracted': False, 'ai_human_labels_assigned': False}.items():
        compare(receipt[key], value, 'source receipt.'+key)
    metadata = read_json(source/'nsynth-test/examples.json')
    rows = [json.loads(line) for line in (source/'manifest.jsonl').read_text().splitlines() if line.strip()]
    ids = {r['id'] for r in rows}
    require(len(rows) == len(ids) == 4096 and ids == set(metadata), 'all source notes')
    grouped = defaultdict(list)
    for row in rows:
        note, m = row['id'], row['metadata']
        path = child(source, f'nsynth-test/audio/{note}.wav')
        compare(m, metadata[note], 'source metadata', 0, 0)
        for key, value in {'path': str(path), 'file_sha256': commit['products'][str(path.relative_to(source))]['sha256'],
                           'native_channels': 1, 'native_sample_rate': 16000, 'frames': 64000,
                           'duration_s': 4, 'role': 'external_measurement_control_only', 'ai_human_label': None}.items():
            compare(row[key], value, 'source row.'+key, 0, 0)
        require(m['note_str'] == note and m['sample_rate'] == 16000, 'note metadata')
        require(len(row['pcm_sha256']) == 64 and all(c in '0123456789abcdef' for c in row['pcm_sha256']), 'PCM SHA syntax')
        grouped[m['instrument_str']].append(row)
    require({n for n in commit['products'] if n.endswith('.wav')} == {f'nsynth-test/audio/{n}.wav' for n in ids}, 'all source WAVs')
    ordered = sorted(grouped, key=lambda n: (hashlib.sha256(('Q-pilot-20260907|'+n).encode()).hexdigest(), n))
    require(len(ordered) == 53, '53 source instruments')
    selected = []
    for instrument in ordered[:27]:
        candidates = sorted(grouped[instrument], key=lambda r: (abs(r['metadata']['pitch']-60),
            hashlib.sha256(('Q-note-20260907|'+r['id']).encode()).hexdigest(), r['id']))
        seen = set()
        for row in candidates:
            if row['pcm_sha256'] not in seen:
                selected.append(row)
                seen.add(row['pcm_sha256'])
            if len(seen) == 2:
                break
        require(len(seen) == 2, 'two distinct PCM per instrument')
    pilot_pcm = {r['pcm_sha256'] for n in ordered[:27] for r in grouped[n]}
    reserved_pcm = {r['pcm_sha256'] for n in ordered[27:] for r in grouped[n]}
    require(not pilot_pcm & reserved_pcm, 'pilot reserved PCM disjoint')
    return {'ordered_instrument_ids': ordered, 'pilot_instrument_ids': ordered[:27],
            'reserved_instrument_ids': ordered[27:], 'selected_notes': selected,
            'pilot_reserved_pcm_sets_disjoint': True}


def pcm_audio(source, row):
    path = child(source, f"nsynth-test/audio/{row['id']}.wav")
    require(str(path) == row['path'] and sha(path) == row['file_sha256'], 'selected file SHA')
    with wave.open(str(path), 'rb') as stream:
        require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(),
                 stream.getnframes(), stream.getcomptype()) == (1, 2, 16000, 64000, 'NONE'), 'PCM WAV format')
        data = stream.readframes(64000)
        require(len(data) == 128000 and not stream.readframes(1), 'full PCM length')
    require(hashlib.sha256(data).hexdigest() == row['pcm_sha256'], 'decoded PCM SHA')
    return np.frombuffer(data, dtype='<i2').astype(np.float64)/32768


def reconstruct(audio, note):
    require(audio.dtype == np.float64 and audio.shape == (64000,) and np.isfinite(audio).all(), 'construction input')
    seed = int.from_bytes(hashlib.sha256(('BC-injection-20260907|'+note).encode()).digest()[:8], 'big')
    knots = np.random.Generator(np.random.PCG64(seed)).uniform(-np.pi, np.pi, (3, 33))
    unwrapped = np.unwrap(knots, axis=1)
    kt, t = np.arange(33, dtype=np.float64)/8, np.arange(64000, dtype=np.float64)/16000
    phases = np.array([np.interp(t, kt, p) for p in unwrapped])
    closed_phases = np.array([phases[0], phases[1], phases[0]+phases[1]])
    carriers = 2*np.pi*np.array([500., 750., 1250.])[:, None]*t
    c, i = [.2*np.cos(carriers+p).sum(axis=0) for p in (closed_phases, phases)]
    cr, ir = [float(np.sqrt(np.mean(v*v))) for v in (c, i)]
    require(cr > 0 and ir > 0, 'positive prototype RMS')
    b = audio*.25
    rms = float(np.sqrt(np.mean(b*b)))
    require(math.isfinite(rms) and not (rms == 0 and np.any(b)), 'unsupported background RMS')
    cu, iu = c/cr, i/ir
    waves = {'baseline': b, 'common_gain': b*.1, 'polarity': -b}
    arrays = {'source_audio': audio, 'background': b, 'sample_times_seconds': t,
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
    meta = {'note_id': note, 'seed': seed, 'bit_generator': 'PCG64', 'status': status,
            'background_rms': rms, 'prototype_rms': {'closed': cr, 'independent': ir},
            'prototype_normalization_factors': {'closed': 1/cr, 'independent': 1/ir},
            'injected_full_record_rms_matched': rms > 0, 'framewise_power_matched': False,
            'zero_rms_interpretation': 'zero waveforms retained; relative-level intervention unsupported'}
    return waves, arrays, meta


def sums_for(columns):
    product_column = columns[:, 0]*columns[:, 1]
    third = columns[:, 2]
    triple = np.sum(product_column*np.conjugate(third))
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
    require(math.isfinite(score) and score <= 1+64*np.finfo(float).eps, 'Cauchy Schwarz')
    result['squared_bicoherence'] = min(1., score)
    if triple == 0:
        result['biphase_status'] = 'undefined_zero_resultant'
    else:
        result['biphase_status'] = 'descriptive_only_no_significance'
        result['biphase_radians'] = math.atan2(triple.imag, triple.real)
    return result


def replay_features(audio, construction_status):
    grid = np.asarray([(a,b,a+b) for a in range(8,193,8) for b in range(a,193,8) if a+b<=256], dtype=np.int64)
    window = .5-.5*np.cos(2*np.pi*np.arange(1024)/1024)
    offsets = np.arange(247, dtype=np.int64)*256
    frames = np.array([audio[start:start+1024] for start in offsets])
    means = np.mean(frames, axis=1)
    spectra = np.array([np.fft.rfft((frame-mean)*window)/np.sum(window) for frame,mean in zip(frames,means)])
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


def array_json_agreement(path, metadata):
    """Stored JSON and arrays must agree exactly, independently of replay error."""
    cells = metadata['pools'][0]['cells']
    expected = {
        'energy_floor_mask': np.array([[c['energy_floor_passed'] for c in cells]], dtype=bool),
        'eligible_mask': np.array([[c['eligible'] for c in cells]], dtype=bool),
        'primitive_defined_mask': np.array([[c['primitive']['status']=='ok' and c['primitive']['squared_bicoherence'] is not None for c in cells]], dtype=bool),
    }
    for key in ('squared_bicoherence', 'biphase_radians'):
        for raw in (False, True):
            values = [(c['primitive'] if raw else c)[key] for c in cells]
            expected[('primitive_' if raw else '')+key] = np.array([[np.nan if v is None else v for v in values]])
    with np.load(path, allow_pickle=False) as stored:
        for key, values in expected.items():
            compare(stored[key], values, 'stored JSON/array '+key, 0, 0)
        for c in cells:
            triad = c['frequency_bins']
            fractions = stored['bin_energy_fraction'][0,triad]
            expected_fraction = np.array([np.nan if v is None else v for v in c['energy_fractions']])
            compare(fractions, expected_fraction, 'stored JSON/array fractions', 0, 0)


def descriptor(meta):
    cells = meta['pools'][0]['cells']
    return {'target': next(c for c in cells if c['frequency_bins'] == [32,48,80]),
            'grid_cell_count': len(cells), 'eligible_cell_count': sum(c['eligible'] for c in cells),
            'grid_eligibility': [c['eligible'] for c in cells],
            'grid_squared_bicoherence': [c['squared_bicoherence'] for c in cells]}


def nuisance(a, b):
    states = Counter(zip(a['grid_eligibility'], b['grid_eligibility']))
    delta = [abs(x-y) for x,y in zip(a['grid_squared_bicoherence'], b['grid_squared_bicoherence'], strict=True) if x is not None and y is not None]
    ta, tb = a['target'], b['target']
    return {'grid_denominator': len(a['grid_eligibility']), 'grid_eligibility_agreement_count': states[False,False]+states[True,True],
            'missing_to_finite_count': states[False,True], 'finite_to_missing_count': states[True,False],
            'common_finite_cell_count': len(delta), 'maximum_finite_b2_difference': max(delta) if delta else None,
            'target_eligibility_agreement': ta['eligible'] == tb['eligible'],
            'target_difference': tb['squared_bicoherence']-ta['squared_bicoherence'] if ta['eligible'] and tb['eligible'] else None}


def balanced(rows):
    instruments = sorted({r['instrument'] for r in rows})
    per = []
    for instrument in instruments:
        group = [r['difference'] for r in rows if r['instrument'] == instrument]
        observed = [v for v in group if v is not None]
        per.append({'instrument': instrument, 'note_denominator': len(group), 'covered_notes': len(observed),
                    'mean_difference': math.fsum(observed)/len(observed) if observed else None})
    means = [p['mean_difference'] for p in per if p['mean_difference'] is not None]
    values = [r['difference'] for r in rows if r['difference'] is not None]
    return {'note_denominator': len(rows), 'covered_notes': len(values), 'instrument_denominator': len(instruments),
            'covered_instruments': len(means), 'equal_instrument_mean_difference': math.fsum(means)/len(means) if means else None,
            'positive': sum(v>0 for v in values), 'zero': sum(v==0 for v in values),
            'negative': sum(v<0 for v in values), 'per_instrument': per}


def summarize(notes):
    baseline = [n['conditions']['baseline'] for n in notes]
    result = {'note_denominator': len(notes), 'policy': POLICY, 'levels': {},
              'independent_replay_status': 'required_before_accepting_values',
              'baseline': {'target_covered_notes': sum(x['target']['eligible'] for x in baseline),
                           'grid_cell_denominator': sum(x['grid_cell_count'] for x in baseline),
                           'grid_eligible_cells': sum(x['eligible_cell_count'] for x in baseline)}, 'per_note': notes}
    for level in ('minus6db', '0db'):
        rows = []
        for n in notes:
            c, i = [n['conditions'][kind+'_'+level]['target'] for kind in ('closed', 'independent')]
            rows.append({k:n[k] for k in ('note_id','instrument','source_category','instrument_family')} |
                        {'difference': c['squared_bicoherence']-i['squared_bicoherence'] if c['eligible'] and i['eligible'] else None,
                         'closed_eligible': c['eligible'], 'independent_eligible': i['eligible']})
        strata = {key: {v:balanced([r for r in rows if r[key]==v]) for v in sorted({r[key] for r in rows})}
                  for key in ('source_category', 'instrument_family')}
        result['levels'][level] = {**balanced(rows), 'strata': strata, 'per_note': rows}
    result['nuisance'] = {}
    for name in ('common_gain', 'polarity'):
        checks = [n['nuisance'][name] for n in notes]
        values = [c['maximum_finite_b2_difference'] for c in checks if c['maximum_finite_b2_difference'] is not None]
        result['nuisance'][name] = {key:sum(c[key] for c in checks) for key in
            ('grid_denominator','grid_eligibility_agreement_count','missing_to_finite_count','finite_to_missing_count','common_finite_cell_count')}
        result['nuisance'][name].update(maximum_finite_b2_difference=max(values) if values else None,
            target_eligibility_agreement_notes=sum(c['target_eligibility_agreement'] for c in checks))
    return result


def audit(result_dir, draft_path, freeze_sha256, commit_sha256, output):
    root, draft_path, output = Path(result_dir).resolve(strict=True), Path(draft_path).resolve(strict=True), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(root) and not output.is_relative_to(draft_path.parent), 'new outside receipt required')
    require(sha(draft_path) == freeze_sha256, 'reviewed draft SHA')
    draft = read_json(draft_path)
    draft_commit = inventory(draft_path.parent, sha(draft_path.parent/'COMMIT.json'))
    compare(draft_commit, {'status':'committed', 'kind':'BC_development_draft',
        'draft_sha256':freeze_sha256, 'products':{'draft.json':product(draft_path)}}, 'draft inventory', 0, 0)
    source = Path(draft['source']['source_dir']).resolve(strict=True)
    require(not output.is_relative_to(source), 'receipt outside source')
    commit = inventory(root, commit_sha256)
    compare({k:v for k,v in commit.items() if k!='products'}, {'status':'committed', 'kind':'BC_external_note_development_only',
        'draft_sha256':freeze_sha256, 'classifier_admitted':False, 'external_gate_passed':False, 'independent_replay_passed':False}, 'run COMMIT', 0, 0)
    require(sha(root/'frozen_draft.json') == freeze_sha256, 'frozen draft bytes')
    compare(read_json(root/'frozen_draft.json'), draft, 'frozen draft', 0, 0)
    for key,value in {'version':'bicoherence_nsynth_development_pilot_v1', 'stage':'draft_requires_reviewed_sha256',
                      'policy':POLICY, 'planned_run_output_dir':str(root), 'features_extracted':False,
                      'selected_pcm_verified':True, 'reserved_pcm_decoded':False}.items():
        compare(draft[key], value, 'draft.'+key, 0, 0)
    verify_bindings(draft['bindings'])
    selection = selection_from_source(source, draft['source'])
    compare(draft['selection'], selection, 'selection', 0, 0)
    compare(read_json(draft['bindings']['roster']['path'])['selection'], selection, 'old Q roster', 0, 0)
    rows = selection['selected_notes']
    require(len(rows)==54 and len(selection['reserved_instrument_ids'])==26, '54 / 26 accounting')
    expected_products = {'frozen_draft.json', 'summary.json', 'verification.json'}
    for row in rows:
        expected_products.update(row['id']+'/'+name for name in ('construction.npz','construction.json','summary.json'))
        expected_products.update(row['id']+'/'+name+'.'+ext for name in CONDITIONS for ext in ('wav','npz','json'))
    require(set(commit['products']) == expected_products, 'complete 378-condition inventory')
    notes, max_stft_error = [], 0.
    for index, row in enumerate(rows):
        note_id, m = row['id'], row['metadata']
        require(m['instrument_str'] not in selection['reserved_instrument_ids'], 'reserved waveform forbidden')
        audio = pcm_audio(source, row)
        waves, controls, construction = reconstruct(audio, note_id)
        directory = root/note_id
        npz_check(directory/'construction.npz', controls, exact=True)
        compare(read_json(directory/'construction.json'), construction, 'construction metadata', 0, 0)
        record = {'note_id':note_id, 'instrument':m['instrument_str'], 'source_category':m['instrument_source_str'],
                  'instrument_family':m['instrument_family_str'], 'pitch':m['pitch'], 'velocity':m['velocity'],
                  'source_record':row, 'construction_status':construction['status'], 'conditions':{}}
        for name in CONDITIONS:
            rate, decoded = wavfile.read(directory/(name+'.wav'))
            require(rate==16000, 'derivative sample rate')
            compare(decoded, waves[name], 'full derivative WAV', 0, 0)
            arrays, metadata = replay_features(decoded, construction['status'])
            npz_check(directory/(name+'.npz'), arrays)
            with np.load(directory/(name+'.npz'), allow_pickle=False) as saved:
                max_stft_error = max(max_stft_error, float(np.max(abs(saved['spectra']-arrays['spectra']))))
            recorded_metadata = read_json(directory/(name+'.json'))
            compare(recorded_metadata, metadata, note_id+'.'+name)
            array_json_agreement(directory/(name+'.npz'), recorded_metadata)
            for actual_cell, expected_cell in zip(recorded_metadata['pools'][0]['cells'], metadata['pools'][0]['cells'], strict=True):
                compare(actual_cell['primitive']['coefficient_scales'], expected_cell['primitive']['coefficient_scales'], 'relative coefficient scales', RTOL, 0)
                if expected_cell['primitive']['raw_sums'] is not None:
                    compare(actual_cell['primitive']['raw_sums'], expected_cell['primitive']['raw_sums'], 'binary raw sums', RTOL, 0)
            record['conditions'][name] = descriptor(metadata)
        record['nuisance'] = {name:nuisance(record['conditions']['baseline'],record['conditions'][name]) for name in ('common_gain','polarity')}
        compare(read_json(directory/'summary.json'), record, 'per-note summary')
        notes.append(record)
        print(f'Independent BC replay {index+1}/54: {note_id}', file=sys.stderr, flush=True)
    compare(read_json(root/'summary.json'), summarize(notes), 'complete aggregate')
    compare(read_json(root/'verification.json'), {'frozen_start_end_equal':True,
        'derivative_float64_wavs_exact_roundtrip':True, 'notes':54, 'conditions':378, 'reserved_pcm_decoded':False,
        'independent_replay_status':'required_before_accepting_values'}, 'producer verification', 0, 0)
    # Re-hash all immutable inputs and outputs after replay, not only COMMIT.
    inventory(root, commit_sha256)
    selection_from_source(source, draft['source'])
    verify_bindings(draft['bindings'])
    require(sha(draft_path)==freeze_sha256, 'draft changed during audit')
    receipt = {'passed':True, 'scope':'independent_bytes_construction_full_waveform_stft_all_cells_and_aggregates',
               'result_dir':str(root), 'result_commit_sha256':commit_sha256, 'draft_sha256':freeze_sha256,
               'audit_code_sha256':sha(__file__), 'audit_tests_sha256':sha(Path(__file__).with_name('test_audit_bicoherence_nsynth_pilot_v1.py')),
               'notes':54, 'instruments':27, 'reserved_instruments_excluded':26, 'conditions':378,
               'cells_checked':378*228, 'stft_complex_coefficients_checked':378*247*513,
               'construction_and_wav_bit_exact':True, 'maximum_stft_absolute_error':max_stft_error,
               'float_rtol':RTOL, 'float_atol':ATOL, 'producer_or_extractor_imported':False,
               'overlapping_frames_independent':False, 'biphase_interpretation':'descriptive_only_no_significance',
               'positive_response_interpretation':'known controlled injection sensitivity on selected backgrounds only',
               'baseline_causal_label':None, 'external_gate_passed':False, 'classifier_admitted':False, 'classifier_fits':0,
               'null_calibrated':False, 'python':platform.python_version(), 'numpy':np.__version__, 'scipy':scipy.__version__}
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('result-dir','draft','freeze-sha256','commit-sha256','output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.result_dir,args.draft,args.freeze_sha256,args.commit_sha256,args.output), sort_keys=True))


if __name__ == '__main__':
    main()
