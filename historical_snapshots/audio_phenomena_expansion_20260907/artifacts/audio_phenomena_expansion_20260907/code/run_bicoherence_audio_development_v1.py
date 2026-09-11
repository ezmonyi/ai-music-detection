"""Preserve fixed continuous synthetic BC controls; no external music or fits."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

import bicoherence_audio_v1 as bc
import bicoherence_primitive_v2 as primitive

SEED = 20260907
SAMPLE_RATE = 16000
SAMPLES = 16 * SAMPLE_RATE
TARGET = [32, 48, 80]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def fixtures():
    rng = np.random.default_rng(SEED)
    t = np.arange(SAMPLES, dtype=np.float64) / SAMPLE_RATE
    knots = np.arange(129, dtype=np.float64) * 0.125
    knot_phases = np.unwrap(rng.uniform(-np.pi, np.pi, (3, 129)), axis=1)
    phases = np.asarray([np.interp(t, knots, row) for row in knot_phases])
    constant = np.array([0.2, 0.7, -1.1])[:, None]
    frequencies = np.array([500., 750., 1250.])[:, None]

    def compose(freq=frequencies, phase=constant, amplitudes=0.2):
        return np.sum(amplitudes * np.cos(2 * np.pi * freq * t + phase), axis=0)

    closed_phase = np.vstack([phases[:2], phases[0] + phases[1]])
    closed = compose(phase=closed_phase)
    off_parents = np.array([500. + 0.3 * 15.625, 750. + 0.2 * 15.625])
    off_frequencies = np.r_[off_parents, off_parents.sum()][:, None]
    changed_amplitude = np.full((3, SAMPLES), 0.2)
    changed_amplitude[2, np.arange(SAMPLES) % 64000 >= 32000] = 2.0
    low_amplitude = np.array([0.2, 0.2, 0.2e-5])[:, None]
    signals = {
        'static_linear_on_bin': compose(),
        'static_linear_off_bin_sum': compose(freq=off_frequencies),
        'static_linear_nonsum_detuned': compose(freq=np.array([500., 750., 1255.])[:, None]),
        'continuous_phase_closed': closed,
        'continuous_phase_independent': compose(phase=phases),
        'constant_phase_amplitude_mismatch': compose(amplitudes=changed_amplitude),
        'closed_gain_0p1': closed * 0.1,
        'closed_polarity_inverted': -closed,
        'closed_circular_shift_19': np.roll(closed, 19),
        'gaussian_noise': rng.normal(0.0, np.sqrt(3 * 0.2**2 / 2), SAMPLES),
        'silence': np.zeros(SAMPLES, dtype=np.float64),
        'static_third_tone_below_energy_floor': compose(amplitudes=low_amplitude),
    }
    controls = {'knot_times_seconds': knots, 'unwrapped_knot_phases': knot_phases,
                'continuous_phase_trajectories': phases, 'base_frequencies_hz': frequencies[:, 0],
                'off_bin_frequencies_hz': off_frequencies[:, 0]}
    return signals, controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--extractor-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if sha(args.protocol) != args.protocol_sha256:
        raise ValueError('protocol hash mismatch')
    if sha(Path(bc.__file__)) != args.extractor_sha256:
        raise ValueError('extractor hash mismatch')
    if sha(Path(primitive.__file__)) != '9cedc47b0ee42775c996bbf3e9c8eb237aa1acde4a051634b077fd72fd7614d5':
        raise ValueError('unreviewed primitive')
    root = args.output.resolve()
    root.mkdir(exist_ok=False)
    try:
        command = [sys.executable, '-B', '-W', 'error', '-m', 'unittest',
                   'test_bicoherence_audio_v1', 'test_bicoherence_primitive_v2', '-v']
        tests = subprocess.run(command, cwd=Path(__file__).resolve().parent,
                               capture_output=True, text=True, timeout=300)
        (root / 'tests.stdout.txt').write_text(tests.stdout)
        (root / 'tests.stderr.txt').write_text(tests.stderr)
        if tests.returncode:
            raise RuntimeError('parent pre-experiment tests failed')
        (root / 'development_protocol.md').write_bytes(args.protocol.read_bytes())
        signals, controls = fixtures()
        np.savez_compressed(root / 'construction_controls.npz', **controls)
        summary = []
        for name, waveform in signals.items():
            if waveform.dtype != np.float64 or waveform.shape != (SAMPLES,) or not np.isfinite(waveform).all():
                raise ValueError('invalid synthetic waveform: ' + name)
            np.save(root / (name + '.waveform.npy'), waveform, allow_pickle=False)
            result = bc.extract(waveform, SAMPLE_RATE)
            np.savez_compressed(root / (name + '.arrays.npz'), **result['arrays'])
            save_json(root / (name + '.measurement.json'), result['metadata'])
            index, = np.flatnonzero(np.all(result['arrays']['frequency_bins'] == TARGET, axis=1))
            target = [p['cells'][index] for p in result['metadata']['pools']]
            values = [c['squared_bicoherence'] for c in target if c['eligible']]
            entry = {'condition': name, 'samples': SAMPLES, 'sample_rate_hz': SAMPLE_RATE,
                'waveform_peak_abs': float(np.max(np.abs(waveform))),
                'target_frequency_bins': TARGET, 'pool_count': len(target),
                'eligible_target_pools': len(values), 'target_mean_b2': float(np.mean(values)) if values else None,
                'target_min_b2': min(values) if values else None,
                'target_max_b2': max(values) if values else None,
                'target_pools': [{'pool_index': i, 'eligible': c['eligible'], 'status': c['status'],
                    'b2': c['squared_bicoherence'], 'raw_b2': c['primitive']['squared_bicoherence'],
                    'biphase_radians': c['biphase_radians'], 'energy_fractions': c['energy_fractions']}
                    for i, c in enumerate(target)],
                'all_grid_eligible_cells_per_pool': [p['eligible_cell_count'] for p in result['metadata']['pools']]}
            summary.append(entry)
            print(json.dumps({'condition': name, 'target_mean_b2': entry['target_mean_b2'],
                              'eligible_target_pools': len(values)}), flush=True)
        save_json(root / 'summary.json', {'conditions': summary, 'external_gate_passed': False,
                  'classifier_fits': 0, 'null_calibrated': False, 'phase_controls_stft_power_matched': False})
        code_names = ['run_bicoherence_audio_development_v1.py', 'bicoherence_audio_v1.py',
                      'test_bicoherence_audio_v1.py', 'bicoherence_primitive_v2.py',
                      'test_bicoherence_primitive_v2.py']
        code_root = Path(__file__).resolve().parent
        if sha(args.protocol) != args.protocol_sha256 or sha(Path(bc.__file__)) != args.extractor_sha256:
            raise ValueError('protocol or extractor changed during execution')
        save_json(root / 'run_manifest.json', {'status': 'completed_synthetic_development_only',
            'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'seed': SEED,
            'protocol_sha256': args.protocol_sha256, 'code_sha256': {n: sha(code_root / n) for n in code_names},
            'test_command': command, 'test_returncode': tests.returncode,
            'conditions': len(summary), 'pools_per_condition': 4, 'grid_cells_per_pool': 228,
            'external_audio_used': False, 'external_gate_passed': False, 'classifier_fits': 0,
            'waveform_storage': 'NumPy float64, unnormalized, no clipping or codec',
            'ai_human_labels_assigned': False})
        products = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in root.iterdir()}
        save_json(root / 'COMMIT.json', {'status': 'committed', 'kind': 'synthetic_audio_development', 'products': products})
        print(json.dumps({'status': 'committed', 'products': len(products), 'commit_sha256': sha(root / 'COMMIT.json')}))
    except BaseException as exc:
        save_json(root / 'EXECUTION_FAILURE.json', {'error': str(exc), 'type': type(exc).__name__, 'commit_published': False})
        raise


if __name__ == '__main__':
    main()
