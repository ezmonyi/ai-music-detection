"""Independent waveform/STFT/sufficient-sum replay; no producer imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, context):
    if not value:
        raise ValueError(context)


def close(actual, expected, context, atol=2e-14, rtol=2e-12):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True), context)


def direct_cell(spectra, pair):
    a, b, c = pair
    u, v = spectra[:, a] * spectra[:, b], spectra[:, c]
    product_energy = float(np.sum(np.abs(u)**2))
    third_energy = float(np.sum(np.abs(v)**2))
    triple = complex(np.sum(u * np.conj(v)))
    if product_energy == 0 or third_energy == 0:
        return None, None, None
    b2 = min(1., abs(triple)**2 / (product_energy * third_energy))
    phase = math.atan2(triple.imag, triple.real) if triple != 0 else None
    sums = {'product_energy_sum': product_energy, 'sum_frequency_energy_sum': third_energy,
            'triple_sum_real': triple.real, 'triple_sum_imag': triple.imag}
    return b2, phase, sums


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--commit-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    require(not output.exists() and not output.is_relative_to(root), 'new outside receipt required')
    marker = root / 'COMMIT.json'
    require(sha(marker) == args.commit_sha256, 'commit identity')
    products = json.loads(marker.read_text())['products']
    require(len(products) == 42, 'product count')
    require(not any(p.is_symlink() for p in root.rglob('*')), 'symlink')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(actual == set(products) | {'COMMIT.json'}, 'inventory')
    for name, item in products.items():
        require((root / name).resolve().is_relative_to(root), 'path escape')
        require((root / name).stat().st_size == item['bytes'] and sha(root / name) == item['sha256'], name)
    manifest = json.loads((root / 'run_manifest.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    require(manifest['seed'] == 20260907 and manifest['classifier_fits'] == 0, 'scope/seed')
    require(not manifest['external_audio_used'] and not manifest['external_gate_passed'], 'admission scope')
    require(sha(root / 'development_protocol.md') == manifest['protocol_sha256'], 'protocol bytes')
    require(len(summary['conditions']) == 12, 'conditions')

    # Independent waveform construction from the fixed design, no runner import.
    rng = np.random.default_rng(20260907)
    t = np.arange(256000, dtype=np.float64) / 16000
    times = np.arange(129) / 8
    knot_phases = np.unwrap(rng.uniform(-np.pi, np.pi, (3, 129)), axis=1)
    trajectories = np.array([np.interp(t, times, p) for p in knot_phases])
    with np.load(root / 'construction_controls.npz', allow_pickle=False) as stored:
        close(stored['knot_times_seconds'], times, 'knot times', 0, 0)
        close(stored['unwrapped_knot_phases'], knot_phases, 'knot phases', 0, 0)
        close(stored['continuous_phase_trajectories'], trajectories, 'phase trajectories', 0, 0)
    tones = np.array([np.cos(2 * np.pi * f * t + phase)
                      for f, phase in zip([500, 750, 1250], [.2, .7, -1.1])])
    closed = .2 * sum(np.cos(2 * np.pi * f * t + p)
                     for f, p in zip([500, 750, 1250],
                                     [trajectories[0], trajectories[1], trajectories[0] + trajectories[1]]))
    off = [500 + .3 * 15.625, 750 + .2 * 15.625]
    third_amp = np.where(np.arange(256000) % 64000 < 32000, .2, 2.)
    expected_waveforms = {
        'static_linear_on_bin': .2 * sum(tones),
        'static_linear_off_bin_sum': .2 * sum(np.cos(2*np.pi*f*t + p)
            for f, p in zip([*off, sum(off)], [.2, .7, -1.1])),
        'static_linear_nonsum_detuned': .2 * sum(np.cos(2*np.pi*f*t + p)
            for f, p in zip([500,750,1255], [.2,.7,-1.1])),
        'continuous_phase_closed': closed,
        'continuous_phase_independent': .2 * sum(np.cos(2*np.pi*f*t + p)
            for f, p in zip([500,750,1250], trajectories)),
        'constant_phase_amplitude_mismatch': .2*tones[0] + .2*tones[1] + third_amp*tones[2],
        'closed_gain_0p1': closed * .1,
        'closed_polarity_inverted': -closed,
        'closed_circular_shift_19': np.concatenate([closed[-19:], closed[:-19]]),
        'gaussian_noise': rng.normal(0, np.sqrt(3 * .2**2 / 2), 256000),
        'silence': np.zeros(256000),
        'static_third_tone_below_energy_floor': .2*tones[0] + .2*tones[1] + .2e-5*tones[2]}
    require({s['condition'] for s in summary['conditions']} == set(expected_waveforms), 'condition identities')
    grid = np.array([(a, b, a+b) for a in range(8,193,8) for b in range(a,193,8) if a+b<=256])
    window = .5 - .5 * np.cos(2*np.pi*np.arange(1024)/1024)
    target_index = next(i for i, p in enumerate(grid.tolist()) if p == [32,48,80])
    maxima = {'waveform_absolute_error': 0., 'stft_absolute_error': 0., 'b2_absolute_error': 0.}
    total_cells = 0
    for row in summary['conditions']:
        name = row['condition']
        waveform = np.load(root / (name + '.waveform.npy'), allow_pickle=False)
        require(waveform.shape == (256000,) and waveform.dtype == np.float64, name + ' shape/dtype')
        close(waveform, expected_waveforms[name], name + ' constructed waveform')
        maxima['waveform_absolute_error'] = max(maxima['waveform_absolute_error'], float(np.max(abs(waveform-expected_waveforms[name]))))
        metadata = json.loads((root / (name + '.measurement.json')).read_text())
        require(metadata['pool_count'] == 4 and metadata['discarded_tail_samples'] == 0, 'pool accounting')
        require(metadata['frames_overlap_and_are_not_asserted_independent'] and not metadata['classifier_admitted'], 'scope')
        values = []
        with np.load(root / (name + '.arrays.npz'), allow_pickle=False) as arrays:
            close(arrays['frequency_bins'], grid, 'grid', 0, 0)
            close(arrays['window'], window, 'window', 0, 0)
            for pool in range(4):
                starts = range(pool*64000, (pool+1)*64000-1024+1, 256)
                frames = [waveform[i:i+1024] for i in starts]
                calculated = np.array([np.fft.rfft((x-np.mean(x))*window)/sum(window) for x in frames])
                spectra = arrays['spectra'][pool]
                close(spectra, calculated, 'independent waveform STFT')
                maxima['stft_absolute_error'] = max(maxima['stft_absolute_error'], float(np.max(abs(spectra-calculated))))
                close(arrays['frame_start_samples'][pool], list(starts), 'frame positions', 0, 0)
                close(arrays['frame_means'][pool], [np.mean(x) for x in frames], 'frame means')
                energies = np.sum(abs(spectra)**2, axis=0)
                total = sum(energies[1:])
                fraction = energies / total if total else np.full(513, np.nan)
                close(arrays['bin_coefficient_energy'][pool], energies, 'energies')
                close(arrays['bin_energy_fraction'][pool], fraction, 'fractions')
                pool_meta = metadata['pools'][pool]
                require(len(pool_meta['cells']) == 228, 'all cells retained')
                eligible_count = 0
                for index, triad in enumerate(grid):
                    cell = pool_meta['cells'][index]
                    require(cell['frequency_bins'] == triad.tolist(), 'cell identity')
                    b2, phase, raw = direct_cell(spectra, triad)
                    eligible = bool(np.all(fraction[triad] >= 1e-6) and b2 is not None)
                    require(cell['eligible'] == eligible == bool(arrays['eligible_mask'][pool,index]), 'eligibility')
                    eligible_count += eligible
                    recorded = cell['primitive']['squared_bicoherence']
                    if b2 is None:
                        require(recorded is None, 'undefined raw b2')
                    else:
                        close(recorded, b2, 'raw b2', atol=2e-12)
                        maxima['b2_absolute_error'] = max(maxima['b2_absolute_error'], abs(recorded-b2))
                        for key, expected in raw.items():
                            encoded = cell['primitive']['raw_sums'][key]
                            value = math.ldexp(encoded['mantissa'], encoded['exponent2'])
                            close(value, expected, 'raw sufficient sum ' + key, atol=1e-60, rtol=2e-11)
                        if phase is None:
                            require(cell['primitive']['biphase_radians'] is None, 'undefined phase')
                        else:
                            phase_error = math.atan2(math.sin(cell['primitive']['biphase_radians']-phase), math.cos(cell['primitive']['biphase_radians']-phase))
                            close(phase_error, 0., 'descriptive phase', atol=2e-11)
                    if eligible:
                        close(cell['squared_bicoherence'], b2, 'eligible score', atol=2e-12)
                        close(arrays['squared_bicoherence'][pool,index], cell['squared_bicoherence'], 'array score', 0, 0)
                    else:
                        require(cell['squared_bicoherence'] is None and np.isnan(arrays['squared_bicoherence'][pool,index]), 'missing, not zero')
                    total_cells += 1
                require(pool_meta['eligible_cell_count'] == eligible_count, 'eligible count')
                target = pool_meta['cells'][target_index]
                require(row['target_pools'][pool]['b2'] == target['squared_bicoherence'], 'target summary')
                if target['eligible']:
                    values.append(target['squared_bicoherence'])
        require(row['eligible_target_pools'] == len(values), 'target coverage')
        if values:
            close(row['target_mean_b2'], np.mean(values), 'summary mean', 0, 0)
        else:
            require(row['target_mean_b2'] is None, 'summary missing')
    require(sha(marker) == args.commit_sha256, 'commit changed')
    receipt = {'passed': True, 'scope': 'independent_synthetic_waveform_stft_and_stored_sum_replay',
        'conditions': 12, 'pools': 48, 'cells_checked': total_cells,
        'stft_complex_coefficients_checked': 48*247*513, 'maximum_errors': maxima,
        'commit_sha256': args.commit_sha256, 'audit_code_sha256': sha(Path(__file__)),
        'producer_imported': False, 'python': sys.version, 'numpy': np.__version__,
        'external_gate_passed': False, 'classifier_fits': 0, 'null_calibrated': False}
    with output.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
