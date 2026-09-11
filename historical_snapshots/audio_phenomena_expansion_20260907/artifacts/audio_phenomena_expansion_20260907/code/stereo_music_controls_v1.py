#!/usr/bin/env python3
"""Frozen, non-classifying real-music stereo arithmetic and codec controls."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.io import wavfile
from scipy.signal import correlate, correlation_lags
from stereo_candidate_v1 import BANDS_HZ, CONFIG, extract_stereo_candidate

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
MANIFEST = WORKSPACE / 'artifacts/musdb18_oracle_vocal_eval_20260901/manifest.jsonl'
SNAPSHOT = ROOT / 'sources/stereo_external_v1/musdb_tracklist.csv'
PROTOCOL = ROOT / 'preregistration/stereo_music_controls_v1.md'
FREEZE = ROOT / 'preregistration/stereo_music_controls_frozen_v1.json'
OUTPUT = ROOT / 'results/stereo_music_controls_v1'
FFMPEG = '/opt/homebrew/bin/ffmpeg'
FFPROBE = '/opt/homebrew/bin/ffprobe'
TOL = 1e-8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    with Path(path).open('x') as out:
        json.dump(clean(value), out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')


def rank(title):
    return hashlib.sha256(('stereo_controls_20260907:' + title).encode()).hexdigest()


def probe(path):
    result = json.loads(subprocess.check_output([FFPROBE, '-v', 'error',
        '-select_streams', 'a:0', '-show_entries', 'stream=sample_rate,channels,duration',
        '-of', 'json', str(path)], text=True))['streams'][0]
    if int(result['sample_rate']) != 44100 or result['channels'] != 2:
        raise ValueError(f'actual audio header is not stereo 44100 Hz: {path}')
    return result


def prepare():
    licenses = {r['Track Name']: r for r in csv.DictReader(io.StringIO(SNAPSHOT.read_text()))}
    originals = [json.loads(s) for s in MANIFEST.read_text().splitlines() if s]
    inventory = []
    for row in originals:
        credit = licenses.get(row['track_name'], {})
        src = credit.get('Source')
        allowed = (src == 'MedleyDB' and credit.get('License') == 'CC BY-NC-SA') or (
            src == "C't Remix Comp." and credit.get('License') == 'CC BY-NC-SA 3.0')
        artist = row['track_name'].split(' - ')[0]
        if artist.startswith('The Easton Ellises'):
            artist = 'The Easton Ellises'
        inventory.append(dict(row, artist_proxy=artist, selection_hash=rank(row['track_name']),
            license_source=src, license=credit.get('License'),
            license_url=('https://creativecommons.org/licenses/by-nc-sa/' +
                ('4.0/' if src == 'MedleyDB' else '3.0/')) if allowed else None,
            selection_reason='eligible' if allowed else 'excluded_license'))
    representatives = {}
    for row in sorted(inventory, key=lambda r: r['selection_hash']):
        if row['selection_reason'] != 'eligible':
            continue
        if row['artist_proxy'] in representatives:
            row['selection_reason'] = 'excluded_artist_cap'
        else:
            representatives[row['artist_proxy']] = row
    selected = list(representatives.values())[:24]
    for i, row in enumerate(representatives.values()):
        row['selection_reason'] = 'selected' if i < 24 else 'excluded_count_cap'
        if i < 24:
            assert row['channels'] == 2 and row['sample_rate'] == 44100
            row['input_sha256'] = sha(row['mixture'])
            row['actual_header'] = probe(row['mixture'])
    bindings = [Path(__file__), Path(__file__).with_name('stereo_candidate_v1.py'),
                PROTOCOL, MANIFEST, SNAPSHOT]
    freeze = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='reused external measurement controls; no independent authorship test',
        bindings={str(p): sha(p) for p in bindings}, config=CONFIG,
        selection_counts=dict(Counter(r['selection_reason'] for r in inventory)),
        selected=selected, inventory=inventory,
        ffmpeg_version=subprocess.check_output([FFMPEG, '-version'], text=True))
    assert selected and len({r['track_id'] for r in selected}) == len(selected)
    write_json(FREEZE, freeze)
    print(json.dumps(dict(freeze=str(FREEZE), sha256=sha(FREEZE),
                         selection_counts=freeze['selection_counts'])), flush=True)


def decode(path):
    probe(path)
    cmd = [FFMPEG, '-v', 'error', '-i', str(path), '-map', '0:a:0',
           '-f', 'f64le', '-acodec', 'pcm_f64le', '-']
    data = subprocess.check_output(cmd)
    return np.frombuffer(data, dtype='<f8').reshape(-1, 2).copy()


def extract(y):
    return extract_stereo_candidate(y, 44100, native_channels=2)


def rms_match(y, original):
    a, b = float(np.mean(original**2)), float(np.mean(y**2))
    if b <= 0:
        raise ValueError('silent transformed input')
    return y * np.sqrt(a / b)


def scalar_pair(a, b):
    rows = {}
    for key, av in a['features'].items():
        bv = b['features'][key]
        valid = np.isfinite(av) and np.isfinite(bv)
        rows[key] = dict(original=av, transformed=bv,
            absolute_delta=abs(bv-av) if valid else None,
            missingness_changed=bool(np.isfinite(av) != np.isfinite(bv)))
    return rows


def frame_pair(a, b):
    rows = []
    for bi, band in enumerate(BANDS_HZ):
        for metric, values in a['per_frame'].items():
            original, transformed = values[bi], b['per_frame'][metric][bi]
            mask = np.isfinite(original) & np.isfinite(transformed)
            delta = transformed[mask]-original[mask]
            if metric == 'ipd_increment_rad':
                delta = (delta+np.pi) % (2*np.pi)-np.pi
            delta = np.abs(delta)
            rows.append(dict(band=list(band), primary=bi<4, metric=metric,
                common_frames=int(mask.sum()), common_fraction=float(mask.mean()) if len(mask) else 0,
                original_valid_frames=int(np.isfinite(original).sum()),
                transformed_valid_frames=int(np.isfinite(transformed).sum()),
                frame_mask_changes=int(np.count_nonzero(np.isfinite(original)!=np.isfinite(transformed))),
                median_absolute_delta=float(np.median(delta)) if len(delta) else None,
                max_absolute_delta=float(delta.max()) if len(delta) else None))
    return rows


def frame_error(actual, expected):
    mask = np.isfinite(actual) & np.isfinite(expected)
    delta = np.abs(actual[mask] - expected[mask])
    n = int(mask.sum())
    coverage = n / len(mask) if len(mask) else 0
    mx = float(delta.max()) if n else None
    return dict(common_frames=n, common_fraction=coverage,
        median_absolute_error=float(np.median(delta)) if n else None,
        max_absolute_error=mx,
        coverage_pass=bool(n >= 30 and coverage >= .5),
        arithmetic_pass=bool(mx is not None and mx <= TOL),
        passed=bool(n >= 30 and coverage >= .5 and mx is not None and mx <= TOL))


def align(original, decoded):
    a = original-original.mean(axis=0)
    b = decoded-decoded.mean(axis=0)
    norm = np.sqrt(float(np.sum(a*a)))*np.sqrt(float(np.sum(b*b)))
    if norm <= 1e-12:
        raise ValueError('codec alignment ineligible: insufficient centered stereo energy')
    cc = sum(correlate(b[:, c], a[:, c], method='fft') for c in range(2))/norm
    lags = correlation_lags(len(b), len(a))
    keep = np.abs(lags) <= 4096
    cc, lags = cc[keep], lags[keep]
    peak = int(np.argmax(cc))
    lag = int(lags[peak])
    distant = np.abs(lags-lag) > 32
    second = float(np.max(cc[distant])) if np.any(distant) else -1.
    if cc[peak] < .1 or cc[peak]-second <= 1e-6:
        raise ValueError('codec alignment ineligible: weak or ambiguous stereo correlation peak')
    start_a, start_b = max(0, -lag), max(0, lag)
    count = min(len(a)-start_a, len(b)-start_b)
    if count < .9 * len(original):
        raise ValueError('codec alignment overlap below frozen minimum')
    return original[start_a:start_a+count], decoded[start_b:start_b+count], dict(
        lag_samples=lag, original_start=start_a, decoded_start=start_b,
        overlap_samples=count, original_samples=len(original), decoded_samples=len(decoded),
        normalized_peak=float(cc[peak]), runner_up_outside_32_samples=second,
        method='sum_of_centered_channel_correlations', eligible=True)


def persist_extraction(directory, name, y, result):
    wavfile.write(directory / (name + '.wav'), 44100, np.asarray(y, dtype=np.float64))
    np.savez_compressed(directory / (name + '_frames.npz'),
        **result['per_frame'], **result['valid_masks'])
    write_json(directory / (name + '_features.json'),
        {k: v for k, v in result.items() if k not in {'per_frame', 'valid_masks'}})


def run():
    freeze = json.loads(FREEZE.read_text())
    for path, digest in freeze['bindings'].items():
        assert sha(path) == digest, ('binding changed', path)
    for row in freeze['selected']:
        assert sha(row['mixture']) == row['input_sha256']
    assert subprocess.check_output([FFMPEG, '-version'], text=True) == freeze['ffmpeg_version']
    OUTPUT.mkdir(exist_ok=False)
    records = []
    for index, row in enumerate(freeze['selected']):
        directory = OUTPUT / row['track_id']
        directory.mkdir()
        write_json(directory / 'ATTRIBUTION.json', dict(row,
            modifications='gain, channel swap, M/S width, or codec round trip; see variant name',
            derived_audio_license=row['license_url'], dataset_doi='10.5281/zenodo.1117372'))
        y = decode(row['mixture'])
        assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all()
        baseline = extract(y)
        persist_extraction(directory, 'original', y, baseline)
        variants = {'gain_half': y*.5, 'swap': y[:, ::-1]}
        for db in [3, 6, 12]:
            modified = y.copy()
            modified[:, 0] *= 10**(db/20)
            variants[f'iid_plus_{db}db'] = rms_match(modified, y)
        mid, side = .5*(y[:, 0]+y[:, 1]), .5*(y[:, 0]-y[:, 1])
        for width in [.5, 2.]:
            variants[f'width_{width:g}'] = rms_match(np.column_stack(
                [mid+width*side, mid-width*side]), y)
        track = dict(track_id=row['track_id'], title=row['track_name'],
            artist_proxy=row['artist_proxy'], license_source=row['license_source'],
            sample_count=len(y), comparisons={})
        for name, modified in variants.items():
            result = extract(modified)
            persist_extraction(directory, name, modified, result)
            comparison = dict(scalars=scalar_pair(baseline, result),
                              frame_differences=frame_pair(baseline, result), band_checks=[])
            if name in {'gain_half', 'swap'}:
                for bi, (low, high) in enumerate(BANDS_HZ):
                    entries = {k: v for k, v in comparison['scalars'].items()
                               if k.startswith(f'SC_{low}_{high}hz_')}
                    finite = [v['absolute_delta'] for v in entries.values()
                              if v['absolute_delta'] is not None]
                    changes = sum(v['missingness_changed'] for v in entries.values())
                    comparison['band_checks'].append(dict(band=[low, high],
                        primary=bi<4, common_finite_features=len(finite),
                        missingness_changes=changes, max_absolute_error=max(finite) if finite else None,
                        passed=bool(finite and max(finite)<=TOL and changes==0)))
            elif name.startswith('iid') or name.startswith('width'):
                for bi, band in enumerate(BANDS_HZ):
                    if name.startswith('iid'):
                        db = float(name.split('_')[-1][:-2])
                        base = baseline['per_frame']['iid_db'][bi]
                        expected = base+db
                        expected[(abs(base)>=60) | (abs(expected)>=60)] = np.nan
                        actual = result['per_frame']['iid_db'][bi]
                    else:
                        width = float(name.split('_')[-1])
                        q = baseline['per_frame']['side_energy_fraction'][bi]
                        expected = width**2*q / (1-q+width**2*q)
                        actual = result['per_frame']['side_energy_fraction'][bi]
                    comparison['band_checks'].append(dict(band=list(band), primary=bi<4,
                                                         **frame_error(actual, expected)))
            track['comparisons'][name] = comparison
        for encoder, extension in [('libmp3lame', 'mp3'), ('aac', 'm4a')]:
            for bitrate in [128, 256]:
                name = f'{extension}_{bitrate}k'
                encoded = directory / f'{name}.{extension}'
                cmd = [FFMPEG, '-v', 'error', '-n', '-i', str(directory/'original.wav'),
                       '-c:a', encoder, '-b:a', f'{bitrate}k', '-ar', '44100', '-ac', '2', str(encoded)]
                subprocess.run(cmd, check=True, capture_output=True)
                decoded = decode(encoded)
                wavfile.write(directory/(name+'_full_decode.wav'), 44100, decoded)
                try:
                    a, b, alignment = align(y, decoded)
                except ValueError as exc:
                    track['comparisons'][name] = dict(command=cmd,
                        alignment=dict(eligible=False, reason=str(exc)),
                        scalars={}, frame_differences=[], target_response_ratios={}, band_checks=[])
                    continue
                ar, br = extract(a), extract(b)
                persist_extraction(directory, name+'_reference', a, ar)
                persist_extraction(directory, name, b, br)
                scalars = scalar_pair(ar, br)
                # Compute intervention denominators on the IDENTICAL aligned reference crop.
                target_iid = a.copy()
                target_iid[:, 0] *= 10**(.3)
                target_iid = rms_match(target_iid, a)
                crop_mid = a.mean(axis=1)
                crop_side = .5*(a[:, 0]-a[:, 1])
                target_width = rms_match(np.column_stack(
                    [crop_mid+2*crop_side, crop_mid-2*crop_side]), a)
                iid_result, width_result = extract(target_iid), extract(target_width)
                write_json(directory/(name+'_matched_target_features.json'), dict(
                    iid_plus_6db=iid_result['features'], width_2=width_result['features'],
                    reconstruction='Use saved matched reference WAV and frozen intervention code.'))
                target_pairs = {'iid': scalar_pair(ar, iid_result),
                                'width': scalar_pair(ar, width_result)}
                ratios = {}
                for key, value in scalars.items():
                    target = ('iid' if key.endswith('abs_iid_db_median') else
                              'width' if key.endswith('side_energy_fraction_median') else None)
                    if target is None:
                        continue
                    denominator = target_pairs[target][key]['absolute_delta']
                    numerator = value['absolute_delta']
                    available = numerator is not None and denominator is not None and denominator > TOL
                    ratios[key] = dict(codec_delta=numerator, target_delta=denominator,
                        ratio=numerator/denominator if available else None,
                        target='iid_plus_6db' if target == 'iid' else 'width_2')
                track['comparisons'][name] = dict(command=cmd, alignment=alignment,
                    scalars=scalars, frame_differences=frame_pair(ar, br),
                    target_response_ratios=ratios, band_checks=[])
        write_json(directory/'comparisons.json', track)
        records.append(track)
        print(f'{index+1}/{len(freeze["selected"])} {row["track_name"]}', flush=True)
    arithmetic = []
    for track in records:
        for name, comp in track['comparisons'].items():
            for band in comp['band_checks']:
                if band['primary']:
                    arithmetic.append(dict(track_id=track['track_id'], variant=name, **band))
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        freeze_sha256=sha(FREEZE), tracks=len(records),
        artist_proxies=len({r['artist_proxy'] for r in records}),
        source_counts=dict(Counter(r['license_source'] for r in records)),
        primary_checks=len(arithmetic), passed_checks=sum(r['passed'] for r in arithmetic),
        failed_checks=[r for r in arithmetic if not r['passed']],
        measurement_arithmetic_pass=all(r['passed'] for r in arithmetic),
        classifier_admitted=False, runtime=dict(python=sys.version, numpy=np.__version__,
            scipy=scipy.__version__, platform=platform.platform()),
        scope='reused CC-licensed music controls, not independent attribution evidence')
    write_json(OUTPUT/'summary.json', summary)
    files = {str(p.relative_to(OUTPUT)): dict(sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(OUTPUT.rglob('*')) if p.is_file()}
    write_json(OUTPUT/'COMMIT.json', dict(freeze_sha256=sha(FREEZE), products=files))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else run()
