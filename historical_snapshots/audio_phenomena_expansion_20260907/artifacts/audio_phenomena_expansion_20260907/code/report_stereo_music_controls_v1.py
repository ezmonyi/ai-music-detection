#!/usr/bin/env python3
"""Descriptive report only. Never fits or admits a classifier; no frozen edits."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'results/stereo_music_controls_v1'
OUT = ROOT/'results/stereo_music_controls_report_v1'
CODECS = ['mp3_128k', 'mp3_256k', 'm4a_128k', 'm4a_256k']
BANDS = [(80, 500), (500, 2000), (2000, 6000), (6000, 12000), (12000, 20000)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def main():
    summary = json.loads((RAW/'summary.json').read_text())
    commit = json.loads((RAW/'COMMIT.json').read_text())
    for relative, record in commit['products'].items():
        path = RAW/relative
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
    OUT.mkdir(exist_ok=False)
    tracks = [json.loads(p.read_text()) for p in sorted(RAW.glob('*/comparisons.json'))]
    alignment = []
    target_bindings = []
    for track in tracks:
        directory = RAW/track['track_id']
        for codec in CODECS:
            comparison = track['comparisons'][codec]
            alignment.append(dict(track_id=track['track_id'], codec=codec, **comparison['alignment']))
            if not comparison['alignment']['eligible']:
                continue
            sr, reference = wavfile.read(directory/(codec+'_reference.wav'))
            assert sr == 44100
            iid = reference.copy()
            iid[:, 0] *= 10**(.3)
            mid = reference.mean(axis=1)
            side = .5*(reference[:, 0]-reference[:, 1])
            width = np.column_stack([mid+2*side, mid-2*side])
            for name, target in [('iid_plus_6db', iid), ('width_2', width)]:
                target *= np.sqrt(np.mean(reference**2)/np.mean(target**2))
                existing = directory/(name+'.wav')
                old_sr, old = wavfile.read(existing)
                identical = old_sr == 44100 and np.array_equal(old, target)
                if identical:
                    resolved = existing
                else:
                    resolved = OUT/f'{track["track_id"]}_{codec}_{name}.wav'
                    wavfile.write(resolved, 44100, target)
                target_bindings.append(dict(track_id=track['track_id'], codec=codec,
                    target=name, reference=str(directory/(codec+'_reference.wav')),
                    resolved_wav=str(resolved), sha256=sha(resolved),
                    reused_identical_persisted_waveform=bool(identical)))
    table = []
    for codec in CODECS:
        for low, high in BANDS:
            for metric in ['abs_iid_db', 'side_energy_fraction']:
                key = f'SC_{low}_{high}hz_{metric}_median'
                entries = [t['comparisons'][codec]['target_response_ratios'].get(key) for t in tracks]
                entries = [e for e in entries if e is not None]
                delta = [e['codec_delta'] for e in entries if e['codec_delta'] is not None]
                ratios = [e['ratio'] for e in entries if e['ratio'] is not None]
                table.append(dict(codec=codec, band_low=low, band_high=high,
                    primary=high<=12000, metric=metric,
                    valid_delta_recordings=len(delta), valid_ratio_recordings=len(ratios),
                    codec_delta_median=float(np.median(delta)) if delta else None,
                    codec_delta_q25=float(np.percentile(delta, 25)) if delta else None,
                    codec_delta_q75=float(np.percentile(delta, 75)) if delta else None,
                    ratio_median=float(np.median(ratios)) if ratios else None,
                    ratio_above_one_recordings=sum(r>1 for r in ratios)))
    target_checks = [b for t in tracks for name,c in t['comparisons'].items()
                     for b in c['band_checks'] if b['primary'] and
                     (name.startswith('iid') or name.startswith('width'))]
    level = [b['max_absolute_error'] for t in tracks for n,c in t['comparisons'].items()
             if n.startswith('iid') for b in c['band_checks'] if b['primary']]
    width = [b['max_absolute_error'] for t in tracks for n,c in t['comparisons'].items()
             if n.startswith('width') for b in c['band_checks'] if b['primary']]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        raw_commit_sha256=sha(RAW/'COMMIT.json'), source_summary=summary,
        alignment_eligible=sum(a['eligible'] for a in alignment), alignment_total=len(alignment),
        alignment_rejections=[a for a in alignment if not a['eligible']],
        target_waveforms=len(target_bindings),
        identical_existing_targets=sum(b['reused_identical_persisted_waveform'] for b in target_bindings),
        maximum_primary_iid_target_error_db=max(level),
        maximum_primary_width_target_error=max(width),
        minimum_primary_target_common_fraction=min(b['common_fraction'] for b in target_checks),
        minimum_primary_target_common_frames=min(b['common_frames'] for b in target_checks),
        codec_table=table, alignment=alignment,
        interpretation='Mechanism controls passed; codec sensitivity observed. No classifier admission.')
    save(OUT/'report_data.json', report)
    save(OUT/'matched_target_waveform_bindings.json', target_bindings)
    header = [
        '# Stereo relationship controls on real music', '',
        '## Outcome', '',
        f'The frozen run completed {summary["primary_checks"]} primary numerical checks '
        f'on {summary["tracks"]} recordings, with {summary["passed_checks"]} passing. '
        f'All {len(alignment)} codec pairs were alignment-eligible; '
        f'{len(report["alignment_rejections"])} were rejected. '
        'These are measurement checks, not AI/Human classification accuracy.', '',
        'The substantive limitation is codec sensitivity: AAC at128 kbps changes '
        'high-band side-energy fraction substantially. Consequently, arithmetic '
        'success does not justify admitting all50 stereo descriptors as attribution evidence.', '',
        '## Data and controlled processing', '',
        '24 title-derived artist groups;23 MedleyDB and1 Easton Ellises recording. '
        'These are existing approximately6.8-second MUSDB previews, not24 newly '
        'collected full songs. The144-entry inventory yielded42 CC-licensed candidates; '
        '102 license-restricted/unknown entries were excluded,11 by the one-artist cap, '
        'and7 by the count cap. Selection was hash-based and frozen before extraction. '
        'The control library has been used before; MedleyDB overlap with the detector '
        'cohort is possible. It supplies paired mechanism checks, not an independent '
        'authorship generalization test.', '',
        'Original audio headers were verified as stereo44.1 kHz. All processing was '
        'CPU-only: periodic Hann4096, hop1024, unpadded STFT. Original files were '
        'unchanged. Conditions: original; gainx0.5; channel swap; left-channel '
        '+3/+6/+12 dB with common RMS matching; M/S widthx0.5/x2 with common RMS '
        'matching; MP3 and AAC round trips at128/256 kbps. Stereo-summed correlation '
        'avoids antiphase mono cancellation; all observed lags were0 samples. '
        'Both codec sides use identical overlapping crops. AAC preview recompression '
        'is not conversion from a lossless master.', '',
        '## Arithmetic validation', '',
        'Primary bands were80-500,500-2000,2000-6000 and6000-12000 Hz. The12000-20000 Hz '
        'band was exploratory because the original codec has limited high bandwidth.', '',
        f'- Maximum primary signed-IID target error: {max(level):.6g} dB.',
        f'- Maximum primary side-fraction target error: {max(width):.6g}.',
        f'- Minimum primary target coverage: {report["minimum_primary_target_common_fraction"]:.2%} '
        f'({report["minimum_primary_target_common_frames"]} frames minimum).',
        '- Frozen tolerance:1e-8; at least30 common frames and50% coverage.', '',
        'The target equations were IID\u2032=IID+d and '
        'q\u2032=w²q/(1−q+w²q), with common-valid-frame masking. Gain and swap '
        'checks concern scalar invariance, not signed panning-direction invariance. '
        'Frame errors are not independent observations, and no inferential p-values '
        'or classifier thresholds were fitted.', '',
        '## Codec sensitivity by band', '',
        'Each cell is the equal-recording median absolute change in a per-recording '
        'median descriptor. IID is in dB; side fraction is unitless. '
        'The ratio is computed within recording against +6dB IID or widthx2 on the '
        'identical reference crop, then summarized across recordings. '
        'It is not the ratio of two group medians. A ratio above1 means that this '
        'codec perturbation exceeded that specific intervention\u2019s scalar response, '
        'not that codec distortion exceeds every possible spatial change.', '',
        '| Codec | Band (Hz) | IID change (dB) | Width-fraction change | Width response ratio | Ratio >1 |',
        '|---|---:|---:|---:|---:|---:|']
    texrows = []
    for codec in CODECS:
        for low, high in BANDS:
            entries = {r['metric']:r for r in table if r['codec']==codec and r['band_low']==low}
            a, b = entries['abs_iid_db'], entries['side_energy_fraction']
            values = [codec.replace('_', ' '), f'{low}-{high}'+(' (expl.)' if high>12000 else ''),
                f'{a["codec_delta_median"]:.6f}', f'{b["codec_delta_median"]:.6f}',
                f'{b["ratio_median"]:.4f}', f'{b["ratio_above_one_recordings"]}/{b["valid_ratio_recordings"]}']
            header.append('| '+' | '.join(values)+' |')
            texrows.append(' & '.join(values)+r' \\')
    header += ['', '## Interpretation and next gate', '',
        'Level imbalance and M/S width measurements recover their known interventions. '
        'This does not validate panning-motion complexity or IPD as a physical velocity. '
        'The observed AAC128 high-band width sensitivity is a specific nuisance in '
        'these24 excerpts and this FFmpeg encoder, not a universal AAC claim. '
        'Below6 kHz is a prospective robustness candidate, not a selected detector. '
        'Before extending S/D/R/P/F/H/M combinations, freeze a stereo subfamily and '
        'validate its nuisance robustness on a separately inventoried corpus. '
        'Then compare baseline and augmented models on the identical eligible cohort '
        '(native stereo provenance matters; duplicated mono must not become AI evidence). '
        'The existing seven-family v5 results and all protected evaluation data remain unchanged.', '',
        '## Reproducibility and thesis references', '',
        '- Protocol: `preregistration/stereo_music_controls_v1.md`.',
        '- Freeze: `preregistration/stereo_music_controls_frozen_v1.json`.',
        '- Experiment: `code/stereo_music_controls_v1.py`.',
        '- Extractor: `code/stereo_candidate_v1.py`.',
        '- Tests: `code/test_stereo_music_controls_v1.py` and `code/test_stereo_candidate_v1.py`.',
        '- Outputs: `results/stereo_music_controls_v1/COMMIT.json`, per-track WAVs, codec '
        'files, frame NPZs, scalar JSONs and paired comparisons.',
        '- Analysis: `results/stereo_music_controls_report_v1/report_data.json` and this builder.',
        '- Matched intervention persistence: `matched_target_waveform_bindings.json`; '
        f'all {len(target_bindings)} targets have an exact retained waveform '
        f'({report["identical_existing_targets"]} reuse byte-identical sample arrays).',
        '- Independent audit is a separate receipt; this report does not substitute for it.', '',
        'Attribution and licensing: Rafii et al., The MUSDB18 corpus for music separation '
        '([dataset DOI](https://doi.org/10.5281/zenodo.1117372)); '
        '[SigSep source/license documentation](https://sigsep.github.io/datasets/musdb.html). '
        'Per-track `ATTRIBUTION.json` names each artist/title and retains '
        '[CC BY-NC-SA4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) or '
        '[CC BY-NC-SA3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/). '
        'Derived audio is marked as modified and retains its original license. '
        'No public redistribution or neural training was performed.', '']
    with (OUT/'REPORT_EN.md').open('x') as f:
        f.write('\n'.join(header))
    tex = r'''% Includable English thesis section; requires longtable, hyperref.
\section{Stereo relationship controls and codec sensitivity}
Twenty-four existing CC-licensed MUSDB preview recordings (23 MedleyDB and one
Easton Ellises recording; one title-derived artist group each) were used for
paired mechanism controls. This is a reused control library, not an independent
AI/Human test. No authorship labels or classifier fits were introduced.
The frozen stereo front end uses44.1\,kHz, a4096-sample periodic Hann window and
a1024-sample hop without padding. Known targets are
\(\mathrm{IID}'=\mathrm{IID}+d\) and
\(q'=w^2q/(1-q+w^2q)\), for level difference\(d\) and M/S width\(w\).
All672 primary arithmetic/invariance checks passed. This is not a classification
accuracy. MP3/AAC recompression pairs were aligned by summed per-channel
correlation; all96 passed alignment eligibility with zero-sample estimated lag.
The12--20\,kHz band remains exploratory because of preview bandwidth limitations.

\begin{longtable}{llrrrr}
Codec & Band (Hz) & IID (dB) & Side fraction & Width ratio & Ratio$>1$ \\
\hline
'''+ '\n'.join(texrows)+r'''
\end{longtable}
Values are medians across recordings of absolute changes in their median
descriptors. Ratios compare codec changes against a width-times-two intervention
on each identical aligned reference crop, then take the recording median.
AAC128 substantially perturbs high-band side fraction in these excerpts;
the result is encoder- and corpus-specific. Low-band stereo observables are
candidates for a new prospective robustness gate, not admitted attribution
evidence. The frozen seven-family detector results remain unchanged.

\paragraph{Reproducible artifacts.}
See \path{preregistration/stereo_music_controls_frozen_v1.json},
\path{code/stereo_music_controls_v1.py},
\path{results/stereo_music_controls_v1/COMMIT.json}, and
\path{results/stereo_music_controls_report_v1/report_data.json} under the
\path{artifacts/audio_phenomena_expansion_20260907} project directory.
The matching English report retains intervention definitions, coverage, numerical
errors, code references, per-track attribution, and limitations.
Dataset attribution: Rafii et al., The MUSDB18 corpus for music separation,
\url{https://doi.org/10.5281/zenodo.1117372}. The source/licensing documentation is
\url{https://sigsep.github.io/datasets/musdb.html}; retained audio follows its
per-track CC BY-NC-SA4.0 or3.0 license. No commercial use is authorized here.
'''
    with (OUT/'thesis_section_en.tex').open('x') as f:
        f.write(tex)
    products = {str(p.relative_to(OUT)): dict(sha256=sha(p), bytes=p.stat().st_size)
                for p in sorted(OUT.rglob('*')) if p.is_file()}
    save(OUT/'COMMIT.json', dict(raw_commit_sha256=sha(RAW/'COMMIT.json'),
        builder_sha256=sha(__file__), products=products))
    print(json.dumps({k:v for k,v in report.items() if k not in {'codec_table','alignment','source_summary'}}))


if __name__ == '__main__':
    main()
