#!/usr/bin/env python3
"""Separate prospective SC_L6 gate, reusing the immutable v1 control runner."""
import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import stereo_music_controls_v1 as V1

ROOT = Path(__file__).resolve().parents[1]
OLD_FREEZE = ROOT/'preregistration/stereo_music_controls_frozen_v1.json'
FREEZE = ROOT/'preregistration/stereo_lowband_gate_frozen_v2.json'
PROTOCOL = ROOT/'preregistration/stereo_lowband_gate_v2.md'
OUTPUT = ROOT/'results/stereo_lowband_controls_v2'
GATE = ROOT/'results/stereo_lowband_gate_v2'
KEYS = [f'SC_{lo}_{hi}hz_{metric}_median'
        for lo,hi in [(80,500),(500,2000),(2000,6000)]
        for metric in ['abs_iid_db','side_energy_fraction']]
CODECS = ['mp3_128k','mp3_256k','m4a_128k','m4a_256k']
OLD_FREEZE_SHA = '2a09c65a3c6a9400771b47bfc30b1e59e60e69dd3910f9c59f313ba568e2721f'
RUNNER_SHA = '4376ffebd8256e38376a794a296f9d550c0ee6cc166525b742e3a7cd5819c75e'


def prepare():
    assert V1.sha(OLD_FREEZE)==OLD_FREEZE_SHA and V1.sha(V1.__file__)==RUNNER_SHA
    old = json.loads(OLD_FREEZE.read_text())
    selected = [dict(r) for r in old['inventory'] if r['selection_reason']=='excluded_count_cap']
    assert len(selected)==7
    assert not {r['track_id'] for r in selected}&{r['track_id'] for r in old['selected']}
    assert not {r['artist_proxy'] for r in selected}&{r['artist_proxy'] for r in old['selected']}
    for row in selected:
        assert row['license_source']=='MedleyDB' and row['license_url'].endswith('/4.0/')
        row['selection_reason']='all_seven_remaining_artist_representatives'
        row['input_sha256']=V1.sha(row['mixture'])
        row['actual_header']=V1.probe(row['mixture'])
    paths = [Path(__file__),Path(V1.__file__),Path(V1.__file__).with_name('stereo_candidate_v1.py'),
             OLD_FREEZE,PROTOCOL,V1.SNAPSHOT,V1.MANIFEST]
    freeze = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        selected=selected, bindings={str(p):V1.sha(p) for p in paths}, config=V1.CONFIG,
        ffmpeg_version=subprocess.check_output([V1.FFMPEG,'-version'],text=True),
        candidate='SC_L6', candidate_features=KEYS,
        thresholds=dict(primary_checks=196,codec_pairs=28,ratios=168,
            minimum_fraction_at_most_one_tenth=.9,maximum_any_ratio=1),
        prior_controls_freeze_sha256=OLD_FREEZE_SHA,
        scope='new stereo controls within reused source corpus; no classification test')
    V1.write_json(FREEZE,freeze)
    print(json.dumps(dict(freeze=str(FREEZE),sha256=V1.sha(FREEZE),tracks=len(selected))))


def run():
    V1.FREEZE, V1.OUTPUT = FREEZE, OUTPUT
    V1.run()


def assess():
    frozen=json.loads(FREEZE.read_text())
    for path,digest in frozen['bindings'].items():
        assert V1.sha(path)==digest
    commit=json.loads((OUTPUT/'COMMIT.json').read_text())
    assert commit['freeze_sha256']==V1.sha(FREEZE)
    for name,record in commit['products'].items():
        path=OUTPUT/name
        assert path.stat().st_size==record['bytes'] and V1.sha(path)==record['sha256']
    GATE.mkdir(exist_ok=False)
    original=json.loads((OUTPUT/'summary.json').read_text())
    rows=[json.loads((OUTPUT/r['track_id']/'comparisons.json').read_text()) for r in frozen['selected']]
    target_bindings=[]
    feature_rows=[]
    alignments=[r['comparisons'][c]['alignment'] for r in rows for c in CODECS]
    for row in rows:
        directory=OUTPUT/row['track_id']
        for codec in CODECS:
            if not row['comparisons'][codec]['alignment']['eligible']:
                continue
            sr,reference=wavfile.read(directory/(codec+'_reference.wav'))
            assert sr==44100
            iid=reference.copy(); iid[:,0]*=10**.3
            mid=reference.mean(axis=1); side=.5*(reference[:,0]-reference[:,1])
            width=np.column_stack([mid+2*side,mid-2*side])
            for name,target in [('iid_plus_6db',iid),('width_2',width)]:
                target=V1.rms_match(target,reference)
                existing=directory/(name+'.wav')
                old_sr,old_wave=wavfile.read(existing)
                identical=old_sr==44100 and np.array_equal(old_wave,target)
                resolved=existing if identical else GATE/(row['track_id']+'_'+codec+'_'+name+'.wav')
                if not identical:
                    wavfile.write(resolved,44100,target)
                target_bindings.append(dict(track_id=row['track_id'],codec=codec,target=name,
                    resolved_wav=str(resolved),sha256=V1.sha(resolved),reused_identical=bool(identical)))
    for key in KEYS:
        pairs=[]
        for row in rows:
            for codec in CODECS:
                ratio=row['comparisons'][codec]['target_response_ratios'].get(key,{})
                value=ratio.get('ratio')
                pairs.append(dict(track_id=row['track_id'],codec=codec,**ratio,
                    available=value is not None,at_most_one_tenth=value is not None and value<=.1))
        available=[p['ratio'] for p in pairs if p['available']]
        n_small=sum(p['at_most_one_tenth'] for p in pairs)
        passed=(len(available)==28 and n_small/28>=.9 and max(available)<=1)
        feature_rows.append(dict(feature=key,available=len(available),expected=28,
            count_at_most_one_tenth=n_small,fraction_at_most_one_tenth=n_small/28,
            maximum_ratio=max(available) if available else None,
            median_ratio=float(np.median(available)) if available else None,passed=passed,pairs=pairs))
    passed=(original['primary_checks']==196 and original['passed_checks']==196 and
        len(alignments)==28 and all(a['eligible'] for a in alignments) and
        all(r['passed'] for r in feature_rows))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),candidate='SC_L6',
        candidate_features=KEYS,measurement_gate_passed=passed,classifier_fitted=False,
        freeze_sha256=V1.sha(FREEZE),raw_commit_sha256=V1.sha(OUTPUT/'COMMIT.json'),
        primary_checks=original['primary_checks'],primary_passed=original['passed_checks'],
        codec_pairs=len(alignments),eligible_codec_pairs=sum(a['eligible'] for a in alignments),
        feature_checks=feature_rows,target_waveforms=len(target_bindings),
        scope='prospective engineering margin on seven new stereo controls; reused MedleyDB source',
        decision='eligible_for_separately_frozen_exploratory_study' if passed else
                 'gate_failed_do_not_relax_or_admit_as_passed')
    V1.write_json(GATE/'gate_result.json',result)
    V1.write_json(GATE/'target_waveform_bindings.json',target_bindings)
    lines=['# Prospective SC_L6 stereo measurement gate','',
        f'Gate outcome: **{"PASS" if passed else "FAIL"}**. No classifier was fitted.','',
        'Seven new stereo-control tracks/artists from the existing CC-licensed MUSDB inventory. '
        'These are not a source-independent AI/Human test. The six-descriptor candidate '
        'and thresholds were fixed after the first24-track exploration but before this run.','',
        '| Descriptor | Ratios available | Ratio <=0.10 | Maximum ratio | Gate |',
        '|---|---:|---:|---:|---|']
    for r in feature_rows:
        maximum=f'{r["maximum_ratio"]:.6f}' if r['maximum_ratio'] is not None else 'NA'
        lines.append(f'| {r["feature"]} | {r["available"]}/28 | '
            f'{r["count_at_most_one_tenth"]}/28 | {maximum} | {"PASS" if r["passed"] else "FAIL"} |')
    lines += ['',f'Arithmetic: {original["passed_checks"]}/{original["primary_checks"]} primary checks; '
        f'alignment: {sum(a["eligible"] for a in alignments)}/{len(alignments)} eligible.', '',
        'A pass permits only a separately frozen exploratory matched-cohort study. '
        'A failure must not be repaired by dropping tracks/bands or changing thresholds on this set. '
        'Ratios are within-recording codec changes relative to the declared target response; '
        'small target responses can amplify ratios. Neither outcome establishes authorship accuracy.', '',
        'Code: `code/stereo_lowband_gate_v2.py`; protocol and exact roster: '
        '`preregistration/stereo_lowband_gate_v2.md` and `stereo_lowband_gate_frozen_v2.json`. '
        'Raw retained outputs: `results/stereo_lowband_controls_v2/COMMIT.json`. '
        'Exact metrics: `gate_result.json`; matched target WAVs: `target_waveform_bindings.json`.', '',
        'Attribution: Rafii et al., MUSDB18, DOI10.5281/zenodo.1117372; '
        'per-track titles/artists and CC BY-NC-SA4.0 licenses remain in raw ATTRIBUTION.json files.','']
    with (GATE/'REPORT_EN.md').open('x') as f:
        f.write('\n'.join(lines))
    V1.write_json(GATE/'COMMIT.json',dict(freeze_sha256=V1.sha(FREEZE),
        raw_commit_sha256=V1.sha(OUTPUT/'COMMIT.json'),products={p.name:dict(
            sha256=V1.sha(p),bytes=p.stat().st_size) for p in GATE.iterdir() if p.is_file()}))
    print(json.dumps({k:v for k,v in result.items() if k not in {'feature_checks','candidate_features'}}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','assess'])
    action=parser.parse_args().action
    {'prepare':prepare,'run':run,'assess':assess}[action]()
