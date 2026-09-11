#!/usr/bin/env python3
"""Parent delivery checks: standard-library scalar table replay and bindings."""
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'results/stereo_music_controls_v1'
REPORT = ROOT/'results/stereo_music_controls_report_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    audit_path = ROOT/'audit/stereo_music_controls_independent_audit_v1.json'
    audit = json.loads(audit_path.read_text())
    data = json.loads((REPORT/'report_data.json').read_text())
    assert audit['passed'] and not audit['issues']
    assert audit['commit_sha256']==sha(RAW/'COMMIT.json')==data['raw_commit_sha256']
    assert sha(ROOT/'code/audit_stereo_music_controls_v1.py')==audit['audit_code_sha256']
    report_commit = json.loads((REPORT/'COMMIT.json').read_text())
    assert report_commit['builder_sha256']==sha(ROOT/'code/report_stereo_music_controls_v1.py')
    for rel, record in report_commit['products'].items():
        p = REPORT/rel
        assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256']
    tracks = [json.loads(p.read_text()) for p in RAW.glob('*/comparisons.json')]
    checked = 0
    for row in data['codec_table']:
        key = f'SC_{row["band_low"]}_{row["band_high"]}hz_{row["metric"]}_median'
        pairs = [r['comparisons'][row['codec']]['target_response_ratios'][key] for r in tracks]
        delta = [v['codec_delta'] for v in pairs if v['codec_delta'] is not None]
        ratio = [v['ratio'] for v in pairs if v['ratio'] is not None]
        assert abs(statistics.median(delta)-row['codec_delta_median'])<=1e-12
        assert abs(statistics.median(ratio)-row['ratio_median'])<=1e-12
        assert sum(v>1 for v in ratio)==row['ratio_above_one_recordings']
        assert len(delta)==row['valid_delta_recordings'] and len(ratio)==row['valid_ratio_recordings']
        checked += 1
    targets = json.loads((REPORT/'matched_target_waveform_bindings.json').read_text())
    for target in targets:
        assert sha(target['resolved_wav'])==target['sha256']
    protected = {
        ROOT/'results/equal60_exploratory_v5_results_v1/COMMIT.json':
            '7e1e5d7126169ac1c553ac1f47291ecdeed053bb85852f613b6b20506a5409c6',
        ROOT/'code/stereo_candidate_v1.py':
            '132bd5fe225258b72b913238bd456e3b18279b6b5a664b2191b0292aede9481c',
    }
    for p, expected in protected.items():
        assert sha(p)==expected, str(p)
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        checker_sha256=sha(__file__), independent_audit_sha256=sha(audit_path),
        raw_commit_sha256=sha(RAW/'COMMIT.json'), report_commit_sha256=sha(REPORT/'COMMIT.json'),
        table_rows_independently_replayed=checked, matched_target_wav_hashes_checked=len(targets),
        report_product_hashes_checked=len(report_commit['products']),
        protected_hashes={str(p):value for p,value in protected.items()},
        limitations=['No classifier fitting or eighth-family admission.',
                     'English LaTeX section saved; not merged into or compiled as the main thesis.',
                     'Figure visual QA and remote mirror acceptance are separate receipts.'])
    output = ROOT/'audit/stereo_controls_parent_delivery_v1.json'
    with output.open('x') as f:
        json.dump(receipt,f,indent=2,sort_keys=True)
        f.write('\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
