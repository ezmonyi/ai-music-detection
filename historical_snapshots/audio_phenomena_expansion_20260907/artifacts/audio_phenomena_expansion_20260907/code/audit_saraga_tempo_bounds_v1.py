#!/usr/bin/env python3
"""Join candidate annotation intervals to physical duration; no audio selection."""
import argparse
import hashlib
import json
from pathlib import Path

SUMMARY_SHA = '59dc718293c45c492e945da825d379969836ff13dfb0fb2a894d7570391c73c3'
TEMPO_SHA = '42d9cc52feb3c4fc930199c6a03b10143215e0cf08857fc4cdcc30f2e5a589d2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def join(summary, tempo):
    records = summary['records']
    physical = {r['item']['mbid']: r for r in records}
    if len(physical) != len(records):
        raise ValueError('Duplicate physical identity')
    candidates = []
    for row in tempo['rows']:
        for change in row['changes']:
            item = physical[row['mbid']]
            m = item['measurement']
            meta = item['item']['reconciled_metadata']
            windows = [change['before_window_s'], change['after_window_s']]
            candidates.append(dict(mbid=row['mbid'], windows_s=windows,
                actual_duration_s=m['actual_duration_seconds'],
                within_physical_bounds=all(0 <= a < b <= m['actual_duration_seconds'] for a,b in windows),
                speech_title_flag=meta['speech_title_review_flag'],
                missing_performer_credits=not meta['performer_credits']))
    return dict(status='completed_arithmetic_bounds_not_alignment_or_selected_stimuli',
        candidate_pairs=len(candidates), recordings=len({r['mbid'] for r in candidates}),
        physical_bounds_pass=sum(r['within_physical_bounds'] for r in candidates),
        speech_flag_pairs=sum(r['speech_title_flag'] for r in candidates),
        missing_performer_pairs=sum(r['missing_performer_credits'] for r in candidates),
        rows=candidates, audio_opened=False, alignment_verified=False,
        cohort_selected=False, classifier_admission=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--tempo-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing existing output')
    paths = [a.summary.resolve(), a.tempo_report.resolve(), Path(__file__).resolve()]
    before = {str(x): sha(x) for x in paths}
    if before[str(paths[0])] != SUMMARY_SHA or before[str(paths[1])] != TEMPO_SHA:
        raise ValueError('Wrong frozen physical/annotation reports')
    result = join(json.loads(paths[0].read_text()), json.loads(paths[1].read_text()))
    if before != {str(x): sha(x) for x in paths}:
        raise ValueError('Evidence/code changed')
    result['input_sha256'] = before
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2))


if __name__ == '__main__':
    main()
