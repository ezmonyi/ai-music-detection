#!/usr/bin/env python3
"""Compare two authored tempo inventories; no audio, model or accuracy claims."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

COMMIT = 'bd103295689ad11a5ed19038b7469e91b5c3c124'
CATALOG_SHA = '2ef89a32a62085c6bf6f33e857309a1e33a7cab646548ca73368dcc9641e3a28'
TAR_SHA = '9b5d0cf735bf09c93442c9c08dc4318f9aa998819be35d3a23eac499c47fb3f7'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse(data, width):
    rows, errors = [], []
    for line, raw in enumerate(csv.reader(io.StringIO(data.decode('utf-8-sig'))), 1):
        if not raw or all(not s.strip() for s in raw):
            continue
        try:
            if len(raw) != width:
                raise ValueError('column_count')
            vals = [float(s) if s.strip() not in ('-', '') else None for s in raw]
            if any(v is not None and not math.isfinite(v) for v in vals):
                raise ValueError('nonfinite')
            tempo = None if vals[0] in (None, -1) else vals[0]
            start, end = vals[-2:]
            if start is None or end is None or start < 0 or end <= start:
                raise ValueError('interval')
            if tempo is not None and tempo <= 0:
                raise ValueError('tempo')
            if rows and start < rows[-1]['end_s'] - .001:
                raise ValueError('overlap_or_order')
            if width == 6 and tempo is not None and any(v is None or v <= 0 for v in vals[1:4]):
                raise ValueError('metrical_fields')
            rows.append(dict(line=line, tempo=tempo, start_s=start, end_s=end,
                             metrical_fields=vals[1:4] if width == 6 else None))
        except ValueError as exc:
            errors.append(dict(line=line, raw=raw, error=str(exc)))
    return rows, errors


def compare(bpm_data, tempo_data):
    coarse, ce = parse(bpm_data, 3)
    detailed, de = parse(tempo_data, 6)
    pairs, changes = [], []
    matched = not ce and not de and len(coarse) == len(detailed)
    if matched:
        for a, b in zip(coarse, detailed):
            same = a['tempo'] == b['tempo'] and abs(a['start_s'] - b['start_s']) <= .001
            matched = matched and same
            pairs.append(dict(coarse=a, detailed=b, tempo_and_start_match=same,
                              coarse_minus_detailed_end_s=a['end_s'] - b['end_s']))
    if matched:
        for i in range(1, len(coarse)):
            a, b = coarse[i-1:i+1]
            if a['tempo'] is None or b['tempo'] is None or a['tempo'] == b['tempo']:
                continue
            if abs(a['end_s'] - b['start_s']) > .001:
                continue
            x, y = detailed[i-1:i+1]
            gap = y['start_s'] - x['end_s']
            changes.append(dict(coarse_change_time_s=b['start_s'],
                tempo_before=x['tempo'], tempo_after=y['tempo'],
                detailed_unannotated_gap_s=gap,
                detailed_adjacent=abs(gap) <= .001,
                thirty_second_windows_inside_both_detailed_intervals=(
                    x['end_s']-x['start_s'] >= 30 and y['end_s']-y['start_s'] >= 30),
                before_window_s=[x['end_s']-30, x['end_s']],
                after_window_s=[y['start_s'], y['start_s']+30]))
    return dict(status='matched_tempo_and_starts' if matched else 'requires_manual_review',
                coarse_rows=len(coarse), detailed_rows=len(detailed),
                coarse_errors=ce, detailed_errors=de, pairs=pairs, changes=changes)


def audit(catalog_path, tar_path):
    inputs = {str(p): sha(p) for p in (catalog_path, tar_path, Path(__file__).resolve())}
    if inputs[str(catalog_path)] != CATALOG_SHA or inputs[str(tar_path)] != TAR_SHA:
        raise ValueError('Pinned catalog/tar changed')
    catalog = json.loads(catalog_path.read_text())
    results = []
    with tarfile.open(tar_path, 'r:gz') as archive:
        for row in catalog['rows']:
            if row['tradition'] != 'hindustani' or 'bpm-manual.txt' not in row['companion_files']:
                continue
            stem = 'saraga-' + COMMIT + '/' + row['metadata_path'][:-5]
            data, hashes = [], {}
            for suffix in ('.bpm-manual.txt', '.tempo-manual.txt'):
                member = archive.getmember(stem + suffix)
                if not member.isfile() or member.size > 100000:
                    raise ValueError('Unexpected annotation member')
                content = archive.extractfile(member).read()
                data.append(content)
                hashes[stem + suffix] = hashlib.sha256(content).hexdigest()
            results.append(dict(mbid=row['mbid'], title=row['title'],
                                annotation_sha256=hashes, **compare(*data)))
    if {p: sha(p) for p in inputs} != inputs:
        raise ValueError('Evidence/code changed while reading')
    changes = [c for r in results for c in r['changes']]
    return dict(status='completed_annotation_concordance_not_accuracy', input_sha256=inputs,
        files=len(results), matched_files=sum(r['status']=='matched_tempo_and_starts' for r in results),
        review_files=sum(r['status']!='matched_tempo_and_starts' for r in results),
        coarse_contiguous_changes_in_matched_files=len(changes),
        changes_also_contiguous_in_detailed_annotations=sum(c['detailed_adjacent'] for c in changes),
        changes_with_separate_30s_detailed_context=sum(c['thirty_second_windows_inside_both_detailed_intervals'] for c in changes),
        rows=results, audio_opened=False, classifier_admission=False,
        limits=['Equal tempo/start fields do not prove equivalence of annotation intervals.',
                'Detailed Hindustani tempo is documented in matras per minute, not automatically a Western beat rate.',
                'Separate stable-context windows are annotation candidates, not selected or audio-aligned stimuli.',
                'Gaps prevent treating the coarse boundary as a precisely annotated instantaneous change.',
                'No unlabelled interval is promoted to a rhythm-change negative.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--metadata-tar', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing existing report')
    result = audit(a.catalog.resolve(), a.metadata_tar.resolve())
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2))


if __name__ == '__main__':
    main()
