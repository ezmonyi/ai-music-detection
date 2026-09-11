#!/usr/bin/env python3
"""Inspect authored BPM interval annotations; not a model/accuracy benchmark."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile

COMMIT = 'bd103295689ad11a5ed19038b7469e91b5c3c124'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_bpm(data):
    segments, errors = [], []
    for number, row in enumerate(csv.reader(io.StringIO(data.decode('utf-8-sig'))), 1):
        if not row or all(not x.strip() for x in row):
            continue
        try:
            if len(row) != 3:
                raise ValueError('expected bpm,start,end columns')
            raw, start, end = [x.strip() for x in row]
            start, end = float(start), float(end)
            if not all(math.isfinite(v) for v in (start, end)) or start < 0 or end <= start:
                raise ValueError('invalid interval')
            bpm = None if raw in ('-', '-1', '') else float(raw)
            if bpm is not None and (not math.isfinite(bpm) or bpm <= 0):
                raise ValueError('invalid BPM')
            if segments and start < segments[-1]['end_s'] - 0.001:
                raise ValueError('overlapping or out-of-order interval')
            segments.append(dict(line=number, raw_bpm=raw, bpm=bpm, start_s=start, end_s=end))
        except ValueError as exc:
            errors.append(dict(line=number, raw_columns=row, error=str(exc)))
    transitions = []
    # An invalid annotation file cannot supply a positive measurement label.
    if not errors:
        for left, right in zip(segments, segments[1:]):
            if left['bpm'] is None or right['bpm'] is None:
                continue
            if abs(left['end_s'] - right['start_s']) > 0.001:
                continue
            ratio = right['bpm'] / left['bpm']
            if ratio == 1:
                continue
            log_ratio = math.log2(ratio)
            transitions.append(dict(time_s=right['start_s'], bpm_before=left['bpm'],
                bpm_after=right['bpm'], ratio=ratio, log2_ratio=log_ratio,
                annotation_change_only=True,
                near_half_or_double_within5pct=any(abs(ratio / r - 1) <= 0.05 for r in (0.5, 2.0)),
                context_30s_both_sides_within_adjacent_annotations=(
                    left['end_s'] - left['start_s'] >= 30 and right['end_s'] - right['start_s'] >= 30)))
    return dict(status='invalid_annotation' if errors else 'parsed_annotation',
        segments=segments, errors=errors, contiguous_known_bpm_changes=transitions,
        physical_audio_alignment_verified=False)


def audit(catalog_path, tar_path):
    inputs = {str(p): sha(p) for p in (catalog_path, tar_path, Path(__file__))}
    catalog = json.loads(catalog_path.read_text())
    if catalog['repository_commit'] != COMMIT or catalog['status'] != 'passed_metadata_only':
        raise ValueError('Wrong catalog snapshot')
    if catalog['source_inputs_sha256'][tar_path.name] != inputs[str(tar_path)]:
        raise ValueError('Changed pinned metadata tar')
    results = []
    with tarfile.open(tar_path, 'r:gz') as archive:
        for row in catalog['rows']:
            if row['tradition'] != 'hindustani' or 'bpm-manual.txt' not in row['companion_files']:
                continue
            name = f"saraga-{COMMIT}/" + row['metadata_path'][:-5] + '.bpm-manual.txt'
            member = archive.getmember(name)
            if not member.isfile() or member.size > 100000:
                raise ValueError('Unexpected annotation member')
            data = archive.extractfile(member).read()
            result = parse_bpm(data)
            results.append(dict(mbid=row['mbid'], title=row['title'], annotation_member=name,
                annotation_sha256=hashlib.sha256(data).hexdigest(),
                advertised_length_ms=row['advertised_length_ms'], **result))
    if inputs != {p: sha(Path(p)) for p in inputs}:
        raise ValueError('Inputs/code changed during annotation audit')
    transitions = [t for r in results for t in r['contiguous_known_bpm_changes']]
    return dict(status='completed_annotation_content_inventory_not_performance_benchmark',
        input_sha256=inputs, files=len(results),
        parsed_files=sum(r['status'] == 'parsed_annotation' for r in results),
        invalid_files=sum(r['status'] == 'invalid_annotation' for r in results),
        files_with_contiguous_known_bpm_change=sum(bool(r['contiguous_known_bpm_changes']) for r in results),
        annotated_change_count=len(transitions),
        changes_with_30s_context_both_sides=sum(t['context_30s_both_sides_within_adjacent_annotations'] for t in transitions),
        near_half_or_double_change_count=sum(t['near_half_or_double_within5pct'] for t in transitions),
        rows=results, audio_opened=False, neural_inference=False, classifier_admission=False,
        limits=['A BPM label change is not necessarily rhythmic-pattern diversity.',
                'Half/double tempo conventions and tala-cycle changes require musical review.',
                'An unchanging coarse BPM annotation is not a verified negative for local rhythm changes.',
                'Annotation coverage is not physical audio alignment or model accuracy.',
                'No assay/test cohort or thresholds selected from detector scores.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog-audit', type=Path, required=True)
    p.add_argument('--repository-metadata-tar', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing existing output')
    result = audit(a.catalog_audit.resolve(), a.repository_metadata_tar.resolve())
    with a.output.open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
