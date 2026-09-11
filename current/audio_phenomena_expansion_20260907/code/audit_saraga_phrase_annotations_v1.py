#!/usr/bin/env python3
"""Inventory phrase-label semantics and time scales, without model scoring."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import tarfile

COMMIT = 'bd103295689ad11a5ed19038b7469e91b5c3c124'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_phrases(data):
    events, errors = [], []
    for line, raw in enumerate(data.decode('utf-8-sig').splitlines(), 1):
        if not raw.strip():
            continue
        try:
            fields = raw.split()
            if len(fields) != 4:
                raise ValueError('expected start,flag,duration,notes')
            start, flag, duration, label = fields
            start, duration, flag = float(start), float(duration), int(flag)
            if not math.isfinite(start) or not math.isfinite(duration) or start < 0 or duration <= 0:
                raise ValueError('invalid phrase interval')
            events.append(dict(line=line, start_s=start, duration_s=duration,
                end_s=start+duration, flag=flag, notes=label))
        except ValueError as exc:
            errors.append(dict(line=line, raw=raw, error=str(exc)))
    by_label = defaultdict(list)
    for event in events:
        by_label[event['notes']].append(event)
    pairs = []
    for label, group in sorted(by_label.items()):
        for left, right in itertools.combinations(sorted(group, key=lambda e:(e['start_s'],e['line'])), 2):
            lag = right['start_s'] - left['start_s']
            if 8 <= lag <= 56:
                pairs.append(dict(notes=label, first_line=left['line'], second_line=right['line'],
                    lag_s=lag, both_duration_at_least4s=left['duration_s']>=4 and right['duration_s']>=4,
                    both_flags_1_or_2=left['flag'] in (1,2) and right['flag'] in (1,2),
                    flags=[left['flag'],right['flag']]))
    return dict(status='invalid_annotation' if errors else 'parsed_annotation', events=events,
        errors=errors, flag_counts=dict(Counter(str(e['flag']) for e in events)),
        unknown_flag_values=sorted({e['flag'] for e in events if e['flag'] not in (0,1,2)}),
        repeated_note_labels=sum(len(v)>=2 for v in by_label.values()),
        note_labels_with_mixed_flags=sorted(k for k,v in by_label.items() if len({e['flag'] for e in v})>1),
        same_label_pairs_lag8to56s=pairs,
        duplicate_event_count=len(events)-len({(e['start_s'],e['duration_s'],e['flag'],e['notes']) for e in events}))


def audit(catalog_path, tar_path):
    inputs = {str(p):sha(p) for p in (catalog_path,tar_path,Path(__file__))}
    catalog = json.loads(catalog_path.read_text())
    if catalog['repository_commit'] != COMMIT or catalog['source_inputs_sha256'][tar_path.name] != inputs[str(tar_path)]:
        raise ValueError('Wrong pinned metadata')
    rows = []
    with tarfile.open(tar_path,'r:gz') as archive:
        for row in catalog['rows']:
            if row['tradition']!='hindustani' or 'mphrases-manual.txt' not in row['companion_files']:
                continue
            name = f'saraga-{COMMIT}/'+row['metadata_path'][:-5]+'.mphrases-manual.txt'
            member = archive.getmember(name)
            if not member.isfile() or member.size > 1000000:
                raise ValueError('Unexpected phrase member')
            data = archive.extractfile(member).read()
            rows.append(dict(mbid=row['mbid'],title=row['title'],annotation_member=name,
                annotation_sha256=hashlib.sha256(data).hexdigest(),**parse_phrases(data)))
    if inputs != {p:sha(Path(p)) for p in inputs}:
        raise ValueError('Source/code changed during audit')
    events = [e for r in rows for e in r['events']]
    pairs = [p for r in rows if r['status']=='parsed_annotation' for p in r['same_label_pairs_lag8to56s']]
    durations = [e['duration_s'] for e in events]
    return dict(status='completed_phrase_annotation_inventory_not_motif_benchmark',input_sha256=inputs,
        files=len(rows),invalid_files=sum(r['status']!='parsed_annotation' for r in rows),
        empty_files=sum(not r['events'] for r in rows),events=len(events),
        flag_counts=dict(Counter(str(e['flag']) for e in events)),
        files_with_mixed_flags_for_same_notes=sum(bool(r['note_labels_with_mixed_flags']) for r in rows),
        duration_s=dict(min=min(durations),median=statistics.median(durations),max=max(durations)) if durations else None,
        events_at_least4s=sum(e['duration_s']>=4 for e in events),
        same_label_pairs_lag8to56s=len(pairs),
        same_label_pairs_lag8to56s_both_at_least4s=sum(p['both_duration_at_least4s'] for p in pairs),
        same_label_pairs_lag8to56s_both_at_least4s_flags1or2=sum(p['both_duration_at_least4s'] and p['both_flags_1_or_2'] for p in pairs),
        rows=rows, classifier_admission=False, physical_audio_alignment_verified=False,
        limitations=['Flag0 remains semantically unresolved, not a negative example.',
          'Exact solfege strings retain case and are not acoustic motif identity.',
          'Annotations are incomplete; unannotated intervals are not verified negatives.',
          'Pair lag/duration screens only assess scale overlap; they do not validate whole-window M metrics.',
          'Pairs within one track or performer group are not independent observations.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog-audit',type=Path,required=True)
    p.add_argument('--repository-metadata-tar',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing existing output')
    result=audit(a.catalog_audit.resolve(),a.repository_metadata_tar.resolve())
    with a.output.open('x') as handle:
        json.dump(result,handle,indent=2,ensure_ascii=False,allow_nan=False)
        handle.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__': main()
