#!/usr/bin/env python3
"""Read-only independent metadata arithmetic; no cohort freeze or audio reads."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / 'manifests/saraga_hindustani_physical_v1/summary.json'
PACKAGE = ROOT / 'results/equal60_package_v4_with_recovery_v1'
HISTORICAL = ROOT / 'evaluation_inputs/sdrfh_10s_v2/metadata_10s.csv'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def audit():
    assert sha(SUMMARY) == '59dc718293c45c492e945da825d379969836ff13dfb0fb2a894d7570391c73c3'
    assert sha(PACKAGE / 'evidence/native.csv') == '00ff8671c589ba280f2fc2a5bf3019f2d907fdf7a8933afa573f433112a8ee1b'
    assert sha(PACKAGE / 'evidence/fhm.csv') == '46ea2a81d4653fa8d830e7f31a858ad75999105e8b6ce813032862743c7e51fa'
    rows = json.loads(SUMMARY.read_text())['records']
    assert len(rows) == 108
    eligible, excluded = [], []
    for row in rows:
        md = row['item']['reconciled_metadata']
        reasons = []
        if md['speech_title_review_flag']:
            reasons.append('speech_title_flag')
        if not md['performer_credits']:
            reasons.append('missing_performer_credits')
        (excluded if reasons else eligible).append((row, reasons))
    assert len(eligible) == 103 and len(excluded) == 5
    ids = sorted(row['item']['mbid'] for row, _ in eligible)
    digest = hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    assert digest == 'a0df862441fb0cf9f214d739a6fca55dc54626457fdd410810aeb1690d35321a'
    parent = {i: i for i in ids}
    by_key = collections.defaultdict(list)
    performers, albums, releases = set(), set(), set()

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for row, _ in eligible:
        md, i = row['item']['reconciled_metadata'], row['item']['mbid']
        a = {c['artist_mbid'] for c in md['performer_credits']}
        b = {c['mbid'] for c in md['album_artists']}
        c = {x['mbid'] for x in md['release_or_concert_groups']}
        performers |= a
        albums |= b
        releases |= c
        for identity in a | b | c:
            assert re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', identity)
        for key in [('artist', x) for x in a | b] + [('release', x) for x in c]:
            by_key[key].append(i)
    for members in by_key.values():
        for other in members[1:]:
            a, b = find(members[0]), find(other)
            parent[max(a, b)] = min(a, b)
    groups = collections.defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    components = [{'component_id': 'performer-connected-' + hashlib.sha256('\n'.join(v).encode()).hexdigest()[:16],
                   'rows': len(v), 'mbids': v} for _, v in sorted(groups.items())]
    assert sorted(len(v) for v in groups.values()) == [2, 5, 10, 14, 72]
    new_hashes = {r['measurement']['raw_hashes_after_decode']['sha256'] for r, _ in eligible}
    dev = [r for r in csv_rows(PACKAGE / 'evidence/native.csv') if r['role'] == 'development']
    assert len(dev) == 1604
    dev_ids = {r['id'] for r in dev}
    old = [r for r in csv_rows(PACKAGE / 'evidence/fhm.csv') if r['id'] in dev_ids]
    assert len(old) == len({r['id'] for r in old}) == 1604
    old_hashes = {r['source_audio_sha256'] for r in old}
    assert all(re.fullmatch('[0-9a-f]{64}', h) for h in old_hashes)
    assert not new_hashes & old_hashes
    historical = [r for r in csv_rows(HISTORICAL) if r['source_group'] == 'human_hindustani_raag_hf']
    assert len(historical) == len({r['id'] for r in historical}) == 100
    assert all(re.fullmatch('[0-9a-f]{64}', r['raw_sha256']) for r in historical)
    assert not new_hashes & {r['raw_sha256'] for r in historical}
    measurements = [r['measurement'] for r, _ in eligible]
    inputs = [SUMMARY, PACKAGE / 'evidence/native.csv', PACKAGE / 'evidence/fhm.csv', HISTORICAL]
    return dict(status='passed_metadata_only_parent_review_not_frozen_cohort',
        code_sha256=sha(Path(__file__)), input_sha256={str(p): sha(p) for p in inputs},
        eligible_count=103, eligible_sorted_mbid_sha256_no_trailing_newline=digest,
        excluded=[dict(mbid=r['item']['mbid'], reasons=reasons, title=r['item']['reconciled_metadata']['title']) for r, reasons in excluded],
        raw_bytes=sum(m['raw_hashes_after_decode']['bytes'] for m in measurements),
        actual_duration_seconds=math.fsum(m['actual_duration_seconds'] for m in measurements),
        sample_rate_counts=dict(collections.Counter(m['sample_rate'] for m in measurements)),
        header_mismatch_count=sum(not m['header_matches_actual_eof'] for m in measurements),
        performer_ids=len(performers), album_artist_ids=len(albums), release_ids=len(releases),
        components=components, v4_dev_rows=len(dev),
        v4_hash_source='accepted FHM evidence source_audio_sha256, exact1604 ID join; not incomplete native physical_sha256',
        registered_exact_raw_hash_intersection_v4=0,
        historical_hf_hindustani_10s_rows=len(historical),
        registered_exact_raw_hash_intersection_historical100=0,
        perceptual_or_excerpt_overlap='unresolved', raw_files_rehashed_in_this_review=False,
        audio_opened_in_this_review=False, new_roles_frozen=False, classifier_admission=False)


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, allow_nan=False))
