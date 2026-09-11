#!/usr/bin/env python3
"""Acquisition-only500, separate from frozen detector cohorts; resumable receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import unicodedata

import probe_music8k_mureka_long as PROBE

SEED = 'music8k-mureka-v9-formal500-20260907'


def text_hash(value):
    normalized = ' '.join(unicodedata.normalize('NFKC', str(value)).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def formal_rank(candidates, excluded):
    rows = [{**r, 'rank': hashlib.sha256((SEED+'|'+r['id']).encode()).hexdigest()}
            for r in candidates if r['id'] not in excluded]
    if len(rows) != 650 or len(excluded) != 12 or len({r['id'] for r in rows}) != 650:
        raise ValueError('Wrong exclusion/candidate population')
    return sorted(rows, key=lambda r:r['rank'])


def verify_raw(path, row):
    if path.stat().st_size != row['bytes'] or PROBE.sha(path) != row['sha256']:
        raise ValueError('LFS file size/hash mismatch: '+row['id'])


def freeze(root, output):
    if (output/'contract.json').exists():
        raise ValueError('Existing acquisition contract is immutable')
    catalog = root/'external_validation/long_source_catalogs_v1/homura23__MUSIC8K.json'
    probe = root/'external_validation/music8k_mureka_probe_v1'
    summary = json.loads((probe/'summary.json').read_text())
    if summary['passed'] != 12 or summary['eligible_native60'] != 12 or summary['contract_sha256'] != PROBE.sha(probe/'contract.json'):
        raise ValueError('Completed physical feasibility proof required')
    for name, expected in summary['receipts_sha256'].items():
        if PROBE.sha(probe/name) != expected:
            raise ValueError('Probe receipt changed')
    excluded = {r['id'] for r in json.loads((probe/'contract.json').read_text())['selected']}
    ranked = formal_rank(PROBE.select(json.loads(catalog.read_text())), excluded)
    metadata = root/'external_validation/music8k_metadata_v1/metadata'
    original = {r['id']:r for r in csv.DictReader((metadata/'pop_1000_unique_artist.csv').open())}
    captions = {str(r['id']):r for r in map(json.loads,(metadata/'songs_musiccaps.jsonl').open())}
    for row in ranked:
        a, b = original[row['id']], captions[row['id']]
        row.update(role='reserved_unscored', source='Mureka v9',
                   reference_group_id='music8k_reference_'+row['id'],
                   reference_artist_hash=text_hash(a['artist']),
                   reference_lyrics_hash=text_hash(a['lyrics']),
                   generation_lyrics_hash=text_hash(b['lyrics']),
                   caption_hash=text_hash(b['caption']))
    selected = ranked[:500]
    if len({r['reference_artist_hash'] for r in selected}) != 500:
        raise ValueError('Unexpected repeated reference artist')
    contract = dict(status='frozen_for_acquisition_only', frozen_utc=datetime.now(timezone.utc).isoformat(),
        repo='homura23/MUSIC8K', revision=PROBE.REVISION, seed=SEED, selected_count=500,
        code_sha256=PROBE.sha(__file__), probe_helper_sha256=PROBE.sha(PROBE.__file__),
        protocol_sha256=PROBE.sha(root/'MUSIC8K_MUREKA500_ACQUISITION_EN.md'),
        catalog_sha256=PROBE.sha(catalog), probe_summary_sha256=PROBE.sha(probe/'summary.json'),
        metadata_sha256={p.name:PROBE.sha(p) for p in metadata.iterdir() if p.is_file()},
        ffmpeg_version=subprocess.check_output(['/opt/homebrew/bin/ffmpeg','-version'],text=True),
        excluded_probe_ids=sorted(excluded), ranked_candidates=ranked, selected=selected,
        intended_bytes=sum(r['bytes'] for r in selected), classifier_authorized=False)
    PROBE.write_new(output/'contract.json', contract)
    print(json.dumps({k:v for k,v in contract.items() if k not in ('ranked_candidates','selected','ffmpeg_version')}), flush=True)


def materialize(root, output):
    contract = json.loads((output/'contract.json').read_text())
    if (contract['status'] != 'frozen_for_acquisition_only' or contract['classifier_authorized']
        or contract['code_sha256'] != PROBE.sha(__file__) or contract['probe_helper_sha256'] != PROBE.sha(PROBE.__file__)
        or contract['protocol_sha256'] != PROBE.sha(root/'MUSIC8K_MUREKA500_ACQUISITION_EN.md')
        or contract['ffmpeg_version'] != subprocess.check_output(['/opt/homebrew/bin/ffmpeg','-version'],text=True)):
        raise ValueError('Frozen acquisition code/runtime/protocol mismatch')
    selected = contract['selected']
    if len(selected) != 500 or any(r['role'] != 'reserved_unscored' for r in selected):
        raise ValueError('Wrong acquisition population')
    cmd = ['hf','download',contract['repo'],*(r['path'] for r in selected), '--repo-type','dataset',
           '--revision',contract['revision'],'--local-dir',str(output/'raw'),'--max-workers','4']
    with (output/'download.log').open('a') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    (output/'items').mkdir(exist_ok=True)
    (output/'decode_logs').mkdir(exist_ok=True)
    def one(row):
        path = output/'raw'/row['path']
        receipt_path = output/'items'/(row['id']+'.json')
        verify_raw(path,row)
        if receipt_path.exists():
            result = json.loads(receipt_path.read_text())
            if result['contract_sha256'] != PROBE.sha(output/'contract.json') or result['sha256'] != row['sha256']:
                raise ValueError('Resume receipt changed')
            return result
        result = dict(row,contract_sha256=PROBE.sha(output/'contract.json'))
        try:
            result.update(PROBE.decode(path, output/'decode_logs'/(row['id']+'.log')))
            result['status']='passed'
        except Exception as e:
            result.update(status='failed',error=str(e),eligible_native60=False)
        PROBE.write_new(receipt_path,result)
        print(json.dumps({'id':row['id'],'status':result['status'],'eligible_native60':result['eligible_native60']}),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(one,selected))
    summary = dict(status='completed',selected=500,passed=sum(r['status']=='passed' for r in records),
        eligible_native60=sum(r['eligible_native60'] for r in records),bytes=sum(r['bytes'] for r in records),
        contract_sha256=PROBE.sha(output/'contract.json'), classifiers_fitted=0, neural_inference=False,
        receipts_sha256={p.name:PROBE.sha(p) for p in sorted((output/'items').glob('*.json'))})
    PROBE.write_new(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='receipts_sha256'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('freeze','materialize'),required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parent.parent
    output=root/'external_validation/music8k_mureka500_v1'
    output.mkdir(parents=True,exist_ok=True)
    with (output/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (freeze if a.stage=='freeze' else materialize)(root,output)


if __name__=='__main__':
    main()
