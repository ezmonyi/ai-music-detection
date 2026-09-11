#!/usr/bin/env python3
"""Independent selection/receipt/raw-byte audit of the reserved Mureka500.

Does not import acquisition code, fit models, infer features, or assign roles.
Full original audio is rehashed; native decode assertions are audited against
the completed acquisition receipts, not claimed as a second full decode.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import unicodedata

REVISION='05232438ba76a7bc55cbebfdc6d5f4011c980bba'
SEED='music8k-mureka-v9-formal500-20260907'
PROBE_SEED='music8k-mureka-v9-physical-probe-20260907'


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''): h.update(part)
    return h.hexdigest()


def text_hash(value):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC',str(value)).casefold().split()).encode()).hexdigest()


def rank(ids,seed):
    return sorted(ids,key=lambda i:hashlib.sha256((seed+'|'+i).encode()).hexdigest())


def check_decode(r):
    require(r['status']=='passed' and r['all_samples_finite'] is True,'Failed/nonfinite decode')
    rate,channels,frames=r['sample_rate'],r['channels'],r['decoded_frames']
    require(type(rate) is int and type(channels) is int and type(frames) is int,'Decode count/type mismatch')
    require(rate>0 and channels>0 and frames>0,'Nonpositive decoded format')
    require(math.isclose(r['decoded_duration_s'],frames/rate,rel_tol=0,abs_tol=1e-9),'Frame/duration mismatch')
    eligible=frames>=60*rate and rate>=16000 and channels in (1,2)
    require(r['eligible_native60'] is eligible,'Wrong native60 eligibility')
    require(len(r['decoded_native_float32_sha256'])==64 and all(c in '0123456789abcdef' for c in r['decoded_native_float32_sha256']),'Invalid PCM hash')
    return eligible


def audit(root,acquisition):
    contract_path=acquisition/'contract.json'; summary_path=acquisition/'summary.json'
    require(summary_path.is_file(),'Acquisition is not complete')
    c=json.loads(contract_path.read_text()); s=json.loads(summary_path.read_text())
    ch=sha(contract_path); sh=sha(summary_path)
    require(c['status']=='frozen_for_acquisition_only' and c['classifier_authorized'] is False,'Invalid acquisition role')
    require(c['repo']=='homura23/MUSIC8K' and c['revision']==REVISION and c['seed']==SEED,'Unpinned source/selection')
    for path,key in [(root/'code/materialize_music8k_mureka500.py','code_sha256'),
                     (root/'code/probe_music8k_mureka_long.py','probe_helper_sha256'),
                     (root/'MUSIC8K_MUREKA500_ACQUISITION_EN.md','protocol_sha256')]:
        require(sha(path)==c[key],'Frozen acquisition code/protocol changed')
    require(s['status']=='completed' and s['contract_sha256']==ch and s['selected']==500 and s['classifiers_fitted']==0 and s['neural_inference'] is False,'Invalid completion summary')
    catalog_path=root/'external_validation/long_source_catalogs_v1/homura23__MUSIC8K.json'
    require(sha(catalog_path)==c['catalog_sha256'],'Catalog hash changed')
    catalog=json.loads(catalog_path.read_text())
    require(catalog['sha']==REVISION,'Catalog revision mismatch')
    catalog_rows={}
    for row in catalog['siblings']:
        p=Path(row['rfilename'])
        if p.parts[0]!='mureka_v9': continue
        require(len(p.parts)==2 and p.suffix=='.mp3' and p.stem.isdigit(),'Unsafe/unexpected member path')
        require(row['size']==row['lfs']['size'] and p.stem not in catalog_rows,'Invalid/duplicate LFS entry')
        catalog_rows[p.stem]=dict(id=p.stem,path=str(p),bytes=row['size'],sha256=row['lfs']['sha256'])
    require(len(catalog_rows)==662,'Wrong source population')
    excluded=rank(catalog_rows,PROBE_SEED)[:12]
    require(sorted(excluded)==c['excluded_probe_ids'],'Probe exclusions changed')
    ranked=rank(set(catalog_rows)-set(excluded),SEED)
    require([r['id'] for r in c['ranked_candidates']]==ranked and [r['id'] for r in c['selected']]==ranked[:500],'Frozen random ranking changed')
    require(c['selected']==c['ranked_candidates'][:500] and c['selected_count']==500,'Selected candidate mismatch')
    metadata=root/'external_validation/music8k_metadata_v1/metadata'
    for name,expected in c['metadata_sha256'].items():
        require(Path(name).name==name and sha(metadata/name)==expected,'Metadata changed')
    originals={r['id']:r for r in csv.DictReader((metadata/'pop_1000_unique_artist.csv').open())}
    prompts={str(r['id']):r for r in map(json.loads,(metadata/'songs_musiccaps.jsonl').open())}
    for r in c['ranked_candidates']:
        require(all(r[k]==v for k,v in catalog_rows[r['id']].items()),'Selected raw metadata differs from source')
        require(r['rank']==hashlib.sha256((SEED+'|'+r['id']).encode()).hexdigest(),'Rank digest mismatch')
        require(r['role']=='reserved_unscored' and r['source']=='Mureka v9' and r['reference_group_id']=='music8k_reference_'+r['id'],'Role/group changed')
        a,b=originals[r['id']],prompts[r['id']]
        for key,value in [('reference_artist_hash',a['artist']),('reference_lyrics_hash',a['lyrics']),('generation_lyrics_hash',b['lyrics']),('caption_hash',b['caption'])]:
            require(r[key]==text_hash(value),'Text/group hash mismatch')
    selected=c['selected']; expected_items={r['id']+'.json' for r in selected}
    require({p.name for p in (acquisition/'items').iterdir()}==expected_items==set(s['receipts_sha256']),'Missing/extra item receipts')
    require({str(p.relative_to(acquisition/'raw')) for p in (acquisition/'raw').rglob('*.mp3')}=={r['path'] for r in selected},'Missing/extra raw audio')
    records=[]
    for row in selected:
        item=acquisition/'items'/(row['id']+'.json')
        require(sha(item)==s['receipts_sha256'][item.name],'Changed item receipt')
        r=json.loads(item.read_text())
        require(all(r[k]==v for k,v in row.items()) and r['contract_sha256']==ch,'Receipt selection mismatch')
        path=acquisition/'raw'/row['path']
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Raw file bytes/hash mismatch')
        eligible=check_decode(r)
        command=r['command']
        require(command==['/opt/homebrew/bin/ffmpeg','-v','error','-xerror','-i',str(path),'-map','0:a:0','-c:a','pcm_f32le','-f','f32le','-'],'Unexpected native decode command')
        records.append(dict(id=row['id'],sha256=row['sha256'],bytes=row['bytes'],sample_rate=r['sample_rate'],channels=r['channels'],frames=r['decoded_frames'],duration_s=r['decoded_duration_s'],eligible_native60=eligible,decoded_native_float32_sha256=r['decoded_native_float32_sha256']))
    require(sum(r['bytes'] for r in records)==c['intended_bytes']==s['bytes'],'Total bytes mismatch')
    require(s['passed']==len(records)==500 and s['eligible_native60']==sum(r['eligible_native60'] for r in records),'Summary decode counts mismatch')
    require(sha(contract_path)==ch and sha(summary_path)==sh,'Contract/summary changed during audit')
    require(all(sha(acquisition/'items'/n)==v for n,v in s['receipts_sha256'].items()),'Receipts changed during audit')
    return dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),
        source_revision=REVISION,contract_sha256=ch,summary_sha256=sh,selected=500,
        bytes=sum(r['bytes'] for r in records),eligible_native60=sum(r['eligible_native60'] for r in records),
        native_format_counts=dict(Counter(f"{r['sample_rate']}Hz/{r['channels']}ch" for r in records)),
        duration_s_min=min(r['duration_s'] for r in records),duration_s_max=max(r['duration_s'] for r in records),
        all_original_files_rehashed=True,second_full_decode_performed=False,
        decode_scope='All500 original full-native decode receipts checked; not a second independent decoder.',
        selection_independently_reconstructed=True,rows=records,classifier_admission_authorized=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.root,a.root/'external_validation/music8k_mureka500_v1')
    with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
