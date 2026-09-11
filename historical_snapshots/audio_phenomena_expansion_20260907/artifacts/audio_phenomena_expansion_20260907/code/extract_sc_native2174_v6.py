#!/usr/bin/env python3
"""Bounded CPU-only SC measurement, separate from classifier authorization."""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
import soundfile as sf
import stereo_candidate_v1 as SC
import summarize_v5_channel_provenance as INV

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907')
PACKAGE=ROOT/'results/equal60_exploratory_package_v5_v1'
GATE=ROOT/'results/stereo_lowband_gate_v2/gate_result.json'
GATE_AUDIT=ROOT/'audit/stereo_lowband_gate_parent_replay_v2.json'
PROTOCOL=ROOT/'SC_NATIVE2174_EXTRACTION_PROTOCOL_EN.md'
KEYS=[f'SC_{lo}_{hi}hz_{metric}_median' for lo,hi in [(80,500),(500,2000),(2000,6000)]
      for metric in ['abs_iid_db','side_energy_fraction']]
GATE_SHA='da5c82c052c779fca840b194c2fbef5b42b8bf827141e49a375b0066ab020a4e'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def clean(value):
    if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.generic): return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def write_json(path,value):
    with Path(path).open('x') as f:
        json.dump(clean(value),f,sort_keys=True,allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())


def read(path): return json.loads(Path(path).read_text())
def csvrows(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))


def roster(data):
    inventory=INV.summarize(ROOT,data)
    assert inventory['rows']==2207 and sha(GATE)==GATE_SHA
    gate,audit=read(GATE),read(GATE_AUDIT)
    assert gate['measurement_gate_passed'] and gate['candidate_features']==KEYS
    assert audit['passed'] and audit['gate_receipt_sha256']==GATE_SHA
    paths=[PACKAGE/'COMMIT.json',PACKAGE/'metadata_60s.csv',GATE,GATE_AUDIT,
           Path(__file__),Path(SC.__file__),Path(INV.__file__),PROTOCOL]
    for name in INV.PINS: paths.append(data/name.removeprefix('manifests/'))
    all_sources={}
    for p in paths:
        if p.name=='inference_manifest.csv':
            for row in csvrows(p):
                assert row['item_id'] not in all_sources
                all_sources[row['item_id']]=row
    selected=[]; excluded=[]
    for m in csvrows(PACKAGE/'metadata_60s.csv'):
        source=all_sources[m['id']]
        assert source['group_id']==m['group_id'] and source['label']==m['label']
        if m['source_group']=='human_saraga_hindustani_v1':
            proof_path=data/'saraga_external103_intervals_v1/items'/m['id']/'proof.json'
            proof=read(proof_path); native=proof['interval_proof']['native_channels']; paths.append(proof_path)
        else: native=int(source['source_channels'])
        row=dict(metadata=m,native_channels=native,standardized_path=source['standardized_path'],
            standardized_file_sha256=source['standardized_file_sha256'],
            standardized_sr=int(source['standardized_sr']),standardized_frames=int(source['standardized_frames']),
            standardized_channels=int(source['standardized_channels']))
        assert row['standardized_sr']==44100 and row['standardized_frames']==2646000 and row['standardized_channels']==2
        assert Path(row['standardized_path']).is_relative_to(data)
        (selected if native==2 else excluded).append(row)
    assert len(selected)==2174 and len(excluded)==33
    assert Counter(r['metadata']['label'] for r in selected)=={'0':1278,'1':896}
    assert all(r['native_channels']==1 and r['metadata']['source_group']=='human_urmp' for r in excluded)
    return dict(selected=selected,excluded=excluded,bindings={str(p):sha(p) for p in paths},
        source_inventory=inventory,candidate_features=KEYS,config=SC.CONFIG,
        runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,soundfile=sf.__version__,
            platform=platform.platform(),executable=str(Path(sys.executable).resolve())),
        scope='measurement_only_no_classifier_fit',raw_audio_decoded_during_draft=False)


def extract_one(row,output):
    identity=row['metadata']['id']; path=Path(row['standardized_path'])
    assert identity and '/' not in identity and '\\' not in identity
    assert sha(path)==row['standardized_file_sha256'], 'input bytes changed: '+identity
    y,sr=sf.read(path,dtype='float64',always_2d=True)
    assert sr==44100 and y.shape==(2646000,2) and np.isfinite(y).all()
    result=SC.extract_stereo_candidate(y,sr,native_channels=2)
    directory=output/'items'/identity; directory.mkdir()
    np.savez_compressed(directory/'frames.npz',**result['per_frame'],**result['valid_masks'])
    quality={k:v for k,v in result.items() if k not in {'per_frame','valid_masks'}}
    provenance=dict(id=identity,source_group=row['metadata']['source_group'],
        group_id=row['metadata']['group_id'],label=row['metadata']['label'],
        native_channels=2,input_file_sha256=row['standardized_file_sha256'],
        input_path=str(path),same_channels_exact=bool(np.array_equal(y[:,0],y[:,1])),
        opposite_channels_exact=bool(np.array_equal(y[:,0],-y[:,1])),
        channel_rms=[float(np.sqrt(np.mean(y[:,c]**2))) for c in range(2)],
        status=result['status'],candidate_missing=[k for k in KEYS if not np.isfinite(result['features'][k])],
        candidate_features={k:result['features'][k] for k in KEYS},quality=quality,
        diagnostic_fields_are_not_predictors=True)
    assert sha(path)==row['standardized_file_sha256'], 'input changed during decode: '+identity
    write_json(directory/'measurement.json',provenance)
    return clean(provenance)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['draft','run'])
    parser.add_argument('--data-root',type=Path,default=DATA)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--freeze-sha256')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    assert 1<=args.workers<=4
    if args.action=='draft':
        contract=roster(args.data_root)
        contract.update(created_utc=datetime.now(timezone.utc).isoformat(),workers=args.workers,
                        output=str(args.output),status='frozen_for_measurement_only')
        write_json(args.freeze,contract)
        print(json.dumps(dict(freeze=str(args.freeze),sha256=sha(args.freeze),selected=2174,excluded=33)))
        return
    assert args.freeze_sha256 and sha(args.freeze)==args.freeze_sha256
    contract=read(args.freeze)
    assert contract['output']==str(args.output) and contract['workers']==args.workers
    assert contract['status']=='frozen_for_measurement_only' and contract['candidate_features']==KEYS
    for p,digest in contract['bindings'].items(): assert sha(p)==digest, p
    assert contract['runtime']==roster(args.data_root)['runtime']
    args.output.mkdir(exist_ok=False); (args.output/'items').mkdir()
    collected=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator=pool.map(lambda row:extract_one(row,args.output),contract['selected'])
        with (args.output/'selected_features.jsonl').open('x') as f:
            for index,result in enumerate(iterator,1):
                f.write(json.dumps({k:result[k] for k in ['id','source_group','group_id','label','native_channels',
                    'input_file_sha256','input_path','candidate_features','candidate_missing']},sort_keys=True,allow_nan=False)+'\n')
                collected.append(result)
                if index%50==0 or index==2174: print(f'extracted {index}/2174',flush=True)
    assert len(collected)==2174
    for p,digest in contract['bindings'].items(): assert sha(p)==digest,p
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),rows=len(collected),
        freeze_sha256=sha(args.freeze),candidate_features=KEYS,classifier_fitted=False,
        source_counts=dict(Counter(r['source_group'] for r in collected)),
        candidate_missing_counts={k:sum(k in r['candidate_missing'] for r in collected) for k in KEYS},
        exact_equal_channels=sum(r['same_channels_exact'] for r in collected),
        exact_opposite_channels=sum(r['opposite_channels_exact'] for r in collected),
        same_channel_counts_by_source=dict(Counter(r['source_group'] for r in collected if r['same_channels_exact'])),
        quality_did_not_filter_rows=True,selected_jsonl_sha256=sha(args.output/'selected_features.jsonl'))
    write_json(args.output/'summary.json',summary)
    products={str(p.relative_to(args.output)):dict(sha256=sha(p),bytes=p.stat().st_size)
              for p in sorted(args.output.rglob('*')) if p.is_file()}
    write_json(args.output/'COMMIT.json',dict(freeze_sha256=sha(args.freeze),products=products))
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
