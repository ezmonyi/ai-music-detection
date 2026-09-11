#!/usr/bin/env python3
"""Preserve and audit the upstream empty-beat/nonempty-downbeat TSV failure.

Only an already logged failed beat shard is eligible. Replays the same neural
model on an idle RTX5090, retaining raw arrays. No thresholds or model changes.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import fcntl

import numpy as np
from materialize_equal60_inputs_v2 import sha_file, preserve_json
import run_equal60_inference_batches as B


def validate_failure_arrays(beats, downbeats):
    beats, downbeats = np.asarray(beats), np.asarray(downbeats)
    if beats.ndim != 1 or downbeats.ndim != 1 or len(beats) != 0 or len(downbeats) == 0:
        raise ValueError('Not the reviewed empty-beat/nonempty-downbeat serialization failure')
    if not np.isfinite(downbeats).all() or (downbeats < 0).any() or (downbeats > 60).any() or (np.diff(downbeats) <= 0).any():
        raise ValueError('Invalid raw downbeat times')


def recover(prepared, output, gpu):
    import torch
    from beat_this.inference import File2Beats
    from beat_this.utils import infer_beat_numbers
    import beat_this.inference as inference
    import beat_this.utils as utils
    import beat_this.model.postprocessor as postprocessor

    contract_path = output/'run_contract.json'
    contract = json.loads(contract_path.read_text())
    contract_sha = sha_file(contract_path)
    if contract['code_sha256'] != sha_file(B.__file__):
        raise ValueError('Original runner changed')
    for name, expected in contract['dependencies'].items():
        if sha_file(Path(__file__).with_name(name)) != expected:
            raise ValueError('Original dependency changed')
    manifest = prepared/'inference_manifest.csv'
    if sha_file(manifest) != contract['manifest_sha256']:
        raise ValueError('Prepared manifest changed')
    rows = list(csv.DictReader(manifest.open()))
    by_path = {r['standardized_path']:r for r in rows}
    runtime = Path(contract['runtime_root'])
    repo=runtime/'repos/beat_this'
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() != 'b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c':
        raise ValueError('Frozen Beat This revision changed')
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip():
        raise ValueError('Beat This checkout has unreviewed changes')
    checkpoint = runtime/'checkpoints/hub/checkpoints/beat_this-final0.ckpt'
    if sha_file(checkpoint) != '8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331':
        raise ValueError('Frozen checkpoint changed')
    recovered = []
    model = None
    for shard in contract['shards']:
        index = shard['index']
        receipt_path=output/'receipts'/f'beats_{index:02d}.json'
        if receipt_path.exists():
            continue
        shard_path=prepared/f'inference_shard_{index:02d}.txt'
        if sha_file(shard_path) != shard['sha256']:
            raise ValueError('Shard input changed')
        inputs=shard_path.read_text().splitlines()
        paths=[output/'beats'/(by_path[p]['item_id']+'.beats') for p in inputs]
        log_path=output/'logs'/f'beats_{index:02d}.log'
        if not log_path.exists() and not any(p.exists() for p in paths):
            continue
        if not log_path.exists():
            raise ValueError('Unlogged partial shard')
        log_before=sha_file(log_path)
        failed = re.findall(r'Could not process "([^"]+)"\. Rerun with this file alone for details\.',log_path.read_text())
        missing=[p for p,target in zip(inputs,paths) if not target.exists()]
        if not missing or set(missing) != set(failed) or len(failed) != len(set(failed)):
            raise ValueError('Partial shard does not exactly match logged failures')
        preexisting={str(p):sha_file(p) for p in paths if p.exists()}
        for p in paths:
            if p.exists(): B.inspect_beats(p)
        for p in inputs:
            if sha_file(p) != by_path[p]['standardized_file_sha256']:
                raise ValueError('Frozen audio input changed')
        B.require_idle_5090(gpu)
        if not torch.cuda.is_available() or 'RTX 5090' not in torch.cuda.get_device_name(gpu):
            raise ValueError('Only RTX5090 neural inference allowed')
        if model is None:
            model=File2Beats(str(checkpoint),device=f'cuda:{gpu}',float16=True,dbn=False)
        sidecars={}
        for path in missing:
            row=by_path[path]
            beats,downbeats=model(path)
            validate_failure_arrays(beats,downbeats)
            try:
                infer_beat_numbers(beats,downbeats)
            except ValueError as exc:
                if str(exc) != 'Not all downbeats are beats.': raise
            else:
                raise ValueError('Serialization failure not reproduced')
            sidecar=output/'recovery_empty_beats'/f"{row['item_id']}.json"
            preserve_json(sidecar,dict(status='verified_unavailable_no_beat_events',item_id=row['item_id'],
                input_path=path,input_sha256=sha_file(path),run_contract_sha256=contract_sha,
                recovery_code_sha256=sha_file(__file__), checkpoint_sha256=sha_file(checkpoint),
                dependency_sha256={str(p):sha_file(p) for p in (Path(inference.__file__),Path(utils.__file__),Path(postprocessor.__file__))},
                gpu=gpu,gpu_name=torch.cuda.get_device_name(gpu),float16=True,dbn=False,
                beats=beats.tolist(),downbeats=downbeats.tolist(),error='Not all downbeats are beats.',
                interpretation='Beat sequence unavailable; isolated raw downbeats retained only in this sidecar. No artificial beat or zero-variability feature is introduced.',
                original_log_sha256=log_before,utc=B.now()))
            target=output/'beats'/f"{row['item_id']}.beats"
            # Empty is the frozen extractor's existing unavailable-sequence encoding.
            with target.open('x'):
                pass
            sidecars[str(sidecar)]={'sha256':sha_file(sidecar),'bytes':sidecar.stat().st_size}
        del model
        model=None
        torch.cuda.empty_cache()
        outputs=B.verify_products('beats',[by_path[p] for p in inputs],output)
        for p in inputs:
            if sha_file(p) != by_path[p]['standardized_file_sha256']: raise ValueError('Input changed during replay')
        if preexisting != {p:sha_file(p) for p in preexisting} or sha_file(log_path) != log_before:
            raise ValueError('Existing shard outputs or log changed')
        preserve_json(receipt_path,dict(status='passed',stage='beats',shard_index=index,run_contract_sha256=contract_sha,
            item_ids=[by_path[p]['item_id'] for p in inputs],gpu=gpu,
            command=B.command('beats',inputs,runtime,output),outputs={**outputs,**sidecars},
            log_sha256=log_before,completed_utc=B.now(),
            recovery_status='same_model_replay_verified_empty_unavailable',recovery_code_sha256=sha_file(__file__),
            recovery_sidecars=list(sidecars),preserved_existing_sha256=preexisting))
        recovered.append(index)
    if not recovered:
        raise ValueError('No eligible logged failure recovered')
    print(json.dumps({'status':'recovered_empty_unavailable','shards':recovered,'classifier_fitting':False}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-dir',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=5)
    a=p.parse_args()
    with (a.output_root/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        recover(a.prepared_dir,a.output_root,a.gpu)


if __name__ == '__main__': main()
