"""YuE2 native60: frozen neural commands with original exact60 product checks.

Independent stage processes use GPUs 6 and 7. No fitting, imputation, or
scientific-missingness recovery is performed here. Failed logs are retained.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RC = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
RD = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
sys.path.insert(0, str(RC / 'code'))
import run_native30_inference_batches_v1 as core
import run_equal60_inference_batches as exact60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['allinone', 'beats'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--input-audit', type=Path, required=True)
    parser.add_argument('--input-audit-sha256', required=True)
    args = parser.parse_args()
    assert core.digest(RC/'code/run_native30_inference_batches_v1.py') == '84dce45426596888b3a4e3b20e4ad1c61dc976645506f269c7ec7990b12f3933'
    assert core.digest(RC/'code/run_equal60_inference_batches.py') == 'a33f667ac9ef5cc5b4e5e646adfb752a68380b1af5f42b0768b1a10e3d031858'
    assert core.digest(RC/'code/verify_extract_equal60.py') == 'c921bc04988f58a382efea080df2c5493b1d4faca3e48bc38797b59981316771'
    assert core.digest(args.input_audit) == args.input_audit_sha256
    audit = core.read_json(args.input_audit)
    assert audit['status'] == 'independent_native60_membership_and_full_decode_crop_verified'
    assert audit['originals_verified'] == 500 and audit['crops_verified'] == 276
    freeze = core.read_json(RC/'preregistration/native30_inference_parent_freeze_v2.json')
    runtime_checks = core.verify_runtime(freeze)
    source = RD/'native60_inputs_v1'
    commit = core.read_json(source/'COMMIT.json')
    assert core.digest(source/'COMMIT.json') == audit['source_commit_sha256']
    assert commit['status'] == 'native60_inputs_only_no_features_or_inference'
    assert core.digest(source/'metadata.json') == commit['products']['metadata.json']['sha256']
    out = RD/'native60_neural_v1'
    out.mkdir(exist_ok=True)
    lock = (out/(args.stage+'.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    views = out/('inputs_'+args.stage)
    views.mkdir(exist_ok=True)
    rows = []
    for r in core.read_json(source/'metadata.json'):
        path = Path(r['input_path'])
        assert core.digest(path) == r['input_sha256']
        audio, sr = sf.read(path,dtype='float64',always_2d=True)
        assert sr == 48000 and audio.shape == (2880000,2) and np.isfinite(audio).all()
        wave = resample_poly(audio,147,160,axis=0,window=('kaiser',5.0),padtype='constant').astype('<f4')
        assert wave.shape == (2646000,2) and np.isfinite(wave).all()
        path = views/(r['id']+'.wav')
        if not path.exists():
            sf.write(path,wave,44100,format='WAV',subtype='FLOAT')
        replay, rate = sf.read(path,dtype='float32',always_2d=True)
        assert rate == 44100 and sf.info(path).subtype == 'FLOAT' and np.array_equal(replay,wave)
        rows.append({'id': r['id'], 'item_id': r['id'], 'standardized_path': str(path),
                     'input': core.binding(path)})
    assert len(rows) == 276
    for folder in ['logs', 'receipts']:
        (out/folder).mkdir(exist_ok=True)
    contract = dict(stage=args.stage, gpu=args.gpu, rows=rows,
                    source_commit=core.binding(source/'COMMIT.json'),
                    driver=core.binding(Path(__file__).resolve()),
                    runtime_checks=runtime_checks, command_policy='unchanged exact60 commands; stereo FLOAT 44100Hz views',
                    input_resampling='float64 scipy resample_poly 147/160 axis0 kaiser5 constant then float32; no gain or DC change',
                    input_audit=core.binding(args.input_audit),
                    classifier_fits=0, recovery_applied=False)
    path = out/(args.stage+'_contract.json')
    if path.exists():
        assert core.read_json(path) == contract
    else:
        core.write_new(path, contract)
    env = dict(freeze['child_environment']['exact_base'])
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # Task-owned temporary storage avoids the nearly full system disk.
    tmp = out/('tmp_'+args.stage)
    tmp.mkdir(exist_ok=True)
    env['TMPDIR'] = str(tmp)
    core.require_idle_5090(args.gpu)
    products = {}
    for index, offset in enumerate(range(0, len(rows), 25)):
        batch = rows[offset:offset+25]
        receipt = out/'receipts'/f'{args.stage}_{index:03d}.json'
        if receipt.exists():
            prior = core.read_json(receipt)
            assert prior['status'] == 'passed' and prior['contract_sha256'] == core.digest(path)
            actual = exact60.verify_products(args.stage, batch, out)
            assert prior['products'] == actual
            products.update(actual)
            continue
        log = out/'logs'/f'{args.stage}_{index:03d}.log'
        command = exact60.command(args.stage, [r['input']['path'] for r in batch],
                               Path(freeze['runtime_root']), out)
        print(json.dumps({'stage':args.stage, 'batch':index, 'count':len(batch), 'event':'start'}), flush=True)
        assert not any(p.exists() for r in batch for p in exact60.product_paths(args.stage,r['id'],out)), 'Unreceipted partial outputs require review'
        with log.open('xb') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        actual = exact60.verify_products(args.stage, batch, out)
        core.write_new(receipt, dict(status='passed', products=actual,
                                     command=command, log=core.binding(log),
                                     contract_sha256=core.digest(path)))
        products.update(actual)
        print(json.dumps({'stage':args.stage, 'completed':offset+len(batch)}), flush=True)
    for row in rows:
        assert core.binding(Path(row['input']['path'])) == row['input']
    core.write_new(out/(args.stage+'_COMMIT.json'), dict(status='completed_verified_neural_products',
                   rows=len(rows), products=products, contract=core.binding(path), classifier_fits=0))


if __name__ == '__main__':
    main()
