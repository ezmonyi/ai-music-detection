"""YuE2 extension: unchanged native30 neural commands and product checks.

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

RC = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
RD = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
sys.path.insert(0, str(RC / 'code'))
import run_native30_inference_batches_v1 as core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['allinone', 'beats'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    assert core.digest(RC/'code/run_native30_inference_batches_v1.py') == '84dce45426596888b3a4e3b20e4ad1c61dc976645506f269c7ec7990b12f3933'
    freeze = core.read_json(RC/'preregistration/native30_inference_parent_freeze_v2.json')
    runtime_checks = core.verify_runtime(freeze)
    source = RD/'native30_fhsc_v1'
    commit = core.read_json(source/'COMMIT.json')
    rows = []
    for uid, expected in sorted(commit['items'].items()):
        receipt = source/'items'/(uid+'.json')
        assert core.digest(receipt) == expected
        r = core.read_json(receipt)
        assert r['id'] == uid and core.digest(r['view_path']) == r['view_sha256']
        core.inspect_audio(Path(r['view_path']), input_audio=True)
        rows.append({'id': uid, 'input': core.binding(Path(r['view_path']))})
    assert len(rows) == 498
    out = RD/'native30_neural_v1'
    out.mkdir(exist_ok=True)
    lock = (out/(args.stage+'.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for folder in ['logs', 'receipts']:
        (out/folder).mkdir(exist_ok=True)
    contract = dict(stage=args.stage, gpu=args.gpu, rows=rows,
                    source_commit=core.binding(source/'COMMIT.json'),
                    driver=core.binding(Path(__file__).resolve()),
                    runtime_checks=runtime_checks, command_policy='unchanged native30 v1',
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
            actual = core.verify_products(args.stage, batch, out)
            assert prior['products'] == actual
            products.update(actual)
            continue
        log = out/'logs'/f'{args.stage}_{index:03d}.log'
        command = core.command(args.stage, [r['input']['path'] for r in batch],
                               Path(freeze['runtime_root']), out)
        print(json.dumps({'stage':args.stage, 'batch':index, 'count':len(batch), 'event':'start'}), flush=True)
        with log.open('xb') as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        actual = core.verify_products(args.stage, batch, out)
        core.write_new(receipt, dict(status='passed', products=actual,
                                     command=command, log=core.binding(log),
                                     contract_sha256=core.digest(path)))
        products.update(actual)
        print(json.dumps({'stage':args.stage, 'completed':offset+len(batch)}), flush=True)
    core.write_new(out/(args.stage+'_COMMIT.json'), dict(status='completed_verified_neural_products',
                   rows=len(rows), products=products, contract=core.binding(path), classifier_fits=0))


if __name__ == '__main__':
    main()
