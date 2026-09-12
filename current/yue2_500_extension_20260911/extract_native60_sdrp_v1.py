"""Consume exact60-verified YuE2 batches with the unchanged duration-aware SDRP extractor."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import argparse
import fcntl
import json
import math
import os
import sys
import time

RC = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
RD = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
sys.path.insert(0, str(RC/'code'))
# Avoid importing this driver under the same name as the pinned cohort module.
import importlib.util
spec = importlib.util.spec_from_file_location('_original_sdrp_adapter', RC/'code/extract_native30_sdrp_v1.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
core = adapter.physical
import run_equal60_inference_batches as exact60


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [clean(v) for v in value]
    return value


def live(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aio-pid', type=int, required=True)
    parser.add_argument('--beats-pid', type=int, required=True)
    args = parser.parse_args()
    freeze = core.read_json(RC/'preregistration/native30_inference_parent_freeze_v2.json')
    extractor = adapter.load_extractor(freeze['old_code_root'])
    bias_path = Path(freeze['bias']['path'])
    assert core.binding(bias_path) == freeze['bias']
    bias = extractor.load_bias(bias_path)
    source = RD/'native60_inputs_v1'
    neural = RD/'native60_neural_v1'
    out = RD/'native60_sdrp_v1'
    out.mkdir(exist_ok=True)
    (out/'items').mkdir(exist_ok=True)
    lock = (out/'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    commit = core.read_json(source/'COMMIT.json')
    assert commit['status'] == 'native60_inputs_only_no_features_or_inference'
    assert core.digest(source/'metadata.json') == commit['products']['metadata.json']['sha256']
    assert core.digest(RC/'code/run_equal60_inference_batches.py') == 'a33f667ac9ef5cc5b4e5e646adfb752a68380b1af5f42b0768b1a10e3d031858'
    for stage,pid in [('allinone',args.aio_pid),('beats',args.beats_pid)]:
        while not (neural/(stage+'_contract.json')).exists():
            if not live(pid):raise RuntimeError(f'{stage} stopped before publishing its contract')
            time.sleep(30)
    rows = []
    for row in core.read_json(source/'metadata.json'):
        uid=row['id']
        path=neural/'inputs_allinone'/(uid+'.wav')
        rows.append({'id':uid,'item_id':uid,'standardized_path':str(path),'input':core.binding(path),
                     'prompt_id':row['prompt_id'],'group_id':row['group_id'],'split':row['role']})
    assert len(rows) == 276
    contract = dict(driver=core.binding(Path(__file__).resolve()),
                    input_commit=core.binding(source/'COMMIT.json'),
                    extractor=core.binding(Path(extractor.__file__)),
                    bias=freeze['bias'], duration=60, classifier_fits=0,
                    inference_contracts={stage:core.binding(neural/(stage+'_contract.json'))
                                         for stage in ('allinone','beats')}, rows=rows)
    path = out/'contract.json'
    if path.exists():
        assert core.read_json(path) == contract
    else:
        core.write_new(path, contract)
    fingerprint = core.digest(path)
    opts = SimpleNamespace(duration=60,demix_root=[neural/'demix'],
                           beat_root=[neural/'beats'],structure_root=[neural/'structure'])

    def one(row):
        destination = out/'items'/(row['id']+'.json')
        if destination.exists():
            old = core.read_json(destination)
            assert old['contract_sha256'] == fingerprint and old['row'] == row
            return old
        mapped = dict(item_id=row['id'],source_id='YuE2',label='ai',
                      group_id=row['group_id'],standardized_path=row['input']['path'],
                      native_sample_rate_hz='48000',duration='60',audio_offset_s='0')
        assert core.binding(Path(row['input']['path'])) == row['input']
        result = extractor.process(mapped,opts,bias,freeze['bias']['sha256'],fingerprint)
        assert result['status'] == 'complete', result['errors']
        record = dict(row=row,contract_sha256=fingerprint,features=clean(result),
                      status='extracted_not_independently_audited_not_classifier_admitted')
        core.write_new(destination,record)
        return record

    for batch_index, offset in enumerate(range(0,len(rows),25)):
        batch = rows[offset:offset+25]
        for stage, pid in [('allinone',args.aio_pid),('beats',args.beats_pid)]:
            receipt = neural/'receipts'/f'{stage}_{batch_index:03d}.json'
            while not receipt.exists():
                if not live(pid):
                    raise RuntimeError(f'{stage} producer {pid} stopped without batch {batch_index}; retain outputs for diagnosis')
                time.sleep(30)
            evidence = core.read_json(receipt)
            assert evidence['status'] == 'passed'
            assert evidence['contract_sha256'] == contract['inference_contracts'][stage]['sha256']
            assert exact60.verify_products(stage,batch,neural) == evidence['products']
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(one,batch))
        print(f'SDRP completed {offset+len(batch)}/276',flush=True)
    core.write_new(out/'COMMIT.json',dict(status='completed_extraction_not_independent_audit',
                   rows=276,contract=core.binding(path),classifier_fits=0,
                   items={row['id']:core.binding(out/'items'/(row['id']+'.json')) for row in rows}))


if __name__ == '__main__':
    main()
