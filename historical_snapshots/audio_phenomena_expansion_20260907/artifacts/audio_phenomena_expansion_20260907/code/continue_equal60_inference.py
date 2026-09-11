#!/usr/bin/env python3
"""Resume original audited inference; recover only reviewed empty-beat failures.

All classifier fitting remains outside this operational supervisor.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import fcntl

from materialize_equal60_inputs_v2 import sha_file, preserve_json


def commands():
    root=Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
    bulk=Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/equal60_development_v2')
    runtime='/mnt/nfs-code/users/yi/dynamics_rhythm_external_benchmark_20260903'
    old='/mnt/nfs-code/users/yi/source_diversity_expansion_20260905/code'
    shared='/mnt/nfs-data/users/yi/source_diversity_expansion_20260905/results'
    common=['--runtime-contract',shared+'/inference_runtime_contract.json',
            '--checkpoint-manifest',shared+'/inference_checkpoint_manifest.json',
            '--old-code-root',old,
            '--bias','/mnt/nfs-code/users/yi/demucs_bias_corrected_1000_20260901/demucs_frequency_bias.npz']
    infer=[sys.executable,'-u',str(root/'code/run_equal60_inference_batches.py'),
        '--prepared-dir',str(bulk),'--output-root',str(bulk/'inference'),
        '--materialization-audit',str(root/'audit/equal60_development_materialization_v2.json'),
        '--runtime-root',runtime,*common]
    recover=[sys.executable,'-u',str(root/'code/recover_equal60_empty_beats_v2.py'),
        '--prepared-dir',str(bulk),'--output-root',str(bulk/'inference'),'--gpu','5']
    extract=[sys.executable,'-u',str(root/'code/verify_extract_equal60.py'),
        '--prepared-dir',str(bulk),'--inference-root',str(bulk/'inference'),
        '--output-dir',str(root/'results/equal60_full_validation_v1'),*common,'--extract','--workers','8']
    return root,bulk,infer,recover,extract


def main():
    root,bulk,infer,recover,extract=commands()
    code_hashes={p:sha_file(p) for p in (infer[2],recover[2],extract[2],__file__)}
    contract=dict(code_sha256=code_hashes,infer=infer,recover=recover,extract=extract,
        recovery_limit=67,classifier_fitting_authorized=False,
        original_inference_contract_sha256=sha_file(bulk/'inference/run_contract.json'))
    with (bulk/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        preserve_json(bulk/'inference_continuation_v1.json',contract)
        for attempt in range(68):
            if code_hashes != {p:sha_file(p) for p in code_hashes}:
                raise ValueError('Supervisor/inference/recovery/extraction code changed')
            print('original_runner_resume_attempt',attempt,flush=True)
            with (root/'logs/equal60_full_inference_v1.log').open('a') as log:
                result=subprocess.run(infer,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode == 0:
                break
            if attempt == 67:
                raise RuntimeError('Finite recovery bound exhausted')
            print('runner_stopped_checking_only_reviewed_empty_beat_case',flush=True)
            with (root/'logs/equal60_empty_beats_recovery_v1.log').open('a') as log:
                subprocess.run(recover,stdout=log,stderr=subprocess.STDOUT,check=True)
        completion=json.loads((bulk/'inference/completion.json').read_text())
        if completion['status'] != 'passed' or completion['rows'] != 1604 or completion['completed_stage_shards'] != 134:
            raise ValueError('No complete inference proof')
        print('all134_stage_shards_verified_start_strict_audit_and_extraction',flush=True)
        with (root/'logs/equal60_full_validation_v1.log').open('a') as log:
            subprocess.run(extract,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('strict_audit_and_old_features_complete_no_classifier_fitted',flush=True)


if __name__=='__main__': main()
