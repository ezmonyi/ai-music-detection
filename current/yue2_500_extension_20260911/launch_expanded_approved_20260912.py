"""Launch the reviewed YuE2 contract once, without changing the evaluator."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

RC = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
RD = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
PY = '/mnt/nfs-code/users/yi/dynamics_rhythm_external_benchmark_20260903/venv/bin/python'

def main():
    raw = (RD / 'expanded_native30_evaluation_preflight_v1.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '8c1646e724ac020b07de79fc32ed44387085d70a94cb0e06017fff524b9e6e44'
    preflight = json.loads(raw)
    assert preflight['status'] == 'preflight_passed_no_fitting'
    assert preflight['contract_sha256'] == 'f50c854c9c525940f49a5ba5c9daa5e4b3e3ba5b58c345dc5dd4090a295b4d23'
    contract = preflight['contract']
    for key in ('driver', 'evaluator', 'plan_commit', 'feature_commit'):
        binding = contract[key]
        data = Path(binding['path']).read_bytes()
        assert len(data) == binding['bytes']
        assert hashlib.sha256(data).hexdigest() == binding['sha256']
    assert not (RD / 'expanded_native30_evaluation_v1/contract.json').exists()
    freeze = RC / 'expanded_native30_parent_freeze_20260912.json'
    payload = dict(fitting_authorized=True, contract=contract,
                   contract_sha256=preflight['contract_sha256'],
                   review='Exact preflight, four passing synthetic tests, development-only protocol reviewed; user requested remaining experiments to start.')
    with freeze.open('x') as handle:
        json.dump(payload, handle, sort_keys=True)
    digest = hashlib.sha256(freeze.read_bytes()).hexdigest()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    with (RC / 'expanded_evaluation_run_20260912.log').open('xb') as log:
        process = subprocess.Popen([PY, '-u', str(RC / 'evaluate_expanded_native30_v1.py'),
            '--mode', 'run', '--freeze', str(freeze), '--freeze-sha256', digest],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            env=env, cwd=RC, start_new_session=True)
    receipt = dict(pid=process.pid, freeze_sha256=digest, expected_models=112455)
    with (RC / 'expanded_evaluation_launch_20260912.json').open('x') as handle:
        json.dump(receipt, handle, sort_keys=True)
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
