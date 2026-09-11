"""Bind existing authorities, assemble BC package, then preflight; never fit."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

RC = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
RD = Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907')
sys.path.insert(0, str(RC / 'code'))
import prepare_native30_evaluation_inputs_bc_v1 as adapter


def main():
    audit = RC / 'audit/native30_bc_v1_independent_receipt_20260911_r6.json'
    assert hashlib.sha256(audit.read_bytes()).hexdigest() == 'c2c2c7a1ba70b7afbe5c766386b872a54e1deb96b5c2463ff0ceb1107bc021cc'
    proof = json.loads(audit.read_text())
    assert proof['passed'] is True and proof['rows_replayed'] == 3830
    refs = {
        'parent_commit': (RD/'native30_evaluation_package_v3_20260911/COMMIT.json', '361d499824cf1340ef5b65073685bbf2b981002545ee80485fe07374e1607863'),
        'bc_freeze': (RC/'preregistration/native30_bc_parent_freeze_v1.json', 'c7fcf8cdf2d3bb0e4673a7aa6b2e3468a67e31cd0e7fc4d4ec30575c9d874ea0'),
        'bc_commit': (RD/'native30_bc_v2/COMMIT.json', '573293da2c4159a5a30c204b08881eea86c2137bfce352fd6db3acf8765f398e'),
        'schedule_draft': (RD/'native30_bc_evaluation_schedule_draft_v1/schedule_draft.json', '0ec6fd6ec218dae003a4007a6c39e435923490736a632397c922562891615fb8'),
        'schedule_commit': (RD/'native30_bc_evaluation_schedule_draft_v1/COMMIT.json', '92d094fc2a436ba9661f97bd0d0a8c56536e5536000304ab5513514122aeebc1'),
    }
    request = {k: adapter.binding(p, sha) for k, (p, sha) in refs.items()}
    path = RC/'preregistration/native30_bc_package_request_20260911.json'
    raw = adapter.canonical(request)
    if path.exists():
        assert path.read_bytes() == raw
    else:
        with path.open('xb') as f: f.write(raw)
    package = RD/'native30_bc_evaluation_package_v1_20260911'
    cmd = [sys.executable, str(RC/'code/prepare_native30_evaluation_inputs_bc_v1.py'), '--mode', 'assemble', '--request', str(path), '--request-sha256', hashlib.sha256(raw).hexdigest(), '--output', str(package)]
    if not (package/'COMMIT.json').exists():
        with (RC/'logs/native30_bc_package_20260911.log').open('xb') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    sha = hashlib.sha256((package/'COMMIT.json').read_bytes()).hexdigest()
    print(json.dumps({'stage': 'package_complete', 'commit_sha256': sha}), flush=True)
    with (RC/'preregistration/native30_bc_evaluation_preflight_20260911.json').open('xb') as out, (RC/'logs/native30_bc_preflight_20260911.log').open('xb') as err:
        subprocess.run([sys.executable, str(RC/'code/evaluate_native30_bc_v1.py'), '--package', str(package), '--package-commit-sha256', sha, '--output', str(RD/'native30_bc_evaluation_v1_20260911'), '--mode', 'preflight'], stdout=out, stderr=err, check=True)
    print('Preflight complete; separate root review and fitting freeze still required.', flush=True)


if __name__ == '__main__': main()
