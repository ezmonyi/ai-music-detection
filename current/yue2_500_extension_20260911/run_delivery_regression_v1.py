"""Run six offline delivery suites and preserve an explicit, scoped receipt."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

SUITES = ['test_native60_score_kernel_v1.py', 'test_publication_backoff_v1.py',
    'test_500_publication_receipts_v1.py', 'test_yue2_publication_helpers_v1.py',
    'test_processed_upload_backoff_v1.py', 'test_external_maestro_publication_v1.py']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists; preserve previous receipt')
    rows = []
    for name in SUITES:
        r = subprocess.run([sys.executable, '-m', 'unittest', 'discover',
            '-s', str(a.source), '-p', name, '-v'], capture_output=True, text=True)
        rows.append(dict(suite=name, returncode=r.returncode,
            stdout=r.stdout, stderr=r.stderr))
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(a.source.glob('*.py'))}
    receipt = dict(python=platform.python_version(),
        huggingface_hub=importlib.metadata.version('huggingface_hub'),
        suites=rows, source_python_sha256=hashes,
        passed=all(r['returncode'] == 0 for r in rows),
        scope='Six offline unit suites only; not live upload or end-to-end music inference',
        whole_project_complete=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(dict(passed=receipt['passed'], suites=len(rows))))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
