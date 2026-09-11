"""Run math/acquisition/byte-verifier tests and preserve scoped provenance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    base = Path(__file__).resolve().parents[1]
    output = base / 'audit' / 'bc_development_checks_v1'
    output.mkdir(exist_ok=False)
    modules = ['test_bicoherence_primitive_v2', 'test_bicoherence_math_probe_v1',
               'test_acquire_guitarset_archives_v1', 'test_verify_committed_byte_mirror_v2']
    command = [sys.executable, '-B', '-W', 'error', '-m', 'unittest', *modules, '-v']
    run = subprocess.run(command, cwd=base / 'code', capture_output=True, text=True)
    (output / 'tests.stdout.txt').write_text(run.stdout)
    (output / 'tests.stderr.txt').write_text(run.stderr)
    if run.returncode:
        raise RuntimeError('development tests failed, output preserved')
    code = ['bicoherence_primitive_v2.py', 'bicoherence_math_probe_v1.py',
            'acquire_guitarset_archives_v1.py', 'verify_committed_byte_mirror_v2.py',
            *[name + '.py' for name in modules]]
    receipt = {'passed': True, 'test_count': 37, 'command': command,
        'code_sha256': {name: sha(base / 'code' / name) for name in code},
        'runner_sha256': sha(Path(__file__)), 'scope': 'math_and_offline_operational_tests_only',
        'audio_decode_validated': False, 'external_bc_gate_passed': False,
        'bc_classifier_admitted': False, 'returncode': run.returncode}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    products = {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in output.iterdir()}
    (output / 'COMMIT.json').write_text(json.dumps({'status': 'committed', 'products': products}, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
