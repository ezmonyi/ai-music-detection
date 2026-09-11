"""Publish the reviewed, narrowly scoped v2 re-audit authority, not admission."""
from pathlib import Path
import hashlib
import importlib.util
import json

AUDITOR = 'c5f5ca2f64224a5114063d51dbb674d921ea4efe5ced9bc70488ad413daa5c41'
TESTS = 'c660fd001e5c795c9405c8e251405d8228ea6d66a03cb1d46972f9e5fedab09a'


def main():
    rc = Path(__file__).resolve().parent.parent
    source = rc / 'code/audit_bc_guitarset_reserved_admission_v2.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == AUDITOR
    spec = importlib.util.spec_from_file_location('reviewed_bc_v2', source)
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    own = r.binding(source, AUDITOR)
    tests = r.binding(rc / 'code/test_audit_bc_guitarset_reserved_admission_v2.py', TESTS)
    log = rc / 'logs/bc_reserved_auditor_v2_parent_tests.log'
    text = log.read_text()
    assert 'Ran 23 tests in 120.805s' in text and text.rstrip().endswith('OK')
    freeze, freeze_entry = r.read_json(rc / 'preregistration/bc_reserved_parent_freeze_v1.json', r.ORIGINAL_FREEZE_SHA)
    commit = r.binding(Path(freeze['output_root']) / 'COMMIT.json', r.ORIGINAL_COMMIT_SHA)
    original = r.binding(rc / 'code/audit_bc_guitarset_reserved_admission_v1.py', r.ORIGINAL_AUDITOR_SHA)
    failed = r.binding(rc / 'logs/bc_reserved_independent_actual_v1.log', r.ORIGINAL_FAILED_LOG_SHA)
    diagnosis = {key: r.binding(rc / key, value) for key, value in r.DIAGNOSIS_PINS.items()}
    authority = {'version': r.CORRECTION_VERSION,
        'status': 'authorized_independent_auditor_v2_replay_only_not_admission',
        'original_parent_freeze': freeze_entry, 'original_result_COMMIT': commit,
        'original_auditor': original, 'original_failed_audit_log': failed,
        'corrected_auditor': own, 'corrected_tests': tests,
        'diagnosis': diagnosis, 'scope': r.CORRECTION_SCOPE,
        'correction': r.CORRECTION_DESCRIPTION}
    r.a.check_file_bindings(authority)
    output = rc / 'preregistration/bc_reserved_auditor_correction_v1.json'
    with output.open('xb') as stream:
        stream.write(r.canonical(authority) + b'\n')
    proof = r.verify_correction_authority(output, r.digest(output), rc=rc,
        original_freeze=freeze_entry, original_commit=commit,
        original_auditor=original, own=own, tests=tests)
    print(json.dumps({'authority': proof['correction_authority'],
        'parent_tests': r.binding(log), 'BC_admitted': False}, sort_keys=True))


if __name__ == '__main__':
    main()
