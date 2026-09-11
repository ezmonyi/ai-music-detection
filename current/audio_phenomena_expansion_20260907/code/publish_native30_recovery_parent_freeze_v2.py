"""Publish reviewed v2 authority without changing historical evidence."""
from pathlib import Path
import hashlib
import importlib

ROOT = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
DRIVER_SHA = '91f5fec0fb25b163f69d6b46e032a5f9b27fb7206f93ac7aa4afb4933ae0e6bf'
driver_path = ROOT / 'code/continue_native30_empty_beats_v2.py'
assert hashlib.sha256(driver_path.read_bytes()).hexdigest() == DRIVER_SHA
r = importlib.import_module('continue_native30_empty_beats_v2')
assert Path(r.__file__).resolve() == driver_path
b = r.b
draft_path = ROOT / 'preregistration/native30_recovery_parent_draft_v2.json'
audit_path = ROOT / 'audit/native30_recovery_v2_epoch_draft_inventory_v1.json'
b.require_hash(draft_path, '55932c925afa98ed1b3d6c689e4c6f98678a0a49dc1d7a6ba4c27241dab90ee4')
b.require_hash(audit_path, '6448067f9a2c4a7ea157b34a9b771ba598139e7e175bf5428a16601345b22ae5')
draft, audit = b.read_json(draft_path), b.read_json(audit_path)
assert draft['status'] == 'draft_requires_parent_review_not_execution_authorized'
assert draft['driver']['sha256'] == DRIVER_SHA
assert draft['tests']['sha256'] == 'd73cde748119301e2a878301e5ac0309ec0e291fc9f6577e359d8f466b6e7a6f'
assert audit['draft'] == b.binding(draft_path)
assert audit['both_exclusive_writer_locks_held_and_unchanged'] is True
for name, field, count, sha in (
    ('original', 'resume_original_inventory', 3790, '60c8bd87a9e1a3f4b2daae8859e21b8926962d7116995f9ac8a4143ad8e1b2ee'),
    ('continuation', 'resume_continuation_inventory', 22, '4d4f98fc8d7b714e32e54a384c107318da759a16b07dc98900fbaf3b85744cb3')):
    assert len(draft[field]) == count and b.value_hash(draft[field]) == sha
    assert audit['snapshot_counts'][name]['sha256'] == sha
assert all(draft['resume_original_inventory'][p] == entry for p, entry in draft['initial_partial_inventory'].items())
assert draft['previous_continuation_freeze']['sha256'] == '310bcb8093ffd82d4e6df143693df5626ec61540ddd683900b878fc57501b7ba'
assert draft['policy'] == r.POLICY and draft['environment_transition'] == r.ENVIRONMENT_TRANSITION
freeze = {**draft, 'status': r.FREEZE_STATUS}
target = ROOT / 'preregistration/native30_recovery_parent_freeze_v2.json'
b.write_new(target, freeze)
r.validate_freeze(target, b.digest(target), audio=False)
print(b.canonical(b.binding(target)).decode())
