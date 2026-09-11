"""Publish the reviewed operational authority from explicitly pinned metadata.

Does not launch inference. The caller must review the completed draft and audit
before supplying their exact SHA256 values. Existing targets are never replaced.
"""
import argparse
from pathlib import Path

import resume_native30_gpu_guard_v2 as s

ROOT = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
DRIVER_SHA = '102875006494669f289e8501e66f37d276385f9f371a0de40815d37a718e4e11'
TEST_SHA = '56357e706d14d57bb92e69aa32a07123098750f30fad4b4f34145dd8fd3b149e'
BUILDER_SHA = 'bf2e01ba46a06924fc4a123e99315a3b94ea707953925bdbd28985e0eddff7db'
SNAPSHOT_SHA = '17db12d2dece3a98c3e9837748e703df48d2cf52cca58db657b4ce18ae124e02'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft-sha256', required=True)
    parser.add_argument('--audit-sha256', required=True)
    args = parser.parse_args()
    b, require = s.b, s.require
    driver = ROOT / 'code/resume_native30_gpu_guard_v2.py'
    require(Path(s.__file__).resolve() == driver, 'unexpected driver origin')
    b.require_hash(driver, DRIVER_SHA)
    draft_path = ROOT / 'preregistration/native30_gpu_guard_parent_draft_v2.json'
    audit_path = ROOT / 'audit/native30_gpu_guard_draft_inventory_v2.json'
    b.require_hash(draft_path, args.draft_sha256)
    b.require_hash(audit_path, args.audit_sha256)
    draft, audit = b.read_json(draft_path), b.read_json(audit_path)
    require(draft['version'] == s.FREEZE_VERSION and
            draft['status'] == 'draft_nonauthorizing_pending_parent_publication', 'draft status')
    require(draft['driver'] == b.binding(driver) and draft['tests']['sha256'] == TEST_SHA
            and draft['base_freeze']['sha256'] == s.BASE_FREEZE_SHA, 'reviewed authority pins')
    require(draft['snapshot_audit'] == b.binding(audit_path) and
            draft['snapshot_builder'] == audit['builder'] and
            audit['builder']['sha256'] == BUILDER_SHA, 'reviewed snapshot builder')
    require(audit['status'] == 'readonly_stopped_snapshot_for_nonauthorizing_operational_draft'
            and audit['snapshot_sha256'] == SNAPSHOT_SHA and audit['files'] == 6887
            and audit['bytes'] == 16846709529 and audit['unmatched_intents'] == ['beats_096.json']
            and audit['locks']['unchanged_at_end'] is True
            and audit['locks']['mode'] == 'exclusive_nonblocking_no_writes'
            and audit['inference_launched'] is False and audit['freeze_authorized'] is False,
            'stopped snapshot audit mismatch')
    original, historical = draft['stopped_original_inventory'], draft['stopped_recovery_inventory']
    require(not set(original).intersection(historical) and
            b.value_hash({**original, **historical}) == SNAPSHOT_SHA, 'snapshot maps mismatch')
    require(draft['placement'] == s.PLACEMENT and draft['policy'] == s.r.POLICY
            and all(draft.get(k) == v for k, v in s.r.SCOPE.items()), 'scientific scope changed')
    require(draft['output_root'] == '/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/native30_gpu_guard_resume_v2',
            'unexpected operational output root')
    for key in ('base_freeze', 'base_driver', 'driver', 'tests', 'terminal_log', 'blocked_intent', 'snapshot_builder'):
        b._binding_matches(draft[key], audio=True)
    target = ROOT / 'preregistration/native30_gpu_guard_parent_freeze_v2.json'
    b.write_new(target, {**draft, 'status': s.FREEZE_STATUS})
    s.setup_context(target, b.digest(target), audio=False)
    print(b.canonical({'status': 'parent_authority_published_no_inference', 'freeze': b.binding(target)}).decode())


if __name__ == '__main__':
    main()

