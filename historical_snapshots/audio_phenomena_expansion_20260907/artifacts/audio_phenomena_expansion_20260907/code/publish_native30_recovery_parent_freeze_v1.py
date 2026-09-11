"""Publish the exact root-reviewed recovery authority, without running models."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRAFT_SHA = '1c1397cec1605aa92f56c78c87e87c4c0ad97c734426a02aad077145bbb59831'
AUDIT_SHA = '9c98d46b8004e038be651f3a797749a40e87d946c5e3716e097efcf77d224a69'


def checked(path, sha):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('reviewed artifact changed: ' + str(path))
    return json.loads(raw)


def main():
    driver = ROOT / 'code/continue_native30_empty_beats_v1.py'
    if hashlib.sha256(driver.read_bytes()).hexdigest() != '2e10ff670964a3dc6840c858565fb2056d09ee067a3fcd9ed3334d3c3dcb60a8':
        raise ValueError('approved driver changed')
    import continue_native30_empty_beats_v1 as r
    draft = checked(ROOT / 'preregistration/native30_recovery_parent_draft_v1.json', DRAFT_SHA)
    audit = checked(ROOT / 'audit/native30_recovery_eligibility_inventory_v1.json', AUDIT_SHA)
    r.require(audit['anomalies'] == [] and audit['writer_lock_before_after_identical'] is True
              and audit['snapshot_files'] == 3076 and audit['snapshot_bytes'] == 7219515866,
              'reviewed inventory requirements')
    r.require(r.b.value_hash(draft['initial_partial_inventory']) == audit['snapshot_sha256']
              and draft['policy'] == r.POLICY and draft['version'] == r.FREEZE_VERSION
              and draft['status'] == 'draft_requires_parent_review_not_execution_authorized',
              'reviewed draft requirements')
    failed = [s for s in audit['shards'] if s['status'].startswith('candidate_')]
    r.require(len(failed) == 1 and failed[0]['index'] == 48 and failed[0]['missing_ids'] ==
              ['aime_mtg_jamendo_06003', 'aime_mtg_jamendo_06005'], 'reviewed failure boundary')
    frozen = {**draft, 'status': r.FREEZE_STATUS}
    output = ROOT / 'preregistration/native30_recovery_parent_freeze_v1.json'
    r.b.write_new(output, frozen)
    print(r.b.canonical({'status': 'parent_freeze_published_no_models_run',
                         'freeze': r.b.binding(output), 'draft_sha256': DRAFT_SHA,
                         'inventory_audit_sha256': AUDIT_SHA}).decode().strip())


if __name__ == '__main__':
    main()
