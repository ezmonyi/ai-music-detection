"""Reconcile exact processed objects with accepted public and local-private delivery."""
import hashlib
import json
from collections import Counter
from reconcile_processed_delivery_v1 import REPORT, ACCEPTED

OUT = REPORT / 'datasets/processed_delivery_reconciliation_v2'


def main():
    assert not OUT.exists()
    source = REPORT / 'datasets/historical_test_view_objects_v1/objects.json'
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'a56c71a3ddce44a71563c8023f4a64906043f09d81132f7aa6dcc235a9a96a92'
    objects = json.loads(raw)
    accepted, bindings = {}, {}
    names = ACCEPTED + ['hf_open_model_test_views_v1', 'hf_fma_test_views_v1',
                        'hf_aime_test_views_v1', 'hf_suno_test_views_v1']
    for name in names:
        folder = REPORT / 'current_results' / name
        receipt = json.loads((folder / 'INDEPENDENT_ACCEPTANCE.json').read_text())
        manifest = (folder / 'manifest.json').read_bytes()
        assert receipt['final_acceptance'] and hashlib.sha256(manifest).hexdigest() == receipt['manifest_sha256']
        rows = json.loads(manifest)
        assert len(rows) == receipt['verified_audio_objects']
        bindings[name] = receipt
        for row in rows:
            assert row['sha256'] not in accepted
            accepted[row['sha256']] = dict(bytes=row['bytes'], path=row['path'], revision=receipt['revision'])
    folder = REPORT / 'private_rights_hold_4484_v1'
    local_receipt = json.loads((folder / 'LOCAL_ACCEPTANCE.json').read_text())
    assert local_receipt['status'] == 'local_archive_all_audio_sha256_verified'
    raw = (folder / 'manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == local_receipt['manifest_sha256']
    private = {r['sha256']: r for r in json.loads(raw)}
    assert len(private) == local_receipt['objects'] == 4484
    assert not set(private).intersection(accepted)
    counts, records, used_public, used_private = Counter(), [], set(), set()
    for obj in objects:
        key = obj['sha256']
        if obj['public_paths']:
            status = 'public_snapshot_matched'
        elif key in accepted:
            assert accepted[key]['bytes'] == obj['bytes']
            status = 'public_supplement_accepted'; used_public.add(key)
        elif key in private:
            assert private[key]['bytes'] == obj['bytes']
            status = 'local_private_archive_accepted'; used_private.add(key)
        else:
            status = 'unresolved'
        counts[status] += 1
        records.append(dict(sha256=key, bytes=obj['bytes'], status=status,
            public_receipt=accepted.get(key), local_member=private.get(key, {}).get('archive_path')))
    assert used_public == set(accepted) and used_private == set(private)
    assert counts == {'public_snapshot_matched': 966, 'public_supplement_accepted': 8794,
                      'local_private_archive_accepted': 4484}
    OUT.mkdir()
    raw = (json.dumps(records, separators=(',', ':')) + '\n').encode()
    (OUT / 'records.json').write_bytes(raw)
    summary = dict(objects=len(objects), counts=dict(counts), records_sha256=hashlib.sha256(raw).hexdigest(),
        public_bindings=bindings, local_binding=local_receipt,
        scope='Frozen 14,244 historical audio_path byte objects only. Snapshot matches are historical; stems and other scopes remain separate.',
        preservation_coverage_complete_for_this_scope=True, whole_project_complete=False)
    (OUT / 'COMMIT.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(objects=len(objects), counts=dict(counts))))


if __name__ == '__main__':
    main()
