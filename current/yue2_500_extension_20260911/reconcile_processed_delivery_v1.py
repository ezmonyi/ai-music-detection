"""Reconcile the frozen processed-object gap against accepted supplements."""
import hashlib
import json
from collections import Counter
from pathlib import Path

REPORT=Path('/Users/yi/Documents/report/music_ai_detection_20260912')
OUT=REPORT/'datasets/processed_delivery_reconciliation_v1'
ACCEPTED=['hf_diffrhythm_test_views_v1','hf_ishizaka_test_views_v1',
          'hf_maestro_historical_views_v1','hf_mtt_test_views_v1']


def main():
    assert not OUT.exists()
    raw=(REPORT/'datasets/historical_test_view_objects_v1/objects.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='a56c71a3ddce44a71563c8023f4a64906043f09d81132f7aa6dcc235a9a96a92'
    objects=json.loads(raw); accepted={}; bindings={}
    for name in ACCEPTED:
        p=REPORT/'current_results'/name
        receipt=json.loads((p/'INDEPENDENT_ACCEPTANCE.json').read_text())
        assert receipt['final_acceptance'] is True
        manifest=(p/'manifest.json').read_bytes()
        assert hashlib.sha256(manifest).hexdigest()==receipt['manifest_sha256']
        rows=json.loads(manifest)
        assert len(rows)==receipt['verified_audio_objects']
        bindings[name]=receipt
        for r in rows:
            assert r['sha256'] not in accepted
            accepted[r['sha256']]=(r['bytes'],r['path'],receipt['revision'])
    records=[]; counts=Counter(); byte_counts=Counter(); seen=set()
    for obj in objects:
        digest=obj['sha256']
        if obj['public_paths']: status='matched_at_original_snapshot'
        elif digest in accepted:
            assert obj['bytes']==accepted[digest][0]
            status='accepted_supplement'; seen.add(digest)
        else: status='not_yet_accepted'
        counts[status]+=1; byte_counts[status]+=obj['bytes']
        records.append(dict(sha256=digest,bytes=obj['bytes'],status=status,
            sources=sorted({m['source_group'] for m in obj['memberships']}),
            accepted_publication=accepted.get(digest)))
    assert seen==set(accepted) and len(seen)==1278
    OUT.mkdir()
    data=(json.dumps(records,indent=2)+'\n').encode()
    (OUT/'records.json').write_bytes(data)
    summary=dict(objects=len(objects),counts=dict(counts),bytes_by_status=dict(byte_counts),
        records_sha256=hashlib.sha256(data).hexdigest(),accepted_bindings=bindings,
        scope='Frozen historical audio_path objects only; running uploads not counted as accepted',
        whole_project_complete=False)
    (OUT/'COMMIT.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(objects=len(objects),counts=dict(counts),bytes_by_status=dict(byte_counts))))


if __name__=='__main__': main()
