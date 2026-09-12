"""Recover attribution by exact UUID from a pinned upstream metadata Parquet."""
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

BASE=Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911/early_humair_upstream_v1')
AUDIT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/early_original_audio_audit_v1')
PIN='c544c5274e19cd89a69d8afd3ad850649a055ffa3ec8a1c1a98b923d8900807f'


def main():
    path=BASE/'source.parquet'
    with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==PIN
    raw=(AUDIT/'originals.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==json.loads((AUDIT/'COMMIT.json').read_text())['manifest_sha256']
    selected={Path(r['relative_source_path']).stem:r for r in json.loads(raw) if r['memberships'][0]['source']=='humair_suno'}
    assert len(selected)==100
    columns=['id','title','display_name','handle','model_name','major_model_version','created_at','status']
    records={};duplicates=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=10000,columns=columns):
        for row in batch.to_pylist():
            if row['id'] not in selected:continue
            if row['id'] in records:
                duplicates.append(row['id'])
                if row not in records[row['id']]['metadata_variants']:
                    records[row['id']]['metadata_variants'].append(row)
                continue
            records[row['id']]=dict(id=row['id'],metadata_variants=[row],original_sha256=selected[row['id']]['sha256'],
                original_bytes=selected[row['id']]['bytes'],
                relative_source_path=selected[row['id']]['relative_source_path'],
                historical_memberships=selected[row['id']]['memberships'])
    missing=sorted(set(selected)-set(records))
    out=BASE/'recovered_metadata_v2';out.mkdir(exist_ok=False)
    encoded=json.dumps([records[k] for k in sorted(records)],ensure_ascii=False,indent=2).encode()
    (out/'records.json').write_bytes(encoded)
    result=dict(status='upstream_uuid_attribution_recovered' if not missing else 'partial_upstream_uuid_recovery',
        selected=100,recovered=len(records),missing_ids=missing,source_dataset='nyuuzyou/suno',
        source_revision='dd95495c415eea043c250f12da595de2ad4cad7f',source_parquet_sha256=PIN,
        records_sha256=hashlib.sha256(encoded).hexdigest(),
        duplicate_source_rows=len(duplicates),multi_variant_ids=sum(len(r['metadata_variants'])>1 for r in records.values()),
        empty_creator_count=sum(not any(v['display_name'] or v['handle'] for v in r['metadata_variants']) for r in records.values()),
        historical_metadata_equivalence_claimed=False,prompts_or_lyrics_exported=False,audio_uploaded=False)
    (out/'COMMIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
