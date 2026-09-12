"""Stream selected AIME originals from preserved parquet; no extraction/upload."""
import collections
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

BASE=Path('/mnt/nfs-code/users/yi/external_generator_500_testset_20260904')
OUT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/aime_generated_byte_audit_v1')


def main():
    manifest=BASE/'testset/manifest.jsonl'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='49dbe01b8be98857361b3a437db44ee937698e014bf70ccc7d443c24ad342fb4'
    selected=[r for r in map(json.loads,manifest.open()) if r['label']==1]
    assert len(selected)==5000
    shards=collections.defaultdict(dict)
    for r in selected:
        assert r['source_row_number'] not in shards[r['source_parquet']]
        shards[r['source_parquet']][r['source_row_number']]=r
    OUT.mkdir(exist_ok=False)
    count=0;total=0
    with (OUT/'verified_rows.jsonl').open('x') as stream:
        for shard,expected in sorted(shards.items()):
            assert Path(shard).name==shard
            seen=set()
            file=pq.ParquetFile(BASE/'source/aime_parquet/default/train'/shard)
            for index,batch in enumerate(file.iter_batches(batch_size=1,columns=['id','model','audio'])):
                if index not in expected:continue
                r=expected[index];row=batch.to_pylist()[0]
                assert row['id']==r['hf_original_id'] and row['model']==r['model']
                raw=row['audio']['bytes'];assert raw
                digest=hashlib.sha256(raw).hexdigest();assert digest==r['raw_sha256'],r['id']
                stream.write(json.dumps(dict(id=r['id'],model=r['model'],shard=shard,row=index,
                    sha256=digest,bytes=len(raw),source_filename=row['audio']['path']))+'\n')
                count+=1;total+=len(raw);seen.add(index)
            assert seen==set(expected)
            stream.flush();print(f'verified {count}/5000',flush=True)
    assert count==5000
    receipt=dict(status='all_selected_original_bytes_verified',files=count,bytes=total,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        verified_rows_sha256=hashlib.sha256((OUT/'verified_rows.jsonl').read_bytes()).hexdigest(),
        audio_extracted=False,audio_uploaded=False)
    (OUT/'COMMIT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
