"""Materialize selected original audio bytes for publication without decoding."""
import collections
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
AUDIT=ROOT/'aime_generated_byte_audit_v1'
SOURCE=Path('/mnt/nfs-code/users/yi/external_generator_500_testset_20260904/source/aime_parquet/default/train')
OUT=ROOT/'aime_originals_publication_v1'


def main():
    commit=json.loads((AUDIT/'COMMIT.json').read_text())
    raw=(AUDIT/'verified_rows.jsonl').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==commit['verified_rows_sha256']
    assert commit['files']==5000 and commit['status']=='all_selected_original_bytes_verified'
    rows=[json.loads(line) for line in raw.splitlines()]
    shards=collections.defaultdict(dict)
    for r in rows:shards[r['shard']][r['row']]=r
    OUT.mkdir(exist_ok=False);(OUT/'audio').mkdir()
    manifest=[]
    for shard,selection in sorted(shards.items()):
        seen=set()
        for index,batch in enumerate(pq.ParquetFile(SOURCE/shard).iter_batches(batch_size=1,columns=['audio'])):
            if index not in selection:continue
            r=selection[index];data=batch.to_pylist()[0]['audio']['bytes']
            assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
            # Preserve actual container bytes; do not guess format from source filenames.
            extension='.wav' if data[:4]==b'RIFF' and data[8:12]==b'WAVE' else '.flac' if data[:4]==b'fLaC' else '.bin'
            filename=r['id']+extension
            assert Path(filename).name==filename
            with (OUT/'audio'/filename).open('xb') as stream:stream.write(data)
            manifest.append(dict(r,filename=filename,path='audio/aime_generated_originals_v1/'+filename,
                license='CC-BY-4.0',license_basis='AIME publisher generated-audio declaration; excludes MTG human subset',
                source_dataset='disco-eth/AIME',source_parquet_revision='1bdacac93127439e361bdd19d575d8b596bca4e3',
                modification='None; original embedded audio bytes, renamed to experiment ID'))
            seen.add(index)
        assert seen==set(selection)
        print(f'materialized {len(manifest)}/5000',flush=True)
    assert len(manifest)==5000
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'README.md').write_text('''# AIME generated originals selected for this project

5,000 original recordings, 500 each from ten AIME generator labels. Source:
disco-eth/AIME, https://huggingface.co/datasets/disco-eth/AIME . Cite the AIME
authors and Benchmarking Music Generation Models and Metrics via Human Preference
Studies, https://ieeexplore.ieee.org/abstract/document/10887745 . The publisher
explicitly licenses its generated audio under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/ . This relies on its declaration,
not independent clearance of every underlying service or potential third-party
right. No endorsement is implied. All files retain original embedded bytes.

The MTG-labelled human subset is excluded: its per-track licenses are separate.
Prompt description tags have separate terms and are not reproduced here. The
manifest preserves generator labels, raw hashes, source shard/row and stable
experiment IDs. Historical roles/groups and derived-view hashes are available
in the project's separate selected-generated catalogue; do not infer a new
untouched test split from publication. This planned manifest does not prove all
audio arrived: use batch receipts and terminal verification for upload status.
''')
    (OUT/'COMMIT.json').write_text(json.dumps(dict(status='original_bytes_materialized_not_uploaded',files=5000,
        bytes=sum(r['bytes'] for r in manifest),
        audit_commit_sha256=hashlib.sha256((AUDIT/'COMMIT.json').read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
