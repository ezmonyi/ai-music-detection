"""Rehash earlier external benchmark inputs and match fixed public object hashes."""
import csv
import hashlib
import json
from pathlib import Path

BASE=Path('/mnt/nfs-code/users/yi/dynamics_rhythm_external_benchmark_20260903')
ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
OUT=ROOT/'early_external_audio_audit_v1'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    assert not OUT.exists()
    snapshot=ROOT/'public_audio_snapshot_20260913_v1'
    assert sha(snapshot/'files.json')=='72183ebaf1b1445b1560abe3252b40e7e105bcc2057713f6348be91b6653d07f'
    by_hash={}
    for row in json.loads((snapshot/'files.json').read_text()):
        by_hash.setdefault(row['sha256'],[]).append(row['path'])
    m=BASE/'manifests'; inputs=[m/name for name in ['maestro_transport_verification.json','maestro_recordings.csv','salami_selected.csv','salami_downloads.json']]
    verification=json.loads(inputs[0].read_text());assert verification['all_accepted'] and verification['n_files']==100
    maestros={r['audio_filename']:r for r in csv.DictReader(inputs[1].open())}
    salamis={r['song_id']:r for r in csv.DictReader(inputs[2].open())}
    downloads=json.loads(inputs[3].read_text())['records']
    assert len(maestros)==50 and len(salamis)==len(downloads)==44
    records=[]
    for row in verification['records']:
        assert row['accepted'] and row['size_match'] and row['crc32_match']
        path=BASE/'data/maestro/hf_transport'/row['relative_path']
        identity=row['relative_path'] if row['kind']=='audio' else str(Path(row['relative_path']).with_suffix('.wav'))
        credit=maestros[identity]
        records.append(dict(dataset='MAESTRO',id=credit['recording_id'],kind=row['kind'],path=str(path),
            expected_sha256=row['transport_sha256'],expected_bytes=row['transport_size'],
            source_url=row['official_archive'],source_member=row['official_archive_member'],
            title=credit['canonical_title'],composer=credit['canonical_composer']))
    for row in downloads:
        credit=salamis[row['song_id']]
        records.append(dict(dataset='SALAMI',id=row['song_id'],kind='audio',path=row['local_path'],
            expected_sha256=row['sha256'],expected_bytes=row['bytes'],source_url=row['source_url'],
            title=credit['title'],artist=credit['artist'],album=credit['album']))
    assert len(records)==144
    for index,row in enumerate(records):
        path=Path(row['path']);before=path.stat()
        actual=sha(path);after=path.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        assert actual==row['expected_sha256'] and after.st_size==row['expected_bytes']
        row['actual_sha256']=actual;row['public_paths']=by_hash.get(actual,[])
        if (index+1)%25==0:print(f'Checked {index+1}/144',flush=True)
    OUT.mkdir()
    data=(json.dumps(records,indent=2)+'\n').encode();(OUT/'records.json').write_bytes(data)
    summary=dict(status='144_external_inputs_rehashed',audio_files=94,midi_files=50,
        public_revision=json.loads((snapshot/'COMMIT.json').read_text())['revision'],
        matched_audio=sum(bool(r['public_paths']) for r in records if r['kind']=='audio'),
        matched_midi=sum(bool(r['public_paths']) for r in records if r['kind']!='audio'),
        bytes=sum(r['expected_bytes'] for r in records),records_sha256=hashlib.sha256(data).hexdigest(),
        source_manifest_sha256={p.name:sha(p) for p in inputs},whole_project_complete=False)
    (OUT/'COMMIT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
