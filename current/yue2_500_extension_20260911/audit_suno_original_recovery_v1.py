"""Reconcile selected Suno originals with historical crop/source metadata."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
METADATA=ROOT/'historical_metadata_10s_for_source_audit.csv'
MANIFEST=Path('/mnt/nfs-code/users/yi/demucs_bias_corrected_1000_20260901/manifest.jsonl')
RECOVERY=Path('/mnt/nfs-code/users/yi/macbook_reproducibility_20260909/workspace')
OUT=ROOT/'suno_original_recovery_audit_v1'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    assert sha(METADATA)=='a03aec930130fcb942c40b35ba5fb49109ceee588894124010448f041d9d1ef4'
    history={r['id']:r for r in csv.DictReader(METADATA.open()) if r['source_group']=='Suno'}
    source=[json.loads(line) for line in MANIFEST.read_text().splitlines()]
    selected=[r for r in source if r['source'] in ('humair_suno','suno_unknown')]
    assert len(selected)==len(history)==500
    assert {'ai_'+r['source']+'_'+r['id'] for r in selected}==set(history)
    assert all(sum(r['source']==s for r in selected)==250 for s in ('humair_suno','suno_unknown'))
    OUT.mkdir(exist_ok=False)
    records=[]
    for index,r in enumerate(selected):
        identifier='ai_'+r['source']+'_'+r['id'];h=history[identifier]
        path=RECOVERY/r['path']
        allowed='humair025-suno-audio-mp3' if r['source']=='humair_suno' else 'suno-ai-music-dataset-audio'
        assert path.is_relative_to(RECOVERY/allowed) and '..' not in path.parts
        item=dict(id=identifier,source_collection=r['source'],source_relative_path=r['path'],creator=r.get('creator'),
            model_name=r['model_name'],source_duration_recorded_s=r['duration'],
            historical_crop_start_s=r['crop_start'],historical_group_id=h['group_id'],
            historical_role=h['role'],expected_original_sha256=h['raw_sha256'])
        try:
            item.update(original_bytes=path.stat().st_size,actual_original_sha256=sha(path))
            item['status']='verified' if item['actual_original_sha256']==h['raw_sha256'] else 'hash_mismatch'
        except FileNotFoundError:item['status']='missing'
        records.append(item)
        if (index+1)%100==0:print(f'Audited {index+1}/500 Suno original paths',flush=True)
    (OUT/'records.json').write_text(json.dumps(records,indent=2))
    counts={s:sum(r['status']==s for r in records) for s in sorted({r['status'] for r in records})}
    (OUT/'COMMIT.json').write_text(json.dumps(dict(status='selected_original_recovery_audit_completed',
        files=500,counts=counts,verified_bytes=sum(r['original_bytes'] for r in records if r['status']=='verified'),
        records_sha256=sha(OUT/'records.json'),historical_manifest_sha256=sha(MANIFEST),
        metadata_sha256=sha(METADATA),audio_uploaded=False,whole_project_complete=False),indent=2))
    print(json.dumps(counts),flush=True)


if __name__=='__main__':main()
