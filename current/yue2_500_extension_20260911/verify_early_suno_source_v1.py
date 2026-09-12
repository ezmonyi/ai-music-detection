"""Verify early selected Suno bytes against a fixed public source revision."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi

ROOT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
REV='bdff424e70c10ef62dca13ba43659ad9e7e1fcdb'
REPO='Kukedlc/suno-ai-music-dataset'


def main():
    audit=ROOT/'early_original_audio_audit_v1';raw=(audit/'originals.json').read_bytes()
    commit=json.loads((audit/'COMMIT.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==commit['manifest_sha256']
    rows=[r for r in json.loads(raw) if r['memberships'][0]['source']=='suno_unknown']
    assert len(rows)==100
    expected={'audio/'+Path(r['relative_source_path']).name:r for r in rows}
    assert len(expected)==100
    api=HfApi(token=False);assert not api.dataset_info(REPO,revision=REV).private
    names=sorted(expected)
    for offset in range(0,100,50):
        entries=api.get_paths_info(REPO,paths=names[offset:offset+50],repo_type='dataset',revision=REV)
        assert {r.path for r in entries}==set(names[offset:offset+50])
        for entry in entries:
            row=expected[entry.path]
            assert entry.size==row['bytes'] and entry.lfs and entry.lfs.sha256==row['sha256']
        print(f'Public source bytes verified: {offset+50}/100',flush=True)
    receipt=dict(status='100_early_suno_originals_match_pinned_public_source',source_repo=REPO,
        source_revision=REV,verified_files=100,verified_bytes=sum(r['bytes'] for r in rows),
        local_audit_manifest_sha256=commit['manifest_sha256'],
        files=[dict(source_path=p,bytes=expected[p]['bytes'],sha256=expected[p]['sha256']) for p in names],
        project_audio_uploaded=False,historical_labels_changed=False)
    out=ROOT/'early_suno_source_verification_v1.json'
    with out.open('x') as stream:json.dump(receipt,stream,indent=2)
    print('Fixed-source verification complete; no project audio upload')


if __name__=='__main__':main()
