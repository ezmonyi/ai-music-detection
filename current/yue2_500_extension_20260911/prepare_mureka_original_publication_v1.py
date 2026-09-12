"""Bind all 500 Mureka originals to their frozen acquisition receipts."""
import hashlib
import json
from pathlib import Path

SOURCE=Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/music8k_mureka500_v1')
OUT=Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/mureka_original_publication_plan_v1')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    contract=SOURCE/'contract.json'
    assert sha(contract)=='e985adc956116851a7358f3ed9279241a6f92b65d28f22b0eb40e94023de8d24'
    summary=json.loads((SOURCE/'summary.json').read_text())
    assert summary['selected']==summary['passed']==500
    receipts=summary['receipts_sha256']
    assert len(receipts)==500
    rows=[]
    for name,expected in sorted(receipts.items()):
        receipt=SOURCE/'items'/name
        assert sha(receipt)==expected
        r=json.loads(receipt.read_text())
        assert r['status']=='passed' and r['contract_sha256']==sha(contract)
        path=SOURCE/'raw'/r['path']
        assert path.is_relative_to(SOURCE/'raw') and '..' not in path.parts
        assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256']
        rows.append(dict(id='music8k_mureka_v9_'+r['id'],source_id=r['id'],
            path='audio/mureka_originals_v1/'+r['id']+'.mp3',source_relative_path=r['path'],
            bytes=r['bytes'],sha256=r['sha256'],sample_rate=r['sample_rate'],channels=r['channels'],
            duration_s=r['decoded_duration_s'],reference_group_id=r['reference_group_id'],
            acquisition_role=r['role'],source_dataset='homura23/MUSIC8K',
            source_revision='05232438ba76a7bc55cbebfdc6d5f4011c980bba',
            publisher_declared_license='CC-BY-4.0',modification='None; preserved original MP3 bytes'))
        if len(rows)%100==0:print(f'Verified {len(rows)}/500 Mureka originals',flush=True)
    assert len({r['id'] for r in rows})==500
    assert sum(r['bytes'] for r in rows)==summary['bytes']==2411879065
    OUT.mkdir(exist_ok=False)
    (OUT/'manifest.json').write_text(json.dumps(rows,indent=2))
    (OUT/'README.md').write_text('''# Selected Mureka v9 originals from MUSIC8K

500 original MP3 files selected for this research project, from homura23/MUSIC8K
revision 05232438ba76a7bc55cbebfdc6d5f4011c980bba. Source and attribution:
https://huggingface.co/datasets/homura23/MUSIC8K
The publisher declares CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ .
No endorsement is implied. This relies on the publisher's declaration; it is
not independent clearance of the generation service or every third-party right.
The source card also reminds publishers to ensure redistribution rights.

Original MP3 bytes are unchanged. No source reference recordings, lyric text,
artist-reference metadata or prompts are copied into this audio folder.
Reference-group IDs are retained for grouping. Acquisition role reserved_unscored
describes the original selection stage, not current untouched-test status:
these 500 identities were subsequently used in development evaluations.
Consult the project's cross-experiment memberships for actual later roles.

This manifest is a verified publication plan, not confirmation that upload
completed. Use terminal and batch receipts plus independent remote verification.
''')
    (OUT/'COMMIT.json').write_text(json.dumps(dict(status='500_originals_verified_not_uploaded',
        files=500,bytes=2411879065,manifest_sha256=sha(OUT/'manifest.json'),
        source_contract_sha256=sha(contract),source_summary_sha256=sha(SOURCE/'summary.json'),
        whole_project_complete=False),indent=2))


if __name__=='__main__':main()
