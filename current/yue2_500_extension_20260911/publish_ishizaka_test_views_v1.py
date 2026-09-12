"""Publish and anonymously verify the 78 source-bound CC0 Ishizaka test views."""
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi, CommitOperationAdd
import publish_maestro_native30_v1 as transport
from publish_yue2_originals_v1 import digest, save
from publish_external_maestro_v1 import verify

ROOT = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911')
OUT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/hf_ishizaka_test_views_v1')
PREFIX = 'audio/musicnet_ishizaka_cc0_v1/historical_test_views_v1/'
NOTICE = '''# Ishizaka historical processed test inputs

78 cropped/standardized historical 10s/30s input files from the 39 previously
identified MusicNet recordings by Kimiko Ishizaka, Bach Well-Tempered Clavier
Book 1. These are experimental versions, not 78 independent performances.
Original record metadata, attribution, hashes and crop parameters are retained.
Files are copied unchanged from the audited experiment outputs.

Recording dedication: CC0 1.0.
https://creativecommons.org/publicdomain/zero/1.0/
Performer and source statement:
https://kimikoishizaka.bandcamp.com/album/bach-well-tempered-clavier-book-1
MusicNet provided the source WAV distribution and identifying metadata.
No byte identity with the separately distributed high-resolution album is
claimed. No album artwork, annotations or scores are included. No endorsement
is implied. The other MusicNet recordings are not covered by this source
review. This notice does not license unrelated dataset content.
'''


def worker(token):
    plan = ROOT/'ishizaka_test_views_objects_v1.json'
    assert digest(plan) == 'ec366c7ca823fc8addec461886eabae7b29e7bb061bab10212f662478455eb89'
    objects = json.loads(plan.read_text())
    assert len(objects) == 78
    rows = []; paths = {}
    for obj in objects:
        assert obj['recording_license'] == 'CC0-1.0' and not obj['public_paths']
        p = Path(obj['source_paths'][0]); name = PREFIX+obj['sha256']+p.suffix
        paths[name] = p
        rows.append(dict(path=name, sha256=obj['sha256'], bytes=obj['bytes'],
            memberships=obj['memberships'], original_records=obj['original_records'],
            recording_license='CC0-1.0', modification=obj['modification']))
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    manifest = OUT/'manifest.json'
    if manifest.exists(): assert json.loads(manifest.read_text()) == rows
    else: save(manifest, rows)
    terminal = OUT/'COMMIT.json'
    if terminal.exists():
        revision = json.loads(terminal.read_text())['revision']
    else:
        ops = []
        for r in rows:
            p = paths[r['path']]
            assert p.stat().st_size == r['bytes'] and digest(p) == r['sha256']
            ops.append(CommitOperationAdd(path_in_repo=r['path'], path_or_fileobj=str(p)))
        ops += [CommitOperationAdd(path_in_repo=PREFIX+'manifest.json', path_or_fileobj=str(manifest)),
                CommitOperationAdd(path_in_repo=PREFIX+'README.md', path_or_fileobj=NOTICE.encode())]
        revision = api.create_commit(repo_id=transport.REPO, repo_type='dataset', operations=ops,
            commit_message='Preserve 78 Ishizaka historical test inputs with CC0 attribution').oid
        save(terminal, dict(revision=revision, objects=78, manifest_sha256=digest(manifest),
            status='uploaded_pending_independent_acceptance', whole_project_complete=False))
    anonymous = HfApi(token=False)
    assert not anonymous.dataset_info(transport.REPO, revision=revision).private
    for name, expected in [('manifest.json', manifest.read_bytes()), ('README.md', NOTICE.encode())]:
        url = f'https://huggingface.co/datasets/{transport.REPO}/resolve/{revision}/{PREFIX}{name}'
        with urllib.request.urlopen(url, timeout=60) as response:
            assert response.read() == expected
    for start in range(0, 78, 40): verify(anonymous, rows[start:start+40], revision)
    receipt = dict(revision=revision, verified_audio_objects=78, original_ids=39,
        checked_bytes=sum(r['bytes'] for r in rows), manifest_sha256=digest(manifest),
        final_acceptance=True, verification='Anonymous exact manifest/notice and every LFS hash/size; no full audio redownload',
        whole_project_complete=False)
    save(OUT/'INDEPENDENT_ACCEPTANCE.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    from bounded_hf_http_v1 import install
    install()
    transport.OUT = OUT
    transport.worker = worker
    transport.main()
