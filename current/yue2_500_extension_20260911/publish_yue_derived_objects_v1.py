"""Publish remaining byte-deduplicated YuE views; preserve all original aliases."""
import json
from pathlib import Path
import time
from huggingface_hub import HfApi, CommitOperationAdd
import publish_maestro_native30_v1 as transport
from publish_yue2_originals_v1 import digest, save, verify

ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
PLAN = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911/yue_derived_publication_plan_v1')
OUT = ROOT/'hf_yue_derived_objects_v1'
PREFIX = 'audio/yue2/derived_objects_v1/'
PINS = {'objects.json': '3008a5aeef5f1788585f185936e613357775596662ca1ae7637eea7e10f13815',
        'memberships.json': '9de65e002bf978a40f34800f7a844d660cd0afab8797b0ba4b1fdf1691a74195'}
NOTICE = '''# YuE preserved derived audio objects

This folder preserves 4,900 additional unique byte objects from existing YuE
experiments. Content-identical aliases are stored once, not counted as new
recordings. The complete mapping contains 6,922 path memberships and 5,900
objects; 1,000 objects reside separately under ../legacy_demucs_v1/.
The original population is 500 generated recordings, not thousands of songs.

These are exact saved standardized inputs and separated stems from legacy,
Native30 and Native60 experiments. No new inference, cropping or normalization
was performed for publication. Relative source paths identify the experimental
stage; memberships retain original IDs, roles and grouping. Native60 eligibility
differs from Native30, and legacy short-input padding is not reversed here.
Objects may represent different views of the same recording. Never treat them
as independent train/test examples. Reconstruct historical paths using the
membership and object maps; their publication_verified=false flags describe
the immutable planning checkpoint, not current upload receipts.

Original provenance and notices: ../originals_v1/README.md and manifest.json
at original acceptance revision 4389bbd7906137b9abbc340fcd743bca451ab092.
YuE: https://github.com/multimodal-art-projection/YuE
Demucs: https://github.com/facebookresearch/demucs
Only project-generated YuE audio is included. No human-source audio, model
weights, or raw text prompts are included. Model licensing is not asserted as
an automatic license for generated outputs; no blanket third-party clearance
or endorsement is granted. Preserve source notices and experiment identities.
The manifest is a planned roster; receipts establish verified arrival.
'''


def records():
    for name, pin in PINS.items(): assert digest(PLAN/name) == pin
    all_objects = json.loads((PLAN/'objects.json').read_text())
    rows = [dict(o, path=o['proposed_path']) for o in all_objects if not o['existing_legacy_demucs_path']]
    assert len(rows) == len({r['path'] for r in rows}) == 4900
    assert sum(r['bytes'] for r in rows) == 46265373990
    assert all(r['path'].startswith(PREFIX) for r in rows)
    return rows


def worker(token):
    try: publish(token)
    except Exception as exc: raise RuntimeError(type(exc).__name__) from None


def publish(token):
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ezmonyi'
    assert not api.dataset_info(transport.REPO).private
    rows = records(); save(OUT/'manifest.json', rows)
    for index, offset in enumerate(range(0, len(rows), 50)):
        batch = rows[offset:offset+50]; receipt = OUT/f'batch_{index:03d}.json'
        if receipt.exists():
            prior = json.loads(receipt.read_text())
            assert prior['files'] == [r['path'] for r in batch]
            verify(api, batch, prior['revision']); continue
        operations = []
        for row in batch:
            source = ROOT/row['source_paths'][0]
            assert source.resolve().is_relative_to(ROOT.resolve())
            assert source.stat().st_size == row['bytes'] and digest(source) == row['sha256']
            operations.append(CommitOperationAdd(path_in_repo=row['path'], path_or_fileobj=str(source)))
        if index == 0:
            operations.extend([CommitOperationAdd(path_in_repo=PREFIX+'manifest.json', path_or_fileobj=str(OUT/'manifest.json')),
                CommitOperationAdd(path_in_repo=PREFIX+'README.md', path_or_fileobj=NOTICE.encode())])
            operations.extend(CommitOperationAdd(path_in_repo=PREFIX+name, path_or_fileobj=str(PLAN/name)) for name in PINS)
        while True:
            try:
                result = api.create_commit(repo_id=transport.REPO, repo_type='dataset', operations=operations,
                    commit_message=f'Preserve additional YuE derived objects {offset+1}-{offset+len(batch)} of 4900')
                break
            except Exception as exc:
                response = getattr(exc, 'response', None)
                if response is None or response.status_code != 429: raise
                retry = response.headers.get('Retry-After', '')
                delay = max(3900, int(retry)) if retry.isdigit() else 3900
                print(f'HTTP 429; retry in {delay} seconds', flush=True); time.sleep(delay)
        verify(api, batch, result.oid)
        save(receipt, dict(revision=result.oid, files=[r['path'] for r in batch], sha256_verified=True))
        print(f'Uploaded and verified {offset+len(batch)}/4900', flush=True)
    save(OUT/'COMMIT.json', dict(status='4900_additional_yue_objects_uploaded_hash_verified',
        audio_objects=4900, manifest_sha256=digest(OUT/'manifest.json'), plan_pins=PINS,
        whole_project_complete=False))


if __name__ == '__main__':
    transport.OUT = OUT; transport.worker = worker; transport.main()
