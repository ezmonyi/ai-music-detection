"""Create the requested HF dataset and publish verified report artifacts.

Authentication is accepted only through stdin, never recorded in artifacts.
This stage uploads no audio and makes no complete-archive claim.
"""
import hashlib
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, CommitOperationAdd


def main():
    payload = json.load(sys.stdin)
    api = HfApi(token=payload['token'])
    assert api.whoami()['name'].lower() == 'ezmonyi'
    repo = 'EZMONYI/music-ai-human-test-audio'
    root = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911/delivery_snapshot_v1')
    provenance = json.loads((root/'provenance.json').read_text())
    for name, expected in provenance['outputs'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
    api.create_repo(repo_id=repo,repo_type='dataset',private=False,exist_ok=True)
    assert not api.dataset_info(repo).private
    card = '''---
pretty_name: Interpretable AI and Human Music Evaluation Archive
language:
- en
tags:
- music
- ai-music-detection
- audio
- research
---

# Interpretable AI and Human Music Evaluation Archive

This public dataset is being assembled for an interpretable music-detection
thesis. It is **incomplete**. The initial commit contains verified intermediate
experimental reports and tables, **not the full tested audio collection**.

## Current contents

`reports/overnight_snapshot_20260912_v1/` contains an English report, all 1,905
primary seven-family protocol cells, and provenance hashes. This analysis
covers 3,830 recordings across 11 sources. BC and YuE2 extension experiments
are still in progress and are not represented as completed here.

## Intended audio archive

Future additions will inventory tested original recordings and derived views,
with source, AI/human label, generator, split membership, hashes, duration,
preprocessing, and source-specific redistribution terms. A public source is
not automatically permission to redistribute it. Restricted recordings will
be documented separately rather than silently published.

No blanket license is assigned to all recordings. Source licenses and generated
output terms must be assessed individually. No credentials or recovery keys
belong in this dataset.

## Scientific limitations

Balanced accuracy is not ordinary accuracy. Source-held-out results and grouped
descriptive results answer different questions. Feature separability does not
prove an intrinsic AI fingerprint. See the reports for weighting, missingness,
coverage, and limitations. No final winner or final thesis claim is made yet.
'''
    additions = [CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=card.encode())]
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name in {'REPORT_EN.md','all_seven_family_primary_cells.csv','provenance.json'}:
            additions.append(CommitOperationAdd(path_in_repo='reports/overnight_snapshot_20260912_v1/'+path.name,
                                               path_or_fileobj=str(path)))
    result = api.create_commit(repo_id=repo,repo_type='dataset',operations=additions,
                               commit_message='Preserve verified intermediate seven-family report and full primary tables')
    files = api.list_repo_files(repo,repo_type='dataset',revision=result.oid)
    expected = [op.path_in_repo for op in additions]
    assert all(name in files for name in expected)
    receipt = dict(repo_id=repo,revision=result.oid,url=str(result.commit_url),
                   verified_paths=expected,audio_uploaded_by_this_stage=0,final_delivery_complete=False)
    path = root.parent/'hf_delivery_publication_v1.json'
    with path.open('x') as stream:
        json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
