"""Publish the committed full fold-metric export, not a final accuracy claim."""
import hashlib
import json
from pathlib import Path
import sys
from huggingface_hub import HfApi, CommitOperationAdd

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    token = json.load(sys.stdin)['token']
    api = HfApi(token=token)
    assert api.whoami()['name'].lower() == 'ezmonyi'
    root = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911/expanded_fold_metrics_export_v1')
    assert sha(root/'COMMIT.json') == '59ea6be944b72ca8255b4fdd9ce4658172b61f6957f94b12147c3b741f4650bf'
    commit = json.loads((root/'COMMIT.json').read_text())
    assert commit['receipts'] == 112455 and commit['rows'] == 1012095
    files = ['COMMIT.json', *commit['products']]
    for name, entry in commit['products'].items():
        assert sha(root/name) == entry['sha256'] and (root/name).stat().st_size == entry['bytes']
    repo = 'EZMONYI/music-ai-human-test-audio'
    prefix = 'reports/yue2_expanded_fold_metrics_v1/'
    notice = b'''# YuE2-expanded full fold-metric export

1,012,095 rows from 112,455 verified model receipts: 112,455 within-fold
descriptive records and 899,640 source-pair records. These are evaluation
instances, NOT song counts or independent observations. Do not average all
rows or pool scores across models to claim overall accuracy. The separate
group/source/class-weighted final report is still pending. Locked YuE2 test
recordings were not scored. No model winner or confidence interval is claimed.
COMMIT.json binds the complete CSV and original evaluation COMMIT.
'''
    result = api.create_commit(repo_id=repo, repo_type='dataset',
        operations=[CommitOperationAdd(path_in_repo=prefix+n,path_or_fileobj=str(root/n)) for n in files]
          + [CommitOperationAdd(path_in_repo=prefix+'README.md',path_or_fileobj=notice)],
        commit_message='Archive all YuE2 expanded fold metrics with immutable provenance')
    remote = {x.path:x for x in api.get_paths_info(repo, paths=[prefix+n for n in files],
                      repo_type='dataset', revision=result.oid)}
    for name in files:
        assert remote[prefix+name].size == (root/name).stat().st_size
    assert remote[prefix+'fold_metrics.csv'].lfs.sha256 == commit['products']['fold_metrics.csv']['sha256']
    receipt = dict(repo=repo, revision=result.oid, csv_lfs_sha256_verified=True,
                   paths=[prefix+n for n in files], final_delivery_complete=False)
    target = Path('/mnt/nfs-code/users/yi/yue2_500_extension_20260911/expanded_metrics_publication_v1.json')
    with target.open('x') as stream:
        json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
