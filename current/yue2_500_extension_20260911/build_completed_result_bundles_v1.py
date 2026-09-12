"""Bundle committed numerical results and the verified thesis without audio."""
import hashlib
import json
from pathlib import Path
import tarfile

BASE=Path('/mnt/nfs-data/users/yi')
YUE=BASE/'yue2_500_extension_20260911'
OUT=YUE/'completed_results_publication_v1'
PACKAGES={
 'bc_report':(BASE/'audio_phenomena_expansion_20260907/native30_bc_evaluation_report_v3_20260912','ad5223a5f1f0fb667fe38a1e0c214b28ab8a8d0b57f01f912521885aa1c0213b'),
 'expanded_yue2_report':(YUE/'expanded_native30_report_v1','b7934f2c649e5d0fbada832c335d36839a1aacb3bd7d6a674db83af5372bc911'),
 'native60_predictions':(YUE/'native60_transfer_scores_v1','34782ccfa467d5304ec2a4af1000d390eb467f004bb591ddce1474d06a068e64'),
 'native60_features':(YUE/'native60_feature_package_v1','5724ed5190996583ee37b45dadba830aebd771b1c5e5c99d7bde220d3f233547'),
}
THESIS={'thesis_v6.pdf':'54976713a162cd69902a24d6a313f3247f1595d4327333821e0746a03ce4c667',
 'thesis_v6_sources.tar.gz':'e6b318e7da40a7206e41adb0b2b11ec09a26d51fe807ee61a6adfd128044c836'}


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    bound={}
    for name,pin in THESIS.items():
        path=OUT/name;assert sha(path)==pin
        bound[name]=dict(bytes=path.stat().st_size,sha256=pin)
    for name,(source,pin) in PACKAGES.items():
        assert sha(source/'COMMIT.json')==pin
        commit=json.loads((source/'COMMIT.json').read_text())
        expected=dict(commit['products']);expected['COMMIT.json']=dict(sha256=pin,bytes=(source/'COMMIT.json').stat().st_size)
        archive=OUT/(name+'.tar.gz')
        assert not archive.exists(), 'Preserve pre-existing archive for inspection'
        with tarfile.open(archive,'x:gz') as tar:
            for relative,entry in expected.items():
                path=source/relative
                assert path.resolve().is_relative_to(source.resolve()) and not path.is_symlink()
                assert sha(path)==entry['sha256'] and path.stat().st_size==entry['bytes']
                tar.add(path,arcname=relative,recursive=False)
        with tarfile.open(archive,'r:gz') as tar:
            members=tar.getmembers();assert {m.name for m in members}==set(expected)
            assert len(members)==len(expected)
            for member in members:
                assert member.isfile() and member.size==expected[member.name]['bytes']
                with tar.extractfile(member) as stream:
                    assert hashlib.file_digest(stream,'sha256').hexdigest()==expected[member.name]['sha256']
        bound[archive.name]=dict(bytes=archive.stat().st_size,sha256=sha(archive),
                                source_commit_sha256=pin,verified_archive_members=len(expected))
        print('Verified archive '+archive.name,flush=True)
    with (OUT/'COMMIT.json').open('x') as f:
        json.dump(dict(status='all_bundle_members_verified',products=bound,audio_included=False,
                       whole_project_delivery_complete=False),f,indent=2)
    print('Completed-result bundles committed',flush=True)


if __name__=='__main__':main()
