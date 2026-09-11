"""Parent-reviewed, no-fit independent lineage/schedule acceptance and freeze."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json
import audit_equal60_v6_results as A

ROOT=Path(__file__).resolve().parents[1]


def run(draft,output):
    require=A.require
    require(not output.exists(),'New freeze output required')
    require(A.sha(Path(A.__file__))=='ef322f740acc545398746f65f7103983b466ee23658c3fd32e10a3d4062ee5f6','Auditor changed')
    require(A.sha(Path(A.A5.__file__))==A.HELPER_SHA,'Auditor helper changed')
    marker=A.read_json(draft/'COMMIT.json')
    # The no-fit draft is an immutable publication, not a free-standing JSON.
    require(marker['status']=='committed' and set(marker['files'])==
        {'fold_registry.jsonl','omitted_fold_cells.json','preregistration_draft.json'},'Draft publication inventory')
    require({p.name for p in draft.iterdir()}==set(marker['files'])|{'COMMIT.json'},'Unexpected draft products')
    for name,record in marker['files'].items():
        path=draft/name
        require(path.is_file() and not path.is_symlink() and A.sha(path)==record['sha256']
                and path.stat().st_size==record['bytes'],'Draft product changed')
    frozen=A.read_json(draft/'preregistration_draft.json')
    require(frozen['status']=='draft' and frozen['fitting_started'] is False,'Not an unfit draft')
    c=frozen['contract']; require(frozen['contract_sha256']==A.digest(c),'Draft contract hash')
    package=Path(c['input_package'])
    table,prep,package_sha=A.load_package(package,False)
    schedule,omitted,counts=A.build_schedule(table)
    A.validate_contract(c,table,prep,package,schedule,omitted,counts,False)
    tests_path=ROOT/'audit/v6_preparation_evaluator_auditor_synthetic_tests_v1.json'
    require(A.sha(tests_path)=='032b0e10868e5d9c7831444be7bb50e4c484d07bd62cb31414fcb8dfd6158986','Tests receipt changed')
    tests=A.read_json(tests_path)
    require(tests['status']=='passed' and tests['tests_passed']==tests['tests_run']==51,'Tests failed')
    for name,h in tests['code_sha256'].items(): require(A.sha(ROOT/'code'/name)==h,'Tested code changed: '+name)
    require(A.sha(ROOT/tests['log_path'])==tests['log_sha256'],'Test log changed')
    replay_path=ROOT/'audit/sc_native2174_frame_replay_v6.json'; replay=A.read_json(replay_path)
    require(replay['passed'] is True and replay['rows']==2174 and replay['scalar_checks']==13044
            and replay['maximum_absolute_error']==0 and replay['classifier_fitted'] is False,'SC frame replay failed')
    require(replay['extraction_commit_sha256']==prep['contract']['extraction_commit_sha256'],'Frame replay origin mismatch')
    require(replay['audit_code_sha256']==A.sha(ROOT/'code/audit_sc_native2174_frames_v6.py'),'Frame replay code changed')
    now=datetime.now(timezone.utc).isoformat()
    frozen.update(status='frozen',independent_review=dict(approved=True,reviewer='root',reviewed_utc=now,
        real_package_independently_reconstructed=True,real_schedule_independently_reconstructed=True,
        classification_fitted_during_review=False,package_commit_sha256=package_sha,
        draft_commit_sha256=A.sha(draft/'COMMIT.json'),auditor_sha256=A.sha(Path(A.__file__)),
        synthetic_tests_sha256=A.sha(tests_path),stored_frame_replay_sha256=A.sha(replay_path),
        review_driver_sha256=A.sha(__file__),
        limits='Exploratory development only; 7 reused-corpus control recordings; no unseen-source confirmation or winner selection'))
    A.bindings_check(c['input_files_sha256'])
    with output.open('x') as stream:
        json.dump(frozen,stream,sort_keys=True,separators=(',',':'),allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status='frozen_no_fit',receipt=str(output),sha256=A.sha(output),
        contract_sha256=A.digest(c),accounting=counts)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--draft',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args(); run(a.draft,a.output)
