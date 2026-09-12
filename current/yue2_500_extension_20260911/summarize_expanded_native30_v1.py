"""Use the unchanged source/group-weighted reporting core for YuE2 expansion.

No fitting and no locked scoring. All intended cells, including omissions,
are retained. This adapter verifies the expanded contract instead of pretending
it is the original 3,830-row package.
"""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import sys

RC = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/code')
ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
sys.path.insert(0, str(RC))
import summarize_native30_evaluation_bc_v3 as report
import evaluate_expanded_native30_v1 as expanded

PIN = '11f074f867d0e3721c6a586e8c8f86116ff1b0198147ba14f6379b5d6f55570d'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    source = ROOT / 'expanded_native30_evaluation_v1'
    output = ROOT / 'expanded_native30_report_v1'
    assert not output.exists()
    commit_binding = report.binding(source / 'COMMIT.json', PIN)
    commit = report.read(source / 'COMMIT.json')
    contract, table, tasks = expanded.build()
    contract_sha = report.value_hash(contract)
    assert report.read(source / 'contract.json') == contract
    assert commit['models'] == len(commit['model_receipts']) == 112455
    assert commit['locked_test_scored'] is False
    # The approved launcher wrote ordinary sorted JSON, not newline-canonical
    # JSON. Its exact raw-byte hash remains mandatory below.
    freeze = report.read(commit['freeze']['path'], canonical_required=False)
    assert report.binding(commit['freeze']['path']) == commit['freeze']
    assert freeze['contract'] == contract and freeze['contract_sha256'] == contract_sha
    assert freeze['fitting_authorized'] is True
    expected = dict(report.EXPECTED, rows=4228,
                    sources=dict(Counter(table['__source'])),
                    labels=dict(Counter(int(x) for x in table['__label'])),
                    eligible=112455, intended=116025, primary=80325, diagnostic=32130)
    pools, caps, fold_rows, omissions = defaultdict(list), {}, [], []
    seen = set()
    for index, task in enumerate(tasks, 1):
        uid = expanded.evaluator.task_uid(task, contract_sha)
        entry = None
        if task['status'] == 'eligible_metadata_cell':
            entry = commit['model_receipts'][uid]
            record = report.receipt(Path(entry['path']), entry, contract_sha, expected)
            actual = record['task']
            for key in ['fold_uid', 'schedule_uid', 'combination', 'feature_mode',
                        'fold_type', 'heldout_source', 'quantity', 'train_ids', 'test_ids']:
                assert actual[key] == task[key], key
            assert uid not in seen
            seen.add(uid)
            if args.preflight:
                print(json.dumps(dict(status='single_receipt_and_expanded_contract_preflight_passed',
                                      expected_models=expected['eligible'], receipt_id=uid)), flush=True)
                return
            predictions = record['predictions']
            fold_rows.append({'model_uid': uid, **{k: actual[k] for k in
                ['fold_type', 'heldout_source', 'quantity', 'combination', 'feature_mode',
                 'opposite_group_fold', 'schedule_uid']},
                'train_rows': len(actual['train_ids']), 'test_rows': len(predictions),
                'single_model_fold_auc': report.auc(predictions),
                'single_model_recording_weighted_BA': report.recording_recalls(predictions)['balanced_accuracy']})
            if len(seen) % 1000 == 0:
                print(f'Expanded report verified {len(seen)}/112455 receipts', flush=True)
        else:
            actual = task
            omissions.append(task)
        reference = {k: actual[k] for k in ['opposite_group_fold', 'train_ids', 'test_ids',
                                           'status', 'omission_reasons']}
        cap_id = actual['schedule_uid']
        assert cap_id not in caps or caps[cap_id] == reference
        caps[cap_id] = reference
        pools[report.CORE.task_key(actual)].append({'entry': entry, 'schedule_uid': cap_id})
    assert seen == set(commit['model_receipts']) and len(omissions) == 3570
    published = dict(fold_rows=fold_rows, pools=dict(pools), cap_records=caps,
                     expected=expected, contract_sha256=contract_sha,
                     contract={'accounting': {k:expected[k] for k in ['intended','eligible','primary','diagnostic']}})
    result = report.summarize(published)
    result.update(version='summarize_expanded_native30_v1',
                  evaluation_COMMIT=commit_binding, locked_test_scored=False,
                  independent_implementation_audit=False)
    assert len(result['primary_protocol_cells']) == 3825
    assert len(result['diagnostic_protocol_cells']) == 1530
    assert report.binding(source / 'COMMIT.json') == commit_binding
    output.mkdir()
    report.write_new(output / 'summary.json', result)
    report.write_new(output / 'per_fold_auc.json', fold_rows)
    report.write_new(output / 'omitted_cells.json', omissions)
    for name, text in report.markdown_products(result, published).items():
        report.write_new(output / name, text.encode())
    # Rehash all receipt files again before committing the report.
    for entry in commit['model_receipts'].values():
        assert report.binding(entry['path']) == entry
    assert report.binding(source / 'COMMIT.json') == commit_binding
    products = {p.name:report.binding(p) for p in sorted(output.iterdir())}
    report.write_new(output / 'COMMIT.json', dict(status='committed_reporting_only_no_selection',
        evaluation_COMMIT=commit_binding, products=products, classifier_fits=0,
        locked_test_scored=False, all_receipts_end_rehashed=True))
    print('Expanded source-balanced report committed', flush=True)

if __name__ == '__main__':
    main()
