"""Lossless metric-table export from committed YuE2 model receipts; no fitting.

These are fold-level tables, not a source-balanced headline report. Never pool
scores across fitted models or select a winner using these exports.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
PIN = '11f074f867d0e3721c6a586e8c8f86116ff1b0198147ba14f6379b5d6f55570d'
TASK = ['analysis_role', 'combination', 'feature_mode', 'fold_type', 'fold_uid',
        'heldout_source', 'opposite_group_fold', 'quantity', 'schedule_uid']
METRICS = ['ai_sensitivity', 'human_specificity', 'balanced_accuracy', 'roc_auc',
           'tp', 'tn', 'fp', 'fn']

def main():
    source = ROOT / 'expanded_native30_evaluation_v1'
    raw = (source / 'COMMIT.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PIN
    commit = json.loads(raw)
    assert commit['models'] == len(commit['model_receipts']) == 112455
    out = ROOT / 'expanded_fold_metrics_export_v1'
    out.mkdir(exist_ok=False)
    fields = ['receipt_id'] + TASK + ['metric_scope', 'ai_source', 'human_source'] + METRICS
    with (out / 'fold_metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        rows = 0
        for count, (uid, entry) in enumerate(sorted(commit['model_receipts'].items()), 1):
            path = source / 'models' / (uid + '.json')
            assert str(path) == entry['path']
            data = path.read_bytes()
            assert len(data) == entry['bytes']
            assert hashlib.sha256(data).hexdigest() == entry['sha256']
            payload = json.loads(data)['payload']
            base = {'receipt_id': uid, **{k: payload['task'][k] for k in TASK}}
            metrics = payload['metrics']
            for scope, values in [('within_fold_pooled_descriptive', [metrics['within_fold_pooled_descriptive']]),
                                  ('source_pair', metrics['source_pairs'])]:
                for value in values:
                    writer.writerow({**base, 'metric_scope': scope,
                        **{k: value.get(k) for k in ['ai_source', 'human_source'] + METRICS}})
                    rows += 1
            if count % 1000 == 0:
                print(f'Hash-verified and exported {count}/112455 receipts', flush=True)
    assert (source / 'COMMIT.json').read_bytes() == raw
    table = out / 'fold_metrics.csv'
    with table.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = dict(status='complete_fold_metric_export_not_headline_report', receipts=count,
                  rows=rows, input_commit_sha256=PIN, classifier_fits=0,
                  cross_model_score_pooling=False, winner_selection=False,
                  products={'fold_metrics.csv': {'bytes': table.stat().st_size, 'sha256': sha}})
    with (out / 'COMMIT.json').open('x') as stream:
        json.dump(result, stream, sort_keys=True)
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
