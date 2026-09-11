#!/usr/bin/env python3
"""Root independent percentile/hash reconciliation; does not refit or redraw."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def rows(path):
    return list(csv.DictReader(path.open()))


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered)-1)*q
    lo = math.floor(index)
    return ordered[lo] + (ordered[math.ceil(index)]-ordered[lo])*(index-lo)


def main():
    root = Path(__file__).resolve().parent.parent
    result = root/'results/uncertainty_sdrfh10_v1'
    manifest = json.loads((result/'run_manifest.json').read_text())
    for name, digest in manifest['outputs_sha256'].items():
        assert hashlib.sha256((result/name).read_bytes()).hexdigest() == digest, name
    errors, checks = [], 0
    alias = {'delta_J': 'delta_equal_mean_auc', 'delta_human_balanced_accuracy': 'delta_human_ba',
             'delta_generator_balanced_accuracy': 'delta_generator_ba',
             'delta_equal_mean_balanced_accuracy': 'delta_equal_mean_ba'}
    for directory, replicate_name, summary_name, expected in (
        (result, 'bootstrap_replicates.csv', 'uncertainty_summary.csv', 1000),
        (result/'source_sensitivity', 'replicates.csv', 'sensitivity_summary.csv', 10000)):
        draws = defaultdict(list)
        for row in rows(directory/replicate_name):
            draws[row['added_combination']].append(row)
        assert len(draws) == 3
        for samples in draws.values():
            assert {int(r['replicate']) for r in samples} == set(range(expected))
            assert len(samples) == expected
        for row in rows(directory/summary_name):
            column = alias.get(row['metric'], row['metric']) if expected == 1000 else row['metric']
            values = [float(r[column]) for r in draws[row['added_combination']]]
            assert all(math.isfinite(x) for x in values)
            fields = (('percentile_95_ci_low', .025), ('percentile_95_ci_high', .975)) if expected == 1000 else (
                ('empirical_source_95_low', .025), ('empirical_source_95_high', .975))
            for field, q in fields:
                errors.append(abs(percentile(values, q)-float(row[field])))
                checks += 1
            if expected == 1000:
                errors.append(abs(statistics.mean(values)-float(row['bootstrap_mean'])))
                errors.append(abs(statistics.stdev(values)-float(row['bootstrap_standard_error'])))
                checks += 2
    assert max(errors) < 1e-12
    receipt = {'status': 'passed', 'declared_output_hashes_checked': len(manifest['outputs_sha256']),
               'summary_numeric_checks': checks, 'max_absolute_error': max(errors),
               'scope': 'independent standard-library quantiles/moments from saved draws; not an independent redraw or model fit',
               'run_manifest_sha256': hashlib.sha256((result/'run_manifest.json').read_bytes()).hexdigest()}
    out = root/'audit/uncertainty_sdrfh10_root_summary_audit.json'
    with out.open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
