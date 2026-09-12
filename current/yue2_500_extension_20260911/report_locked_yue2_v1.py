"""Export every locked-test combination without selecting a winner or refitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

from verify_local_package_v1 import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('package', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    receipt = verify(args.package)
    summary = json.loads((args.package / 'summary.json').read_text())
    assert summary['unique_test_items'] == 100
    assert summary['classifier_fits'] == 0
    rows = summary['per_model']
    assert len(rows) == 1275
    groups = {}
    for row in rows:
        assert all(row[k] is None for k in ('balanced_accuracy', 'human_specificity', 'roc_auc'))
        groups.setdefault(row['combination'], []).append(row)
    assert len(groups) == 255
    records = []
    for combination, members in groups.items():
        assert sorted(r['fold'] for r in members) == list(range(5))
        values = [r['ai_sensitivity'] for r in sorted(members, key=lambda r: r['fold'])]
        records.append(dict(combination=combination, **{f'fold_{i}': v for i, v in enumerate(values)},
                            descriptive_mean=mean(values), minimum=min(values), maximum=max(values)))
    args.output.mkdir(exist_ok=False, parents=True)
    with (args.output / 'all_255_combinations.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    text = '''# Locked YuE2 sensitivity — prediction-only results

All 255 feature subsets were evaluated using five existing old-cohort models
per subset (1,275 models), each on the same 100 locked AI recordings. There
were no new classifier fits or outcome-based model selection. These are
127,500 prediction occurrences, not independent recordings. The five-model
mean and range below are descriptive, not confidence intervals.

No human recordings are included in this test: specificity, balanced accuracy
and ROC AUC cannot be estimated. High sensitivity alone does not establish a
useful AI/human detector. A detector predicting AI for everything also has
100% sensitivity. Grouped old-model variants are not five independent test sets.

Ordering deviation: locked scoring happened after expanded development,
contrary to the earlier proposed order. The scoring catalogue was exhaustive
and fixed before this scoring run; this does not retroactively establish
preregistration before expanded development.

## Pre-existing families and combinations (descriptive)

| Family/subset | Mean sensitivity | Range across five models |
|---|---:|---:|
'''
    selected = ['S', 'D', 'R', 'P', 'F', 'H', 'SC', 'BC', 'S+D+R+P', 'S+D+R+P+F+H+SC+BC']
    for combo in selected:
        r = next(r for r in records if r['combination'] == combo)
        text += f"| {combo} | {r['descriptive_mean']:.1%} | {r['minimum']:.0%}–{r['maximum']:.0%} |\n"
    text += '''
S denotes spectral statistics; D dynamics; R rhythm; P phrase/section proxies;
F phase/group delay; H pitch-class structure; SC stereo coherence; BC the
fixed bicoherence statistic. This summary cannot establish causal differences
between human performance and generation. All 255 subsets, including negative
results, are retained in the adjacent CSV. BC's fold variability warrants
inspection alongside human false-positive rates, not a sensitivity-only claim.
'''
    (args.output / 'REPORT_EN.md').write_text(text)
    manifest = dict(source_verification=receipt, classifier_fits=0, products={})
    for path in sorted(args.output.iterdir()):
        raw = path.read_bytes()
        manifest['products'][path.name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (args.output / 'COMMIT.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(combinations=len(records), source_products_verified=receipt['products'])))


if __name__ == '__main__':
    main()
