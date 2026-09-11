#!/usr/bin/env python3
"""Render the independently audited 31 x 5 exact10 lattice; never fit a model."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ORDER = ('S', 'D', 'R', 'F', 'H')
QUANTITIES = ('25', '50', '100', '200', 'all')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return list(csv.DictReader(Path(path).open()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    report = json.loads(a.audit.read_text())
    if report.get('status') != 'passed' or report.get('errors'):
        raise ValueError('Independent full audit has not passed')
    if (a.output_dir / 'presentation_receipt.json').exists():
        raise ValueError('Preserve existing presentation; use a new version')
    manifest = json.loads((a.results_dir / 'run_manifest.json').read_text())
    if manifest['authorization']['status'] != 'frozen_verified':
        raise ValueError('Missing frozen evaluation authorization')
    source_path = a.results_dir / 'development_source_holdout_summary.csv'
    group_path = a.results_dir / 'development_group_cv_summary.csv'
    sources, groups = read(source_path), read(group_path)
    combinations = ['+'.join(c) for size in range(1, 6) for c in itertools.combinations(ORDER, size)]
    source_lookup = {(r['combination'], r['quantity'], r['fold_type']): r for r in sources}
    group_lookup = {(r['combination'], r['quantity']): r for r in groups}
    expected = {(c, q) for c in combinations for q in QUANTITIES}
    expected_source = {(c, q, direction) for c, q in expected for direction in ('human_source_holdout', 'generator_holdout')}
    if len(sources) != 310 or set(source_lookup) != expected_source or len(groups) != 155 or set(group_lookup) != expected:
        raise ValueError('Incomplete or duplicated 31 x 5 lattice')
    if {r['cohort_id'] for r in sources + groups} != {'dev-d418be0b6d721b2e'}:
        raise ValueError('Unexpected cohort or cohort changes between candidates')
    rows = []
    for c in combinations:
        for q in QUANTITIES:
            h = source_lookup[c, q, 'human_source_holdout']
            ai = source_lookup[c, q, 'generator_holdout']
            g = group_lookup[c, q]
            if int(h['heldout_sources']) != 7 or int(ai['heldout_sources']) != 13:
                raise ValueError('Unexpected source coverage')
            row = {'combination': c, 'quantity': q, 'cohort_id': h['cohort_id'], 'eligible_rows': 8361,
                   'human_held_auc': float(h['roc_auc_source_macro']),
                   'ai_held_auc': float(ai['roc_auc_source_macro']),
                   'human_held_ba': float(h['balanced_accuracy_source_macro']),
                   'ai_held_ba': float(ai['balanced_accuracy_source_macro']),
                   'descriptive_group_auc': float(g['roc_auc_source_macro']),
                   'descriptive_group_ba': float(g['balanced_accuracy_source_macro'])}
            row['J'] = (row['human_held_auc'] + row['ai_held_auc']) / 2
            row['mean_BA'] = (row['human_held_ba'] + row['ai_held_ba']) / 2
            if not np.isfinite([row[k] for k in ('J', 'mean_BA', 'descriptive_group_auc', 'descriptive_group_ba')]).all():
                raise ValueError('Undefined metric in published cell')
            rows.append(row)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    numeric = a.output_dir / 'all_31x5_numeric_results.csv'
    with numeric.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lookup = {(r['combination'], r['quantity']): r for r in rows}
    md = ['# Exact-10-second S/D/R/F/H development lattice', '',
          'All 31 nonempty combinations share 8,361 development recordings. Caps are training global groups per source, not total songs. Each cell is J / mean balanced accuracy; J is the equal mean of Human-held and AI-held source-macro AUC. The two directions contain 7 Human and 13 AI sources. Threshold is fixed at 0.5. P is unavailable at 10 seconds.', '',
          'These are source-held-out development results, not an untouched external test. The 10-second and 30-second populations differ; do not interpret their difference as a duration effect. No multiplicity correction or new confidence interval is claimed. Ordinary group-CV numbers are retained separately in the CSV.', '',
          '| Combination | 25 | 50 | 100 | 200 | All |', '|---|---:|---:|---:|---:|---:|']
    latex = [r'\begingroup\footnotesize', r'\begin{longtable}{lrrrrr}',
             r'\caption{Audited exact-10-second development lattice. Each cell is $J$/mean BA; caps count training global groups per source. All cells share 8,361 recordings, seven Human sources and thirteen AI sources.}\label{tab:sdrfh10-full}\\',
             r'\toprule Combination & 25 & 50 & 100 & 200 & All \\ \midrule\endfirsthead',
             r'\toprule Combination & 25 & 50 & 100 & 200 & All \\ \midrule\endhead',
             r'\bottomrule\endlastfoot']
    for c in combinations:
        cells = [f"{lookup[c,q]['J']:.4f}/{lookup[c,q]['mean_BA']:.4f}" for q in QUANTITIES]
        md.append('| ' + c + ' | ' + ' | '.join(cells) + ' |')
        latex.append(c + ' & ' + ' & '.join(cells) + r' \\')
    latex.extend([r'\end{longtable}', r'\endgroup', ''])
    (a.output_dir / 'RESULTS_SDRFH_10S_EN.md').write_text('\n'.join(md) + '\n')
    (a.output_dir / 'results_sdrfh10_en.tex').write_text('\n'.join(latex))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 12), sharey=True)
    for ax, metric, title in zip(axes, ('J', 'mean_BA'), ('Equal-direction source-macro AUC (J)', 'Mean balanced accuracy (threshold 0.5)')):
        values = np.array([[lookup[c, q][metric] for q in QUANTITIES] for c in combinations])
        im = ax.imshow(values, vmin=.45, vmax=.95, cmap='viridis', aspect='auto')
        ax.set_xticks(range(5), ('25', '50', '100', '200', 'All'))
        ax.set_yticks(range(31), combinations)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Training group cap per source')
        for i in range(31):
            for j in range(5):
                ax.text(j, i, f'{values[i,j]:.3f}', ha='center', va='center', fontsize=7, color='white' if values[i,j] < .74 else 'black')
    fig.suptitle('Exact 10 s: all 31 combinations x 5 caps\nSource-held-out development evidence; 8,361 recordings', fontsize=12)
    fig.subplots_adjust(left=.16, right=.91, top=.93, bottom=.07, wspace=.10)
    fig.colorbar(im, ax=axes, fraction=.025, pad=.03)
    fig.savefig(a.output_dir / 'all_31x5_auc_ba.png', dpi=150)
    plt.close(fig)
    outputs = {f.name: sha(f) for f in a.output_dir.iterdir() if f.is_file()}
    receipt = {'status': 'complete', 'numeric_cells': 155, 'classifiers_fitted': False,
               'independent_audit_sha256': sha(a.audit), 'code_sha256': sha(__file__),
               'inputs': {str(p): sha(p) for p in (source_path, group_path, a.results_dir / 'run_manifest.json')},
               'outputs': outputs, 'visual_qa': 'pending'}
    (a.output_dir / 'presentation_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
