#!/usr/bin/env python3
"""Post-hoc crossed-source sensitivity of frozen paired F/H increments.

This is NOT a model-refit or a new confirmatory test. The same Human and AI
source multiplicities reweight both evaluation directions and all comparisons.
It exposes empirical source-composition sensitivity, conditional on the saved
models, folds and recordings. Ten observed sources are not a population sample.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE = 'S+D+R'
ADDED = ('S+D+R+F', 'S+D+R+H', 'S+D+R+F+H')
DIRECTIONS = ('human_source_holdout', 'generator_holdout')
METRICS = ('roc_auc', 'balanced_accuracy')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reweight(matrix, human_weights, ai_weights):
    """[direction, human, ai, comparison, metric] -> [comparison, metric]."""
    if np.any(human_weights < 0) or np.any(ai_weights < 0):
        raise ValueError('Negative source multiplicity')
    weight = np.outer(human_weights, ai_weights)
    if weight.sum() <= 0:
        raise ValueError('Empty source sample')
    return np.einsum('dhacm,ha->cm', matrix, weight) / (2 * weight.sum())


def write_csv(path, records):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(args):
    if args.output_dir.exists():
        raise ValueError('Refusing existing output directory')
    eval_manifest = args.evaluation_dir / 'run_manifest.json'
    manifest = json.loads(eval_manifest.read_text())
    authorization = manifest.get('authorization', {})
    canonical_contract = json.dumps(manifest['contract'], sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    if (authorization.get('status') != 'frozen_verified' or
        hashlib.sha256(canonical_contract.encode()).hexdigest() != authorization.get('contract_sha256')):
        raise ValueError('Only completed, frozen verified evaluation is eligible')
    data = args.evaluation_dir / 'development_source_holdout_matched_deltas.csv'
    cells = defaultdict(list)
    keys = set()
    with data.open() as f:
        for row in csv.DictReader(f):
            if row['quantity'] != 'all' or row['baseline_combination'] != BASE or row['added_combination'] not in ADDED:
                continue
            key = (row['fold_type'], row['human_source'], row['ai_source'], row['added_combination'])
            foldkey = (*key, row['fold_uid'])
            if foldkey in keys:
                raise ValueError('Duplicate paired fold')
            keys.add(foldkey)
            cells[key].append([float(row['delta_' + metric + '__added_minus_baseline']) for metric in METRICS])
    humans = sorted({key[1] for key in cells})
    ais = sorted({key[2] for key in cells})
    if len(humans) != 6 or len(ais) != 4:
        raise ValueError('This analysis is explicitly scoped to six Human/four AI sources')
    matrix = np.empty((2, 6, 4, 3, 2))
    source_pairs = []
    for d, direction in enumerate(DIRECTIONS):
        for h, human in enumerate(humans):
            for a, ai in enumerate(ais):
                for c, added in enumerate(ADDED):
                    values = cells[(direction, human, ai, added)]
                    if len(values) != 5:
                        raise ValueError('Expected five opposite-group folds per source pair')
                    matrix[d, h, a, c] = np.mean(values, axis=0)
                    source_pairs.append({'fold_type': direction, 'human_source': human, 'ai_source': ai,
                                         'added_combination': added, 'folds': len(values),
                                         'delta_auc': matrix[d,h,a,c,0], 'delta_ba': matrix[d,h,a,c,1]})
    if not np.isfinite(matrix).all():
        raise ValueError('Nonfinite paired source values')
    point = reweight(matrix, np.ones(6), np.ones(4))
    group_summary = args.group_uncertainty_dir / 'uncertainty_summary.csv'
    reference = {}
    with group_summary.open() as f:
        for row in csv.DictReader(f):
            reference[(row['added_combination'], row['metric'])] = float(row['point_estimate'])
    errors = [abs(point[c,m] - reference[(added, metric)]) for c,added in enumerate(ADDED)
              for m,metric in enumerate(('delta_J','delta_equal_mean_balanced_accuracy'))]
    if max(errors) > 1e-12:
        raise ValueError('Source-grid point does not reproduce original paired estimate')
    rng = np.random.default_rng(args.seed)
    human_draws = rng.multinomial(6, np.ones(6)/6, size=args.replicates)
    ai_draws = rng.multinomial(4, np.ones(4)/4, size=args.replicates)
    estimates = np.stack([reweight(matrix, h, a) for h,a in zip(human_draws, ai_draws)])
    summary, replicates = [], []
    for c, added in enumerate(ADDED):
        for m, metric in enumerate(('delta_J','delta_mean_BA')):
            low, high = np.quantile(estimates[:,c,m], [.025,.975])
            summary.append({'baseline':BASE, 'added':added, 'metric':metric, 'point_estimate':point[c,m],
                            'empirical_source_95_low':low, 'empirical_source_95_high':high,
                            'positive_fraction':float(np.mean(estimates[:,c,m]>0)), 'replicates':args.replicates})
        for i, values in enumerate(estimates[:,c]):
            replicates.append({'replicate':i, 'added':added, 'delta_J':values[0], 'delta_mean_BA':values[1]})
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir/'source_pair_deltas.csv', source_pairs)
    write_csv(args.output_dir/'sensitivity_summary.csv', summary)
    write_csv(args.output_dir/'replicates.csv', replicates)
    np.savez_compressed(args.output_dir/'source_multiplicities.npz', human=human_draws, ai=ai_draws)
    (args.output_dir/'LIMITATIONS_EN.md').write_text(
        '# Post-hoc empirical source-composition sensitivity\n\n'
        'Six Human and four generator source labels are sampled with replacement, independently '
        'between classes. A single pair of multiplicity vectors reweights both evaluation directions '
        'and every paired increment. Five opposite-group folds are averaged within each source pair; '
        'they are not independent replicates. This checks sensitivity to the observed source mix. '
        'It was added after inspecting point results, is not a preregistered discovery, and does not '
        'refit models, combine recording-group uncertainty, quantify training instability, or provide '
        'future-domain coverage. Models share training data; the small, purposively collected source '
        'set is not an iid sample from all Human music or AI generators. Treat the percentile ranges '
        'as conditional empirical sensitivity, not a guarantee of generalization.\n')
    receipt = {'status':'complete', 'analysis':'posthoc_crossed_source_composition_sensitivity',
               'seed':args.seed, 'replicates':args.replicates, 'human_sources':humans, 'ai_sources':ais,
               'frozen_models':True, 'no_historical_labels':True, 'no_model_refit':True,
               'code_sha256':sha(__file__), 'numpy':np.__version__, 'max_point_reproduction_error':max(errors),
               'inputs':{str(p):sha(p) for p in (eval_manifest,data,group_summary)},
               'outputs':{p.name:sha(p) for p in args.output_dir.iterdir() if p.is_file()}}
    (args.output_dir/'run_manifest.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--evaluation-dir', type=Path, required=True)
    p.add_argument('--group-uncertainty-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20260907)
    p.add_argument('--replicates', type=int, default=10000)
    run(p.parse_args())
