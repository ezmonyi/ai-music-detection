#!/usr/bin/env python3
"""Independently reconcile saved V development artifacts, without pitch inference."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-root', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Existing audit is immutable')
    checks = 0
    def check(condition, message):
        nonlocal checks
        if not condition:
            raise ValueError(message)
        checks += 1
    run = json.loads((a.result_root / 'run_manifest.json').read_text())
    review = json.loads(a.review.read_text())
    summary = json.loads((a.result_root / 'summary.json').read_text())
    check(run['status'] == 'complete' and run['role'] == review['role'] == 'development', 'Role/status mismatch')
    check(sha(a.review) == run['contract']['review_receipt_sha256'], 'Review hash mismatch')
    check(sha(a.pairs) == review['pair_manifest_sha256'], 'Pair manifest mismatch')
    check(sha(a.result_root / 'summary.json') == run['summary_sha256'], 'Summary hash mismatch')
    check(hashlib.sha256(json.dumps(run['contract'], sort_keys=True, separators=(',', ':')).encode()).hexdigest() == run['contract_hash'], 'Run contract hash mismatch')
    for relative, expected in run['output_file_sha256'].items():
        check(sha(a.result_root / relative) == expected, 'Output hash mismatch: ' + relative)
    for key in ('runner_code', 'presence_code', 'prior_pyin_code'):
        check(sha(review[key + '_path']) == review[key + '_sha256'], 'Frozen code changed: ' + key)
    pairs = list(csv.DictReader(a.pairs.open()))
    dev = [r for r in pairs if r['split'] == 'development']
    held = [r for r in pairs if r['split'] == 'evaluation']
    check(len(dev) == 43 and len(held) == 47, 'Pair population changed')
    check(not ({r['singer'] for r in dev} & {r['singer'] for r in held}), 'Singer leakage')
    expected_ids = {r['pair_id'] + '::' + t for r in dev for t in ('straight', 'vibrato')}
    receipts = [json.loads(p.read_text()) for p in (a.result_root / 'receipts').glob('*.json')]
    check(len(receipts) == 86 and {r['clip_id'] for r in receipts} == expected_ids, 'Unexpected recorded clip identities')
    inventory = {r['clip_id']: r for r in review['selected_source_inventory']}
    values = {}
    coverage = Counter()
    for r in receipts:
        check(r['status'] == 'ok' and r['role'] == 'development' and r['contract_hash'] == run['contract_hash'], 'Clip contract mismatch')
        source = inventory[r['clip_id']]
        check(sha(source['source_path']) == source['source_sha256'] == r['source_sha256'], 'Source hash mismatch')
        contour = a.result_root / 'contours' / (hashlib.sha256(r['clip_id'].encode()).hexdigest() + '.npz')
        check(sha(contour) == r['contour_sha256'], 'Contour hash mismatch')
        with np.load(contour, allow_pickle=False) as npz:
            times, f0, voiced = npz['time_sec'], npz['f0_hz'], npz['voiced']
            check(len(times) == len(f0) == len(voiced), 'Contour length mismatch')
            check(np.allclose(times, np.arange(len(times)) * .01, atol=1e-10, rtol=0), 'Contour time grid mismatch')
        eligible = [w for w in r['window_details'] if w['eligible']]
        detected = [w for w in eligible if w['detected']]
        check(len(eligible) == r['V_eligible_stable_pitch_window_count'] and len(detected) == r['V_periodic_window_count'], 'Window count mismatch')
        value = len(detected) / len(eligible) if len(eligible) >= 3 else None
        actual = r['V_periodic_modulation_window_fraction']
        check((value is None and actual is None) or (value is not None and actual is not None and abs(value-actual) < 1e-12), 'Primary descriptor mismatch')
        check(r['V_status'] == ('ok' if value is not None else 'missing'), 'Missingness status mismatch')
        values[r['clip_id']] = value
        coverage[r['technique']] += value is not None
    directions = Counter()
    context = {}
    differences = []
    for pair in dev:
        straight, vibrato = (values[pair['pair_id'] + '::' + t] for t in ('straight', 'vibrato'))
        entry = context.setdefault(pair['context'], {'valid': 0, 'total': 0})
        entry['total'] += 1
        if straight is None or vibrato is None:
            directions['missing'] += 1
            continue
        delta = vibrato - straight
        differences.append(delta)
        entry['valid'] += 1
        directions['positive' if delta > 0 else 'negative' if delta < 0 else 'tie'] += 1
    check(dict(directions) == summary['pair_direction_counts_including_missing'], 'Pair direction mismatch')
    check(len(differences) == summary['paired_valid_pairs'], 'Paired coverage mismatch')
    check(abs(float(np.median(differences)) - summary['paired_difference_median']) < 1e-12, 'Median mismatch')
    for technique in ('straight', 'vibrato'):
        check(coverage[technique] == summary['numeric_coverage_by_technique'][technique]['numeric'], 'Technique coverage mismatch')
    gate = review['frozen_config']['development_gate']
    gap = abs(coverage['straight'] - coverage['vibrato']) / 43
    independently_passed = (min(coverage.values()) / 43 >= gate['minimum_numeric_coverage_each_technique']
        and gap <= gate['maximum_absolute_coverage_difference']
        and all(v['valid'] / v['total'] >= gate['minimum_paired_coverage_each_context'] for v in context.values())
        and np.median(differences) > 0
        and directions['positive'] / len(differences) >= gate['minimum_strictly_positive_pair_fraction'])
    check(bool(independently_passed) == summary['development_gate_passed'], 'Independent gate mismatch')
    check(summary['heldout_evaluation_opened_or_scored'] is False and summary['ai_detector_fitted'] is False, 'Unauthorized score claim')
    out = {'status': 'passed', 'checks_passed': checks, 'auditor_sha256': sha(__file__),
           'run_manifest_sha256': sha(a.result_root / 'run_manifest.json'), 'review_sha256': sha(a.review),
           'rows': len(receipts), 'pairs': len(dev), 'paired_valid': len(differences),
           'directions': dict(directions), 'coverage_counts': dict(coverage), 'absolute_coverage_gap': gap,
           'median_paired_difference': float(np.median(differences)),
           'development_gate_passed': bool(independently_passed), 'heldout_evaluation_scored': False,
           'scope': 'Artifact identity, source/contour hashes, saved-window accounting and gate recomputation; not continuous-F0 ground-truth validation'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
