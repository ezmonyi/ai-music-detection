"""Compare two thesis tables with retained Equal60 CSV, without fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ARMS = ('human_source_holdout', 'generator_holdout',
        'ordinary_group_holdout_descriptive')
SEVEN = 'S+D+R+P+F+H+M'
EIGHT = SEVEN + '+SC'
SETS = {name: name for name in ('S', 'D', 'R', 'P', 'F', 'H', 'M', 'SC', 'S+D+R+P')}
SETS.update(Seven=SEVEN, Eight=EIGHT)


def table_rows(text, label):
    marker = '\\label{' + label + '}'
    if text.count(marker) != 1:
        raise ValueError('table label missing or duplicated: ' + label)
    part = text.split(marker, 1)[1].split('\\end{table}', 1)[0]
    if part.count('\\midrule') != 1 or part.count('\\bottomrule') != 1:
        raise ValueError('unexpected table boundaries')
    part = part.split('\\midrule', 1)[1].split('\\bottomrule', 1)[0]
    result = {}
    for line in part.splitlines():
        if not line.strip():
            continue
        if not line.strip().endswith('\\\\'):
            raise ValueError('unexpected table row syntax')
        cells = [x.strip() for x in line.strip()[:-2].split('&')]
        if cells[0] in result:
            raise ValueError('duplicate table row')
        result[cells[0]] = cells[1:]
    return result


def check(text, rows):
    lookup = {}
    for row in rows:
        if row['summary_level'] != 'fold_type_macro' or row['feature_mode'] != 'values_plus_missing':
            raise ValueError('unexpected CSV scope')
        key = row['combination'], row['fold_type'], row['quantity']
        if key in lookup:
            raise ValueError('duplicate CSV endpoint')
        lookup[key] = row
    errors, checked = [], 0
    main = table_rows(text, 'tab:equal60-main')
    if set(main) != set(SETS):
        raise ValueError('main table roster differs')
    for name, combo in SETS.items():
        expected = []
        for arm in ARMS:
            row = lookup[combo, arm, 'all']
            expected.append(f"{float(row['roc_auc']):.4f} / {100*float(row['balanced_accuracy']):.2f}")
            checked += 2
        if main[name] != expected:
            errors.append({'table': 'main', 'row': name, 'actual': main[name], 'expected': expected})
    caps = table_rows(text, 'tab:equal60-caps')
    if set(caps) != {'25', '50', '100', '200', 'All'}:
        raise ValueError('cap table roster differs')
    for cap, actual in caps.items():
        expected = [f"{100*float(lookup[combo, 'generator_holdout', cap.lower()]['balanced_accuracy']):.2f}"
                    for combo in (SEVEN, EIGHT)]
        checked += 2
        if actual != expected:
            errors.append({'table': 'caps', 'row': cap, 'actual': actual, 'expected': expected})
    return {'status': 'pass' if not errors else 'fail', 'numeric_fields_checked': checked,
            'errors': errors, 'scope': 'two displayed tables versus retained CSV; not upstream model audit'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tex', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    args = parser.parse_args()
    with args.csv.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = check(args.tex.read_text(), rows)
    result['inputs'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (args.tex, args.csv)}
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result['status'] == 'pass' else 1)
