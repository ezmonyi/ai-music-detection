"""Check the displayed Exact10 intervals against two saved CSVs; no resampling."""
import csv
import hashlib
import json
from pathlib import Path
from audit_thesis_equal60_tables_v1 import table_rows

ROOT = Path(__file__).resolve().parent.parent


def check(text, groups, sources):
    def select(rows):
        selected = [r for r in rows if r['metric'] == 'delta_J']
        if len(selected) != 3 or any(r['baseline_combination'] != 'S+D' for r in selected):
            raise ValueError('expected three fixed-baseline delta-J rows')
        result = {r['added_combination']: r for r in selected}
        if set(result) != {'S+D+F', 'S+D+H', 'S+D+F+H'}:
            raise ValueError('unexpected/missing/duplicate comparison')
        return result
    g, s = select(groups), select(sources)
    actual = table_rows(text, 'tab:exact10-conditional')
    if set(actual) != {'F', 'H', 'F+H'}:
        raise ValueError('unexpected displayed comparisons')
    errors = []
    for addition, cells in actual.items():
        a, b = g['S+D+' + addition], s['S+D+' + addition]
        if abs(float(a['point_estimate']) - float(b['point_estimate'])) > 1e-12:
            raise ValueError('group/source point mismatch')
        expected = [f"${float(a['point_estimate']):+.6f}$",
                    f"$[{float(a['percentile_95_ci_low']):.6f},{float(a['percentile_95_ci_high']):.6f}]$",
                    f"$[{float(b['empirical_source_95_low']):.6f},{float(b['empirical_source_95_high']):.6f}]$"]
        if cells != expected:
            errors.append({'row': addition, 'actual': cells, 'expected': expected})
    return {'status': 'pass' if not errors else 'fail', 'numeric_fields_checked': 15,
            'errors': errors, 'scope': 'displayed point/interval table versus saved CSVs only'}


if __name__ == '__main__':
    tex = ROOT / 'latex/final_thesis_completed_transfer_results_en.tex'
    group = ROOT / 'results/uncertainty_sdrfh10_v1/uncertainty_summary.csv'
    source = ROOT / 'results/uncertainty_sdrfh10_v1/source_sensitivity/sensitivity_summary.csv'
    with group.open(newline='') as stream:
        groups = list(csv.DictReader(stream))
    with source.open(newline='') as stream:
        sources = list(csv.DictReader(stream))
    result = check(tex.read_text(), groups, sources)
    result['inputs'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (tex, group, source)}
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result['status'] == 'pass' else 1)
