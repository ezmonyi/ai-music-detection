#!/usr/bin/env python3
"""Build thesis tables from audited numerical artifacts, without fitting."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ('S','D','R','P','F','H')
QUANTITIES = ('25','50','100','200','all')
SELECTED = ('S','D','R','P','F','H','F+H','S+D+R','S+D+R+F','S+D+R+H','S+D+R+F+H','S+D+R+P+F+H')


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data_path = ROOT/'results/presentation_fh_30s_v2/all_63x5_numeric_results.csv'
    group_path = ROOT/'results/uncertainty_fh_30s_v2/uncertainty_summary.csv'
    source_path = ROOT/'results/source_sensitivity_fh_30s_v1/sensitivity_summary.csv'
    data = rows(data_path)
    combos = ['+'.join(c) for n in range(1,7) for c in itertools.combinations(ORDER,n)]
    expected = {(c,q) for c in combos for q in QUANTITIES}
    bykey = {(r['combination'],r['quantity']):r for r in data}
    if len(data) != 315 or set(bykey) != expected:
        raise ValueError('Incomplete or duplicate 63-by-5 result lattice')
    if {r['eligible_rows'] for r in data} != {'3317'} or len({r['cohort_id'] for r in data}) != 1:
        raise ValueError('Unexpected cohort change across cells')
    group = {(r['added_combination'],r['metric']):r for r in rows(group_path)}
    sources = {(r['added'],r['metric']):r for r in rows(source_path)}
    out = [r'\begin{table}[htbp]',r'\centering\small',
           r'\caption{Selected exact-30-second results at the all-groups training cap. AUC columns are source-macro; BA uses the fixed 0.5 threshold. These are development, not untouched test results.}',
           r'\label{tab:fh30-main}',r'\begin{tabular}{lrrrr}',
           r'\toprule Combination & Human-held AUC & AI-held AUC & $J$ & Mean BA \\',r'\midrule']
    for c in SELECTED:
        row = bykey[c,'all']
        nums = [float(row[k]) for k in ('roc_auc_source_macro_human','roc_auc_source_macro_ai','J','mean_BA')]
        out.append(c+' & '+' & '.join(f'{v:.4f}' for v in nums)+r' \\')
    out += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '',
            r'\begin{table}[htbp]',r'\centering\small',
            r'\caption{Predefined paired increments over S+D+R at all groups: 1,000 global-group bootstrap replicates, conditional on frozen models and the observed sources. Intervals are unadjusted percentile intervals.}',
            r'\label{tab:fh30-group-ci}',r'\begin{tabular}{lrrrr}',
            r'\toprule Added & $\Delta J$ & 95\% interval & $\Delta$ Mean BA & 95\% interval \\',r'\midrule']
    for c in ('S+D+R+F','S+D+R+H','S+D+R+F+H'):
        a,b = group[c,'delta_J'],group[c,'delta_equal_mean_balanced_accuracy']
        out.append(c.removeprefix('S+D+R+')+' & '+f"{float(a['point_estimate']):+.4f} & [{float(a['percentile_95_ci_low']):+.4f}, {float(a['percentile_95_ci_high']):+.4f}] & {float(b['point_estimate']):+.4f} & [{float(b['percentile_95_ci_low']):+.4f}, {float(b['percentile_95_ci_high']):+.4f}]"+r' \\')
    out += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '',
            r'Neither F nor F+H has a positive interval bounded away from zero for the equal-direction criterion. H has a negative group-conditional increment here; this is not a universal statement about harmony in generated music. Five replicates contain 40 undefined source-pair cells in total; every macro remains defined using the frozen evaluator\textquotesingle s NaN-skipping rule. All paired point estimates were reproduced to $1.11\times10^{-16}$ or better.', '',
            r'\begin{table}[htbp]',r'\centering\small',
            r'\caption{Additional post-hoc empirical source-composition sensitivity: 10,000 draws of six Human and four AI source labels. The same crossed-source weights apply to both evaluation directions and all comparisons. These ranges do not combine group uncertainty and are not future-source confidence guarantees.}',
            r'\label{tab:fh30-source-sensitivity}',r'\begin{tabular}{lrr}',
            r'\toprule Added & $\Delta J$ empirical 95\% range & $\Delta$ Mean BA empirical 95\% range \\',r'\midrule']
    for c in ('S+D+R+F','S+D+R+H','S+D+R+F+H'):
        a,b = sources[c,'delta_J'],sources[c,'delta_mean_BA']
        out.append(c.removeprefix('S+D+R+')+' & '+f"[{float(a['empirical_source_95_low']):+.4f}, {float(a['empirical_source_95_high']):+.4f}] & [{float(b['empirical_source_95_low']):+.4f}, {float(b['empirical_source_95_high']):+.4f}]"+r' \\')
    out += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '',
            r'All source-reweighted ranges include zero. This sensitivity was added after seeing point estimates and is explicitly post-hoc. It reinforces the source-composition limitation, not a claim of an iid population sample. Models are never refitted and threshold 0.5 remains fixed.', '',
            r'\begingroup\footnotesize\setlength{\tabcolsep}{4pt}',
            r'\begin{longtable}{lrrrrr}',
            r'\caption{Complete 63-combination by five-quantity development lattice. Each cell is $J$/Mean BA. Quantities cap training global groups per source, not total songs. All rows share the same 3,317-recording eligible cohort; realized training sizes vary by fold and source availability.}\label{tab:fh30-full-lattice}\\',
            r'\toprule Combination & 25 & 50 & 100 & 200 & All \\',r'\midrule\endfirsthead',
            r'\toprule Combination & 25 & 50 & 100 & 200 & All \\',r'\midrule\endhead',
            r'\midrule\multicolumn{6}{r}{Continued on next page}\\\endfoot',r'\bottomrule\endlastfoot']
    for c in combos:
        cells = [f"{float(bykey[c,q]['J']):.4f}/{float(bykey[c,q]['mean_BA']):.4f}" for q in QUANTITIES]
        out.append(c+' & '+' & '.join(cells)+r' \\')
    out += [r'\end{longtable}',r'\endgroup','']
    destination = ROOT/'latex/results_fh_table_en.tex'
    destination.write_text('\n'.join(out))
    receipt = {'status':'complete', 'rows':len(data), 'candidate_count':len(combos), 'quantities':list(QUANTITIES),
               'code_sha256':sha(Path(__file__)), 'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (data_path,group_path,source_path)},
               'output':str(destination.relative_to(ROOT)), 'output_sha256':sha(destination), 'no_refit':True}
    (ROOT/'latex/table_generation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
