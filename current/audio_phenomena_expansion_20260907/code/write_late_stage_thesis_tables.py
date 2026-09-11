#!/usr/bin/env python3
"""Render completed exact60 and failed B gate as English LaTeX, no fitting."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result = ROOT/'results/evaluation_fhm_60s_v3'
    manifest_path = result/'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['evaluation_mode'] != 'group_cv_only' or manifest['source_transfer_J'] is not None:
        raise ValueError('Exact60 must be explicitly descriptive, never source-transfer J')
    source = result/'development_group_cv_summary.csv'
    with source.open() as f:
        data = list(csv.DictReader(f))
    combos = ('F','H','M','F+H','F+M','H+M','F+H+M')
    quantities = ('25','50','100','200','all')
    bykey = {(r['combination'],r['quantity']):r for r in data}
    if len(data) != 35 or set(bykey) != {(c,q) for c in combos for q in quantities}:
        raise ValueError('Incomplete exact60 7-by-5 grid')
    if {r['fold_type'] for r in data} != {'ordinary_group_holdout_descriptive'}:
        raise ValueError('Unexpected source-transfer claim')
    out = [r'\subsection{Exact-60-second recurrence association: descriptive only}',
           r'The completed long-context analysis uses 1,604 development recordings (1,208 Human and 396 Suno), five globally blocked group folds and seven F/H/M combinations. The 438 historical locked and 50 pilot recordings are excluded from fitting. The only development AI source is Suno: generator holdout is ineligible, source-transfer $J$ remains undefined, and no cross-generator conclusion can be drawn.', '',
           r'\begin{table}[htbp]',r'\centering\footnotesize\setlength{\tabcolsep}{4pt}',
           r'\caption{Exact60 descriptive group-CV results: each cell is source-pair-averaged AUC/BA, not source-held-out $J$. Training caps are global groups per source. These numbers must not be compared with 30s source-transfer AUC as a causal duration or recurrence gain.}',
           r'\label{tab:fhm60-group-only}',r'\begin{tabular}{lrrrrr}',
           r'\toprule Combination & 25 & 50 & 100 & 200 & All \\',r'\midrule']
    for c in combos:
        cells = [f"{float(bykey[c,q]['roc_auc_source_macro']):.4f}/{float(bykey[c,q]['balanced_accuracy_source_macro']):.4f}" for q in quantities]
        out.append(c+' & '+' & '.join(cells)+r' \\')
    out += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '',
            r'These are within-observed-source associations. Recurrence may reflect repertoire, loops, instrumentation or production in addition to generation. Old S/D/R/P features were not recomputed over this exact60 cohort, so this run does not isolate an M increment over the old pipeline. Missing H/M observations are retained under training-fold imputation and missingness diagnostics, not removed through complete-case filtering.', '']
    long_out = ROOT/'latex/results_fhm60_en.tex'
    long_out.write_text('\n'.join(out))
    breath_path = ROOT/'results/external_breath_events_v2/summary.json'
    audit_path = ROOT/'audit/external_breath_events_v2_independent.json'
    b = json.loads(breath_path.read_text()); audit = json.loads(audit_path.read_text())
    if not audit['artifact_audit_passed'] or b['primary_clips'] != 113 or b['gate_passed']:
        raise ValueError('Unexpected audited breath-gate outcome; do not silently reuse fixed interpretation')
    if audit['recomputed_overall'] != b['overall']:
        raise ValueError('Breath metrics differ from independent recomputation')
    metrics = [('Event precision',b['overall']['precision']),('Event recall',b['overall']['recall']),
               ('Event F1',b['overall']['f1']),('Frame balanced accuracy',b['aggregate_frame_balanced_accuracy']),
               ('Hard-negative frame false-positive fraction',audit['hard_negative_error']['false_positive_fraction'])]
    out = [r'\begin{table}[htbp]',r'\centering\small',
           r'\caption{Completed external B benchmark: leave-one-singer-out, 113 clips, 20 singers and 141 merged reference events. This is event measurement, not AI-music classification.}',
           r'\label{tab:breath-external-failed}',r'\begin{tabular}{lr}',r'\toprule Metric & Value \\',r'\midrule']
    out += [name+' & '+f'{value:.6f}'+r' \\' for name,value in metrics]
    out += [r'TP / FP / FN events & 137 / 1,257 / 4 \\',
            r'Zero-event clips with at least one false event & 38 / 38 \\',
            r'False events in zero-event clips & 352 \\',r'\bottomrule',r'\end{tabular}',r'\end{table}', '',
            r'The predeclared gate requires event precision and F1 both at least 0.70. It fails decisively despite high recall and frame BA. Every zero-event clip has a false alarm; 3,042 of 9,046 labeled hard-negative frames are positive. High frame-level discrimination is therefore not sufficient for reliable breath-event organization. No post-hoc threshold or event-duration rule is tuned on this benchmark, and B is not admitted to the AI/Human classifier. The audit verifies all 230 artifact hashes and event/model records; it does not validate event measurement.', '']
    breath_out = ROOT/'latex/results_breath_en.tex'
    breath_out.write_text('\n'.join(out))
    receipt = {'status':'complete','no_refit':True,'code_sha256':sha(Path(__file__)),
               'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (manifest_path,source,breath_path,audit_path)},
               'outputs':{str(p.relative_to(ROOT)):sha(p) for p in (long_out,breath_out)}}
    (ROOT/'latex/late_stage_table_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
