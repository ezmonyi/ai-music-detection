#!/usr/bin/env python3
"""Render fixed, verified exploratory-v5 tables into English Markdown/LaTeX.

No fitting, new metrics, ranking, calibration, or audio access. All report
numbers are selected from the independently checked presentation publication.
The report is an analysis supplement, not a replacement of earlier PDFs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def tex(s):
    return str(s).replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact-root', type=Path, required=True)
    args = ap.parse_args()
    root = args.artifact_root.resolve()
    p = root/'results/equal60_exploratory_v5_presentation_v1'
    audit = root/'audit/equal60_exploratory_v5_numerical_audit_v1.json'
    check = root/'audit/equal60_v5_presentation_independent_check_v1.json'
    c = json.loads(check.read_text())
    assert c['status'] == 'passed_independent_presentation_table_check'
    assert c['synthetic_test_only'] is False
    assert c['numerical_audit_sha256'] == sha(audit)
    assert c['presentation_commit_sha256'] == sha(p/'COMMIT.json')
    for name, digest in c['presentation_files_sha256'].items():
        assert sha(p/name) == digest, name
    def read(name):
        with (p/name).open(newline='') as f:
            return list(csv.DictReader(f))
    rows = read('primary_pair_macro_all_arms.csv')
    macro = {(r['fold_type'], r['combination'], r['quantity']): r for r in rows if r['summary_level'] == 'fold_type_macro'}
    sources = {(r['source_group'], r['combination']): r for r in read('primary_source_endpoints_all_arms.csv') if r['source_group'] == r['heldout_source'] and r['quantity'] == 'all'}
    diagnostic = {(r['fold_type'], r['combination'], r['feature_mode']): r for r in read('diagnostic_all_cap_pair_macro_all_arms.csv') if r['summary_level'] == 'fold_type_macro'}
    types = ['human_source_holdout', 'generator_holdout', 'ordinary_group_holdout_descriptive']
    labels = ['Human source', 'Generator', 'Ordinary group']
    base = 'S+D+R+P'
    seven = base+'+F+H+M'
    fixed = list('SDRPFHM')+[base, base+'+F', base+'+H', base+'+M', seven]
    short = {base: 'Base', base+'+F': 'Base+F', base+'+H': 'Base+H', base+'+M': 'Base+M', seven: 'All seven'}
    md = ['# Seven-family exploratory results on the expanded cohort', '', '7 September 2026. Completed and independently checked real-music experiment.', '']
    lt = [r'\section{Seven-family results on the expanded cohort}', r'\label{sec:equal60-v5-results}']
    def paragraph(s):
        md.extend([s, ''])
        lt.extend([tex(s), ''])
    def section(s):
        md.extend(['## '+s, ''])
        lt.extend([r'\subsection{'+tex(s)+'}', ''])
    def table(caption, headers, body):
        md.extend([caption, '', '| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'])
        md.extend('| '+' | '.join(map(str, r))+' |' for r in body)
        md.append('')
        lt.extend([r'\begin{table}[htbp]', r'\centering\small', r'\begin{tabular}{@{}l'+'r'*(len(headers)-1)+r'@{}}', r'\toprule', ' & '.join(map(tex, headers))+r' \\', r'\midrule'])
        lt.extend(' & '.join(map(tex, r))+r' \\' for r in body)
        lt.extend([r'\bottomrule', r'\end{tabular}', r'\caption{'+tex(caption)+'}', r'\end{table}', ''])
    paragraph('The additional descriptors improve some comparisons but do not establish a transferable AI-music detector. At the all cap, all seven families reach 90.18% balanced accuracy in ordinary group holdout and 84.20% in Human-source holdout, but only 60.10% in generator holdout. The corresponding four-family baseline reaches 86.98%, 79.54%, and 60.07%. The strong ordinary-group result must not substitute for an unseen-generator result.')
    section('Cohort, parameters, and estimands')
    paragraph('The experiment uses 2,207 exact 60-second excerpts: 1,311 Human and 896 AI recordings, across six Human sources and two generators. The 1,779 recorded global groups define split exclusion and weighting, but singleton provenance keys do not establish independent artists. The old 438 locked and 50 pilot recordings and their groups are excluded. Mureka and Saraga are exposed exploratory development reuse, not untouched external confirmation.')
    paragraph('Feature families are S (15 separated-vocal high-frequency descriptors), D (3 non-vocal dynamics proxies), R (3 rhythm-variation proxies), P (6 section-duration/bar-count proxies), F (15 phase/group-delay descriptors), H (6 pitch-class trajectory descriptors), and M (6 long-range transposition-aware recurrence descriptors). All 127 nonempty subsets are evaluated at caps 25, 50, 100, 200, and all. Each cap limits training groups per class/source, not total songs. All-cap values-only and missingness-only models are diagnostics.')
    paragraph('Seed 20260907 defines five global group-hash buckets. The complete held-out source and every current test-bucket group are excluded from training. There are 28 valid Human-source folds, 10 generator folds, and 5 ordinary-group folds. Two empty Saraga folds are omitted. Each generator-held-out arm trains on one other generator; this is not training on two generators and testing a third.')
    paragraph('Training-only medians, scaling and missing indicators precede weighted ridge regression with penalty 10 and an unpenalized intercept. Class, source, group and recording weights sum to training n. The raw identity-link score is classified as AI at score >= 0.5; it is not a probability. No threshold tuning, full-cohort refit or model selection is performed. Fixed ridge with weights summing to n makes effective mean-loss regularization 10/n, so cap differences change composition and regularization as well as sample quantity.')
    paragraph('AUC and balanced accuracy (BA) use a fixed hierarchy: average Human/generator source pairs within each fold, average valid folds within each held-source arm, then average arms equally within each fold type. These are separate two-class test estimands. There is no pooled out-of-fold AUC across separately fitted score scales. Source-specific correct rates instead use unique out-of-fold recordings within one arm; equal-component rates average each global group once. Fold ranges are not confidence intervals.')
    section('Single families and fixed additions')
    body = []
    for combo in fixed:
        rr = [macro[t, combo, 'all'] for t in types]
        body.append([short.get(combo, combo)]+[f"{float(r[k]):.4f}" for r in rr for k in ('roc_auc','balanced_accuracy')])
    table('All-cap two-class fold-pair macro results. Base means S+D+R+P. H-src denotes Human-source holdout; Gen denotes generator holdout; Group is ordinary group holdout.', ['Features','H-src AUC','H-src BA','Gen AUC','Gen BA','Group AUC','Group BA'], body)
    paragraph('F added to the baseline raises Human-source BA by 2.51 percentage points at all, but lowers generator-held-out BA by 3.58 points. These directions hold across all five caps. H yields a smaller all-cap generator BA increase of 1.52 points; M yields 0.78 points. Neither effect is a significance claim. All seven raise generator AUC from 0.7742 to 0.8031, while BA changes by only +0.04 percentage points. Better within-pair ranking does not establish a transferable fixed-threshold decision boundary.')
    paragraph('The single-family results also depend strongly on the split. S has Human-source AUC 0.8217 but generator AUC 0.6761 and BA 0.5393. F has Human-source AUC 0.8613 but generator AUC 0.7228 and BA 0.6306. R and H have all-cap generator BA 0.6871 and 0.6816, respectively. M alone is weak in this experiment, with Human-source AUC 0.6258 and generator AUC 0.6330. These are descriptive observations about fixed candidates, not a post-hoc winning pipeline.')
    section('Training-cap ablation')
    ranges = {}
    range_rows = read('training_row_group_ranges.csv')
    for cap in ('25','50','100','200','all'):
        q = [r for r in range_rows if r['quantity'] == cap and r['training_source'] == '__total__' and r['summary_level'] == 'held_source_arm']
        assert q
        ranges[cap] = str(min(int(r['min_training_rows']) for r in q))+'-'+str(max(int(r['max_training_rows']) for r in q))
    body = []
    for cap in ranges:
        body.append([cap, ranges[cap]]+[f"{100*float(macro[t,combo,cap]['balanced_accuracy']):.2f}" for t in types for combo in (base, seven)])
    table('Balanced accuracy (%) across five caps. Row ranges cover all 43 folds. B = four-family baseline; 7 = all seven. Changes are not causal quantity-only effects.', ['Cap','Train rows','H-src B','H-src 7','Gen B','Gen 7','Group B','Group 7'], body)
    paragraph('More training groups do not monotonically improve generalization. From cap 25 to all, all-seven ordinary-group BA rises from 88.03% to 90.18% and Human-source BA from 79.32% to 84.20%. Generator BA varies non-monotonically (59.07%, 62.54%, 60.19%, 60.96%, 60.10%). The four-family generator BA drops from 62.94% at cap 25 to 60.07% at all. Data expansion alone is not demonstrated to solve transfer, and this ablation cannot isolate a causal diversity effect.')
    section('Held-out source failures and group dependence')
    names = [('MTG-Jamendo','MTG'),('human_maestro_v3','MAESTRO'),('human_medleydb','MedleyDB'),('human_moisesdb','MoisesDB'),('human_saraga_hindustani_v1','Saraga'),('human_urmp','URMP'),('Mureka_v9','Mureka'),('Suno','Suno')]
    body = []
    for source, label in names:
        a, b = sources[source,base], sources[source,seven]
        body.append([label, a['correct_recordings']+'/'+a['unique_oof_recordings'], b['correct_recordings']+'/'+b['unique_oof_recordings']]+[f"{100*float(r[k]):.2f}" for k in ('recording_rate','equal_component_rate') for r in (a,b)])
    table('All-cap held-source correct counts and rates (%). Human rows measure specificity; AI rows measure sensitivity. B/7 denote baseline/all seven; Rec/Grp denote recording/equal-component weighting. These rows belong to separate held-source arms.', ['Source','B correct','7 correct','Rec B','Rec 7','Grp B','Grp 7'], body)
    paragraph('With all seven families, Human false-positive rates remain 42.83% for held-out MTG and 56.31% for held-out Saraga, versus 2.67% for MAESTRO and 3.03% for URMP. A high macro score therefore coexists with substantial errors on particular Human sources. Saraga has only five components: its apparent baseline-to-seven improvement is +22.33 percentage points by recording and +12.71 points by component. These weightings answer different questions and neither supports significance with five groups.')
    paragraph('Unseen-generator recall is particularly low at the unchanged threshold: all seven detect 178/500 Mureka recordings (35.60%) and 82/396 Suno recordings (20.71%). The baseline detects 196/500 (39.20%) and 85/396 (21.46%). Adding F alone reduces these to 144/500 (28.80%) and 60/396 (15.15%). Generator-holdout BA must not be read as AI recall: correct Human decisions also contribute to BA.')
    paragraph('Earlier frozen-transfer values are not numerically comparable as controlled improvements: that experiment used different training composition, folds and estimands. In particular, the prior Mureka/Saraga evaluation averaged per-model endpoints over the external sets, whereas the current source endpoints pool unique out-of-fold decisions within each new development arm. No old result is overwritten or relabeled as a fresh external test.')
    section('Missingness diagnostics')
    body = []
    for t, label in zip(types,labels):
        for mode in ('values_plus_missing','median_only','missingness_only'):
            r = macro[t,seven,'all'] if mode == 'values_plus_missing' else diagnostic[t,seven,mode]
            body.append([label, {'values_plus_missing':'Values + flags','median_only':'Values only','missingness_only':'Flags only'}[mode], f"{float(r['roc_auc']):.4f}", f"{100*float(r['balanced_accuracy']):.2f}"])
    table('All-seven all-cap diagnostic models. Missing values use training-only median imputation; flags indicate descriptor availability. These are separately fitted diagnostic arms, not selected replacements.', ['Split','Input mode','AUC','BA (%)'], body)
    paragraph('All-seven flags-only BA is approximately 59-60%, showing that availability patterns carry label-associated information in this cohort. Nevertheless, values-only AUC remains 0.9249 for Human-source holdout, 0.7988 for generator holdout, and 0.9562 for ordinary-group holdout; the full result cannot be attributed solely to missing flags. P is more sensitive to this issue: its generator BA is 64.89% with values and flags, 56.07% with values only, and 59.55% with flags only. Missingness and descriptor values are not cleanly additive causal contributions.')
    section('Signal representation and next optimization')
    paragraph('The common native content interval is 60 seconds, but family representations differ. S/D/R/P use a 44.1 kHz, two-channel FLOAT front end, with model-internal conversions recorded separately. F/H/M are derived directly from the native crop, channel-averaged, DC-removed and polyphase-resampled to 16 kHz. F uses a 1,024-sample Hann window and 256-sample hop; its final retained bin is 7,593.75 Hz. F is not an above-8-kHz detector. H/M use a 4,096-sample window and 1,024-sample hop, with 12 pitch classes and one-second summaries. H/M are not ground-truth chords or musical form. D is an amplitude-dynamics proxy, not recovered MIDI velocity.')
    paragraph('The findings motivate prospective transfer-focused optimization, not deployment of the highest ordinary-group score. Priorities are: obtain a genuinely eligible third generator with at least 60 seconds of native context and independent prompt/provenance groups; test source-controlled signal transformations and stem-reliability gates; and separately preregister a normalized-regularization and training-only calibration comparison. No threshold may be tuned on the current held-out predictions and then reported as fresh validation.')
    paragraph('The additional stereo candidate has passed 14 mathematical tests only and is not admitted as an eighth validated family. Its external music-phenomenon benchmark is still pending. Native-channel metadata identifies 33 mono URMP recordings that were duplicated into two channels. A future stereo comparison must use the same eligible population for baseline and augmented models; otherwise removing an entire Human source changes the question. New data collection should prioritize independent sources/groups over many files from an existing dominant component.')
    section('Reproducibility and numerical verification')
    paragraph('The completed run contains 27,305 primary and 10,922 diagnostic model instances, with 7,633,970 and 3,053,588 predictions. The independent numerical audit verified every training transformation, normal-equation stationarity check, score replay, threshold decision and metric. Maximum score replay error is 2.6645352591003757e-15 and every decision matches. It did not independently rerun the optimizer. The subsequent independent presentation checker reconstructs all table denominators, source/fold/arm averages, increments and rendered numbers from audited outputs.')
    refs = [
        'EQUAL60_EXPLORATORY_V5_PROTOCOL_EN.md',
        'preregistration/equal60_exploratory_v5_frozen_v1.json',
        'code/prepare_evaluation_inputs_v5.py', 'code/evaluate_new_phenomena_v5.py',
        'code/audit_equal60_v5_results.py', 'code/present_equal60_v5_results.py',
        'code/check_equal60_v5_presentation.py', 'code/plot_equal60_v5_results.py',
        'code/build_equal60_v5_report.py',
        'results/equal60_exploratory_package_v5_v1/',
        'results/equal60_exploratory_v5_results_v1/',
        'results/equal60_exploratory_v5_presentation_v1/',
        'results/equal60_exploratory_v5_figures_v1/',
        'audit/equal60_exploratory_v5_numerical_audit_v1.json',
        'audit/equal60_v5_presentation_independent_check_v1.json',
        'V5_SIGNAL_REPRESENTATION_CLARIFICATION_EN.md',
        'V5_CHANNEL_PROVENANCE_INVENTORY_EN.md',
        'SARAGA_MUREKA_FROZEN_TRANSFER_RESULTS_EN.md',
    ]
    paragraph('All paths below are relative to artifacts/audio_phenomena_expansion_20260907/. Complete scalar tables preserve every one of the 635 primary candidate-cap cells per reporting arm. The three separate scientific figures show the exhaustive grid, source-specific endpoints/increments, and missingness diagnostics. They are supplementary zoomable figures, not reduced unreadable page thumbnails. The original thesis PDF and earlier supplements remain unchanged.')
    md.extend(['- `'+s+'`' for s in refs]); md.append('')
    lt.append(r'\begin{itemize}\small')
    lt.extend(r'\item \path{'+s+'}' for s in refs)
    lt.extend([r'\end{itemize}', ''])
    for label, digest in [('Result COMMIT', c['raw_result_commit_sha256']),('Numerical audit', c['numerical_audit_sha256']),('Presentation COMMIT', c['presentation_commit_sha256']),('Independent table check', sha(check))]:
        md.extend([label+': `'+digest+'`.', ''])
        lt.extend([tex(label)+r': \path{'+digest+'}.', ''])
    outputs = {
        root/'EQUAL60_V5_RESULTS_EN.md': '\n'.join(md)+'\n',
        root/'latex/equal60_v5_results_en.tex': '\n'.join(lt)+'\n',
        root/'latex/ai_music_detection_v5_results_en_20260907_v1.tex': r'''\documentclass[10pt,a4paper]{article}
\usepackage[a4paper,margin=18mm]{geometry}
\usepackage{amsmath,booktabs,longtable,graphicx,xcolor,xurl}
\usepackage[colorlinks=true,urlcolor=blue,linkcolor=blue]{hyperref}
\usepackage{fancyhdr}
\setlength{\emergencystretch}{4em}
\setlength{\parskip}{0.35em}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.12}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{AI/Human Music Detection: Expanded-Cohort Results}
\fancyfoot[C]{\thepage}\setlength{\headheight}{14pt}
\title{Interpretable AI/Human Music Detection\\\large Seven-Family Expanded-Cohort Results}
\author{Thesis experiment archive}\date{7 September 2026}
\begin{document}\maketitle
\input{equal60_v5_results_en}
\clearpage
\appendix
\section{Fixed-candidate results at every training cap}
The following complete 180-row fixed overview is exported directly from the
independently checked tables. All 127 candidates, all held-source arms, source
denominators, and diagnostics are retained in the accompanying CSV publication.
No candidate is selected by this appendix.
\input{../results/equal60_exploratory_v5_presentation_v1/equal60_v5_presentation_tables_en}
\end{document}
''',
    }
    assert all(not q.exists() for q in outputs), 'Refusing to overwrite an existing report'
    for q, text in outputs.items():
        with q.open('x') as f:
            f.write(text)
    assert sha(p/'COMMIT.json') == c['presentation_commit_sha256']
    print(json.dumps({str(q): sha(q) for q in outputs}, indent=2))


if __name__ == '__main__':
    main()
