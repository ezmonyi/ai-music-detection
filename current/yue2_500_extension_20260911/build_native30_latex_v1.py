"""Render committed Native30 primary endpoints into a new English LaTeX section."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/audio_phenomena_expansion_20260907'
SOURCE=BASE/'native30_evaluation_report_v3_20260911'
TARGET=BASE/'latex/native30_seven_family_results_20260912_en.tex'


def main():
    commit=json.loads((SOURCE/'COMMIT.json').read_text())
    data=(SOURCE/'summary.json').read_bytes()
    assert hashlib.sha256(data).hexdigest()==commit['products']['summary.json']['sha256']
    summary=json.loads(data)
    cells=summary['primary_protocol_cells']
    assert len(cells)==1905
    selections=['S','D','R','P','F','H','SC','S+D+R+P','S+D+R+P+F+H+SC']
    caps=[25,50,100,200,'all']
    text=[r'\section{Native30 seven-family results}',r'\label{sec:native30-seven}',
          r'This section reports the completed seven-family Native30 evaluation, not the subsequent BC or YuE2 extension. '
          r'The cohort contains 3,830 recordings: 1,664 human and 2,166 AI, drawn from eleven sources. '
          r'The committed evaluation contains 51,562 valid model instances (36,830 primary and 14,732 diagnostic), '
          r'with zero recorded failed model instances. There are 127 nonempty feature subsets, five training caps, '
          r'and separately defined source-held-out and grouped-descriptive protocols.',
          r'\subsection{Endpoint and reporting convention}',
          r'The primary mode combines descriptor values with missingness indicators. Training rows alone determine '
          r'medians and weighted scaling. Ridge regularization is fixed at 10 and the decision threshold at 0.5. '
          r'Scores are unbounded decision values, not calibrated probabilities. No held-source threshold tuning '
          r'or post-hoc winner selection is performed.',
          r'Tables report balanced accuracy (BA) in percent. Correctness is averaged within global groups, '
          r'then equally across sources within each class, and finally equally across classes. Protocol summaries '
          r'weight held-source experiments equally. The caps constrain unique global groups per remaining '
          r'training source, not total recordings. Transitive source and test dependencies are excluded from training. '
          r'The exhaustive 1,905 primary cells remain available; the tables below show the singleton families, '
          r'the original four-family set, and the full seven-family set without claiming that these are optimal.',
          r'S denotes common-band vocal spectral descriptors; D onset-conditioned dynamics; R rhythm; '
          r'P phrase/section-length descriptors; F phase/group delay; H pitch-class/chroma organization; '
          r'and SC stereo descriptors. H does not measure acoustic overtone shape.']
    for protocol in sorted({r['fold_type'] for r in cells}):
        lookup={(r['combination'],str(r['quantity'])):r for r in cells if r['fold_type']==protocol}
        if protocol=='ordinary_group_holdout_descriptive':
            text.append(r'\clearpage')
        text += [r'\subsection{'+protocol.replace('_',' ').capitalize()+'}',
                 r'\begin{center}\small',r'\begin{tabular}{lrrrrr}\toprule',
                 r'Features & 25 & 50 & 100 & 200 & All \\ \midrule']
        for selection in selections:
            values=[lookup[(selection,str(cap))]['macro_equal_group_equal_source_BA'] for cap in caps]
            text.append(selection+' & '+' & '.join(f'{100*v:.2f}' for v in values)+r' \\')
        text += [r'\bottomrule\end{tabular}\end{center}']
    text += [r'\subsection{Interpretation}',
             r'At the all-data cap, the four-family model obtains 65.79\% BA under generator holdout, '
             r'65.65\% under human-source holdout, and 74.30\% under ordinary grouped descriptive testing. '
             r'The full seven-family model obtains 61.63\%, 67.99\%, and 75.66\%, respectively. '
             r'Thus the additional families improve the displayed human-source and grouped summaries while '
             r'reducing the displayed held-generator endpoint. More interpretable descriptors are not '
             r'automatically more transferable descriptors.',
             r'The differences are descriptive comparisons within the recorded protocol, not causal effects '
             r'of adding a phenomenon to music. No confidence interval is inferred from five fixed buckets. '
             r'AUC is retained within each fitted fold; raw scores from different models are not pooled '
             r'into one global AUC. Repeated opposite-class predictions across held-source experiments are '
             r'dependent occurrences, not additional independent recordings.',
             r'The training-cap curves are not monotonic. Selecting the largest observed cell after inspecting '
             r'the full lattice would require separate untouched confirmation before an optimality claim. '
             r'These results neither establish a universal AI fingerprint nor justify an accusation against '
             r'an individual musician. Scientific validation of the intended dynamics, rhythm, and phrase '
             r'measurements remains distinct from their classification usefulness.',
             r'\subsection{Reproduction references and pending extensions}',
             r'Code: \path{artifacts/audio_phenomena_expansion_20260907/code/evaluate_native30_v3.py} '
             r'and \path{summarize_native30_evaluation_v3.py} in the same directory. '
             r'Full report outputs: \path{artifacts/audio_phenomena_expansion_20260907/native30_evaluation_report_v3_20260911/}. '
             r'The source \path{summary.json} and \path{COMMIT.json} bind the reported cells. '
             r'The compact CSV is \path{deliverables/overnight_snapshot_20260912_v1/all_seven_family_primary_cells.csv}.',
             r'The new public archive is \href{https://huggingface.co/datasets/EZMONYI/music-ai-human-test-audio}{the EZMONYI HF evaluation dataset}. '
             r'Its first publication contains the intermediate report and tables, not the full audio archive. '
             r'The eight-family BC comparison, YuE2 transfer, and expanded-cohort evaluations remain pending '
             r'in this document version. Their metadata schedules are not treated as completed fits.']
    with TARGET.open('x') as stream:
        stream.write('\n\n'.join(text)+'\n')
    print(TARGET)


if __name__=='__main__':
    main()
