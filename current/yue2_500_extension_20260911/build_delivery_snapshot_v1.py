"""Produce an English, non-selective summary of committed seven-family results."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'artifacts/audio_phenomena_expansion_20260907/native30_evaluation_report_v3_20260911'
OUT = ROOT/'deliverables/overnight_snapshot_20260912_v1'


def main():
    summary = json.loads((SOURCE/'summary.json').read_text())
    commit = json.loads((SOURCE/'COMMIT.json').read_text())
    for name, binding in commit['products'].items():
        artifact = SOURCE/name
        assert artifact.stat().st_size == binding['bytes']
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == binding['sha256']
    assert summary['status'] == 'complete_reporting_only_no_selection'
    cells = summary['primary_protocol_cells']
    assert len(cells) == 1905
    OUT.mkdir(parents=True,exist_ok=True)
    fields = ['fold_type','combination','quantity','macro_equal_group_equal_source_BA',
              'macro_human_recall','macro_ai_recall','descriptive_mean_single_model_fold_auc',
              'eligible_model_count','omitted_model_count','coverage_key']
    with (OUT/'all_seven_family_primary_cells.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(cells)
    selections = ['S','D','R','P','F','H','SC','S+D+R+P','S+D+R+P+F+H+SC']
    caps = [25,50,100,200,'all']
    report = ['# Overnight experiment delivery snapshot (English)', '',
              'This is an intermediate report, not a claim of completed final delivery.', '',
              '## Completed evidence', '',
              'The committed seven-family evaluation covers 3,830 recordings from 11 sources '
              '(1,664 human and 2,166 AI). It contains 51,562 eligible model evaluations: '
              '36,830 primary and 14,732 diagnostic. The reporter provides 1,905 primary '
              'protocol cells. The CSV accompanying this report retains every primary cell, '
              'rather than only the strongest observed combination.', '',
              'S denotes vocal high-frequency spectral descriptors; D dynamics; R rhythm; '
              'P phrase/section-length descriptors; F phase/group delay; H pitch-class/chroma '
              'organization; SC stereo descriptors. H is not an acoustic overtone-shape metric.', '',
              '## Balanced accuracy versus training cap', '',
              'Values below are percentages of macro equal-group/equal-source balanced accuracy. '
              'The cap is the schedule training quantity, not total dataset size. Tables show '
              'predeclared singleton families, the original four-family combination, and the '
              'full seven-family combination; this is not winner selection.', '']
    for protocol in sorted({r['fold_type'] for r in cells}):
        report += ['### '+protocol.replace('_',' '),'',
                   '| Features | 25 | 50 | 100 | 200 | All |',
                   '|---|---:|---:|---:|---:|---:|']
        lookup = {(r['combination'],str(r['quantity'])):r for r in cells if r['fold_type']==protocol}
        for selection in selections:
            values = [lookup[(selection,str(cap))]['macro_equal_group_equal_source_BA'] for cap in caps]
            report.append('| '+selection+' | '+' | '.join(f'{100*v:.2f}' for v in values)+' |')
        report.append('')
    report += ['## Interpretation and limits','',
               'Compare generalization protocols separately. Source-held-out performance '
               'tests transfer beyond the observed source, whereas grouped descriptive folds '
               'do not establish performance on a new generator. Increasing the training cap '
               'need not improve performance monotonically. Missingness, bandwidth, source '
               'composition, production practices, and short temporal support remain potential '
               'confounds; feature separability is not proof of an intrinsic AI signature.', '',
               *['- '+note for note in summary['scope_notes']], '',
               '## Remaining final-delivery work','',
               '- Complete and verify the eight-family BC evaluation and its report.',
               '- Complete YuE2 inference, feature extraction, validation, held-out transfer '
               'and expanded-cohort combination/data-size evaluation.',
               '- Integrate the final findings into the English thesis and preserve all versions.',
               '- Create and verify the requested new HF dataset and GitHub repository; '
               'credentials and redistribution permissions must be resolved without publishing secrets.',
               '- Synchronize final outputs locally and verify their checksums.', '',
               '## Artifact references','',
               f'- Source report: `{SOURCE}`',
               '- Full primary results: `all_seven_family_primary_cells.csv`',
               '- Provenance: `provenance.json`', '']
    (OUT/'REPORT_EN.md').write_text('\n'.join(report))
    provenance = dict(source=str(SOURCE),source_commit_sha256=hashlib.sha256((SOURCE/'COMMIT.json').read_bytes()).hexdigest(),
                      source_summary_sha256=hashlib.sha256((SOURCE/'summary.json').read_bytes()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      primary_cells=len(cells),final_delivery_complete=False,
                      outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()})
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(OUT)


if __name__ == '__main__':
    main()
