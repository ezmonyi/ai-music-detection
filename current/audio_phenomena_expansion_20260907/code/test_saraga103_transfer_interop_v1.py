#!/usr/bin/env python3
"""One persistent synthetic scorer -> independent auditor interoperability run."""
import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import audit_saraga103_transfer_v1 as auditor
import score_saraga103_frozen_v4 as scorer
from test_score_saraga103_frozen_v4 import fixture, put


def write_csv(path, records):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(root):
    root = root.resolve()
    root.mkdir()  # A new isolated synthetic namespace; never overwrite or resume.
    prepared = root/'synthetic_fixture'
    _, meta, raw = fixture(prepared)
    old, fhm = [], []
    for identity, values in zip(meta.to_dict('records'), raw.to_dict('records')):
        index = int(identity['id'].rsplit('_', 1)[1])
        native = '44100' if index < 5 else '48000'
        sd = dict(item_id=identity['id'], label='0', source_id=scorer.SOURCE,
                  group_id=identity['group_id'], status='complete', native_sample_rate_hz=native)
        fh = dict(identity, source_id=scorer.SOURCE, extraction_status='ok', native_sample_rate_hz=native)
        for family, columns in auditor.FAMILIES.items():
            for column in columns:
                (sd if family in ('S','D','R','P') else fh)[column] = values[column]
        old.append(sd)
        fhm.append(fh)
    old_path, fhm_path = prepared/'derived_old4.csv', prepared/'derived_fhm.csv'
    write_csv(old_path, old)
    write_csv(fhm_path, fhm)
    original_load_synthetic = scorer.load_synthetic

    def bound_synthetic_loader(path, bindings):
        result = original_load_synthetic(path, bindings)
        # Extra derived numerical inputs are transparently bound at this synthetic
        # fixture seam; original scoring/authorization/publication code is intact.
        bindings.file(old_path)
        bindings.file(fhm_path)
        return result

    common = ['--synthetic-test-only', '--synthetic-fixture', str(prepared)]
    draft, scored = root/'draft', root/'scored'
    receipt_path = root/'synthetic_only_frozen_receipt.json'
    with patch.object(scorer, 'load_synthetic', side_effect=bound_synthetic_loader), \
         patch.object(scorer, 'validate_measurements', side_effect=AssertionError('real admission forbidden')), \
         patch.object(scorer.M, 'load_old', side_effect=AssertionError('real old model admission forbidden')), \
         patch.object(np.linalg, 'solve', side_effect=AssertionError('model fitting forbidden')), \
         patch.object(np, 'median', side_effect=AssertionError('new median fitting forbidden')):
        scorer.main(['--stage','draft',*common,'--output-dir',str(draft)])
        frozen = scorer.read_json(draft/'scoring_draft.json')
        frozen.update(status='frozen', independent_review=dict(approved=True,reviewer='root',
            reviewed_utc='synthetic_interoperability_test_only_no_real_authorization'))
        put(receipt_path, frozen)
        scorer.main(['--stage','score',*common,'--receipt',str(receipt_path),'--output-dir',str(scored)])
    inputs = dict(old4_csv=old_path, fhm_csv=fhm_path, models=prepared/'fold_models.json', model_index=prepared/'model_index.csv')
    with patch.object(scorer, 'replay', side_effect=AssertionError('auditor may not call scorer replay')), \
         patch.object(scorer.M, 'replay', side_effect=AssertionError('auditor may not call Mureka replay')), \
         patch.object(np.linalg, 'solve', side_effect=AssertionError('auditor may not fit')), \
         patch.object(np, 'median', side_effect=AssertionError('auditor may not fit medians')):
        report, bindings = auditor.audit(scored, receipt_path, auditor.sha(receipt_path), inputs, synthetic=True)
        auditor.structure.publish_report(report, bindings, root/'independent_numerical_audit.json')
    assert report['rows'] == 7 and report['prediction_rows'] == 22225
    assert report['component_summary_rows'] == 15875 and report['aggregate_cells'] == 635
    result = dict(status='passed',synthetic_test_only=True,real_admission_performed=False,
        real_saraga_scoring_performed=False,model_fitting_performed=False,gpu_used=False,
        existing_scorer_fixture=True,scorer_code_modified=False,
        synthetic_loader_extension='bind derived old4/FHM CSVs only; original model/feature results returned unchanged',
        scorer_replay_disabled_during_independent_audit=True,
        rows=7,prediction_rows=22225,model_summary_rows=3175,component_summary_rows=15875,
        aggregate_cells=635,overview_rows=60,all_exact_decisions_passed=report['all_decisions_exact'],
        score_max_absolute_error=report['score_max_absolute_error'],
        per_recording_fp=4,per_recording_tn=3,per_recording_false_positive_rate=4/7,
        equal_component_false_positive_rate=8/15,
        source_files_sha256={str(path):auditor.sha(path) for path in inputs.values()},
        code_sha256={str(Path(module.__file__).resolve()):auditor.sha(Path(module.__file__).resolve()) for module in (scorer,auditor)},
        harness_sha256=auditor.sha(Path(__file__).resolve()),
        synthetic_scoring_receipt_sha256=auditor.sha(receipt_path),
        score_publication_commit_sha256=auditor.sha(scored/'COMMIT.json'),
        independent_audit_sha256=auditor.sha(root/'independent_numerical_audit.json'),
        output_root=str(root))
    put(root/'interop_receipt.json',result)
    print(json.dumps(result,sort_keys=True),flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args().output_dir)
