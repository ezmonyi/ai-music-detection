#!/usr/bin/env python3
"""Retain a synthetic v5 package and no-fit draft; never freezes or runs CV."""
import argparse
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import evaluate_new_phenomena_v5 as evaluator
from test_evaluate_new_phenomena_v5 import package_fixture


def prepare(parent):
    parent=Path(parent)
    evaluator.require(parent.is_absolute() and parent.resolve()==parent and parent.is_dir(),
                      'Existing canonical synthetic output parent required')
    root=Path(tempfile.mkdtemp(prefix='v5_synthetic_interop_',dir=parent)).resolve()
    with patch.object(evaluator.V2,'fit_candidate',side_effect=AssertionError('Synthetic preparation cannot fit')), \
         patch.object(evaluator.V2,'predict_candidate',side_effect=AssertionError('Synthetic preparation cannot predict')), \
         patch.object(evaluator,'authorize',side_effect=AssertionError('Synthetic preparation cannot freeze or authorize')):
        package=package_fixture(root)
        draft=root/'draft'
        evaluator.main(['--stage','draft','--package-dir',str(package),'--output-dir',str(draft),'--synthetic-test-only'])
    marker=evaluator.verify_publication(draft)
    receipt_path=draft/'preregistration_draft.json'
    receipt=evaluator.read_json(receipt_path)
    evaluator.require(receipt['status']=='draft' and receipt['fitting_started'] is False
                      and receipt['independent_review'] is None and receipt['contract']['synthetic_test_only'] is True,
                      'Expected unfrozen synthetic no-fit draft')
    result=dict(status='synthetic_package_and_no_fit_draft_prepared',synthetic_test_only=True,
        root=str(root),source_fixture=str(root/'source'),package=str(package),draft=str(draft),
        receipt=str(receipt_path),receipt_sha256=evaluator.sha(receipt_path),
        contract_sha256=receipt['contract_sha256'],package_commit_sha256=evaluator.sha(package/'COMMIT.json'),
        draft_commit_sha256=evaluator.sha(draft/'COMMIT.json'),rows=receipt['contract']['rows'],
        accounting=receipt['contract']['accounting'],fitting_started=False,predictions_generated=False,
        frozen_authorization_created=False,review_required=True,
        evaluator_sha256=evaluator.sha(Path(evaluator.__file__).resolve()),
        harness_sha256=evaluator.sha(Path(__file__).resolve()),draft_files=sorted(marker['files']))
    evaluator.write_json(root/'synthetic_preparation_receipt.json',result)
    print(json.dumps(result,sort_keys=True),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    prepare(parser.parse_args().parent)
