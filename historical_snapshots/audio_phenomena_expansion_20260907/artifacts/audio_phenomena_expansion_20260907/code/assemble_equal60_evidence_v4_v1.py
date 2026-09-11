#!/usr/bin/env python3
"""Copy the ten authoritative exact60 evidence files; no fitting or freeze."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import prepare_evaluation_inputs_v4 as prep

ROOT = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
DATA = Path('/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907')


def sources():
    strict = ROOT/'results/equal60_full_validation_v1'
    prepared = DATA/'equal60_development_v2'
    return {
        'native.csv': ROOT/'manifests/long_cohort/cohort_60s.csv',
        'fhm.csv': ROOT/'results/features_60s_v1/features.csv',
        'fhm_contract.json': ROOT/'results/features_60s_v1/contract.json',
        'materialization_contract.json': prepared/'materialization_contract.json',
        'materialization_summary.json': prepared/'materialization_summary.json',
        'prepared.csv': prepared/'inference_manifest.csv',
        'strict_inference_audit.json': strict/'strict_inference_audit.json',
        'old_features.csv': strict/'features/expanded_features_60s.csv',
        'old_metadata.json': strict/'features/expanded_features_60s_metadata.json',
        'extraction_receipt.json': ROOT/'results/equal60_recovery_bound_receipt_v1/extraction_receipt.json',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    output = a.output_dir.absolute()
    if output.exists() or output.parent != ROOT/'results':
        raise ValueError('Require a new direct child of the project results directory')
    mapping = sources()
    assert set(mapping) == set(prep.EVIDENCE_NAMES)
    before = {name: prep.sha(path) for name,path in mapping.items()}
    body = prep.read_json(mapping['extraction_receipt.json'])
    binding = body['recovery_evidence_binding']
    assert binding['status'] == 'bound_completed_recovery_evidence'
    assert binding['validated_rows'] == 1604 and binding['validated_stage_shards'] == 134
    temp = Path(tempfile.mkdtemp(prefix=output.name+'.tmp.', dir=output.parent))
    # Retain an incomplete new temp directory on error for inspection; never
    # delete or replace authoritative source/earlier evidence directories.
    for name,path in mapping.items():
        with path.open('rb') as src, (temp/name).open('xb') as dst:
            shutil.copyfileobj(src,dst)
            dst.flush()
            os.fsync(dst.fileno())
        assert prep.sha(temp/name) == before[name], name
    metadata, features = prep.validate_evidence(temp, synthetic=False)
    assert before == {name: prep.sha(path) for name,path in mapping.items()}
    if output.exists():
        raise ValueError('Output appeared while copying')
    os.rename(temp, output)
    print(json.dumps(dict(status='assembled_and_validated_not_authorized_for_scoring',
        rows=len(metadata), feature_rows=len(features), evidence_dir=str(output),
        sources={name:str(path) for name,path in mapping.items()},
        copied_sha256=before, assembler_sha256=prep.sha(Path(__file__))), indent=2))


if __name__ == '__main__':
    main()
