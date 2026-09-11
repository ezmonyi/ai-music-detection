"""Assemble the existing six-authority BC request and metadata contract only."""
import argparse
import json
from pathlib import Path
import run_native30_bc_v2 as producer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference-freeze', required=True)
    parser.add_argument('--inference-freeze-sha256', required=True)
    parser.add_argument('--code-root', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--request-output', required=True)
    parser.add_argument('--prepared-output', required=True)
    parser.add_argument('--measurement-output', required=True)
    args = parser.parse_args()
    base = producer.base
    entry = base.file_binding(producer.physical.safe_path(args.inference_freeze))
    producer.require(entry['sha256'] == args.inference_freeze_sha256, 'inference authority hash')
    prior = producer.pinned_json(entry)
    code = producer.physical.safe_path(args.code_root)
    data = producer.physical.safe_path(args.data_root)
    request = {k: prior[k] for k in ('cohort_contract', 'plan', 'screen')}
    for key, path, sha in (
        ('audit', code / 'audit/bc_reserved_independent_actual_v2.json', producer.AUDIT_SHA),
        ('decision', code / 'preregistration/bc_native30_transfer_decision_v1.json', producer.DECISION_SHA),
        ('producer_commit', data / 'bicoherence_guitarset_reserved_admission_v1/COMMIT.json', producer.RESERVED_COMMIT_SHA)):
        request[key] = base.file_binding(path)
        producer.require(request[key]['sha256'] == sha, 'exact accepted BC evidence: ' + key)
    contract = producer.prepare(request, args.measurement_output)
    for path, value in ((args.request_output, request), (args.prepared_output, contract)):
        target = producer.physical.safe_path(path)
        producer.require(not target.is_relative_to(Path(args.measurement_output)), 'authority outside measurement output')
        base.write_new(target, value)
    print(json.dumps({'status': 'prepared_BC_metadata_only_not_measurement_authorized',
                      'request': base.file_binding(args.request_output),
                      'prepared': base.file_binding(args.prepared_output),
                      'rows': contract['expected_count'], 'audio_reads': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
