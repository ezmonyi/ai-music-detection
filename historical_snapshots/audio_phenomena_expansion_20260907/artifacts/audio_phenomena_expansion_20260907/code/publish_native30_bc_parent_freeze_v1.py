"""Publish reviewed actual BC metadata as measurement-only authority."""
import argparse
import json
from pathlib import Path
import run_native30_bc_v2 as p

PREPARED_SHA = '3509a425dda56ec32c12bd0a481d22e8006f5892edc744ef1de8b350c00f58c9'
RUNNER_SHA = '822f51f3e961bb4edf16389283a91b898393084f34d1a87da0956d7a92cffe0d'
TESTS_SHA = '4d24a5e8689eb4d7599669e560ab504eb29a34b8409e9471b6a304c48ff5c7dd'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    entry = p.base.file_binding(p.physical.safe_path(args.prepared))
    p.require(entry['sha256'] == PREPARED_SHA, 'parent-reviewed prepared hash')
    contract = p.pinned_json(entry)
    p.require(contract['status'] == 'prepared_metadata_only_not_execution_authority'
              and contract['version'] == p.VERSION and contract['expected_count'] == 3830
              and len(contract['rows']) == 3830 and contract['source_counts'] == p.physical.SOURCES
              and contract['feature_names'] == [p.FEATURE] and contract['view'] == p.VIEW
              and all(contract.get(k) == v for k, v in p.SCOPE.items()), 'reviewed fixed measurement scope')
    p.require(contract['code']['runner']['sha256'] == RUNNER_SHA
              and contract['code']['tests']['sha256'] == TESTS_SHA, 'parent tested code')
    for binding in [*contract['request'].values(), *contract['code'].values()]:
        p.require(p.base.file_binding(binding['path']) == binding, 'authority changed')
    p.require(not Path(contract['output_root']).exists(), 'new measurement output required')
    freeze = {'version': p.FREEZE_VERSION, 'status': 'parent_frozen_for_native30_BC_measurement',
              'measurement_authorized': True, 'prepared_contract': entry,
              'runner': contract['code']['runner'], 'tests': contract['code']['tests'],
              'decision': contract['request']['decision'], 'view': contract['view'],
              'runtime': contract['runtime'], **p.SCOPE,
              'parent_authorization': '2026-09-09 user goal: complete existing Native30 BC experiment and English thesis; no new data or conditions',
              'review_note': '31 parent tests passed; actual corrected-v2 gate and metadata prepared; no classifier fit authority'}
    p.base.write_new(p.physical.safe_path(args.output), freeze)
    print(json.dumps({'status': 'BC_measurement_only_parent_freeze_published',
                      'freeze': p.base.file_binding(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
