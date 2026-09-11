"""Publish the parent-reviewed actual v2 draft as extraction-only authority."""
import argparse
import json
from pathlib import Path
import prepare_native30_sdrp_operational_draft_v1 as helper

DRAFT_SHA = '6e7e82c027cea32bde1a0ebe8b421595053ed336acdd254deb86f7271019b249'
COMPLETION_SHA = '3ccf8593327b290ffbf7c61b17f314c8fb10544c1836e933748ed71a286a3eaf'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    draft, entry = helper.read_json(args.draft, DRAFT_SHA)
    backend = helper.load_backend()
    helper.require(draft['status'] == helper.DRAFT_STATUS
                   and draft['feature_extraction_authorized'] is False
                   and draft['inference_completion']['sha256'] == COMPLETION_SHA
                   and draft['terminal_completion_metadata_validation']['rows'] == 3830
                   and all(draft.get(k) == v for k, v in backend.SCOPE.items()),
                   'reviewed extraction-only scope')
    for key, value in draft.items():
        if isinstance(value, dict) and set(value) == {'path', 'bytes', 'sha256'}:
            helper.require(helper.binding(value['path']) == value, 'changed draft binding: ' + key)
    frozen = {**draft, 'status': backend.FREEZE_STATUS,
              'feature_extraction_authorized': True,
              'reviewed_draft': entry,
              'parent_authorization': '2026-09-09 user goal: finish existing Native30 experiments and English thesis; no new data or conditions',
              'scope_note': 'CPU SDRP extraction only; unchanged scientific inputs and numerics; no classifier fitting or neural inference'}
    helper.write_json_new(Path(args.output), frozen)
    print(json.dumps({'status': 'parent_extraction_freeze_published',
                      'freeze': helper.binding(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
