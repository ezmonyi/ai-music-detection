"""Parent-reviewed publication for the exact previously validated candidate."""
import argparse
from pathlib import Path
import prepare_bc_reserved_parent_draft_v1 as b

CANDIDATE_SHA = '454a2b8eed6680f379f1b6da4edc50b55017489b795b89f11311953b10fb77ee'
BUILDER_SHA = 'b7b252a98cdd130a81a8ed2033b70c3eb930015132dfe1954a579748327e3c42'
AUDITOR_SHA = 'cba9e6821e96f5280b21a724559a5b78c43e5b8a1995b8a72e69f3089f6893b7'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    x = p.parse_args()
    b.binding(Path(b.__file__).resolve(), BUILDER_SHA)
    c, origin = b.read_json(x.candidate, CANDIDATE_SHA)
    b.require(c['status'] == b.CANDIDATE_STATUS and
              c['authorized_scope'] == b.NONAUTHORIZING_SCOPE and
              c['independent_auditor']['sha256'] == AUDITOR_SHA,
              'exact nonauthorizing reviewed candidate required')
    rebuilt = b.build(Path(c['draft']['path']).parent, c['draft_COMMIT']['sha256'],
        c['independent_auditor']['path'], c['independent_auditor']['sha256'],
        c['independent_auditor_tests']['path'], c['independent_auditor_tests']['sha256'],
        c['draft_builder_tests']['path'])
    b.require(rebuilt == c, 'candidate metadata/runtime/library graph changed')
    output = Path(x.output)
    for root in (Path(c['output_root']), Path(c['draft']['path']).parent):
        b.require(not output.is_relative_to(root), 'freeze output overlaps result/draft')
    frozen = dict(c)
    frozen.pop('authorization_boundary')
    frozen.update(status='authorized_reserved_measurement_not_admission',
        authorized_scope=dict(b.AUTHORIZED_SCOPE), candidate_origin=origin,
        publisher=b.binding(Path(__file__).resolve()),
        parent_review={'reviewer': 'root', 'approved': True,
            'producer_and_builder_tests': '25 passed, parent session83473, 3.402s',
            'auditor_tests': '20 passed, pin-update session57658, 118.916s; prior parent20 passed119.475s',
            'change_scope': 'three producer module-key lookups and one dependent auditor hash; no numerical changes',
            'scientific_scope': 'fixed90 external reserved measurements only; no classifier fit, scoring, admission, or threshold change'})
    b.write_json_new(output, frozen)
    producer = b.load_producer()
    modules = producer.load_frozen(Path(b.__file__).parent)
    producer.verify_freeze(Path(c['draft']['path']).parent, c['draft_COMMIT']['sha256'],
                           output, b.digest(output), modules)
    print(b.canonical({'status': 'parent_freeze_published_and_launch_entry_verified',
        'freeze': b.binding(output), 'reserved_audio_read': False}).decode())

if __name__ == '__main__':
    main()
