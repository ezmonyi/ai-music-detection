"""Re-audit historical equal60 models for YuE2 transfer; no scoring or fitting."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

RC = Path('/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907')
ROOT = Path('/mnt/nfs-data/users/yi/yue2_500_extension_20260911')
sys.path.insert(0, str(RC/'code'))
import score_mureka60_frozen_v4 as old


def main():
    bindings = old.Bindings()
    bindings.file(Path(__file__).resolve())
    args = SimpleNamespace(
        old_audit=RC/'audit/equal60_v4_with_recovery_results_audit_v1.json',
        old_results=RC/'results/equal60_results_v4_with_recovery_v1',
        old_package=RC/'results/equal60_package_v4_with_recovery_v1',
        old_preregistration=RC/'preregistration/equal60_v4_with_recovery_frozen_v1.json')
    print('Replaying existing independent equal60 audit; no model fitting', flush=True)
    models, index, metadata = old.load_old(args, bindings)
    source = ROOT/'native60_inputs_v1'
    bindings.file(source/'COMMIT.json', 'cde960d1f4ddb29d9751ac8d3bbecf28b6806a5fc9dc05cf299618ec2d8b02ea')
    commit = bindings.json(source/'COMMIT.json')
    bindings.file(source/'metadata.json', commit['products']['metadata.json']['sha256'])
    records = json.loads((source/'metadata.json').read_text())
    assert len(records) == 276 and len({r['id'] for r in records}) == 276
    assert sum(r['role'] == 'locked_test' for r in records) == 53
    for key in ('id', 'group_id'):
        assert {r[key] for r in records}.isdisjoint(set(metadata[key])), key
    for model in models.values():
        assert not any('yue' in source.lower() for source in model['training_source_counts'])
    bindings.recheck()
    output = dict(status='historical_models_reaudited_measurements_not_yet_admitted',
        model_instances=len(index), distinct_saved_models=len(models), rows=276,
        old_development_rows=len(metadata), disjoint_identity_and_group=True,
        classifier_fits=0, predictions_generated=0,
        old_results=str(args.old_results), old_package=str(args.old_package),
        files_sha256=bindings.files, model_index=index.to_dict('records'))
    destination = ROOT/'native60_model_acceptance_v1.json'
    with destination.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print('Historical model acceptance committed; measurement admission remains pending', flush=True)


if __name__ == '__main__':
    main()
