"""No GPU needed to validate immutable-shard and storage-gate contracts."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / ('scripts/' + name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extension = load('run_matpool_offline_extension')


def receipt(n):
    return {'status': 'complete_persistent_dataset_all_members_sha256_verified',
            'prompt_groups': n, 'candidates': n*4, 'paired_wavs': n*8,
            'optimizer_updates': 0, 'proxy_used': False,
            'bytes_verified': n*100, 'files_verified': n*14+11}


def manifest(n, offset):
    return {'prompt_groups': n, 'group_size': 4, 'selection_offset': offset,
            'prompt_ids': [f'p-{i}' for i in range(offset, offset+n)],
            'checkpoint_sha256': extension.BEHAVIOR_SHA, 'trajectory_dtype': 'float32',
            'solver_steps': 50, 'guards_changed': False, 'optimizer_updates': 0,
            'reward_families': ['S', 'R', 'F'], 'config_sha256': 'same',
            'validation_data_sha256': 'validation', 'test_data_sha256': 'test'}


def test_complete_union_requires_all_50_groups():
    out = extension.combine_verified_shards(receipt(24), receipt(26), manifest(24,102), manifest(26,126))
    assert out['prompt_groups'] == 50 and out['candidates'] == 200 and out['paired_wavs'] == 400
    assert out['bytes_verified'] == 5000 and out['files_verified'] == 722
    assert out['original_probe_groups_retained_separately'] == 2


@pytest.mark.parametrize('change', ['overlap', 'dtype', 'boundary', 'updates', 'partial'])
def test_union_rejects_incomplete_changed_or_overlapping_shards(change):
    r, m = receipt(26), manifest(26,126)
    if change == 'overlap': m['prompt_ids'][0] = 'p-102'
    if change == 'dtype': m['trajectory_dtype'] = 'float16'
    if change == 'boundary': m['test_data_sha256'] = 'changed'
    if change == 'updates': r['optimizer_updates'] = 1
    if change == 'partial': r['prompt_groups'] = 25
    with pytest.raises(ValueError):
        extension.combine_verified_shards(receipt(24), r, manifest(24,102), m)


def test_completed_online_files_are_not_reserved_twice():
    base = extension.configure_base(load('run_matpool_offline_persistent'))
    planned = extension.capacity_projection(base, 100, 60_000_000_000)
    done = extension.capacity_projection(base, 100, 60_000_000_000, online_complete=True)
    assert planned['remaining_online_worst_case_bytes'] > 0
    assert done['remaining_online_worst_case_bytes'] == 0
    assert done['environment_snapshot_budget_bytes'] == 26_000_000_000
    assert done['safety_reserve_bytes'] == 4*2**30
    needed = done['needed_free_bytes']
    assert extension.capacity_projection(base,100,needed,online_complete=True)['capacity_gate_passed']
    assert not extension.capacity_projection(base,100,needed-1,online_complete=True)['capacity_gate_passed']
    with pytest.raises(ValueError):
        extension.capacity_projection(base,99,needed,online_complete=True)


def test_original_source_is_hash_pinned():
    source = ROOT / 'scripts/run_matpool_offline_persistent.py'
    base = extension.configure_base(extension.load_base(source))
    assert base.COUNT == 26 and base.OFFSET == 126 and base.OUTPUT == extension.OUTPUT


def test_partial_test_storage_credit_requires_real_100_group_boundary():
    base = extension.configure_base(load('run_matpool_offline_persistent'))
    no_credit = extension.capacity_projection(base,100,60_000_000_000)
    credited = extension.capacity_projection(base,100,60_000_000_000,completed_test_wavs=12)
    assert no_credit['needed_free_bytes'] - credited['needed_free_bytes'] == 12*base.WAV_BYTES
    assert credited['new_dataset_budget_bytes'] == 3_740_106_327
    assert credited['environment_snapshot_budget_bytes'] == 26_000_000_000
    assert credited['safety_reserve_bytes'] == 4*2**30
    for groups, count in [(99,1),(100,101),(100,-1)]:
        with pytest.raises(ValueError):
            extension.capacity_projection(base,groups,60_000_000_000,completed_test_wavs=count)


def test_measured_storage_contract_rejects_changed_precision_shape_or_lyrics():
    files = ([{'path':f'{i}.wav','bytes':11520088} for i in range(192)]
        + [{'path':f'{i}_trajectory.pt','bytes':9796301} for i in range(96)]
        + [{'path':f'{i}/condition.pt','bytes':885841} for i in range(24)]
        + [{'path':f'{i}/group.json','bytes':68453} for i in range(24)])
    config = {'sampling':{'duration_s':30,'steps':50},'model':{'precision':'bfloat16'}}
    selected = [{'duration_s':30,'lyrics':''} for _ in range(26)]
    extension.validate_storage_basis(files,config,selected)
    files[0]['bytes'] += 1
    with pytest.raises(ValueError):extension.validate_storage_basis(files,config,selected)
    files[0]['bytes'] -= 1
    selected[0]['lyrics'] = 'different lyrics'
    with pytest.raises(ValueError):extension.validate_storage_basis(files,config,selected)
