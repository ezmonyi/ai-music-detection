"""Capacity gates and final-payload verification for the reduced MatPool cache."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('persistent_controller', ROOT / 'scripts/run_matpool_offline_persistent.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def test_capacity_preserves_online_worst_case_snapshot_and_safety():
    for groups in (0, 59, 61, 65, 100):
        plan = controller.capacity_projection(groups, 60_000_000_000)
        needed = plan['needed_free_bytes']
        assert controller.capacity_projection(groups, needed)['capacity_gate_passed']
        assert not controller.capacity_projection(groups, needed - 1)['capacity_gate_passed']
        assert plan['environment_snapshot_budget_bytes'] == 26_000_000_000
        assert plan['safety_reserve_bytes'] == 4 * 2**30
        assert plan['new_dataset_budget_bytes'] > 24 * 4 * (2 * controller.WAV_BYTES + controller.TRAJECTORY_BYTES)
    with pytest.raises(ValueError):
        controller.capacity_projection(101, 60_000_000_000)


@pytest.fixture
def completed_cache(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from music_detector.rl.config import load_config
    from music_detector.rl.data import file_sha256
    from music_detector.rl.offline_collect import collect
    from music_detector.rl.trainer import train
    cfg = load_config(ROOT / 'configs/rl/toy_smoke.json')
    cfg = replace(cfg, training=replace(cfg.training, updates=1, group_size=2))
    paths = {}
    for split, count in (('train', 3), ('validation', 1), ('test', 1)):
        rows = [{'prompt_id': f'{split}-{i}', 'caption': f'Distinct fixture {split} number {i}',
                 'lyrics': '', 'duration_s': 30, 'seed': i, 'split': split,
                 'source_dataset': 'synthetic/persistent-test', 'source_revision': 'v1',
                 'source_id': f'{split}-{i}', 'license': 'CC0'} for i in range(count)]
        paths[split] = tmp_path / (split + '.jsonl')
        paths[split].write_text('\n'.join(map(json.dumps, rows)) + '\n')
    train(cfg, paths['train'], tmp_path / 'online')
    checkpoint = tmp_path / 'online/checkpoints/group_000001.pt'
    output = tmp_path / 'cache'
    collect(cfg, paths['train'], paths['validation'], paths['test'], checkpoint,
            file_sha256(checkpoint), output, offset=1, count=1)
    for key, value in (('OUTPUT', output), ('COUNT', 1), ('OFFSET', 1), ('GROUP_SIZE', 2)):
        monkeypatch.setattr(controller, key, value)
    return output


def test_persistent_receipt_covers_actual_files_without_drive(completed_cache):
    receipt = controller.verify_dataset()
    files = [p for p in completed_cache.rglob('*') if p.is_file()]
    assert receipt['status'] == 'complete_persistent_dataset_all_members_sha256_verified'
    assert receipt['candidates'] == 2 and receipt['paired_wavs'] == 4
    assert receipt['optimizer_updates'] == 0 and receipt['proxy_used'] is False
    assert receipt['files_verified'] == len(files)
    assert receipt['bytes_verified'] == sum(p.stat().st_size for p in files)


def test_persistent_receipt_rejects_changed_trajectory(completed_cache):
    (completed_cache / 'groups/g000000/s00_trajectory.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum mismatch'):
        controller.verify_dataset()


def test_persistent_receipt_rejects_unlisted_member(completed_cache):
    (completed_cache / 'unexpected.partial').write_bytes(b'partial')
    with pytest.raises(ValueError, match='Missing, extra or incomplete'):
        controller.verify_dataset()


def test_persistent_receipt_rejects_incomplete_final_state(completed_cache):
    path = completed_cache / 'collection_state.json'
    state = json.loads(path.read_text()); state['phase'] = 'collecting'
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='not complete'):
        controller.verify_dataset()
