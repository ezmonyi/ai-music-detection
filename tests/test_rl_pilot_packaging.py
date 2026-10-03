"""Synthetic artifact packaging checks; never a music-training benchmark."""
import importlib.util
import json
from pathlib import Path
import pytest


spec = importlib.util.spec_from_file_location("rl_pack", Path(__file__).parents[1] / "scripts/pack_rl_pilot_results.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


def fixture(root):
    provision = root / "provisioning"
    provision.mkdir(parents=True)
    (provision / "pipeline_state.json").write_text(json.dumps({
        "phase": "training_and_test_complete_pending_backup_and_snapshot",
        "summary": {"groups_completed": 100, "optimizer_updates": 90},
        "test_summary": {"count": 50, "split": "test"}}))
    test = root / "runs/a6000-srf-final-test"
    test.mkdir(parents=True)
    for i in range(50):
        for kind in ("base", "candidate"):
            (test / f"{i}_{kind}.wav").write_bytes(b"synthetic-container-not-real-audio")
    (root / "data").mkdir()
    (root / "data/test.jsonl").write_text("{}\n")
    # Neither credentials nor base weights may be captured by the allowlist.
    (provision / "secret.txt").write_text("not-a-real-secret")
    weights = root / "checkpoints"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"not-real-weights")


def test_pack_verify_and_omit_unrelated_files(tmp_path):
    root, destination = tmp_path / "remote", tmp_path / "local"
    fixture(root)
    receipt = pack.package(root, destination)
    assert receipt["files"] == 102
    result = pack.verify(destination)
    assert result["status"] == "archive_and_all_local_members_verified"
    assert not (destination / "provisioning/secret.txt").exists()
    assert not (destination / "checkpoints/model.safetensors").exists()


def test_incomplete_test_and_modified_archive_fail(tmp_path):
    root, destination = tmp_path / "remote", tmp_path / "local"
    fixture(root)
    missing = root / "runs/a6000-srf-final-test/0_candidate.wav"
    saved = missing.read_bytes()
    missing.unlink()
    with pytest.raises(ValueError, match="pairs"):
        pack.package(root, destination)
    missing.write_bytes(saved)
    pack.package(root, destination)
    archive = destination / "acestep_a6000_srf_results_20261003.tar.gz"
    with archive.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="archive hash"):
        pack.verify(destination)
