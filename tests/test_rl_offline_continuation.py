"""Real CPU optimizer/replay tests with synthetic models, never music evidence."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")

from music_detector.rl.config import load_config
from music_detector.rl.data import file_sha256
from music_detector.rl.offline_cache import load_shard, merge_caches, score_shard, verify_index
from music_detector.rl.offline_protocol import checkpoint, policy_fingerprint, runtime_config
from music_detector.rl.offline_collect import collect
from music_detector.rl.offline_train import OfflineOptions, evaluate_offline, preflight, train_offline
from music_detector.rl.portable_collect import collect_portable, plan_collection
from music_detector.rl.trainer import train


ROOT = Path(__file__).resolve().parents[1]


def prompts(path, split, count, prefix):
    rows = [{"prompt_id": f"{prefix}-{i}", "caption": f"Distinct synthetic prompt {prefix} number {i}",
             "lyrics": "", "duration_s": 30, "seed": i, "split": split,
             "source_dataset": "synthetic/offline", "source_revision": "fixture-v1",
             "source_id": f"{prefix}-{i}", "license": "CC0"} for i in range(count)]
    path.write_text("\n".join(map(json.dumps, rows))+"\n")
    return path


@pytest.fixture
def cache(tmp_path):
    cfg = load_config(ROOT / "configs/rl/toy_smoke.json")
    cfg = replace(cfg, training=replace(cfg.training, updates=2, checkpoint_every=1))
    old = prompts(tmp_path / "old.jsonl", "train", 6, "old")
    validation = prompts(tmp_path / "validation.jsonl", "validation", 1, "held-val")
    test = prompts(tmp_path / "test.jsonl", "test", 1, "held-test")
    train(cfg, old, tmp_path / "online")
    behavior = tmp_path / "online/checkpoints/group_000001.pt"
    initial = tmp_path / "online/checkpoints/group_000002.pt"
    clouds = []
    for i in range(2):
        cloud = tmp_path / f"cloud{i}"
        collect(cfg, old, validation, test, behavior, file_sha256(behavior), cloud, offset=2+i, count=1)
        clouds.append(cloud)
    new = prompts(tmp_path / "new.jsonl", "train", 3, "new")
    plan = tmp_path / "plan"
    plan_collection(new, validation, test, initial, file_sha256(initial), plan, count=3)
    raw = tmp_path / "raw"
    collect_portable(plan, initial, file_sha256(initial), raw, max_groups=1, disk_reserve_gib=0)
    closed_sha = file_sha256(raw / "groups/g000000/s00_candidate.wav")
    collect_portable(plan, initial, file_sha256(initial), raw, resume=True, disk_reserve_gib=0)
    assert closed_sha == file_sha256(raw / "groups/g000000/s00_candidate.wav")
    scores = tmp_path / "scores"
    score_shard(raw, scores)
    index = tmp_path / "index.json"
    merge_caches(clouds+[raw], validation, test, initial, file_sha256(initial), index,
                 score_files=[scores / "SCORES.json"], behavior_checkpoints=[behavior])
    return dict(cfg=cfg, initial=initial, behavior=behavior, index=index, validation=validation,
                test=test, raw=raw, plan=plan, scores=scores, clouds=clouds, old=old, new=new)


def run(cache, output, options, **kwargs):
    return train_offline(cache["index"], cache["initial"], file_sha256(cache["initial"]),
                         cache["validation"], cache["test"], output, options, **kwargs)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def test_unscored_collection_keeps_full_data_and_has_zero_updates(cache):
    shard = load_shard(cache["raw"])
    assert shard["manifest"]["optimizer_updates"] == 0
    assert len(shard["groups"]) == 3
    assert all(r["reward"] is None for g in shard["groups"] for r in g["group"]["results"])
    assert len(list(cache["raw"].glob("groups/*/*_trajectory.pt"))) == 12
    assert len(list(cache["raw"].glob("groups/*/*.wav"))) == 24


def test_merge_mixes_behavior_versions_and_real_updates_warm_start(cache, tmp_path):
    index = verify_index(cache["index"])
    assert index["prompt_groups"] == 5
    assert len({s["behavior_policy_fingerprint"] for s in index["shards"]}) == 2
    options = OfflineOptions(updates=3, tensorboard=False, checkpoint_every=1, evaluation_every=1)
    summary = run(cache, tmp_path / "continued", options)
    assert summary["status"] == "complete" and summary["offline_optimizer_updates"] == 3
    saved = torch.load(tmp_path / "continued/checkpoints/final.pt", weights_only=True)
    original = torch.load(cache["initial"], weights_only=True)
    assert saved["optimizer_updates"] == original["optimizer_updates"]+3
    assert policy_fingerprint(saved["policy"]) != policy_fingerprint(original["policy"])
    assert saved["offline_resume_boundary"]["initial_checkpoint_sha256"] == file_sha256(cache["initial"])
    metrics = list(map(json.loads, (tmp_path / "continued/metrics.jsonl").read_text().splitlines()))
    assert all(r["gradient_norm"] > 0 and r["updated"] for r in metrics)
    assert (tmp_path / "continued/validation/update_000003/summary.json").exists()


def test_offline_resume_is_bitwise_equal_to_whole_run(cache, tmp_path):
    options = OfflineOptions(updates=4, tensorboard=False, evaluation_every=0, checkpoint_every=1)
    run(cache, tmp_path / "whole", options)
    sliced = run(cache, tmp_path / "slice", options, max_updates=2)
    assert sliced["status"] == "budget_slice_complete" and sliced["offline_optimizer_updates"] == 2
    run(cache, tmp_path / "resumed", options, resume=tmp_path / "slice/checkpoints/final.pt")
    whole = torch.load(tmp_path / "whole/checkpoints/final.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/checkpoints/final.pt", weights_only=True)
    assert equal(whole["policy"], resumed["policy"])
    assert equal(whole["optimizer"], resumed["optimizer"])
    assert whole["offline_cursor"] == resumed["offline_cursor"]


def test_corrupt_payload_rejected_before_training(cache):
    (cache["raw"] / "groups/g000000/s00_trajectory.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_index(cache["index"])


def test_index_cannot_replace_verified_rewards(cache):
    value = json.loads(cache["index"].read_text())
    value["groups"][0]["results"][0]["reward"] += 0.3
    cache["index"].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed indexed group"):
        verify_index(cache["index"])


def test_unscored_cache_cannot_train_and_duplicate_merge_fails(cache, tmp_path):
    with pytest.raises(ValueError, match="Unscored"):
        merge_caches([cache["raw"]], cache["validation"], cache["test"], cache["initial"],
                     file_sha256(cache["initial"]), tmp_path / "unscored.json")
    with pytest.raises(ValueError, match="overlap"):
        merge_caches([cache["clouds"][0]]*2, cache["validation"], cache["test"], cache["initial"],
            file_sha256(cache["initial"]), tmp_path / "duplicate.json", behavior_checkpoints=[cache["behavior"]])


def test_insufficient_signal_is_not_called_1000_updates(cache):
    with pytest.raises(ValueError, match="real updates"):
        preflight(cache["index"], cache["initial"], file_sha256(cache["initial"]),
                  cache["validation"], cache["test"], OfflineOptions(updates=1000, tensorboard=False))


def test_runtime_relocation_preserves_contract_and_rejects_reward_changes(cache):
    _, cfg = checkpoint(cache["initial"], file_sha256(cache["initial"]))
    changed = runtime_config(cfg, {"model": {"checkpoint_dir": "/different/local/path"}})
    assert changed.model.checkpoint_dir == "/different/local/path"
    with pytest.raises(ValueError, match="may not change"):
        runtime_config(cfg, {"reward": {"guards": {"maximum_peak": 2}}})


def test_plan_cannot_reuse_historical_or_heldout_prompts(cache, tmp_path):
    with pytest.raises(ValueError, match="eligible"):
        plan_collection(cache["old"], cache["validation"], cache["test"], cache["initial"],
                        file_sha256(cache["initial"]), tmp_path / "bad-plan", count=1)


def test_evaluate_keeps_native_three_way_audio_and_rejects_training(cache, tmp_path):
    options = OfflineOptions(updates=1, tensorboard=False, evaluation_every=0)
    run(cache, tmp_path / "train", options)
    final = tmp_path / "train/checkpoints/final.pt"
    result = evaluate_offline(cache["initial"], file_sha256(cache["initial"]), final, file_sha256(final),
                              cache["test"], tmp_path / "test")
    assert result["count"] == 1 and len(list((tmp_path / "test").glob("*.wav"))) == 3
    overlap = json.loads(cache["new"].read_text().splitlines()[0])
    overlap.update(split="test", prompt_id="different-id", caption="Different caption same source")
    path = tmp_path / "overlap.jsonl"
    path.write_text(json.dumps(overlap)+"\n")
    with pytest.raises(ValueError, match="overlaps"):
        evaluate_offline(cache["initial"], file_sha256(cache["initial"]), final, file_sha256(final), path, tmp_path / "bad-eval")


def test_all_invalid_and_policy_drift_never_step_optimizer(cache):
    from copy import deepcopy
    from music_detector.rl.offline_train import _group_update
    from music_detector.rl.trainer import create_backend, setup_policy
    from music_detector.rl.policy import export_adapter_state, load_adapter_state
    index = verify_index(cache["index"])
    cfg = cache["cfg"]
    backend = create_backend(cfg)
    _, parameters = setup_policy(cfg, backend)
    initial = torch.load(cache["initial"], weights_only=True)
    behavior = torch.load(cache["behavior"], weights_only=True)["policy"]
    load_adapter_state(backend.decoder, initial["policy"])
    optimizer = torch.optim.AdamW(parameters, lr=3e-5)
    group = deepcopy(index["groups"][0])
    folder = cache["clouds"][0]/group["path"]
    invalid = deepcopy(group)
    invalid["valid_count"] = 0
    for result in invalid["results"]:
        result.update(reward=-1, valid=False)
    state = _group_update(cfg, OfflineOptions(tensorboard=False), invalid, folder,
                          behavior, backend, parameters, optimizer, 0)
    assert not state["updated"] and state["skip_reason"] == "all_invalid"
    assert not optimizer.state
    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(1)
    before = policy_fingerprint(export_adapter_state(backend.decoder))
    state = _group_update(cfg, OfflineOptions(tensorboard=False, max_abs_log_ratio=1e-12), group,
                          folder, behavior, backend, parameters, optimizer, 0)
    assert not state["updated"] and state["skip_reason"] == "log_ratio_drift"
    assert not optimizer.state and all(p.grad is None for p in parameters)
    assert before == policy_fingerprint(export_adapter_state(backend.decoder))


def test_wrong_old_probability_fails_without_overwriting_it(cache):
    from music_detector.rl.offline_train import _group_update
    from music_detector.rl.trainer import create_backend, setup_policy
    from music_detector.rl.policy import load_adapter_state
    group = verify_index(cache["index"])["groups"][0]
    folder = cache["clouds"][0]/group["path"]
    path = folder/"s00_trajectory.pt"
    record = torch.load(path, weights_only=True)
    record["old_logprobs"][record["likelihood_mask"]] += 0.1
    torch.save(record, path)
    tampered = file_sha256(path)
    backend = create_backend(cache["cfg"])
    _, parameters = setup_policy(cache["cfg"], backend)
    initial = torch.load(cache["initial"], weights_only=True)
    behavior = torch.load(cache["behavior"], weights_only=True)["policy"]
    load_adapter_state(backend.decoder, initial["policy"])
    optimizer = torch.optim.AdamW(parameters)
    with pytest.raises(RuntimeError, match="not overwritten"):
        _group_update(cache["cfg"], OfflineOptions(tensorboard=False), group, folder,
                      behavior, backend, parameters, optimizer, 0)
    assert file_sha256(path) == tampered and not optimizer.state


def test_offline_tensorboard_emits_required_real_tags(cache, tmp_path, monkeypatch):
    from music_detector.rl import monitoring
    tags = set()
    class Writer:
        def __init__(self, **kwargs):
            pass
        def add_scalar(self, tag, value, step):
            tags.add(tag)
        def add_text(self, *args):
            pass
        def flush(self):
            pass
        def close(self):
            pass
    monkeypatch.setattr(monitoring, "SummaryWriter", Writer)
    run(cache, tmp_path/"tb", OfflineOptions(updates=1, evaluation_every=1, tensorboard=True))
    assert {"train/loss", "eval/loss", "train/grad_norm", "train/learning_rate",
            "throughput/transitions_per_second", "throughput/optimizer_updates_per_second",
            "system/host_peak_rss_gib"} <= tags
    assert "system/cuda_peak_allocated_gib" not in tags  # unavailable CPU measurement is absent


def test_export_resume_restore_preserves_all_source_hashes(cache, tmp_path):
    from music_detector.rl.portable_delivery import export_batches, inventory, restore_batches
    first = export_batches(cache["raw"], tmp_path/"export", last=2, groups_per_shard=2)
    assert not first["complete"] and first["exported_groups"] == 2
    prefix = restore_batches(tmp_path/"export", tmp_path/"probe-restored", allow_prefix=True)
    assert prefix["closed_groups"] == 2 and not prefix["complete"]
    first_tar = file_sha256(tmp_path/"export/groups_000000_000001.tar")
    final = export_batches(cache["raw"], tmp_path/"export", first=2, groups_per_shard=2)
    assert final["complete"] and final["exported_groups"] == 3
    assert first_tar == file_sha256(tmp_path/"export/groups_000000_000001.tar")
    restored = restore_batches(tmp_path/"export", tmp_path/"restored")
    assert restored["closed_groups"] == 3
    assert inventory(cache["raw"])["files"] == inventory(tmp_path/"restored")["files"]
    load_shard(tmp_path/"restored")


def test_export_corruption_rejected_before_restore_creates_directory(cache, tmp_path):
    from music_detector.rl.portable_delivery import export_batches, restore_batches
    export_batches(cache["raw"], tmp_path/"export")
    path = tmp_path/"export/groups_000000_000002.tar"
    with path.open("r+b") as stream:
        stream.seek(1000); stream.write(b"bad")
    with pytest.raises(ValueError, match="archive checksum"):
        restore_batches(tmp_path/"export", tmp_path/"restored")
    assert not (tmp_path/"restored").exists()


def test_upload_dry_run_is_nonmutating_and_no_network(cache, tmp_path, monkeypatch):
    from music_detector.rl.portable_delivery import publish
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("No network allowed in dry run"))
    result = publish(cache["raw"], "configured:dedicated/rl", tmp_path/"receipts")
    assert result["status"] == "dry_run_no_network_or_writes"
    assert not (tmp_path/"receipts").exists()
    for remote in ("configured:/", "configured:../escape", "https://drive.google.com/folder"):
        with pytest.raises(ValueError):
            publish(cache["raw"], remote, tmp_path/"receipts")


@pytest.mark.parametrize("corrupt", [False, True])
def test_upload_full_readback_success_and_failure_do_not_delete_source(cache, tmp_path, monkeypatch, corrupt):
    import io
    from types import SimpleNamespace
    from music_detector.rl import portable_delivery as delivery
    before = delivery.inventory(cache["raw"])["files"]
    monkeypatch.setenv("HTTPS_PROXY", "http://forbidden-proxy:9999")
    monkeypatch.setattr(delivery.shutil, "which", lambda _: "/fake/rclone")
    def copy(args, **kwargs):
        assert "copy" in args and "--immutable" in args and "sync" not in args
        assert not any(k.lower() in {"http_proxy", "https_proxy", "all_proxy"} for k in kwargs["env"])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(delivery.subprocess, "run", copy)
    class Process:
        def __init__(self, args, **kwargs):
            remote = args[args.index("cat")+1]
            relative = remote.split("/cache-", 1)[1].split("/", 1)[1]
            payload = (cache["raw"]/relative).read_bytes()
            self.stdout = io.BytesIO(b"bad" if corrupt else payload)
            self.stderr = io.BytesIO()
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stdout.close(); self.stderr.close()
        def wait(self):
            return 0
    monkeypatch.setattr(delivery.subprocess, "Popen", Process)
    if corrupt:
        with pytest.raises(ValueError, match="Remote payload SHA-256"):
            delivery.publish(cache["raw"], "configured:dedicated/rl", tmp_path/"receipts", execute=True)
        state = json.loads((tmp_path/"receipts/upload_state.json").read_text())
        assert state["status"] == "failed" and not state["backup_confirmed"]
        assert not (tmp_path/"receipts/UPLOAD_VERIFICATION.json").exists()
    else:
        result = delivery.publish(cache["raw"], "configured:dedicated/rl", tmp_path/"receipts", execute=True)
        assert result["status"] == "all_members_sha256_verified" and result["full_payload_readback"]
    assert before == delivery.inventory(cache["raw"])["files"]


def test_collector_template_dry_run_never_loads_a_gpu_or_writes(monkeypatch):
    from music_detector.rl import collector_cli
    monkeypatch.setattr(collector_cli, "doctor", lambda *a, **kw: pytest.fail("No GPU call in dry run"))
    result = collector_cli.run_session(ROOT/"configs/rl/rtx5060_collection.example.json", "collect")
    assert result["status"] == "dry_run_no_network_or_gpu_or_writes" and result["candidate_count"] == 4000


def test_staged_condition_keeps_parent_numerics_and_separates_text_model():
    from test_rl_acestep_backend import _backend, FakeTokenizer, FakeTextEncoder, FakeConditionModel
    from music_detector.rl.staged_backend import StagedAceStepBackend
    original = _backend()
    original.silence_latent = torch.zeros(1, 750, 64)
    original.text_tokenizer = FakeTokenizer()
    original.text_encoder = FakeTextEncoder()
    original.model = FakeConditionModel()
    staged = object.__new__(StagedAceStepBackend)
    staged.__dict__.update(original.__dict__)
    phases = []
    staged._stage = phases.append
    record = {"caption": "synthetic staged test", "duration_s": 30, "lyrics": ""}
    expected = original.condition(record)
    actual = staged.condition(record)
    assert phases == ["text", "condition", "flow"]
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[k], actual[k]) if isinstance(expected[k], torch.Tensor)
               else expected[k] == actual[k] for k in expected)


def test_staging_offloads_inactive_components_before_loading_gpu(monkeypatch):
    from music_detector.rl.staged_backend import StagedAceStepBackend
    calls = []
    class Component:
        def __init__(self, name):
            self.name = name
        def to(self, *args, **kwargs):
            assert not kwargs  # device only, never cast FP32 adapters to BF16
            calls.append((self.name, str(args[0])))
    backend = object.__new__(StagedAceStepBackend)
    backend.device = torch.device("cuda:0")
    backend.model, backend.text_encoder, backend.vae = [Component(n) for n in ("model", "text", "vae")]
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    backend._stage("text")
    assert calls == [("model", "cpu"), ("vae", "cpu"), ("text", "cuda:0")]
    calls.clear()
    backend._stage("decode")
    assert calls == [("model", "cpu"), ("text", "cpu"), ("vae", "cuda:0")]


def test_archive_rejects_traversal_before_writing(tmp_path):
    import hashlib
    import io
    import tarfile
    from music_detector.rl.portable_delivery import _read_archive
    archive_path = tmp_path/"unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        item = tarfile.TarInfo("../escape")
        item.size = 3
        archive.addfile(item, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="Unsafe archive"):
        _read_archive(archive_path, [{"path": "../escape", "bytes": 3,
                      "sha256": hashlib.sha256(b"bad").hexdigest()}], output=tmp_path/"restore")
    assert not (tmp_path/"escape").exists()


def test_source_verification_rejects_parent_symlink(cache, tmp_path):
    from music_detector.rl.offline_protocol import member
    outside = tmp_path/"outside"
    outside.mkdir()
    (outside/"file").write_text("untrusted")
    (cache["raw"]/"link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        member(cache["raw"], "link/file")


def test_upload_server_sha256_receipt_is_not_claimed_as_readback(cache, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from music_detector.rl import portable_delivery as delivery
    info = delivery.inventory(cache["raw"])
    monkeypatch.setattr(delivery.shutil, "which", lambda _: "/fake/rclone")
    def command(args, **kwargs):
        text = "\n".join(e["sha256"] + "  " + e["path"] for e in info["files"]) if "hashsum" in args else ""
        return SimpleNamespace(returncode=0, stdout=text)
    monkeypatch.setattr(delivery.subprocess, "run", command)
    result = delivery.publish(cache["raw"], "configured:rl/only", tmp_path/"receipts",
                              execute=True, verification="server_sha256")
    assert result["status"] == "all_members_sha256_verified" and not result["full_payload_readback"]


def test_task_lock_prevents_duplicate_active_writer(tmp_path):
    from music_detector.rl.portable_delivery import task_lock
    with task_lock(tmp_path/"task.lock"):
        with pytest.raises(RuntimeError, match="already holds"):
            with task_lock(tmp_path/"task.lock"):
                pytest.fail("Duplicate active writer")
    with task_lock(tmp_path/"task.lock"):
        pass  # stale lock file is harmless after the OS releases its lock


def test_asset_dry_run_never_downloads_or_writes(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("asset_preparation", ROOT/"scripts/prepare_rtx5060_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "build_opener", lambda *a: pytest.fail("No download in dry run"))
    output = tmp_path/"weights"
    result = module.prepare(output)
    assert result["status"] == "inputs_missing_no_download" and not output.exists()
    assert not result["original_full_tree_identity_verified"]


def test_probe_replays_cached_behavior_without_optimizer_updates(cache, tmp_path):
    from music_detector.rl.portable_replay import replay_probe
    result = replay_probe(cache["raw"], cache["initial"], file_sha256(cache["initial"]), tmp_path/"replay.json")
    assert result["status"] == "cached_behavior_replay_passed" and result["optimizer_updates"] == 0
    assert result["device"] == "cpu" and len(result["checks"]) == 8
    assert max(c["maximum_abs_logprob_difference"] for c in result["checks"]) <= 1e-6
