"""CPU-only trajectory/dataset contracts; no claims about music quality."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
torch=pytest.importorskip("torch")

from music_detector.rl.backends import ToyBackend
from music_detector.rl.config import load_config
from music_detector.rl.data import file_sha256
from music_detector.rl.offline_collect import collect
from music_detector.rl.offline_io import select_prompts
from music_detector.rl.offline_trajectory import full_rollout, transition_views, validate_trajectory, verify_replay
from music_detector.rl.rewards import RewardResult
from music_detector.rl.trainer import rollout, setup_policy, train

ROOT=Path(__file__).resolve().parents[1]


def config():
    cfg=load_config(ROOT/"configs/rl/toy_smoke.json")
    return replace(cfg,training=replace(cfg.training,updates=1,group_size=2))


def data(tmp):
    paths={}
    for split,count in [("train",4),("validation",1),("test",1)]:
        rows=[{"prompt_id":f"{split}-{i}","caption":f"Synthetic distinct caption {split} number {i}",
            "lyrics":"","duration_s":30,"seed":i,"split":split,"source_dataset":"synthetic/test",
            "source_revision":"v1","source_id":f"{split}-{i}","license":"CC0"} for i in range(count)]
        path=tmp/(split+".jsonl")
        path.write_text("\n".join(map(json.dumps,rows))+"\n")
        paths[split]=path
    return paths


def test_full_trajectory_exactly_matches_existing_rollout():
    cfg=config(); backend=ToyBackend(); setup_policy(cfg,backend)
    condition=backend.condition({"caption":"fixture"})
    original,_=rollout(backend,condition,cfg,123)
    audio,record=full_rollout(backend,condition,cfg,123)
    assert (audio.samples==original.samples).all()
    assert record["states"].shape==(cfg.sampling.steps+1,1,32,2)
    assert not record["likelihood_mask"][0]
    result=verify_replay(backend,condition,cfg,record)
    assert result["maximum_abs_logprob_difference"]==0
    view=transition_views(record)
    assert torch.equal(view["next_latents"],record["states"][1:][record["likelihood_mask"]])


def test_replay_rejects_corrupt_old_logprob():
    cfg=config(); backend=ToyBackend(); setup_policy(cfg,backend)
    condition=backend.condition({"caption":"fixture"})
    _,record=full_rollout(backend,condition,cfg,321)
    index=int(torch.where(record["likelihood_mask"])[0][0])
    record["old_logprobs"][index]+=0.1
    with pytest.raises(RuntimeError,match="replay logprob mismatch"):
        verify_replay(backend,condition,cfg,record)


def test_record_rejects_lossy_states_and_fake_deterministic_variance():
    cfg=config(); backend=ToyBackend(); setup_policy(cfg,backend)
    _,record=full_rollout(backend,backend.condition({"caption":"fixture"}),cfg,1)
    with pytest.raises(ValueError,match="FP32"):
        validate_trajectory({**record,"states":record["states"].half()})
    record["variance"][~record["likelihood_mask"]]=1
    with pytest.raises(ValueError,match="Deterministic"):
        validate_trajectory(record)


def test_selection_excludes_online_and_holdout(tmp_path):
    paths=data(tmp_path)
    all_rows,chosen=select_prompts(paths["train"],paths["validation"],paths["test"],
        offset=1,count=3,online_groups=1)
    assert len(all_rows)==4 and [r["prompt_id"] for r in chosen]==["train-1","train-2","train-3"]
    with pytest.raises(ValueError,match="unused"):
        select_prompts(paths["train"],paths["validation"],paths["test"],offset=0,count=1,online_groups=1)
    val=json.loads(paths["validation"].read_text())
    val["caption"]=chosen[0]["caption"]
    paths["validation"].write_text(json.dumps(val)+"\n")
    with pytest.raises(ValueError,match="held-out"):
        select_prompts(paths["train"],paths["validation"],paths["test"],offset=1,count=1,online_groups=1)


class InvalidReward:
    def score(self,audio,sample_rate,baseline_audio):
        return RewardResult(-1,False,{"reason":"fixture invalid"})


def test_collection_keeps_invalid_groups_never_updates_and_verifies_files(tmp_path):
    cfg=config(); paths=data(tmp_path)
    train_dir=tmp_path/"online"
    train(cfg,paths["train"],train_dir)
    checkpoint=train_dir/"checkpoints/group_000001.pt"
    output=tmp_path/"offline"
    backend=ToyBackend()
    summary=collect(cfg,paths["train"],paths["validation"],paths["test"],checkpoint,
        file_sha256(checkpoint),output,offset=1,count=2,backend=backend,reward=InvalidReward())
    assert summary["optimizer_updates"]==0 and summary["all_invalid_groups"]==2
    assert summary["candidate_count"]==4
    assert not any(p.requires_grad for p in backend.decoder.parameters())
    for folder in sorted((output/"groups").iterdir()):
        group=json.loads((folder/"group.json").read_text())
        assert group["closed"] and group["advantages"]==[0,0]
        for entry in group["files"]:
            assert file_sha256(folder/entry["path"])==entry["sha256"]
    assert len(list(output.glob("groups/*/*_candidate.wav")))==4
    assert len(list(output.glob("groups/*/*_base.wav")))==4
    assert len(list(output.glob("groups/*/*_trajectory.pt")))==4
    with pytest.raises(FileExistsError,match="never overwrite"):
        collect(cfg,paths["train"],paths["validation"],paths["test"],checkpoint,
            file_sha256(checkpoint),output,offset=1,count=2)


def test_collection_rejects_wrong_policy_hash(tmp_path):
    cfg=config(); paths=data(tmp_path)
    train(cfg,paths["train"],tmp_path/"online")
    with pytest.raises(ValueError,match="SHA-256"):
        collect(cfg,paths["train"],paths["validation"],paths["test"],
            tmp_path/"online/checkpoints/group_000001.pt","0"*64,tmp_path/"bad",offset=1,count=1)
    assert not (tmp_path/"bad").exists()


def test_probe_prefix_resumes_without_overwriting_collected_audio(tmp_path):
    cfg=config(); paths=data(tmp_path)
    train(cfg,paths["train"],tmp_path/"online")
    checkpoint=tmp_path/"online/checkpoints/group_000001.pt"
    output=tmp_path/"offline"
    collect(cfg,paths["train"],paths["validation"],paths["test"],checkpoint,
        file_sha256(checkpoint),output,offset=1,count=3,max_groups=1)
    before=file_sha256(output/"groups/g000000/s00_candidate.wav")
    assert json.loads((output/"collection_state.json").read_text())["phase"]=="probe_complete_pending_upload_route"
    assert not (output/"SUMMARY.json").exists()
    collect(cfg,paths["train"],paths["validation"],paths["test"],checkpoint,
        file_sha256(checkpoint),output,offset=1,count=3,resume=True)
    assert before==file_sha256(output/"groups/g000000/s00_candidate.wav")
    assert json.loads((output/"SUMMARY.json").read_text())["candidate_count"]==6


def test_probe_resume_rejects_corrupt_collected_file(tmp_path):
    cfg=config(); paths=data(tmp_path)
    train(cfg,paths["train"],tmp_path/"online")
    checkpoint=tmp_path/"online/checkpoints/group_000001.pt"
    output=tmp_path/"offline"
    collect(cfg,paths["train"],paths["validation"],paths["test"],checkpoint,
        file_sha256(checkpoint),output,offset=1,count=3,max_groups=1)
    (output/"groups/g000000/s00_trajectory.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="checksum mismatch"):
        collect(cfg,paths["train"],paths["validation"],paths["test"],checkpoint,
            file_sha256(checkpoint),output,offset=1,count=3,resume=True)
