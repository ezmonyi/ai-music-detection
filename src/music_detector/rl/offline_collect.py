"""Frozen-policy trajectory collection alongside (not replacing) online RL.

There is no optimizer. Invalid completions and all-invalid groups are retained.
The dataset is a behavior-policy cache; indefinite reuse is off-policy RL.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .config import load_config
from .data import file_sha256
from .flow import group_advantages
from .monitoring import RunMonitor
from .offline_io import closed_prefix, cpu_condition, file_manifest, save_tensor, select_prompts, wait_for_resources, write_json
from .offline_trajectory import full_rollout, verify_replay
from .policy import export_adapter_state, load_adapter_state
from .trainer import _boundary, _memory, _save_audio, create_backend, create_reward, derive_seed, rollout, setup_policy


def collect(cfg, train_data, validation_data, test_data, checkpoint, checkpoint_sha,
            output, *, offset=100, count=300, main_state=None, monitor_dir=None,
            backend=None, reward=None, max_groups=None, resume=False):
    output, checkpoint = Path(output), Path(checkpoint)
    if output.exists() and not resume:
        raise FileExistsError("Use a new collection directory; never overwrite a partial run")
    if resume and not output.exists():
        raise FileNotFoundError("No collection prefix to resume")
    if max_groups is not None and (type(max_groups) is not int or max_groups<1):
        raise ValueError("max_groups must be a positive group budget")
    if file_sha256(checkpoint) != checkpoint_sha:
        raise ValueError("Behavior checkpoint SHA-256 mismatch")
    records, selected = select_prompts(train_data,validation_data,test_data,
        offset=offset,count=count,online_groups=cfg.training.updates)
    saved = torch.load(checkpoint,map_location="cpu",weights_only=True)
    if saved["config"] != cfg.to_dict():
        raise ValueError("Behavior checkpoint configuration mismatch")
    device = torch.device(cfg.model.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.25,device)
    torch.manual_seed(cfg.training.seed)
    reward = reward or create_reward(cfg)
    backend = backend or create_backend(cfg)
    policy, _ = setup_policy(cfg,backend)
    boundary = _boundary(cfg,Path(train_data),records,backend,reward)
    for name in ("config_sha256","data_sha256","backend","torch_version",
                 "implementation_sha256","analysis_receipt"):
        if boundary[name] != saved["boundary"][name]:
            raise ValueError(f"Behavior checkpoint boundary mismatch: {name}")
    load_adapter_state(backend.decoder,saved["policy"])
    for parameter in backend.decoder.parameters():
        parameter.requires_grad_(False)
    output.mkdir(parents=True,exist_ok=resume)
    started = time.monotonic()
    manifest = {"schema_version":1,"kind":"fixed_behavior_policy_flow_grpo_cache",
        "checkpoint_sha256":checkpoint_sha,"behavior_groups_completed":saved["groups_completed"],
        "config_sha256":cfg.digest(),"prompt_groups":count,"group_size":cfg.training.group_size,
        "prompt_ids":[row["prompt_id"] for row in selected],"selection_offset":offset,
        "validation_data_sha256":file_sha256(validation_data),"test_data_sha256":file_sha256(test_data),
        "trajectory_dtype":"float32","solver_steps":cfg.sampling.steps,
        "deterministic_steps_masked":True,"optimizer_updates":0,
        "guards_changed":False,"reward_families":cfg.reward.families,
        "offline_reuse_is_off_policy":True,"music_quality_validated":False,
        "collector_implementation_sha256":{n:file_sha256(Path(__file__).with_name(n)) for n in
            ("offline_collect.py","offline_io.py","offline_trajectory.py")}}
    if resume:
        if json.loads((output/"COLLECTION_MANIFEST.json").read_text()) != manifest:
            raise ValueError("Collection manifest/config/source changed; do not bypass resume")
        completed,valid_samples,all_invalid=closed_prefix(output,selected,cfg.training.group_size)
        if completed>=count:
            raise ValueError("Collection is already complete")
    else:
        completed=valid_samples=all_invalid=0
        write_json(output/"COLLECTION_MANIFEST.json",manifest)
        write_json(output/"CONFIG.json",cfg.to_dict())
        write_json(output/"SOURCE_BOUNDARY.json",boundary)
        write_json(output/"PROMPTS.json",selected)
        save_tensor(output/"behavior_policy.pt",export_adapter_state(backend.decoder))
        for filename in manifest["collector_implementation_sha256"]:
            destination=output/"source"/filename
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(Path(__file__).with_name(filename).read_bytes())
    monitor=RunMonitor(Path(monitor_dir or output/"tensorboard"),enabled=cfg.monitoring.tensorboard)
    try:
        write_json(output/"collection_state.json",{"phase":"collecting","groups_completed":completed,
            "target_groups":count,"candidate_count":completed*cfg.training.group_size,"optimizer_updates":0})
        stop=min(count,max_groups) if max_groups is not None else count
        for index in range(completed,stop):
            row=selected[index]
            group_start=time.monotonic()
            wait_for_resources(output,device,main_state=main_state)
            with torch.no_grad():
                condition=backend.condition(row)
            directory=output/"groups"/f"g{index:06d}"
            directory.mkdir(parents=True,exist_ok=False)
            save_tensor(directory/"condition.pt",cpu_condition(condition))
            results,replays=[],[]
            for sample in range(cfg.training.group_size):
                wait_for_resources(output,device,main_state=main_state)
                seed=derive_seed(cfg.training.seed,row,offset+index,sample)
                baseline,_=rollout(backend,condition,cfg,seed,reference=True,collect=False)
                candidate,trajectory=full_rollout(backend,condition,cfg,seed)
                if index < 2:
                    replays.append(verify_replay(backend,condition,cfg,trajectory))
                save_tensor(directory/f"s{sample:02d}_trajectory.pt",trajectory)
                _save_audio(directory/f"s{sample:02d}_candidate.wav",candidate)
                _save_audio(directory/f"s{sample:02d}_base.wav",baseline)
                result=reward.score(candidate.samples,candidate.sample_rate,baseline.samples)
                results.append({"sample":sample,"seed":seed,**asdict(result)})
                del trajectory, candidate, baseline
            valid=sum(result["valid"] for result in results)
            values=torch.tensor([r["reward"] for r in results],dtype=torch.float32)
            advantages=group_advantages(values,clip=cfg.training.advantage_clip) if valid else torch.zeros_like(values)
            files=file_manifest(directory)
            write_json(directory/"group.json",{"group":index,"prompt_id":row["prompt_id"],
                "training_record_index":offset+index,"results":results,"valid_count":valid,
                "advantages":advantages.tolist(),"all_invalid":valid==0,
                "replay_checks":replays,"files":files,"closed":True})
            completed=index+1; valid_samples+=valid; all_invalid+=int(valid==0)
            metrics={"phase":"collecting","groups_completed":completed,"target_groups":count,
                "candidate_count":completed*cfg.training.group_size,"valid_candidates":valid_samples,
                "all_invalid_groups":all_invalid,"optimizer_updates":0,
                "group_seconds":time.monotonic()-group_start,"elapsed_seconds":time.monotonic()-started,
                **_memory(device)}
            write_json(output/"collection_state.json",metrics)
            monitor.add_scalars({"collection/groups":completed,"collection/candidates":metrics["candidate_count"],
                "collection/valid_fraction":valid_samples/metrics["candidate_count"],
                "collection/group_seconds":metrics["group_seconds"],
                "collection/gpu_peak_allocated_bytes":metrics.get("cuda_peak_allocated_bytes",0)},completed)
            print(json.dumps({k:metrics[k] for k in ["groups_completed","candidate_count","valid_candidates","group_seconds"]}),flush=True)
        if completed<count:
            write_json(output/"PROBE_SUMMARY.json",{**metrics,"phase":"probe_complete_pending_upload_route"})
            write_json(output/"collection_state.json",{**metrics,"phase":"probe_complete_pending_upload_route"})
            return metrics
        write_json(output/"SUMMARY.json",{**metrics,"phase":"complete","sample_groups":completed,
            "audio_files":completed*cfg.training.group_size*2,"music_quality_validated":False})
        write_json(output/"FILES_SHA256.json",{"files":file_manifest(output)})
        write_json(output/"collection_state.json",{**metrics,"phase":"complete"})
        return metrics
    except Exception as error:
        write_json(output/"collection_state.json",{"phase":"failed","groups_completed":completed,
            "candidate_count":completed*cfg.training.group_size,"error_type":type(error).__name__,
            "message":str(error),"optimizer_updates":0})
        raise
    finally:
        monitor.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","train-data","validation-data","test-data","checkpoint","checkpoint-sha256","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--offset",type=int,default=100)
    parser.add_argument("--count",type=int,default=300)
    parser.add_argument("--main-state")
    parser.add_argument("--monitor-dir")
    parser.add_argument("--max-groups",type=int)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    collect(load_config(args.config),args.train_data,args.validation_data,args.test_data,args.checkpoint,
        args.checkpoint_sha256,args.output,offset=args.offset,count=args.count,
        main_state=args.main_state,monitor_dir=args.monitor_dir,max_groups=args.max_groups,resume=args.resume)


if __name__=="__main__":
    main()
