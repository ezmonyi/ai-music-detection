"""Atomic, checksum-audited collection I/O and strict prompt selection."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import torch

from .data import caption_hash, file_sha256
from .trainer import read_prompts


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+".writing")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    partial.replace(path)


def save_tensor(path: Path, value):
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_suffix(path.suffix+".partial")
    torch.save(value, partial)
    partial.replace(path)


def cpu_condition(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_condition(v) for k,v in value.items()}
    if isinstance(value, (str,int,float,bool)) or value is None:
        return value
    raise TypeError("Condition cache must be tensors or primitive values")


def select_prompts(train, validation, test, *, offset, count, online_groups):
    records = read_prompts(train, "train")
    if offset < online_groups or count <= 0 or offset+count > len(records):
        raise ValueError("Offline selection must fit the unused training-prompt range")
    selected = records[offset:offset+count]
    held_out = read_prompts(validation,"validation")+read_prompts(test,"test")
    keys = lambda r: (r["prompt_id"],caption_hash(r["caption"]),(r["source_dataset"],r["source_id"]))
    held = [set(key[i] for key in map(keys,held_out)) for i in range(3)]
    if any(any(key[i] in held[i] for i in range(3)) for key in map(keys, selected)):
        raise ValueError("Offline data overlaps a held-out prompt/caption/source")
    return records, selected


def file_manifest(directory: Path):
    return [{"path":str(p.relative_to(directory)),"bytes":p.stat().st_size,"sha256":file_sha256(p)}
            for p in sorted(directory.rglob("*")) if p.is_file() and
            p.name not in {"FILES_SHA256.json","collection_state.json"} and
            not p.name.endswith((".writing",".partial"))]


def closed_prefix(output: Path, selected, group_size):
    folders=sorted((output/"groups").iterdir()) if (output/"groups").exists() else []
    valid=all_invalid=0
    for index,folder in enumerate(folders):
        if folder.name != f"g{index:06d}" or not (folder/"group.json").is_file():
            raise ValueError("Resume requires a contiguous prefix of fully closed groups")
        group=json.loads((folder/"group.json").read_text())
        if (group["prompt_id"] != selected[index]["prompt_id"] or not group["closed"] or
                len(group["results"]) != group_size):
            raise ValueError("Collection prefix identity mismatch")
        for entry in group["files"]:
            path=folder/entry["path"]
            if path.is_symlink() or folder.resolve() not in path.resolve().parents:
                raise ValueError("Unsafe collection member path")
            if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
                raise ValueError("Collection prefix file checksum mismatch")
        group_valid=sum(result["valid"] for result in group["results"])
        if group_valid != group["valid_count"]:
            raise ValueError("Collection prefix validity mismatch")
        valid+=group_valid; all_invalid+=int(group_valid==0)
    return len(folders),valid,all_invalid


def wait_for_resources(output, device, *, main_state=None, minimum_free_gib=12,
                       minimum_disk_gib=15, timeout_s=600):
    started = time.monotonic()
    while True:
        if main_state and Path(main_state).exists():
            state = json.loads(Path(main_state).read_text())
            if state.get("phase") == "failed":
                raise RuntimeError("Online pipeline failed; preserve offline progress and inspect first")
        if shutil.disk_usage(output).free < minimum_disk_gib*2**30:
            raise RuntimeError("Offline staging disk is below its free-space reserve")
        if device.type != "cuda" or torch.cuda.mem_get_info(device)[0] >= minimum_free_gib*2**30:
            return
        if time.monotonic()-started > timeout_s:
            raise RuntimeError("GPU headroom wait timed out; online process was not touched")
        time.sleep(10)
