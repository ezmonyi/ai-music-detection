"""Immutable archives of fully closed groups; never package credentials/caches."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            sha.update(block); md5.update(block)
    return {"bytes": Path(path).stat().st_size, "sha256": sha.hexdigest(), "md5": md5.hexdigest()}


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def safe_member(root, name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts or
            str(relative) != name or "\\" in name):
        raise ValueError("Unsafe dataset member path")
    root, path = Path(root), Path(root)/name
    if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent):
        raise ValueError("Dataset symlinks are not archive members")
    if root.resolve() not in path.resolve().parents or not path.is_file():
        raise ValueError("Missing or unsafe dataset member")
    return path


def verified_group(dataset, index, manifest):
    folder = Path(dataset)/"groups"/f"g{index:06d}"
    metadata = safe_member(dataset, f"groups/g{index:06d}/group.json")
    group = json.loads(metadata.read_text())
    size = manifest["group_size"]
    expected = {"condition.pt"} | {f"s{s:02d}_{kind}.{suffix}"
        for s in range(size) for kind,suffix in
        [("candidate","wav"),("base","wav"),("trajectory","pt")]}
    if (group.get("closed") is not True or group["group"] != index or
            group["prompt_id"] != manifest["prompt_ids"][index] or
            len(group["results"]) != size or
            {entry["path"] for entry in group["files"]} != expected or
            len(group["files"]) != len(expected)):
        raise ValueError("Closed group identity or member set mismatch")
    files = []
    for entry in group["files"]:
        path = safe_member(folder, entry["path"])
        actual = hashes(path)
        if any(actual[key] != entry[key] for key in ("bytes","sha256")):
            raise ValueError("Closed group checksum mismatch")
        files.append((path, str(path.relative_to(dataset)), actual))
    files.append((metadata, str(metadata.relative_to(dataset)), hashes(metadata)))
    return files


def shard(dataset, staging, first, last):
    dataset, staging = Path(dataset), Path(staging)
    manifest = json.loads((dataset/"COLLECTION_MANIFEST.json").read_text())
    if not 0 <= first < last <= manifest["prompt_groups"]:
        raise ValueError("Invalid half-open shard group range")
    files = [item for index in range(first,last) for item in verified_group(dataset,index,manifest)]
    filename = f"groups_{first:06d}_{last-1:06d}.tar"
    target, receipt = staging/"shards"/filename, staging/"shards"/(filename+".json")
    members = {name: digest for _,name,digest in files}
    if target.exists() or receipt.exists():
        saved = json.loads(receipt.read_text()) if receipt.exists() else {}
        if not target.is_file() or saved.get("members") != members or saved.get("archive") != hashes(target):
            raise ValueError("Existing shard changed or is incomplete; do not overwrite")
        return target, saved
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix+".partial")
    if partial.exists():
        raise FileExistsError("Preserve interrupted shard for diagnosis")
    with tarfile.open(partial,"w",format=tarfile.PAX_FORMAT) as archive:
        for path,name,_ in files:
            archive.add(path,arcname=name,recursive=False)
    # Read the actual archive, not just the source files, before declaring it durable.
    with tarfile.open(partial,"r") as archive:
        for item in archive:
            stream = archive.extractfile(item)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(4*1024*1024),b""):
                digest.update(block)
            if item.size != members[item.name]["bytes"] or digest.hexdigest() != members[item.name]["sha256"]:
                raise ValueError("Archived member differs from closed-group manifest")
    partial.replace(target)
    saved = {"schema_version":1,"first_group":first,"last_group_exclusive":last,
             "archive":hashes(target),"members":members}
    json_write(receipt,saved)
    return target,saved


STATIC_METADATA = ["COLLECTION_MANIFEST.json","CONFIG.json","SOURCE_BOUNDARY.json",
    "PROMPTS.json","behavior_policy.pt","PROBE_SUMMARY.json",
    "source/offline_collect.py","source/offline_io.py","source/offline_trajectory.py"]
FINAL_METADATA = ["SUMMARY.json","FILES_SHA256.json","collection_state.json"]


def metadata_files(dataset, *, final=False):
    return [(name,safe_member(dataset,name)) for name in
            STATIC_METADATA+(FINAL_METADATA if final else [])]
