"""Explicit immutable export/upload of portable caches, without a GPU or secrets.

MatPool's web/client upload is supported by TAR batches. A configured rclone
remote is optional, not an invented MatPool API. Nothing deletes local files.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import time

from .data import file_sha256
from .offline_archive import hashes, json_write, safe_member


STATIC = ("COLLECTION_MANIFEST.json", "CONFIG.json", "SOURCE_BOUNDARY.json", "PROMPTS.json",
          "behavior_policy.pt", "COLLECTOR_RUNTIME.json")


@contextmanager
def task_lock(path):
    """An OS-released lock; a crashed process cannot leave a false active lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0"); stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another task already holds " + str(path)) from error
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another task already holds " + str(path)) from error
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _entry(root, name):
    if any(c in name for c in ("\n", "\r", "\0")):
        raise ValueError("Unsupported member name")
    path = safe_member(root, name)
    return {"path": name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _verify(root, entries):
    seen = set()
    for expected in entries:
        actual = _entry(root, expected["path"])
        if actual["path"] in seen or actual != {k: expected[k] for k in actual}:
            raise ValueError("Duplicate or changed delivery member: " + actual["path"])
        seen.add(actual["path"])
    return seen


def inventory(root, *, require_complete=True):
    """Rehash only declared closed data, excluding interrupted/quarantined files."""
    root = Path(root).resolve()
    manifest = json.loads(safe_member(root, STATIC[0]).read_text())
    if manifest.get("kind") != "unscored_behavior_flow_grpo_cache" or manifest.get("optimizer_updates") != 0:
        raise ValueError("Delivery expects a portable unscored cache, not an online run")
    rows = json.loads(safe_member(root, "PROMPTS.json").read_text())
    if len(rows) != manifest["prompt_groups"] or manifest["group_size"] != 4:
        raise ValueError("Collection prompt/group-size mismatch")
    entries = [_entry(root, name) for name in STATIC]
    closed = 0
    for index, row in enumerate(rows):
        folder = root / "groups" / f"g{index:06d}"
        if not (folder / "group.json").exists():
            break
        group = json.loads(safe_member(folder, "group.json").read_text())
        expected = {"condition.pt"} | {f"s{s:02d}_{suffix}" for s in range(4)
                    for suffix in ("trajectory.pt", "candidate.wav", "base.wav")}
        names = _verify(folder, group["files"])
        if (group.get("closed") is not True or group["group"] != index or
                group["prompt_id"] != row["prompt_id"] or names != expected or
                [r["sample"] for r in group["results"]] != list(range(4))):
            raise ValueError("Invalid closed collection group")
        entries.extend({**e, "path": f"groups/g{index:06d}/" + e["path"]} for e in group["files"])
        entries.append(_entry(root, f"groups/g{index:06d}/group.json"))
        closed += 1
    complete = (root / "FILES_SHA256.json").exists()
    if complete:
        if closed != len(rows):
            raise ValueError("Completed manifest has missing groups")
        entries.append(_entry(root, "SUMMARY.json"))
        saved = json.loads(safe_member(root, "FILES_SHA256.json").read_text())["files"]
        _verify(root, saved)
        actual_by_name = {e["path"]: e for e in entries}
        if len(saved) != len(actual_by_name) or {e["path"]: {k: e[k] for k in ("path", "bytes", "sha256")} for e in saved} != actual_by_name:
            raise ValueError("Completed file manifest does not exactly cover the portable payload")
        entries.append(_entry(root, "FILES_SHA256.json"))
        entries.append(_entry(root, "collection_state.json"))
        summary = json.loads(safe_member(root, "SUMMARY.json").read_text())
        state = json.loads(safe_member(root, "collection_state.json").read_text())
        if any(v.get("phase") != "complete_unscored" or v.get("groups_completed") != closed or
               v.get("optimizer_updates") != 0 for v in (summary, state)):
            raise ValueError("Cache completion state does not match its members")
    if require_complete and not complete:
        raise ValueError("Collection is not complete; closed-prefix export is available instead")
    return {"root": str(root), "manifest_sha256": file_sha256(root / STATIC[0]),
            "complete": complete, "closed_groups": closed, "target_groups": len(rows),
            "files": sorted(entries, key=lambda e: e["path"]),
            "bytes": sum(e["bytes"] for e in entries), "optimizer_updates": 0}


def _archive(root, path, entries):
    """Create a no-overwrite TAR and verify each actual archived member."""
    path = Path(path)
    receipt = path.with_suffix(path.suffix + ".json")
    if path.exists() or receipt.exists():
        value = json.loads(receipt.read_text()) if receipt.exists() else {}
        if not path.is_file() or value.get("files") != entries or value.get("archive") != hashes(path):
            raise ValueError("Existing export differs; preserve it and use another destination")
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise FileExistsError("Preserve interrupted export archive: " + str(partial))
    if shutil.disk_usage(path.parent).free < sum(e["bytes"] for e in entries) + 2**30:
        raise RuntimeError("Insufficient export staging disk; export fewer closed groups per command")
    with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as archive:
        for entry in entries:
            archive.add(safe_member(root, entry["path"]), arcname=entry["path"], recursive=False)
    _read_archive(partial, entries)
    partial.replace(path)
    value = {"archive": hashes(path), "files": entries}
    json_write(receipt, value)
    return value


def _read_archive(path, entries, *, output=None):
    expected = {e["path"]: e for e in entries}
    seen = set()
    with tarfile.open(path, "r") as archive:
        for item in archive:
            if item.name not in expected or item.name in seen or not item.isfile():
                raise ValueError("Unexpected, duplicate or unsafe archive member")
            relative = PurePosixPath(item.name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in item.name:
                raise ValueError("Unsafe archive member path")
            entry = expected[item.name]
            if item.size != entry["bytes"]:
                raise ValueError("Archive member size differs")
            digest = hashlib.sha256()
            stream = archive.extractfile(item)
            destination = None
            if output is not None:
                target = Path(output) / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                destination = target.open("xb")
            try:
                for block in iter(lambda: stream.read(4*1024*1024), b""):
                    digest.update(block)
                    if destination:
                        destination.write(block)
            finally:
                if destination:
                    destination.close()
                stream.close()
            if digest.hexdigest() != entry["sha256"]:
                raise ValueError("Archived member SHA-256 differs")
            seen.add(item.name)
    if seen != set(expected):
        raise ValueError("Archive is incomplete")


def export_batches(root, destination, *, first=0, last=None, groups_per_shard=25):
    info = inventory(root, require_complete=False)
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Export staging must be outside the immutable cache")
    last = info["closed_groups"] if last is None else last
    if (type(first) is not int or type(last) is not int or type(groups_per_shard) is not int or
            not 0 <= first < last <= info["closed_groups"] or groups_per_shard < 1):
        raise ValueError("Export requires a nonempty range of fully closed groups")
    destination.mkdir(parents=True, exist_ok=True)
    with task_lock(destination / ".export.lock"):
        path = destination / "EXPORT_INDEX.json"
        index = json.loads(path.read_text()) if path.exists() else {
            "schema_version": 1, "kind": "portable_cache_export",
            "manifest_sha256": info["manifest_sha256"], "target_groups": info["target_groups"], "archives": {}}
        if index["manifest_sha256"] != info["manifest_sha256"]:
            raise ValueError("Export destination belongs to a different dataset")
        static = [e for e in info["files"] if e["path"] in STATIC]
        index["archives"]["metadata.tar"] = _archive(root, destination / "metadata.tar", static)
        for begin in range(first, last, groups_per_shard):
            end = min(last, begin+groups_per_shard)
            prefixes = tuple(f"groups/g{g:06d}/" for g in range(begin, end))
            entries = [e for e in info["files"] if e["path"].startswith(prefixes)]
            name = f"groups_{begin:06d}_{end-1:06d}.tar"
            # Different shard layouts must not duplicate the same groups.
            covered = {e["path"] for n, v in index["archives"].items() if n != name for e in v["files"]}
            if covered & {e["path"] for e in entries}:
                raise ValueError("New shard overlaps an existing export; keep a fixed partition")
            index["archives"][name] = _archive(root, destination / name, entries)
            json_write(path, index)
        if info["complete"]:
            completion = [e for e in info["files"] if e["path"] in {"SUMMARY.json", "FILES_SHA256.json", "collection_state.json"}]
            index["archives"]["completion.tar"] = _archive(root, destination / "completion.tar", completion)
        coverage = [e["path"] for v in index["archives"].values() for e in v["files"]]
        index["complete"] = bool(info["complete"] and len(coverage) == len(set(coverage)) and
                                  set(coverage) == {e["path"] for e in info["files"]})
        index["exported_groups"] = sum(n.endswith("/group.json") for n in coverage)
        index["optimizer_updates"] = 0
        json_write(path, index)
        return {k: v for k, v in index.items() if k != "archives"}


def restore_batches(source, output, *, allow_prefix=False):
    source, output = Path(source), Path(output)
    index = json.loads(safe_member(source, "EXPORT_INDEX.json").read_text())
    if index.get("kind") != "portable_cache_export" or (index.get("complete") is not True and not allow_prefix):
        raise ValueError("A complete export index is required before restore")
    archives = {n: r for n, r in index["archives"].items() if index.get("complete") or n != "completion.tar"}
    # Verify every uploaded TAR before creating destination files.
    for name, receipt in archives.items():
        if hashes(safe_member(source, name)) != receipt["archive"]:
            raise ValueError("Export archive checksum mismatch: " + name)
    output.mkdir(parents=True, exist_ok=False)
    for name, receipt in archives.items():
        _read_archive(source / name, receipt["files"], output=output)
    info = inventory(output, require_complete=not allow_prefix)
    if info["manifest_sha256"] != index["manifest_sha256"]:
        raise ValueError("Restored dataset identity mismatch")
    if info["closed_groups"] != index["exported_groups"]:
        raise ValueError("Restored export is not a contiguous closed prefix")
    return {k: info[k] for k in ("complete", "closed_groups", "bytes", "optimizer_updates")}


def _direct_env():
    env = dict(os.environ)
    for name in list(env):
        if name.lower() in {"http_proxy", "https_proxy", "all_proxy", "ftp_proxy"}:
            env.pop(name)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    return env


def _remote(base, identity):
    if not isinstance(base, str) or "://" in base or not re.fullmatch(r"[A-Za-z0-9_.-]+:.+", base):
        raise ValueError("Use a configured rclone REMOTE:dedicated/path, not a URL or credentials")
    tail = base.split(":", 1)[1]
    if any(c in tail for c in ("\n", "\r", "\0")) or ".." in PurePosixPath(tail).parts or not tail.strip("/"):
        raise ValueError("Upload destination must be a dedicated non-root path")
    return base.rstrip("/") + "/cache-" + identity[:16]


def publish(root, remote_base, receipts, *, execute=False, verification="readback", rclone_config=None):
    info = inventory(root)
    target = _remote(remote_base, info["manifest_sha256"])
    root, receipts = Path(root).resolve(), Path(receipts).resolve()
    if receipts == root or root in receipts.parents:
        raise ValueError("Upload receipts must be outside the immutable cache")
    if verification not in {"readback", "server_sha256"}:
        raise ValueError("Verification must be readback or server_sha256")
    result = {"destination": target, "members": len(info["files"]), "bytes": info["bytes"],
              "verification": verification, "execute": execute, "optimizer_updates": 0,
              "local_files_deleted": False, "proxy_environment_removed": True}
    if not execute:
        return {**result, "status": "dry_run_no_network_or_writes"}
    binary = shutil.which("rclone")
    if binary is None:
        raise RuntimeError("Install rclone and securely configure a direct remote first")
    command = [binary] + (["--config", str(rclone_config)] if rclone_config else [])
    env = _direct_env()
    receipts.mkdir(parents=True, exist_ok=True)
    with task_lock(receipts / ".upload.lock"):
        boundary = {"destination": target, "manifest_sha256": info["manifest_sha256"],
                    "files": info["files"], "verification": verification}
        path = receipts / "UPLOAD_BOUNDARY.json"
        if path.exists() and json.loads(path.read_text()) != boundary:
            raise ValueError("Upload resume boundary differs; use another receipt directory")
        json_write(path, boundary)
        file_list = receipts / "FILES_FROM.txt"
        file_list.write_text("\n".join(e["path"] for e in info["files"]) + "\n")
        state_path = receipts / "upload_state.json"
        try:
            json_write(state_path, {**result, "status": "uploading", "verified_members": 0})
            subprocess.run(command + ["copy", str(root), target, "--immutable", "--checksum",
                "--files-from-raw", str(file_list), "--transfers", "2", "--checkers", "4",
                "--contimeout", "15s", "--timeout", "5m", "--stats", "30s", "--retries", "3"],
                env=env, check=True)
            if verification == "server_sha256":
                text = subprocess.run(command + ["hashsum", "SHA-256", target,
                    "--files-from-raw", str(file_list)], env=env, check=True, capture_output=True, text=True).stdout
                remote_hashes = {}
                for line in text.splitlines():
                    match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", line)
                    if not match or match[2] in remote_hashes:
                        raise ValueError("Remote does not provide an unambiguous SHA-256 listing; use explicit readback")
                    remote_hashes[match[2]] = match[1].lower()
                if remote_hashes != {e["path"]: e["sha256"] for e in info["files"]}:
                    raise ValueError("Remote SHA-256 listing differs from the source")
            else:
                for number, entry in enumerate(info["files"], start=1):
                    digest, size = hashlib.sha256(), 0
                    with subprocess.Popen(command + ["cat", target + "/" + entry["path"],
                        "--contimeout", "15s", "--timeout", "5m"], env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE) as process:
                        for block in iter(lambda: process.stdout.read(4*1024*1024), b""):
                            digest.update(block); size += len(block)
                        error = process.stderr.read().decode(errors="replace")
                        if process.wait() != 0:
                            raise RuntimeError("Remote readback failed: " + error[-1000:])
                    if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                        raise ValueError("Remote payload SHA-256 mismatch: " + entry["path"])
                    json_write(state_path, {**result, "status": "verifying", "verified_members": number})
            # No receipt can call a mere copy success a verified cloud backup.
            completed = {**result, "status": "all_members_sha256_verified", "verified_members": len(info["files"]),
                         "full_payload_readback": verification == "readback", "verified_unix": time.time()}
            json_write(state_path, completed)
            json_write(receipts / "UPLOAD_VERIFICATION.json", {**completed, "source_manifest_sha256": info["manifest_sha256"]})
            return completed
        except Exception as error:
            json_write(state_path, {**result, "status": "failed", "error": str(error), "backup_confirmed": False})
            raise
