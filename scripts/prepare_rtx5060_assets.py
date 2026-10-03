"""Explicit public weight preparation through hf-mirror, never via a proxy.

Default is a dry run. Exact historical filenames/bytes/SHA-256 are required;
no Turbo/LM weights, remote code execution or official-HF fallback is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
from urllib.parse import quote
from urllib.request import Request, ProxyHandler, build_opener


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1]/"manifests/rl_5060_model_inputs.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _url(name, components, endpoint):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("Unsafe model manifest member")
    if not relative.parts or relative.parts[0] not in {"acestep-v15-sft", "vae", "Qwen3-Embedding-0.6B"}:
        raise ValueError("Unexpected model component")
    sft = relative.parts[0] == "acestep-v15-sft"
    component = components[0 if sft else 1]
    filename = str(PurePosixPath(*relative.parts[1:])) if sft else name
    return endpoint + "/" + component["repo"] + "/resolve/" + component["revision"] + "/" + quote(filename, safe="/")


def prepare(destination, manifest_path=DEFAULT_MANIFEST, *, execute=False, check_only=False):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["kind"] != "rtx5060_exact_public_model_inputs":
        raise ValueError("Unsupported weight manifest")
    destination = Path(destination).resolve()
    files = manifest["files"]
    missing, verified = [], 0
    for name, entry in files.items():
        path = destination/name
        _url(name, manifest["components"], "https://hf-mirror.com")
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != destination.parent):
            raise ValueError("Model destination symlinks are not allowed")
        if path.exists():
            if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise ValueError("Existing model input differs; preserve it instead of overwriting: " + name)
            verified += 1
        else:
            missing.append(name)
    result = {"model_directory": str(destination), "expected_files": len(files), "verified_files": verified,
              "missing_files": missing, "expected_bytes": manifest["total_bytes"],
              "endpoint": "https://hf-mirror.com", "proxy_used": False, "execution_requested": execute,
              "official_hf_fallback": False, "gpu_inference_started": False,
              "original_full_tree_identity_verified": False,
              "scope_note": "Core public files only. Prefer complete original MyDisk component directories; the collector separately enforces checkpoint provenance tree hashes."}
    if check_only or not execute:
        return {**result, "status": "all_public_model_inputs_verified" if not missing else "inputs_missing_no_download"}
    destination.mkdir(parents=True, exist_ok=True)
    needed = sum(files[n]["bytes"] for n in missing)
    if shutil.disk_usage(destination).free < needed + 2**30:
        raise RuntimeError("Insufficient disk for exact frozen model inputs")
    opener = build_opener(ProxyHandler({}))
    for name in missing:
        entry = files[name]
        path = destination/name
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        offset = partial.stat().st_size if partial.exists() else 0
        if partial.is_symlink() or offset > entry["bytes"]:
            raise ValueError("Unsafe or oversized owned partial; preserve for review")
        if offset != entry["bytes"]:
            headers = {"User-Agent": "music-thesis-5060-collector/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = Request(_url(name, manifest["components"], result["endpoint"]), headers=headers)
            started = last_log = time.monotonic()
            with opener.open(request, timeout=60) as response:
                if response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Mirror returned an inconsistent partial-content range")
                elif response.status == 200:
                    # Server ignored Range; restart only this task-owned partial.
                    offset = 0
                else:
                    raise RuntimeError("Unexpected mirror response")
                with partial.open("ab" if offset else "wb") as stream:
                    for block in iter(lambda: response.read(4*1024*1024), b""):
                        offset += len(block)
                        if offset > entry["bytes"]:
                            raise ValueError("Mirror response exceeds the pinned model size")
                        stream.write(block)
                        if time.monotonic()-last_log > 30:
                            print(json.dumps({"file": name, "downloaded_bytes": offset,
                                              "target_bytes": entry["bytes"]}), flush=True)
                            last_log = time.monotonic()
            print(json.dumps({"file": name, "download_seconds": time.monotonic()-started}), flush=True)
        if partial.stat().st_size != entry["bytes"] or sha256(partial) != entry["sha256"]:
            raise ValueError("Mirror payload differs from the pinned source SHA-256; partial preserved")
        partial.replace(path)
        verified += 1
    return {**result, "verified_files": verified, "missing_files": [], "status": "all_public_model_inputs_verified"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    result = prepare(args.directory, args.manifest, execute=args.execute, check_only=args.check_only)
    print(json.dumps(result, indent=2))
    return 1 if args.check_only and result["missing_files"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
