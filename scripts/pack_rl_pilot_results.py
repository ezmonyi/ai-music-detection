"""Package completed pilot outputs, never generation weights or credentials.

Run after training/test completion on the rented node. Local copies must
verify both the archive and every extracted member before node release.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(root: Path, destination: Path) -> dict:
    state = json.loads((root / "provisioning/pipeline_state.json").read_text())
    if state["phase"] != "training_and_test_complete_pending_backup_and_snapshot":
        raise ValueError("Refuse to package an unfinished or failed pilot as complete")
    if state["summary"]["groups_completed"] != 100 or state["summary"]["optimizer_updates"] < 1:
        raise ValueError("Actual training updates are incomplete")
    if state["test_summary"]["count"] != 50 or state["test_summary"]["split"] != "test":
        raise ValueError("The fixed held-out final test is incomplete")
    test = root / "runs/a6000-srf-final-test"
    if len(list(test.glob("*_base.wav"))) != 50 or len(list(test.glob("*_candidate.wav"))) != 50:
        raise ValueError("Missing held-out audio pairs")
    paths = []
    for directory in (root / "runs", root / "data"):
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("Unexpected output symlink")
            if path.is_file():
                if path.suffix in {".partial", ".writing", ".uploading"}:
                    raise ValueError("Incomplete generated output remains")
                paths.append(path)
    # Explicit allowlist: no process environment, SSH keys, tokens or caches.
    receipts = ["audit.json", "cuda_probe.json", "weights_manifest.json",
                "analysis-preflight-receipt.json", "analysis-runtime-status.json",
                "pipeline_state.json", "pytest-srf.log", "acestep_a6000_srf_resolved.json",
                "pip-freeze.txt", "train_probe.log", "train_main.log", "test_final.log",
                "pipeline.log", "run_a6000_pipeline.py", "RUNBOOK.json"]
    paths.extend(p for name in receipts if (p := root / "provisioning" / name).is_file())
    paths = sorted(set(paths))
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "acestep_a6000_srf_results_20261003.tar.gz"
    if archive.exists():
        raise FileExistsError("Preserve existing archive")
    members = {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in paths}
    manifest = {"schema_version": 1, "status": "training_and_test_outputs_complete",
                "node_release_authorized_by_this_file_alone": False,
                "snapshot_and_local_sha256_verification_still_required": True,
                "source_root": str(root), "members": members,
                "total_uncompressed_bytes": sum(v["bytes"] for v in members.values())}
    manifest_path = destination / "MEMBERS_SHA256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    temporary = archive.with_suffix(".partial")
    with tarfile.open(temporary, "w:gz", compresslevel=3) as output:
        for path in paths:
            output.add(path, arcname=str(path.relative_to(root)), recursive=False)
        output.add(manifest_path, arcname="MEMBERS_SHA256.json", recursive=False)
    temporary.replace(archive)
    receipt = {"archive": archive.name, "bytes": archive.stat().st_size,
               "sha256": sha256(archive), "members_manifest_sha256": sha256(manifest_path),
               "files": len(members), "base_generation_weights_included": False}
    (destination / "ARCHIVE_SHA256.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt


def verify(destination: Path) -> dict:
    manifest_path = destination / "MEMBERS_SHA256.json"
    receipt = json.loads((destination / "ARCHIVE_SHA256.json").read_text())
    manifest = json.loads(manifest_path.read_text())
    archive = destination / receipt["archive"]
    if archive.stat().st_size != receipt["bytes"] or sha256(archive) != receipt["sha256"]:
        raise ValueError("Downloaded archive hash/size mismatch")
    if sha256(manifest_path) != receipt["members_manifest_sha256"]:
        raise ValueError("Member manifest changed")
    # A retry may encounter completed copies, but never overwrite an unrelated
    # or locally edited file under an expected output name.
    for relative, expected in manifest["members"].items():
        existing = destination / relative
        if existing.exists() and (existing.is_symlink() or not existing.is_file() or
                                  sha256(existing) != expected["sha256"]):
            raise FileExistsError("Preserve unexpected local output: " + relative)
    allowed = set(manifest["members"]) | {"MEMBERS_SHA256.json"}
    with tarfile.open(archive, "r:gz") as source:
        names = source.getnames()
        if len(names) != len(set(names)) or set(names) != allowed:
            raise ValueError("Unexpected or duplicate archive members")
        if any(not p.isfile() or Path(p.name).is_absolute() or ".." in Path(p.name).parts for p in source):
            raise ValueError("Unsafe archive member")
        source.extractall(destination, filter="data")
    for relative, expected in manifest["members"].items():
        path = destination / relative
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("Extracted member hash mismatch: " + relative)
    result = {"status": "archive_and_all_local_members_verified", "files": len(manifest["members"]),
              "total_uncompressed_bytes": manifest["total_uncompressed_bytes"],
              "archive_sha256": receipt["sha256"], "snapshot_verified": False,
              "node_released": False}
    (destination / "LOCAL_VERIFICATION.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("package", "verify"))
    parser.add_argument("--root", type=Path, default=Path("/mnt/ai-music-rl-20261003"))
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = package(args.root, args.destination) if args.action == "package" else verify(args.destination)
    print(json.dumps(result, indent=2))
