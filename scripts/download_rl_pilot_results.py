"""Copy the exact packaged member allowlist without a second local archive.

SCP MEMBERS_SHA256.json and either the final ARCHIVE_SHA256.json receipt or
explicitly supply the real manifest SHA-256 read back over the SSH connection.
The latter allows member backup while the optional remote archive compresses;
it does not claim the archive is complete or verified. The existing package
builder supplies the completion gates and excludes secrets.
Rsync copies only listed missing files; every copied/existing file is SHA-256
checked. The archive itself is NOT downloaded or claimed locally verified.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def member_path(destination,name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts or
            str(relative) != name or "\\" in name or "\n" in name or "\x00" in name):
        raise ValueError("Unsafe packaged member path")
    path = destination/name
    if destination.resolve() not in path.resolve().parents:
        raise ValueError("Packaged member escapes destination")
    if any(parent.is_symlink() for parent in [path,*path.parents] if parent != destination.parent):
        raise ValueError("Preserve local symlink; never follow it while copying")
    return path


def copy_verified(destination,host,port,socket,remote_root, *, runner=subprocess.run,
                  expected_manifest_sha256=None):
    destination = Path(destination)
    manifest_path = destination/"MEMBERS_SHA256.json"
    receipt_path = destination/"ARCHIVE_SHA256.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if expected_manifest_sha256 is None:
        if receipt is None:
            raise ValueError("Need an archive receipt or explicit SSH-source manifest SHA-256")
        expected_manifest_sha256 = receipt["members_manifest_sha256"]
        source_verification = "archive_receipt_manifest_sha256"
    else:
        source_verification = "explicit_ssh_source_manifest_sha256"
    if not re.fullmatch(r"[0-9a-f]{64}",expected_manifest_sha256):
        raise ValueError("Invalid manifest SHA-256")
    if receipt is not None and receipt["members_manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("Archive receipt and source manifest SHA-256 disagree")
    if sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Member manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "training_and_test_outputs_complete":
        raise ValueError("Do not copy an unfinished pilot as complete")
    missing = []
    for name,expected in manifest["members"].items():
        path = member_path(destination,name)
        if path.exists():
            if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                raise FileExistsError("Preserve unexpected local output: "+name)
        else:
            missing.append(name)
    if missing:
        missing_bytes = sum(manifest["members"][name]["bytes"] for name in missing)
        free_bytes = shutil.disk_usage(destination).free
        if free_bytes - missing_bytes < 4*1024**3:
            raise ValueError("Insufficient local capacity: preserve 4 GiB safety headroom")
        print(json.dumps({"phase":"copying_exact_missing_members","missing_files":len(missing),
                          "missing_bytes":missing_bytes,"local_free_bytes":free_bytes}),flush=True)
        # No --delete, --inplace, directory-wide recursion, credentials or caches.
        ssh = shlex.join(["ssh","-S",socket,"-o","BatchMode=yes","-o","ConnectTimeout=15","-p",str(port)])
        runner(["rsync","-a","--no-perms","--no-owner","--no-group",
                "--partial-dir=.rl-rsync-partial","--files-from=-","--from0","--relative",
                "-e",ssh,host+":"+remote_root.rstrip("/")+"/",str(destination)+"/"],
               input=("\x00".join(missing)+"\x00").encode(),check=True)
    for name,expected in manifest["members"].items():
        path = member_path(destination,name)
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError("Copied member byte/SHA-256 mismatch: "+name)
    result = {"status":"all_local_members_sha256_verified","files":len(manifest["members"]),
        "total_uncompressed_bytes":manifest["total_uncompressed_bytes"],
        "verification_method":"member_allowlist_rsync_then_sha256",
        "members_manifest_sha256":expected_manifest_sha256,
        "manifest_source_verification":source_verification,
        "archive_receipt_available_at_start":receipt is not None,
        "archive_downloaded_or_locally_verified":False,"snapshot_verified":False,"node_released":False}
    temporary = destination/"LOCAL_VERIFICATION.json.writing"
    temporary.write_text(json.dumps(result,indent=2)+"\n")
    temporary.replace(destination/"LOCAL_VERIFICATION.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--host",required=True)
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--control-socket",required=True)
    parser.add_argument("--remote-root",required=True)
    parser.add_argument("--manifest-sha256",help="Actual source manifest digest read back over SSH; allows parallel member backup without a fabricated archive receipt")
    args = parser.parse_args()
    print(json.dumps(copy_verified(args.destination,args.host,args.port,args.control_socket,args.remote_root,
                                  expected_manifest_sha256=args.manifest_sha256),indent=2))
