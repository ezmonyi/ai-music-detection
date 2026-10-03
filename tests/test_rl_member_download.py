"""Low-disk copying never overwrites edited local results or widens the list."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/"scripts/download_rl_pilot_results.py"
spec=importlib.util.spec_from_file_location("member_download",SCRIPT)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def fixture(tmp_path):
    local,remote=tmp_path/"local",tmp_path/"remote"
    local.mkdir(); remote.mkdir()
    members={}
    for name in ["runs/test/a_candidate.wav","runs/main/checkpoints/group_000100.pt"]:
        source=remote/name; source.parent.mkdir(parents=True,exist_ok=True); source.write_bytes(name.encode())
        members[name]={"bytes":source.stat().st_size,"sha256":module.sha256(source)}
    manifest={"status":"training_and_test_outputs_complete","members":members,
              "total_uncompressed_bytes":sum(r["bytes"] for r in members.values())}
    path=local/"MEMBERS_SHA256.json"; path.write_text(json.dumps(manifest))
    (local/"ARCHIVE_SHA256.json").write_text(json.dumps({"members_manifest_sha256":module.sha256(path)}))
    return local,remote,manifest


def test_copies_only_missing_manifest_members_then_verifies(tmp_path):
    local,remote,manifest=fixture(tmp_path)
    existing=next(iter(manifest["members"]))
    (local/existing).parent.mkdir(parents=True); shutil.copy2(remote/existing,local/existing)
    calls=[]
    def runner(args,*,input,check):
        calls.append(args)
        assert "--delete" not in args and "--inplace" not in args
        names=input.decode().rstrip("\x00").split("\x00")
        assert names==[name for name in manifest["members"] if name!=existing]
        for name in names:
            target=local/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(remote/name,target)
    result=module.copy_verified(local,"node",22,"socket",str(remote),runner=runner)
    assert result["status"]=="all_local_members_sha256_verified"
    assert result["archive_downloaded_or_locally_verified"] is False and len(calls)==1
    module.copy_verified(local,"node",22,"socket",str(remote),runner=lambda *_a,**_k: pytest.fail("Should reuse verified copies"))


def test_preserves_edited_existing_file(tmp_path):
    local,remote,manifest=fixture(tmp_path)
    name=next(iter(manifest["members"]))
    path=local/name; path.parent.mkdir(parents=True); path.write_bytes(b"user edit")
    with pytest.raises(FileExistsError,match="Preserve"):
        module.copy_verified(local,"node",22,"socket",str(remote))
    assert path.read_bytes()==b"user edit"


def test_fails_without_sha256_matched_download(tmp_path):
    local,remote,_=fixture(tmp_path)
    with pytest.raises(ValueError,match="Copied member"):
        module.copy_verified(local,"node",22,"socket",str(remote),runner=lambda *_a,**_k: None)
    assert not (local/"LOCAL_VERIFICATION.json").exists()


def test_unsafe_paths_and_symlinks_are_rejected(tmp_path):
    local,_,_=fixture(tmp_path)
    (local/"runs").symlink_to(tmp_path/"other")
    for name in ["../../private/token","/tmp/x","runs/x","x//y","x\\y"]:
        with pytest.raises(ValueError):
            module.member_path(local,name)


def test_manifest_receipt_corruption_is_rejected(tmp_path):
    local,remote,_=fixture(tmp_path)
    (local/"MEMBERS_SHA256.json").write_text("{}")
    with pytest.raises(ValueError,match="manifest SHA-256"):
        module.copy_verified(local,"node",22,"socket",str(remote))


def test_parallel_copy_needs_real_source_digest_and_does_not_fabricate_receipt(tmp_path):
    local,remote,manifest=fixture(tmp_path)
    (local/"ARCHIVE_SHA256.json").unlink()
    digest=module.sha256(local/"MEMBERS_SHA256.json")
    def runner(args,*,input,check):
        assert "--delete" not in args and "--inplace" not in args
        for name in input.decode().rstrip("\x00").split("\x00"):
            target=local/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(remote/name,target)
    result=module.copy_verified(local,"node",22,"socket",str(remote),runner=runner,
                                expected_manifest_sha256=digest)
    assert result["status"]=="all_local_members_sha256_verified"
    assert result["manifest_source_verification"]=="explicit_ssh_source_manifest_sha256"
    assert result["archive_receipt_available_at_start"] is False
    assert result["archive_downloaded_or_locally_verified"] is False
    assert not (local/"ARCHIVE_SHA256.json").exists()


def test_missing_or_wrong_source_digest_refuses_parallel_copy(tmp_path):
    local,remote,_=fixture(tmp_path)
    (local/"ARCHIVE_SHA256.json").unlink()
    with pytest.raises(ValueError,match="Need an archive receipt"):
        module.copy_verified(local,"node",22,"socket",str(remote))
    with pytest.raises(ValueError,match="manifest SHA-256 mismatch"):
        module.copy_verified(local,"node",22,"socket",str(remote),expected_manifest_sha256="0"*64)


def test_local_capacity_safety_gate_precedes_transfer(tmp_path,monkeypatch):
    local,remote,_=fixture(tmp_path)
    monkeypatch.setattr(module.shutil,"disk_usage",lambda _p: type("Disk",(),{"free":4*1024**3})())
    with pytest.raises(ValueError,match="Insufficient local capacity"):
        module.copy_verified(local,"node",22,"socket",str(remote),runner=lambda *_a,**_k: pytest.fail("Do not transfer without headroom"))


@pytest.mark.skipif(not shutil.which("rsync"),reason="rsync not installed")
def test_actual_installed_rsync_supports_the_exact_copy_flags(tmp_path):
    local,remote,manifest=fixture(tmp_path)
    names=list(manifest["members"])
    subprocess.run(["rsync","-a","--no-perms","--no-owner","--no-group",
        "--partial-dir=.rl-rsync-partial","--files-from=-","--from0","--relative",
        str(remote)+"/",str(local)+"/"],input=("\x00".join(names)+"\x00").encode(),check=True)
    for name in names:
        assert module.sha256(local/name)==module.sha256(remote/name)
