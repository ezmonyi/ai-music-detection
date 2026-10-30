"""No live credentials or network calls: archive and upload gate contracts."""
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from music_detector.rl.offline_archive import hashes, json_write, metadata_files, shard
from music_detector.rl.offline_drive_store import RcloneStore
from music_detector.rl.offline_upload import backup, probe_route


def dataset(tmp_path,count=3,size=2):
    root = tmp_path/"data"
    manifest = {"prompt_groups":count,"group_size":size,"prompt_ids":[f"p{i}" for i in range(count)]}
    for name in ["COLLECTION_MANIFEST.json","CONFIG.json","SOURCE_BOUNDARY.json","PROMPTS.json",
                 "PROBE_SUMMARY.json"]:
        json_write(root/name,manifest if name=="COLLECTION_MANIFEST.json" else {})
    (root/"behavior_policy.pt").write_bytes(b"frozen fixture adapter")
    for name in ["offline_collect.py","offline_io.py","offline_trajectory.py"]:
        (root/"source").mkdir(exist_ok=True)
        (root/"source"/name).write_bytes(b"source fixture")
    for index in range(count):
        folder = root/"groups"/f"g{index:06d}"
        folder.mkdir(parents=True)
        names = ["condition.pt"]+[f"s{s:02d}_{kind}.{suffix}" for s in range(size) for kind,suffix in
            [("candidate","wav"),("base","wav"),("trajectory","pt")]]
        for name in names:
            (folder/name).write_bytes(f"data {index} {name}".encode())
        files = [{"path":name,**hashes(folder/name)} for name in names]
        json_write(folder/"group.json",{"group":index,"prompt_id":f"p{index}","closed":True,
            "results":[{"valid":True}]*size,"files":files})
    json_write(root/"SUMMARY.json",{"candidate_count":count*size,"audio_files":count*size*2,"optimizer_updates":0})
    files = [{"path":str(p.relative_to(root)),**hashes(p)} for p in sorted(root.rglob("*")) if p.is_file()]
    json_write(root/"FILES_SHA256.json",{"files":files})
    json_write(root/"collection_state.json",{"phase":"complete","groups_completed":count})
    return root


class LocalStore:
    folder_id = "approved_fixture_folder"
    def __init__(self,root):
        self.root = root
    def upload(self,path,name):
        target = self.root/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and hashes(target) != hashes(path):
            raise ValueError("Immutable object changed")
        if not target.exists():
            shutil.copy2(path,target)
        return {"path":name,**hashes(target),"drive_size_md5_verified":True,
                "remote_sha256_readback_verified":False}
    def download_sha256(self,name):
        return hashes(self.root/name)["sha256"]


def test_archive_contains_only_closed_group_members(tmp_path):
    root = dataset(tmp_path)
    path,receipt = shard(root,tmp_path/"staging",0,2)
    with tarfile.open(path) as archive:
        assert set(archive.getnames()) == set(receipt["members"])
    assert len(receipt["members"])==16
    assert receipt["archive"]==hashes(path)
    assert shard(root,tmp_path/"staging",0,2)==(path,receipt)


def test_archive_rejects_changed_or_unclosed_group(tmp_path):
    root = dataset(tmp_path)
    (root/"groups/g000000/s00_trajectory.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="checksum"):
        shard(root,tmp_path/"staging",0,1)
    group = root/"groups/g000001/group.json"
    contents = json.loads(group.read_text()); contents["closed"]=False; json_write(group,contents)
    with pytest.raises(ValueError,match="identity"):
        shard(root,tmp_path/"staging",1,2)


def test_archive_rejects_path_traversal_and_extra_member(tmp_path):
    root = dataset(tmp_path)
    metadata = root/"groups/g000000/group.json"
    group = json.loads(metadata.read_text())
    group["files"][0]["path"]="../../private/credential.conf"; json_write(metadata,group)
    with pytest.raises(ValueError,match="member set"):
        shard(root,tmp_path/"staging",0,1)


def test_archive_refuses_corrupt_existing_archive(tmp_path):
    root = dataset(tmp_path)
    path,_ = shard(root,tmp_path/"staging",0,1)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="do not overwrite"):
        shard(root,tmp_path/"staging",0,1)


def test_probe_receipt_requires_actual_sha256_readback(tmp_path):
    root = dataset(tmp_path)
    store = LocalStore(tmp_path/"drive")
    receipt = tmp_path/"route.json"
    probe = probe_route(root,tmp_path/"staging",store,receipt)
    assert probe["sha256_round_trip_verified"]
    assert json.loads(receipt.read_text())["drive_folder_id"] == store.folder_id
    store.download_sha256 = lambda _: "0"*64
    with pytest.raises(ValueError,match="round trip mismatch"):
        probe_route(root,tmp_path/"staging",store,tmp_path/"bad_route.json")
    assert not (tmp_path/"bad_route.json").exists()


def test_full_backup_quantity_checksums_and_honest_verification_method(tmp_path):
    root = dataset(tmp_path)
    store = LocalStore(tmp_path/"drive")
    result = backup(root,tmp_path/"staging",store,tmp_path/"state.json",groups_per_shard=2,poll_seconds=0)
    assert result["phase"]=="complete" and result["groups_uploaded"]==3
    assert result["candidate_count"]==6 and result["audio_files"]==12
    assert result["drive_payload_verification"]=="size_and_md5"
    assert result["full_payload_sha256_readback_performed"] is False
    assert len(list((tmp_path/"drive/shards").glob("*.tar")))==2
    assert all(hashes(store.root/name)["sha256"]==row["sha256"] for name,row in result["objects"].items())


def test_backup_fails_if_source_manifest_omits_a_payload_member(tmp_path):
    root = dataset(tmp_path)
    index = json.loads((root/"FILES_SHA256.json").read_text()); index["files"].pop()
    json_write(root/"FILES_SHA256.json",index)
    with pytest.raises(ValueError,match="not covered"):
        backup(root,tmp_path/"staging",LocalStore(tmp_path/"drive"),tmp_path/"state.json",poll_seconds=0)
    assert json.loads((tmp_path/"state.json").read_text())["phase"]=="failed"


def test_backup_preserves_failed_collection(tmp_path):
    root = dataset(tmp_path)
    json_write(root/"collection_state.json",{"phase":"failed","groups_completed":2})
    with pytest.raises(RuntimeError,match="Collector failed"):
        backup(root,tmp_path/"staging",LocalStore(tmp_path/"drive"),tmp_path/"state.json",poll_seconds=0)
    assert len(list(root.glob("groups/*/*_candidate.wav")))==6


def test_metadata_allowlist_excludes_credentials(tmp_path):
    root = dataset(tmp_path)
    (root/"private").mkdir(); (root/"private/token.json").write_text("never upload")
    assert all("private" not in name for name,_ in metadata_files(root,final=True))


def test_drive_profile_requires_exact_task_root_and_permissions(tmp_path):
    path = tmp_path/"config.conf"
    path.write_text("[offline_drive]\ntype=drive\nroot_folder_id=exact_folder\n")
    path.chmod(0o600)
    with pytest.raises(ValueError,match="approved dataset folder"):
        RcloneStore(path,"another_folder")
    path.chmod(0o644)
    with pytest.raises(ValueError,match="mode 0600"):
        RcloneStore(path,"exact_folder")
    path.chmod(0o600)
    with pytest.raises(ValueError,match="authorized loopback"):
        RcloneStore(path,"exact_folder",proxy="http://unapproved-proxy.example")


def test_remote_path_rejects_unsafe_targets():
    for name in ["../x","/root/x","shards/../x","x\\y","x//y",""]:
        with pytest.raises(ValueError,match="Unsafe"):
            RcloneStore.remote(name)
