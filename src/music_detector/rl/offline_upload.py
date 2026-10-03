"""Incremental immutable Drive backup, decoupled from GPU data collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .offline_archive import hashes, json_write, metadata_files, shard
from .offline_drive_store import RcloneStore


def probe_route(dataset, staging, store, receipt):
    dataset, staging, receipt = Path(dataset), Path(staging), Path(receipt)
    probe = staging/"ROUTE_PROBE.json"
    content = {"kind":"offline_grpo_upload_route_probe",
        "drive_folder_id":store.folder_id,
        "collection_manifest_sha256":hashes(dataset/"COLLECTION_MANIFEST.json")["sha256"]}
    if probe.exists() and json.loads(probe.read_text()) != content:
        raise ValueError("Preserve unrelated upload probe")
    if not probe.exists():
        json_write(probe,content)
    upload = store.upload(probe,"verification/ROUTE_PROBE.json")
    if store.download_sha256(upload["path"]) != upload["sha256"]:
        raise ValueError("Drive route probe SHA-256 round trip mismatch")
    result = {**content,"sha256_round_trip_verified":True,"probe":upload,
              "verified_utc_unix":time.time()}
    if receipt.exists():
        existing = json.loads(receipt.read_text())
        if any(existing.get(key) != result[key] for key in content):
            raise ValueError("Existing route receipt belongs to another dataset")
    json_write(receipt,result)
    return result


def backup(dataset, staging, store, state_path, *, groups_per_shard=10, poll_seconds=20):
    dataset, staging, state_path = Path(dataset), Path(staging), Path(state_path)
    if groups_per_shard < 1:
        raise ValueError("Shard size must be positive")
    uploaded, source_members = {}, {}
    manifest = json.loads((dataset/"COLLECTION_MANIFEST.json").read_text())
    target = manifest["prompt_groups"]
    closed = 0

    def put(path,name):
        if name not in uploaded:
            uploaded[name] = store.upload(path,name)
        return uploaded[name]

    def state(phase,**details):
        value = {"phase":phase,"drive_folder_id":store.folder_id,
            "groups_uploaded":closed,"target_groups":target,
            "uploaded_objects":len(uploaded),"verified_payload_bytes":sum(v["bytes"] for v in uploaded.values()),
            "sha256_file_manifest_preserved":True,"drive_payload_verification":"size_and_md5",
            "full_payload_sha256_readback_performed":False,**details}
        json_write(state_path,value)
        return value

    try:
        for name,path in metadata_files(dataset):
            source_members[name] = put(path,name)
        while closed < target:
            collection = json.loads((dataset/"collection_state.json").read_text())
            if collection["phase"] == "failed":
                raise RuntimeError("Collector failed; preserve source files and partial backup")
            end = min(target,closed+groups_per_shard)
            if collection["groups_completed"] < end:
                state("waiting_for_closed_groups")
                time.sleep(poll_seconds)
                continue
            archive,receipt = shard(dataset,staging,closed,end)
            put(archive,"shards/"+archive.name)
            put(archive.with_name(archive.name+".json"),"shards/"+archive.name+".json")
            source_members.update(receipt["members"])
            closed = end
            state("uploading")
        # The collector writes SUMMARY then the file manifest, then its final state.
        while True:
            phase = json.loads((dataset/"collection_state.json").read_text())["phase"]
            if phase == "complete":
                break
            if phase == "failed":
                raise RuntimeError("Collector failed while finalizing its manifest")
            state("waiting_for_final_manifest")
            time.sleep(poll_seconds)
        summary = json.loads((dataset/"SUMMARY.json").read_text())
        if (summary["candidate_count"] != target*manifest["group_size"] or
                summary["audio_files"] != target*manifest["group_size"]*2 or summary["optimizer_updates"] != 0):
            raise ValueError("Final collection quantity/optimizer boundary mismatch")
        for name,path in metadata_files(dataset,final=True):
            put(path,name)
        source_members["SUMMARY.json"] = uploaded["SUMMARY.json"]
        rows = json.loads((dataset/"FILES_SHA256.json").read_text())["files"]
        if len(rows) != len(source_members) or {r["path"] for r in rows} != set(source_members):
            raise ValueError("Some dataset members are not covered by the Drive backup")
        if any(any(row[key] != source_members[row["path"]][key] for key in ("bytes","sha256")) for row in rows):
            raise ValueError("Final source manifest differs from uploaded payload")
        verification = staging/"DRIVE_UPLOAD_VERIFICATION.json"
        result = state("all_dataset_objects_verified",candidate_count=summary["candidate_count"],
                       audio_files=summary["audio_files"],objects=uploaded)
        json_write(verification,result)
        verified_receipt = store.upload(verification,verification.name)
        if store.download_sha256(verification.name) != verified_receipt["sha256"]:
            raise ValueError("Final Drive verification receipt SHA-256 readback mismatch")
        return state("complete",candidate_count=summary["candidate_count"],
                     audio_files=summary["audio_files"],objects=uploaded,verification_receipt=verified_receipt)
    except Exception as error:
        state("failed",error_type=type(error).__name__,message=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["probe","backup"])
    for name in ("dataset","staging","config","drive-folder-id","receipt"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--proxy",help="Only use after explicit user authorization")
    args = parser.parse_args()
    store = RcloneStore(args.config,args.drive_folder_id,proxy=args.proxy)
    try:
        function = probe_route if args.action == "probe" else backup
        result = function(args.dataset,args.staging,store,args.receipt)
        print(json.dumps({k:v for k,v in result.items() if k not in {"objects","probe"}}),flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
