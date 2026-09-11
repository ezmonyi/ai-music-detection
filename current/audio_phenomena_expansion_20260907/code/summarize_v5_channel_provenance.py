#!/usr/bin/env python3
"""Read-only inventory of stored channel metadata, not a new audio decode."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


PACKAGE_COMMIT = "7cc03aa840c910df919e531153b61daba864e80794abc91ebf29d3859f70947a"
PINS = {
    "manifests/equal60_development_v2/inference_manifest.csv": "f52b93bf4ff820bd35845749b91a2c7139b901588ebde2ebf62f4442f583d3bf",
    "manifests/mureka60_inputs_v2/inference_manifest.csv": "3de2deeb6fe8447c6e1683cdfbeaff36c2dd3fde0e639298bad2fe46e2cad159",
    "manifests/saraga_external103_intervals_v1/final/inference_manifest.csv": "f52ec01d834df7bf73e33e0133adf4d5f93ed81c2a7c913bbdfdd86610642fc9",
    "manifests/saraga_external103_intervals_v1/final/item_proof_inventory.json": "b0dba7785eb626bd06a6f9531907418946cca32291395c7c4f03f677df33712c",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(root, source_manifest_root=None):
    source_manifest_root = source_manifest_root or root / "manifests"
    def manifest_path(name):
        return source_manifest_root / name.removeprefix("manifests/")

    package = root / "results/equal60_exploratory_package_v5_v1"
    require(digest(package / "COMMIT.json") == PACKAGE_COMMIT, "Package changed")
    commit = json.loads((package / "COMMIT.json").read_text())
    metadata_path = package / "metadata_60s.csv"
    require(digest(metadata_path) == commit["files"][metadata_path.name]["sha256"], "Metadata changed")
    metadata = rows(metadata_path)
    require(len(metadata) == len({r["id"] for r in metadata}) == 2207, "Wrong cohort")
    for name, sha in PINS.items():
        require(digest(manifest_path(name)) == sha, f"Manifest changed: {name}")
    source_rows = {}
    for name in PINS:
        if not name.endswith(".csv"):
            continue
        for record in rows(manifest_path(name)):
            identity = record["item_id"]
            require(identity not in source_rows, "Duplicate original identity")
            source_rows[identity] = record
    require(set(source_rows) == {r["id"] for r in metadata}, "Manifest/cohort mismatch")
    proof_root = source_manifest_root / "saraga_external103_intervals_v1"
    inventory = json.loads((proof_root / "final/item_proof_inventory.json").read_text())
    counts = defaultdict(lambda: {"rows": 0, "native_channels": Counter(),
                                 "standardized_channels": Counter(), "native_sample_rates_hz": Counter()})
    proof_paths = {}
    for record in metadata:
        identity = record["id"]
        original = source_rows[identity]
        require(original["group_id"] == record["group_id"], "Group mismatch")
        require(original["label"] == record["label"], "Label mismatch")
        if identity in inventory:
            path = proof_root / "items" / identity / "proof.json"
            require(digest(path) == inventory[identity], "Saraga proof changed")
            proof_paths[path] = inventory[identity]
            proof = json.loads(path.read_text())
            require(proof["item_id"] == identity, "Wrong proof identity")
            interval = proof["interval_proof"]
            require(interval["raw_hashes_after"]["sha256"] == original["original_source_audio_sha256"], "Raw binding mismatch")
            require(proof["wav"]["file_sha256"] == original["standardized_file_sha256"], "WAV binding mismatch")
            native_channels = interval["native_channels"]
            native_sr = interval["native_sample_rate_hz"]
        else:
            native_channels = int(original["source_channels"])
            native_sr = int(original["source_sample_rate"])
        require(native_channels >= 1 and native_sr > 0, "Invalid native metadata")
        require(native_sr == int(record["native_sample_rate_hz"]), "Native rate mismatch")
        current = counts[record["source_group"]]
        current["rows"] += 1
        current["native_channels"][native_channels] += 1
        current["standardized_channels"][int(original["standardized_channels"])] += 1
        current["native_sample_rates_hz"][native_sr] += 1
    require(len(proof_paths) == 103, "Incomplete Saraga proof inventory")
    for path, sha in proof_paths.items():
        require(digest(path) == sha, "Proof changed during inventory")
    for name, sha in PINS.items():
        require(digest(manifest_path(name)) == sha, "Manifest changed during inventory")
    require(digest(package / "COMMIT.json") == PACKAGE_COMMIT, "Package changed during inventory")
    require(digest(metadata_path) == commit["files"][metadata_path.name]["sha256"], "Metadata changed during inventory")
    return {"status": "stored_metadata_inventory_complete", "rows": len(metadata),
            "package_commit_sha256": PACKAGE_COMMIT, "input_manifest_sha256": PINS,
            "saraga_item_proofs_rehashed": len(proof_paths),
            "sources": dict(sorted(counts.items())), "raw_audio_decoded_this_run": False,
            "scope": "Stored original/container channel counts are not proof of acoustic stereo, nonduplicated channels, native bandwidth, or detector eligibility. No feature extraction or scoring."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-manifest-root", type=Path,
                        help="Explicit original-manifest root; default ARTIFACT_ROOT/manifests. Use NFS data root for remote originals.")
    args = parser.parse_args()
    print(json.dumps(summarize(args.artifact_root, args.source_manifest_root), indent=2, sort_keys=True))
