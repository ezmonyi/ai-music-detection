#!/usr/bin/env python3
"""Prospective metadata draft only. Never opens audio, fits, freezes or scores."""
import argparse
import collections
import csv
import ctypes
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile

VERSION = "saraga_external_cohort_v1"
PHYSICAL = "manifests/saraga_hindustani_physical_v1"
PACKAGE = "results/equal60_package_v4_with_recovery_v1"
PINS = {
    f"{PHYSICAL}/summary.json": "59dc718293c45c492e945da825d379969836ff13dfb0fb2a894d7570391c73c3",
    f"{PHYSICAL}/contract.json": "89deda07a35e8aab6c17da2a8339971a89a7295e220ef69d2585f1557f8c3e51",
    "audit/saraga_physical_independent_v1.json": "ffd9158dc2095c918f8bc379c52cc6f1c12eab253de80c261332e9ff120fd03d",
    "audit/saraga_hindustani_reconciliation_v2.json": "d92da602fc6e5dc74469c3fdf1c39d61b4086a4be3a7c4c9ed7f565de6aff2a3",
    "audit/saraga_catalog_v1.json": "2ef89a32a62085c6bf6f33e857309a1e33a7cab646548ca73368dcc9641e3a28",
    "results/equal60_results_v4_with_recovery_v1/run_manifest.json": "03299363a46da5c9704eacbe672e970e63cbce17513a1bfc56cee3213112c9de",
    "audit/equal60_v4_with_recovery_results_audit_v1.json": "9b93b558cff42426218abf8ee79ba19833b2a57918319e8855c54aa03bc5a53c",
    "preregistration/equal60_v4_with_recovery_frozen_v1.json": "200c54d90f863b5174ad0d12801b4d1349838b6f538c05e6da9c713a854dbaad",
}
SPEECH = {"450a6fcc-3c0a-483d-a31b-dde91413dcdd", "4bc5cb16-9808-431b-87a4-4dd63af6bb17", "5c464958-6411-480b-8b36-84da68bea036"}
MISSING = {"2073b056-8120-4b04-b808-3d6a8c6a2bd7", "25d5c348-023b-487b-b149-a46d553cedee"}
ELIGIBLE_ID_SHA = "a0df862441fb0cf9f214d739a6fca55dc54626457fdd410810aeb1690d35321a"
RAW_ROOT = "/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/saraga_hindustani_physical_v1/raw"
ROLE = "external_human_unscored"
SOURCE = "human_saraga_hindustani_v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def seal(obj):
    return sha(canonical(obj))


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(raw):
    obj = json.loads(raw, object_pairs_hook=unique_pairs,
                     parse_constant=lambda x: require(False, f"Nonfinite JSON: {x}"))
    def check(value):
        if isinstance(value, float):
            require(math.isfinite(value), "Nonfinite JSON number")
        elif isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(obj)
    return obj


def no_symlinks(path):
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        require(not part.is_symlink(), f"Symlink forbidden: {part}")
    return path


def read_bytes(path):
    path = no_symlinks(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), f"Not regular: {path}")
        return stream.read()


def ident(value, synthetic=False):
    pattern = r"synthetic_[a-z0-9_]+" if synthetic else r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    require(isinstance(value, str) and re.fullmatch(pattern, value), f"Missing/invalid identity: {value}")
    return value


def positive_int(value, name, allow_zero=False):
    require(type(value) is int and value >= (0 if allow_zero else 1), f"Invalid integer {name}")
    return value


def hash_value(value, size=64):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % size, value), "Invalid digest")
    return value


def csv_rows(raw):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), "Duplicate/empty CSV headers")
    rows = list(reader)
    require(all(None not in r and None not in r.values() for r in rows), "Malformed CSV")
    return rows


def keyed(rows, field):
    result = {r[field]: r for r in rows}
    require(len(result) == len(rows), f"Duplicate {field}")
    return result


def group_records(records, synthetic=False):
    """Union across every performer, every album artist and every release."""
    parents = {}
    keys_by_id = {}
    def find(k):
        parents.setdefault(k, k)
        if parents[k] != k:
            parents[k] = find(parents[k])
        return parents[k]
    for record in records:
        item = record["item"]
        mbid = ident(item["mbid"], synthetic)
        m = item["reconciled_metadata"]
        performers = [ident(p["artist_mbid"], synthetic) for p in m["performer_credits"]]
        album = [ident(a["mbid"], synthetic) for a in m["album_artists"]]
        releases = [ident(r["mbid"], synthetic) for r in m["release_or_concert_groups"]]
        require(performers and album and releases, "Missing group identity; no name imputation")
        require(all(r["kind"] == "release" for r in m["release_or_concert_groups"]), "Unknown release kind")
        keys = sorted({"artist:" + x for x in performers + album} | {"release:" + x for x in releases})
        keys_by_id[mbid] = keys
        for key in keys[1:]:
            parents[find(key)] = find(keys[0])
    components = collections.defaultdict(lambda: {"mbids": [], "keys": set()})
    for mbid, keys in keys_by_id.items():
        component = components[find(keys[0])]
        component["mbids"].append(mbid)
        component["keys"].update(keys)
    groups, assigned = {}, {}
    for component in components.values():
        full_keys = sorted(component["keys"])
        gid = "saraga_group_" + seal(full_keys)
        require(gid not in groups, "Component digest collision")
        members = sorted(component["mbids"])
        groups[gid] = {"full_keys": full_keys, "mbids": members, "record_count": len(members)}
        assigned.update({mbid: gid for mbid in members})
    return dict(sorted(groups.items())), assigned


def make_rows(records, assigned, synthetic=False):
    rows = []
    for record in sorted(records, key=lambda r: r["item"]["mbid"]):
        item, measurement = record["item"], record["measurement"]
        mbid, metadata = item["mbid"], item["reconciled_metadata"]
        sr = positive_int(measurement["sample_rate"], "sample_rate")
        require(sr in (44100, 48000), "Unexpected native rate")
        frames = positive_int(measurement["actual_frames"], "actual_frames")
        header = positive_int(measurement["header_frames"], "header_frames")
        channels = positive_int(measurement["channels"], "channels")
        require(channels == 2, "Unexpected channel count")
        count = 60 * sr
        require(frames >= count and measurement["real_empty_read_observed"] is True, "Missing actual EOF or less than 60 seconds")
        require(measurement["header_matches_actual_eof"] is (frames == header), "Header mismatch flag inconsistent")
        require(measurement["header_minus_actual_frames"] == header - frames, "Header difference inconsistent")
        start = (frames - count) // 2
        offset = repr(start / sr)
        require(round(float(offset) * sr) == start, "Source offset frame roundtrip failed")
        raw = measurement["raw_hashes_before_decode"]
        require(raw == measurement["raw_hashes_after_decode"], "Acquisition hashes changed")
        for field in ("sha256", "md5", "bytes"):
            require(raw[field] == item["archive_member"][field], "Archive/raw identity mismatch")
        hash_value(raw["sha256"])
        hash_value(raw["md5"], 32)
        positive_int(raw["bytes"], "raw bytes")
        require(item["raw_path"] == f"raw/{mbid}.mp3", "Noncanonical raw path")
        root = "/synthetic_test_only/raw" if synthetic else RAW_ROOT
        rows.append({
            "item_id": "saraga_hindustani_" + mbid, "mbid": mbid,
            "label": 0, "source_id": SOURCE, "role": ROLE, "group_id": assigned[mbid],
            "title": metadata["title"], "source_audio_path": f"{root}/{mbid}.mp3",
            "acquisition_raw_sha256": raw["sha256"], "acquisition_raw_md5": raw["md5"], "acquisition_raw_bytes": raw["bytes"],
            "native_sample_rate_hz": sr, "native_channels": channels,
            "actual_eof_frames": frames, "soundfile_header_frames": header,
            "header_minus_actual_frames": header - frames, "header_matches_actual_eof": frames == header,
            "crop_start_frame": start, "crop_frames": count, "crop_end_frame_exclusive": start + count,
            "source_offset_seconds": offset, "source_offset_frame_roundtrip": start,
            "duration_seconds": 60, "crop_authority": "acquisition_receipt_actual_sequential_EOF",
            "performer_artist_mbids": ";".join(sorted({p["artist_mbid"] for p in metadata["performer_credits"]})),
            "album_artist_mbids": ";".join(sorted({a["mbid"] for a in metadata["album_artists"]})),
            "release_mbids": ";".join(sorted({r["mbid"] for r in metadata["release_or_concert_groups"]})),
            "provenance_status": "pinned_acquisition_receipt_metadata_only",
            "current_raw_bytes_verified": False, "crop_materialized": False,
            "classifier_admission": False, "evaluation_allowed": False,
            "synthetic_test_only": synthetic,
        })
    return rows


def rename_exclusive(source, destination):
    """Atomic directory publication with kernel-enforced no-replace."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        fn = libc.renamex_np
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = fn(os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        fn = libc.renameat2
        fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = fn(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise ValueError("Atomic no-replace rename unsupported; fail closed")
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def prepare(root, output, *, _synthetic_pins=None):
    """CLI has no synthetic switch. Private fixture seam requires marked tiny input."""
    root, output = no_symlinks(root), no_symlinks(output)
    require(root.is_dir() and output.parent.is_dir(), "Input/output parent missing")
    require(not output.exists(), "Output conflict")
    synthetic = _synthetic_pins is not None
    pins = dict(PINS if not synthetic else _synthetic_pins)
    require(set(pins) == set(PINS), "Input pin inventory changed")
    inventory, loaded = {}, {}
    def bind(relative, expected=None):
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Unsafe input path")
        path = root / relative
        raw = read_bytes(path)
        digest = sha(raw)
        require(expected is None or digest == expected, f"Stale pin: {relative}")
        inventory[relative] = {"sha256": digest, "bytes": len(raw)}
        return raw
    for relative, expected in pins.items():
        loaded[relative] = strict_json(bind(relative, expected))
        if synthetic:
            require(loaded[relative].get("synthetic_test_only") is True, "Synthetic marker missing")
    summary = loaded[f"{PHYSICAL}/summary.json"]
    contract = loaded[f"{PHYSICAL}/contract.json"]
    audit = loaded["audit/saraga_physical_independent_v1.json"]
    reconciliation = loaded["audit/saraga_hindustani_reconciliation_v2.json"]
    frozen = loaded["preregistration/equal60_v4_with_recovery_frozen_v1.json"]
    run = loaded["results/equal60_results_v4_with_recovery_v1/run_manifest.json"]
    old_audit = loaded["audit/equal60_v4_with_recovery_results_audit_v1.json"]
    require(summary["status"] == "passed_all_physical_acquisition_not_audio_admission", "Physical summary not passed")
    require(audit["status"] == "passed_independent_byte_accounting_audit_not_second_decode_not_admission", "Physical audit not passed")
    require(audit["summary_sha256"] == pins[f"{PHYSICAL}/summary.json"], "Physical summary audit binding")
    require(audit["contract_sha256"] == pins[f"{PHYSICAL}/contract.json"] == summary["contract_sha256"], "Physical contract binding")
    require(audit["reconciliation_sha256"] == pins["audit/saraga_hindustani_reconciliation_v2.json"], "Reconciliation binding")
    require(frozen["status"] == "frozen" and old_audit["status"] == "passed", "Old v4 package not frozen/audited")
    require(frozen["contract"] == run["contract"], "Old v4 contract mismatch")
    require(old_audit["result_manifest_sha256"] == pins["results/equal60_results_v4_with_recovery_v1/run_manifest.json"], "Old result audit binding")
    records = summary["records"]
    require(len(records) == summary["record_count"] == audit["record_count"] == contract["expected_recordings"], "Recording count mismatch")
    require(1 <= len(records) <= 12 if synthetic else len(records) == 108, "Unexpected input cohort size")
    by_id = keyed([dict(r, mbid=r["item"]["mbid"]) for r in records], "mbid")
    for mbid in by_id:
        ident(mbid, synthetic)
    items = keyed(contract["items"], "mbid")
    tracks = keyed(reconciliation["tracks"], "mbid")
    audited = keyed(audit["records"], "mbid")
    require(set(by_id) == set(items) == set(tracks) == set(audited), "Source identity inventory mismatch")
    receipt_dir = no_symlinks(root / PHYSICAL / "receipts")
    names = sorted(p.name for p in receipt_dir.iterdir())
    require(names == sorted(mbid + ".json" for mbid in by_id), "Receipt inventory mismatch")
    receipt_manifest, excluded, eligible = [], [], []
    for mbid in sorted(by_id):
        ident(mbid, synthetic)
        record = {k: v for k, v in by_id[mbid].items() if k != "mbid"}
        require(record["item"] == items[mbid] and record["item"]["reconciled_metadata"] == tracks[mbid], "Reconciled item mismatch")
        require(record["status"] == "passed_physical_acquisition_not_audio_admission" and record["classifier_admission"] is False and record["cohort_selected"] is False, "Physical admission/status mismatch")
        require(record["contract_sha256"] == pins[f"{PHYSICAL}/contract.json"], "Receipt contract mismatch")
        relative = f"{PHYSICAL}/receipts/{mbid}.json"
        receipt = strict_json(bind(relative))
        require(set(receipt) == {"record", "record_sha256"} and receipt["record"] == record and receipt["record_sha256"] == seal(record), "Receipt record seal mismatch")
        receipt_manifest.append({"mbid": mbid, "sha256": inventory[relative]["sha256"]})
        raw = record["measurement"]["raw_hashes_before_decode"]
        require(all(audited[mbid][k] == raw[k] for k in ("sha256", "md5", "bytes")), "Independent raw identity mismatch")
        metadata = record["item"]["reconciled_metadata"]
        require(metadata["mbid"] == mbid, "Metadata recording identity mismatch")
        for credit in metadata["performer_credits"]:
            ident(credit["artist_mbid"], synthetic)
        for credit in metadata["album_artists"]:
            ident(credit["mbid"], synthetic)
        for release in metadata["release_or_concert_groups"]:
            ident(release["mbid"], synthetic)
            require(release["kind"] == "release", "Unknown release kind")
        require(type(metadata["speech_title_review_flag"]) is bool, "Nonboolean speech flag")
        reasons = (["speech_title_review_flag"] if metadata["speech_title_review_flag"] else [])
        if not metadata["performer_credits"]:
            reasons.append("missing_performer_credit")
        if reasons:
            excluded.append({"mbid": mbid, "title": metadata["title"], "reasons": ";".join(reasons), "source_record_json": canonical(record).decode().strip()})
        else:
            eligible.append(record)
    require(seal(receipt_manifest) == audit["receipt_manifest_sha256"], "Receipt manifest audit binding")
    if not synthetic:
        require({r["mbid"] for r in excluded if "speech_title_review_flag" in r["reasons"]} == SPEECH, "Speech exclusion drift")
        require({r["mbid"] for r in excluded if "missing_performer_credit" in r["reasons"]} == MISSING, "Missing-credit exclusion drift")
    eligible_hash = sha("\n".join(sorted(r["item"]["mbid"] for r in eligible)).encode())
    require(eligible, "No eligible recordings")
    require(synthetic or (len(eligible) == 103 and eligible_hash == ELIGIBLE_ID_SHA), "Eligible identity pin mismatch")
    groups, assigned = group_records(eligible, synthetic)
    rows = make_rows(eligible, assigned, synthetic)
    sizes = sorted((g["record_count"] for g in groups.values()), reverse=True)
    rates = dict(sorted(collections.Counter(str(r["native_sample_rate_hz"]) for r in rows).items()))
    dimensions = {key: len({v for row in rows for v in row[key].split(";")}) for key in ("performer_artist_mbids", "album_artist_mbids", "release_mbids")}
    require(synthetic or sizes == [72, 14, 10, 5, 2], "Grouping topology drift")
    require(synthetic or rates == {"44100": 96, "48000": 7}, "Eligible rate drift")
    require(synthetic or list(dimensions.values()) == [36, 11, 35], "Identity cardinality drift")
    package_inputs = {}
    for name in ("metadata_60s.csv", "evidence/native.csv", "evidence/fhm.csv", "evidence/extraction_receipt.json"):
        package_inputs[name] = bind(f"{PACKAGE}/{name}", frozen["contract"]["package_files_sha256"][name])
    old_metadata = keyed(csv_rows(package_inputs["metadata_60s.csv"]), "id")
    native = keyed(csv_rows(package_inputs["evidence/native.csv"]), "id")
    fhm = keyed(csv_rows(package_inputs["evidence/fhm.csv"]), "id")
    extraction = strict_json(package_inputs["evidence/extraction_receipt.json"])
    if synthetic:
        require(extraction.get("synthetic_test_only") is True, "Synthetic extraction marker missing")
        require(len(native) <= 12 and len(fhm) <= 12, "Synthetic native/FHM evidence too large")
        for item_id in set(native) | set(fhm):
            ident(item_id, True)
    require(len(old_metadata) == old_audit["rows"] and (1 <= len(old_metadata) <= 12 if synthetic else len(old_metadata) == 1604), "Old admitted count mismatch")
    require(set(old_metadata) <= set(native), "Old admitted identity absent in native evidence")
    require(set(old_metadata) <= set(fhm), "Old admitted/FHM source hash identity mismatch")
    old_hashes = set()
    for item_id, old in old_metadata.items():
        if synthetic:
            ident(item_id, True)
        require(old["role"] == native[item_id]["role"] == "development", "Old admitted role drift")
        require(fhm[item_id]["role"] == "development", "Old source evidence role drift")
        old_hashes.add(hash_value(fhm[item_id]["source_audio_sha256"]))
        old_hashes.add(hash_value(native[item_id]["registered_raw_sha256"]))
        if native[item_id].get("physical_sha256"):
            old_hashes.add(hash_value(native[item_id]["physical_sha256"]))
    for row in rows:
        require(row["item_id"] not in native and row["mbid"] not in native, "Exact source identity overlap")
        require(row["acquisition_raw_sha256"] not in old_hashes, "Exact raw hash overlap")
    code_path = no_symlinks(Path(__file__).absolute())
    code_bytes = read_bytes(code_path)
    config = {"version": VERSION, "pins": pins, "source_id": SOURCE, "role": ROLE, "label": 0,
              "duration_seconds": 60, "crop_rule": "(actual_eof_frames - 60 * actual_sr) // 2",
              "speech_exclusions": sorted(SPEECH) if not synthetic else "synthetic flags",
              "missing_credit_exclusions": sorted(MISSING) if not synthetic else "synthetic missing credits",
              "synthetic_test_only": synthetic}
    draft = {"version": VERSION, "status": "draft", "synthetic_test_only": synthetic,
             "source_record_count": len(records), "eligible_count": len(rows), "excluded_count": len(excluded),
             "eligible_mbids_sha256": eligible_hash, "group_sizes": sizes, "native_rate_counts": rates,
             "identity_cardinalities": dimensions, "config": config, "config_sha256": seal(config),
             "implementation_sha256": sha(code_bytes), "input_inventory_sha256": seal(inventory),
             "role": ROLE, "classifier_admission": False, "audio_opened": False, "inference_performed": False,
             "classifier_fitted": False, "scoring_performed": False, "frozen": False,
             "current_raw_byte_audit": False, "measurement_acceptance": False,
             "old_v4_admitted_count": len(old_metadata), "exact_id_overlap_count": 0, "exact_recorded_raw_sha256_overlap_count": 0,
             "overlap_scope": "recorded acquisition identities versus pinned old v4 admitted metadata, native registered hashes and complete FHM source_audio_sha256 identity join; feature values unused",
             "perceptual_historical_pretraining_overlap": "unknown",
             "historical_hindustani_caveat": "Earlier HF Hindustani metadata_10s contains 100 rows; historical overlap unresolved, not established disjoint by this v4-only exact check",
             "future_human_only_endpoints": ["specificity", "false_positive_rate"],
             "standalone_BA_or_AUC_supported": False,
             "use_scope": "internal academic analysis only; no redistribution or agreement acceptance",
             "future_materializer_required": "rehash raw bytes, sequentially redecode to actual EOF, prove exact native crop and offset, independently audit; separate authorization required"}
    payloads = {"selection_draft.json": canonical(draft), "groups.json": canonical(groups),
                "input_inventory.json": canonical(inventory), "config.json": canonical(config)}
    for name, data, fields in (("metadata.csv", rows, None), ("exclusions.csv", excluded, ["mbid", "title", "reasons", "source_record_json"])):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(data)
        payloads[name] = stream.getvalue().encode()
    evidence = {"status": "draft_metadata_evidence_only", "synthetic_test_only": synthetic,
                "implementation_sha256": sha(code_bytes), "config_sha256": seal(config),
                "input_inventory_sha256": seal(inventory), "receipt_manifest_sha256": seal(receipt_manifest),
                "outputs_sha256": {name: sha(raw) for name, raw in payloads.items()}, "inputs_rechecked_before_publication": True}
    payloads["evidence_receipt.json"] = canonical(evidence)
    stage = Path(tempfile.mkdtemp(prefix=".saraga-draft-", dir=output.parent))
    try:
        for name, raw in payloads.items():
            with (stage / name).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        require(sha(read_bytes(code_path)) == sha(code_bytes), "Implementation changed before publication")
        for relative, expected in inventory.items():
            require(sha(read_bytes(root / relative)) == expected["sha256"], "Input changed before publication")
        require(sorted(p.name for p in receipt_dir.iterdir()) == names, "Receipt inventory changed before publication")
        no_symlinks(output)
        rename_exclusive(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return draft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; draft only; never freeze")
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
