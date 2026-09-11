#!/usr/bin/env python3
"""Independently audit completed Saraga physical acquisition bytes and receipts.

This standard-library audit does not import the materializer or decode audio.
It independently rehashes every raw MP3 and checks the materializer's decoded
PCM accounting receipts. Passing is neither a second-decoder result nor music,
human-source, cohort, annotation, or classifier admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat


VERSION = "saraga_hindustani_physical_acceptance_audit_v1"
MATERIALIZER_VERSION = "saraga_hindustani_physical_materialization_v1"
COUNT = 108
ARCHIVE_BYTES = 4_109_172_493
ARCHIVE_MD5 = "ea9ed2885ea37a1b10e42f60cf299702"
ARCHIVE_SHA256 = "cd3abd54288efd95e85ae3bf8b13e770e0914bfac12fa07dcba04c6ddc641fb7"
RECON_SHA256 = "d92da602fc6e5dc74469c3fdf1c39d61b4086a4be3a7c4c9ed7f565de6aff2a3"
AUDIT_SHA256 = "1b22ef5a089996af1d45be775b0ef6fbb7d7fc67072b60d1a576ee05f52d3ec1"
MATERIALIZER_SHA256 = "d75eb5a9d6f3d9ca76e770b6a6db686c2b7e690a6ec7da4e45c883dc27fb6f2f"
BLOCK_FRAMES = 65536
PCM_ENCODING = "IEEE754 float64 little-endian; C order [frame, channel]; no header"
MBID_RE = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
MEASUREMENT_KEYS = {
    "sample_rate", "channels", "decoder_format", "decoder_subtype", "header_frames",
    "actual_frames", "header_minus_actual_frames", "header_matches_actual_eof",
    "actual_duration_seconds", "duration_at_least_60_seconds",
    "read_calls_including_empty_eof", "real_empty_read_observed", "read_block_frames",
    "pcm_sha256", "pcm_encoding", "sample_count", "finite_sample_count",
    "nonfinite_sample_count", "sample_min", "sample_max", "peak_absolute",
    "rms_all_samples", "raw_hashes_before_decode", "raw_hashes_after_decode",
}
CONTRACT_KEYS = {
    "version", "expected_recordings", "archive_path", "archive_hashes", "inputs",
    "runtime", "decode_block_frames", "pcm_encoding", "cohort_selected",
    "classifier_admission", "items",
}
SUMMARY_KEYS = {
    "status", "version", "contract_sha256", "record_count",
    "source_archive_hashes_before", "source_archive_hashes_after",
    "input_code_contract_unchanged", "runtime_unchanged",
    "expected_file_inventory_verified", "classifier_admission", "cohort_selected",
    "annotation_alignment_verified", "human_music_usability_claim",
    "speech_title_review_flags", "missing_performer_credits",
    "duration_at_least_60_seconds_count", "header_frame_mismatch_count", "records",
}


@dataclass(frozen=True)
class Pins:
    count: int
    archive_bytes: int
    archive_md5: str
    archive_sha256: str
    reconciliation_sha256: str
    archive_audit_sha256: str
    materializer_sha256: str


PRODUCTION_PINS = Pins(COUNT, ARCHIVE_BYTES, ARCHIVE_MD5, ARCHIVE_SHA256,
                       RECON_SHA256, AUDIT_SHA256, MATERIALIZER_SHA256)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def digest(value, length=64):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value)


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def object_sha256(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def regular_file(path):
    path = Path(path)
    require(not path.is_symlink(), f"Symlink forbidden: {path}")
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
            f"Expected regular, singly linked file: {path}")
    return path


def safe_path(path):
    """Return an absolute path only when no existing component is a symlink."""
    path = Path(os.path.abspath(path))
    for component in [*reversed(path.parents), path]:
        require(not component.is_symlink(), f"Symlink path component forbidden: {component}")
    return path


def stable_hashes(path, both=False):
    """Hash a regular file and reject changes during the read."""
    path = regular_file(path)
    before = path.stat()
    md5 = hashlib.md5() if both else None
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            size += len(block)
            sha.update(block)
            if md5 is not None:
                md5.update(block)
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(identity(before) == identity(after), f"File changed while hashing: {path}")
    result = {"bytes": size, "sha256": sha.hexdigest()}
    if md5 is not None:
        result["md5"] = md5.hexdigest()
    return result


def read_json(path, require_canonical=False):
    path = regular_file(path)

    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def no_constant(value):
        raise ValueError(f"Nonfinite JSON constant in {path}: {value}")

    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=no_duplicates, parse_constant=no_constant)
    if require_canonical:
        require(raw == canonical(value), f"Noncanonical JSON: {path}")
    return value


def read_canonical_json(path):
    return read_json(path, require_canonical=True)


def unique(rows, key, label):
    require(isinstance(rows, list), f"Invalid {label} list")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get(key), str),
                f"Invalid {label} row")
        value = row[key]
        require(value not in result, f"Duplicate {label} {key}: {value}")
        result[value] = row
    return result


def archive_hashes(pins):
    return {"bytes": pins.archive_bytes, "md5": pins.archive_md5,
            "sha256": pins.archive_sha256}


def build_expected_items(reconciliation, archive_audit, pins):
    require(reconciliation.get("status") ==
            "passed_archive_catalog_checksum_path_reconciliation_v2_not_audio_admission"
            and reconciliation.get("failure_reasons") == []
            and reconciliation.get("physical_audio_decoded") is False
            and reconciliation.get("classifier_admission") is False
            and reconciliation.get("cohort_selected") is False,
            "Invalid reconciliation acquisition boundary")
    require(archive_audit.get("status") == "passed_archive_integrity_not_audio_admission"
            and archive_audit.get("source_record") == 4301737
            and archive_audit.get("classifier_admission") is False
            and archive_audit.get("extracted_audio_files") == 0
            and archive_audit.get("physically_decoded_audio_files") == 0,
            "Invalid archive-audit acquisition boundary")
    expected_archive = archive_hashes(pins)
    require(archive_audit.get("archive_bytes") == pins.archive_bytes
            and archive_audit.get("archive_hashes") == {k: expected_archive[k]
                                                         for k in ("md5", "sha256")}
            and reconciliation.get("archive_integrity_evidence", {}).get("archive_bytes") ==
            pins.archive_bytes
            and reconciliation.get("archive_integrity_evidence", {}).get("archive_hashes") ==
            {k: expected_archive[k] for k in ("md5", "sha256")},
            "Source archive identity disagreement")
    records = unique(archive_audit.get("records"), "path", "archive record")
    require(len(records) == archive_audit.get("archive_members_verified"),
            "Archive inventory count mismatch")
    tracks = unique(reconciliation.get("tracks"), "mbid", "track")
    matches = unique(reconciliation.get("matched_audio"), "mbid", "matched audio")
    require(len(tracks) == len(matches) == pins.count and tracks.keys() == matches.keys(),
            "Reconciliation does not contain the exact expected MBIDs")
    items = []
    raw_md5s, raw_shas, archive_paths = set(), set(), set()
    for mbid, track in sorted(tracks.items()):
        require(MBID_RE.fullmatch(mbid), f"Invalid recording MBID: {mbid}")
        match = matches[mbid]
        path = track.get("archive_audio_path")
        require(isinstance(path, str) and path in records and path == match.get("archive_audio_path")
                and path not in archive_paths, f"Invalid archive mapping for {mbid}")
        record = records[path]
        require(integer(record.get("bytes")) and digest(record.get("md5"), 32)
                and digest(record.get("sha256")), f"Invalid archive hashes for {mbid}")
        require(track.get("path_resolution") == match.get("path_resolution") ==
                "exact_unique_mp3_md5"
                and match.get("catalog_md5") == match.get("archive_md5") == record["md5"],
                f"Reconciliation checksum join failed for {mbid}")
        require(record["md5"] not in raw_md5s and record["sha256"] not in raw_shas,
                "Duplicate reconciled raw checksum")
        require(type(track.get("speech_title_review_flag")) is bool
                and isinstance(track.get("performer_credits"), list),
                f"Invalid review metadata for {mbid}")
        archive_paths.add(path)
        raw_md5s.add(record["md5"])
        raw_shas.add(record["sha256"])
        items.append({"mbid": mbid, "raw_path": f"raw/{mbid}.mp3",
                      "archive_member": record, "reconciled_metadata": track,
                      "checksum_match": match})
    return items


def validate_contract(contract, contract_sha, items, paths, pins):
    require(set(contract) == CONTRACT_KEYS, "Invalid contract schema")
    require(contract["version"] == MATERIALIZER_VERSION
            and contract["expected_recordings"] == pins.count
            and contract["archive_hashes"] == archive_hashes(pins)
            and contract["decode_block_frames"] == BLOCK_FRAMES
            and contract["pcm_encoding"] == PCM_ENCODING
            and contract["cohort_selected"] is False
            and contract["classifier_admission"] is False
            and contract["items"] == items, "Contract content mismatch")
    require(digest(contract_sha), "Invalid contract digest")
    expected_inputs = {
        "code": (paths["materializer"], pins.materializer_sha256),
        "reconciliation": (paths["reconciliation"], pins.reconciliation_sha256),
        "archive_audit": (paths["archive_audit"], pins.archive_audit_sha256),
    }
    require(set(contract["inputs"]) == set(expected_inputs), "Invalid contract input set")
    for name, (path, expected_sha) in expected_inputs.items():
        row = contract["inputs"][name]
        current = stable_hashes(path, both=True)
        require(set(row) == {"path", "bytes", "md5", "sha256"}
                and Path(row["path"]).resolve() == path.resolve()
                and {key: row[key] for key in ("bytes", "md5", "sha256")} == current
                and current["sha256"] == expected_sha,
                f"Changed or invalid pinned contract input: {name}")
    runtime = contract["runtime"]
    require(isinstance(runtime, dict)
            and set(runtime) == {"python", "python_executable", "platform", "numpy",
                                 "soundfile", "libsndfile", "numpy_module_sha256",
                                 "soundfile_module_sha256", "decoder_claim"}
            and all(isinstance(runtime[k], str) and runtime[k] for k in
                    ("python", "python_executable", "platform", "numpy", "soundfile",
                     "libsndfile", "decoder_claim"))
            and digest(runtime["numpy_module_sha256"])
            and digest(runtime["soundfile_module_sha256"]), "Invalid decoder runtime descriptor")


def validate_measurement(measurement, expected_raw):
    require(isinstance(measurement, dict) and set(measurement) == MEASUREMENT_KEYS,
            "Invalid measurement schema")
    for name in ("sample_rate", "channels", "actual_frames", "sample_count",
                 "finite_sample_count"):
        require(integer(measurement.get(name), 1), f"Invalid measurement {name}")
    require(integer(measurement.get("header_frames"))
            and integer(measurement.get("read_calls_including_empty_eof"), 2)
            and measurement["read_calls_including_empty_eof"] >=
            math.ceil(measurement["actual_frames"] / BLOCK_FRAMES) + 1
            and measurement.get("real_empty_read_observed") is True
            and measurement.get("read_block_frames") == BLOCK_FRAMES
            and measurement.get("pcm_encoding") == PCM_ENCODING
            and digest(measurement.get("pcm_sha256"))
            and isinstance(measurement.get("decoder_format"), str)
            and bool(measurement["decoder_format"])
            and isinstance(measurement.get("decoder_subtype"), str),
            "Invalid EOF/PCM descriptor")
    require(measurement["channels"] <= 256
            and measurement["sample_count"] == measurement["finite_sample_count"] ==
            measurement["actual_frames"] * measurement["channels"]
            and type(measurement.get("nonfinite_sample_count")) is int
            and measurement["nonfinite_sample_count"] == 0
            and type(measurement.get("header_minus_actual_frames")) is int
            and measurement["header_minus_actual_frames"] ==
            measurement["header_frames"] - measurement["actual_frames"]
            and measurement.get("header_matches_actual_eof") is
            (measurement["header_frames"] == measurement["actual_frames"])
            and type(measurement.get("actual_duration_seconds")) in (int, float)
            and math.isfinite(measurement["actual_duration_seconds"])
            and measurement["actual_duration_seconds"] ==
            measurement["actual_frames"] / measurement["sample_rate"]
            and measurement.get("duration_at_least_60_seconds") is
            (measurement["actual_frames"] >= 60 * measurement["sample_rate"]),
            "Inconsistent frame, finite-sample, or duration accounting")
    for name in ("sample_min", "sample_max", "peak_absolute", "rms_all_samples"):
        require(type(measurement.get(name)) in (int, float)
                and math.isfinite(measurement[name]), f"Nonfinite amplitude statistic: {name}")
    require(measurement["sample_min"] <= measurement["sample_max"]
            and 0 <= measurement["rms_all_samples"] <= measurement["peak_absolute"]
            and measurement["peak_absolute"] ==
            max(abs(measurement["sample_min"]), abs(measurement["sample_max"]))
            and measurement["raw_hashes_before_decode"] == expected_raw
            and measurement["raw_hashes_after_decode"] == expected_raw,
            "Inconsistent amplitude statistics or raw decode binding")


def validate_record(record, item, contract_sha, actual_raw):
    require(isinstance(record, dict)
            and set(record) == {"status", "contract_sha256", "item", "measurement",
                                "classifier_admission", "cohort_selected"},
            "Invalid receipt record schema")
    require(record["status"] == "passed_physical_acquisition_not_audio_admission"
            and record["contract_sha256"] == contract_sha and record["item"] == item
            and record["classifier_admission"] is False
            and record["cohort_selected"] is False, "Receipt identity/admission mismatch")
    expected_raw = {key: item["archive_member"][key] for key in ("bytes", "md5", "sha256")}
    require(actual_raw == expected_raw, f"Raw byte/hash mismatch for {item['mbid']}")
    validate_measurement(record["measurement"], expected_raw)


def inventory(output, items):
    require(output.is_dir() and not output.is_symlink(), "Physical output is not a real directory")
    require(not (output / "writer.lock").exists(), "Physical materialization is busy or stale-locked")
    require({p.name for p in output.iterdir()} == {"contract.json", "summary.json", "raw", "receipts"},
            "Incomplete or unexpected output-root inventory")
    for name in ("raw", "receipts"):
        path = output / name
        require(path.is_dir() and not path.is_symlink(), f"Invalid {name} directory")
    mbids = {item["mbid"] for item in items}
    expected_raw = {mbid + ".mp3" for mbid in mbids}
    expected_receipts = {mbid + ".json" for mbid in mbids}
    require({p.name for p in (output / "raw").iterdir()} == expected_raw,
            "Missing, duplicate, or extra raw file")
    require({p.name for p in (output / "receipts").iterdir()} == expected_receipts,
            "Missing, duplicate, or extra receipt")


def bind(paths, raw_paths):
    return {
        "controls": {name: stable_hashes(path) for name, path in paths.items()},
        "raw": {path.name: stable_hashes(path, both=True) for path in raw_paths},
    }


def verify_bound_snapshot(bound, physical, paths, raw_paths):
    require(not (physical / "writer.lock").exists(),
            "Writer lock appeared during audit")
    for name, expected in bound["controls"].items():
        require(stable_hashes(paths[name]) == expected, f"Evidence changed during audit: {name}")
    for path in raw_paths:
        require(stable_hashes(path, both=True) == bound["raw"][path.name],
                f"Raw file changed during audit: {path.name}")


def _audit(physical, reconciliation_path, archive_audit_path, materializer_path, pins):
    physical = safe_path(physical)
    paths = {"physical": physical, "reconciliation": safe_path(reconciliation_path),
             "archive_audit": safe_path(archive_audit_path),
             "materializer": safe_path(materializer_path),
             "contract": physical / "contract.json", "summary": physical / "summary.json",
             "auditor": Path(__file__).resolve()}
    require(not (physical / "writer.lock").exists(), "Physical materialization is busy or stale-locked")
    expected_control_hashes = {"reconciliation": pins.reconciliation_sha256,
                               "archive_audit": pins.archive_audit_sha256,
                               "materializer": pins.materializer_sha256}
    for name, expected in expected_control_hashes.items():
        require(stable_hashes(paths[name])["sha256"] == expected, f"Pinned {name} hash mismatch")
    # These two fixed-hash source receipts are historically pretty-printed JSON.
    # Their exact pinned bytes provide serialization identity; parsing still rejects
    # duplicate keys and nonfinite constants. Newly materialized evidence below is
    # required to use the materializer's canonical JSON representation.
    reconciliation = read_json(paths["reconciliation"])
    archive_audit = read_json(paths["archive_audit"])
    items = build_expected_items(reconciliation, archive_audit, pins)
    inventory(physical, items)
    raw_paths = [physical / item["raw_path"] for item in items]
    receipt_paths = [physical / "receipts" / (item["mbid"] + ".json") for item in items]
    bind_paths = {name: paths[name] for name in
                  ("reconciliation", "archive_audit", "materializer", "contract", "summary",
                   "auditor")}
    bind_paths.update({"receipt:" + path.name: path for path in receipt_paths})
    bound = bind(bind_paths, raw_paths)
    contract = read_canonical_json(paths["contract"])
    summary = read_canonical_json(paths["summary"])
    contract_sha = bound["controls"]["contract"]["sha256"]
    require(contract_sha == object_sha256(contract), "Contract canonical seal mismatch")
    validate_contract(contract, contract_sha, items, paths, pins)
    require(set(summary) == SUMMARY_KEYS
            and summary["status"] == "passed_all_physical_acquisition_not_audio_admission"
            and summary["version"] == MATERIALIZER_VERSION
            and summary["contract_sha256"] == contract_sha
            and summary["record_count"] == pins.count
            and summary["source_archive_hashes_before"] == archive_hashes(pins)
            and summary["source_archive_hashes_after"] == archive_hashes(pins)
            and summary["input_code_contract_unchanged"] is True
            and summary["runtime_unchanged"] is True
            and summary["expected_file_inventory_verified"] is True
            and summary["classifier_admission"] is False
            and summary["cohort_selected"] is False
            and summary["annotation_alignment_verified"] is False
            and summary["human_music_usability_claim"] is False
            and isinstance(summary["records"], list)
            and len(summary["records"]) == pins.count, "Invalid final summary")
    # Records contain their MBID inside item, so validate identities explicitly without guessed keys.
    by_summary_mbid = {}
    for record in summary["records"]:
        require(isinstance(record, dict) and isinstance(record.get("item"), dict),
                "Invalid summary record")
        mbid = record["item"].get("mbid")
        require(isinstance(mbid, str) and mbid not in by_summary_mbid,
                f"Duplicate/invalid summary MBID: {mbid}")
        by_summary_mbid[mbid] = record
    require(set(by_summary_mbid) == {item["mbid"] for item in items},
            "Summary MBID set mismatch")
    records = []
    receipt_manifest = []
    raw_manifest = []
    for item, receipt_path, raw_path in zip(items, receipt_paths, raw_paths):
        receipt = read_canonical_json(receipt_path)
        require(set(receipt) == {"record", "record_sha256"}
                and receipt["record_sha256"] == object_sha256(receipt["record"]),
                f"Receipt canonical record seal mismatch: {item['mbid']}")
        actual_raw = bound["raw"][raw_path.name]
        validate_record(receipt["record"], item, contract_sha, actual_raw)
        require(by_summary_mbid[item["mbid"]] == receipt["record"],
                f"Summary/receipt disagreement: {item['mbid']}")
        records.append(receipt["record"])
        receipt_manifest.append({"mbid": item["mbid"],
                                 "sha256": bound["controls"]["receipt:" + receipt_path.name]["sha256"]})
        raw_manifest.append({"mbid": item["mbid"], **actual_raw})
    duration60 = sum(r["measurement"]["duration_at_least_60_seconds"] for r in records)
    header_mismatches = sum(not r["measurement"]["header_matches_actual_eof"] for r in records)
    speech_flags = sum(item["reconciled_metadata"]["speech_title_review_flag"] for item in items)
    missing_performers = sum(not item["reconciled_metadata"]["performer_credits"] for item in items)
    speech_mbids = {item["mbid"] for item in items
                    if item["reconciled_metadata"]["speech_title_review_flag"]}
    missing_performer_mbids = {item["mbid"] for item in items
                               if not item["reconciled_metadata"]["performer_credits"]}
    require(summary["duration_at_least_60_seconds_count"] == duration60
            and summary["header_frame_mismatch_count"] == header_mismatches
            and summary["speech_title_review_flags"] == speech_flags
            and summary["missing_performer_credits"] == missing_performers,
            "Summary aggregate mismatch")
    inventory(physical, items)
    verify_bound_snapshot(bound, physical, bind_paths, raw_paths)
    inventory(physical, items)
    durations = [r["measurement"]["actual_duration_seconds"] for r in records]
    return {
        "status": "passed_independent_byte_accounting_audit_not_second_decode_not_admission",
        "version": VERSION, "utc": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": bound["controls"]["auditor"]["sha256"],
        "contract_sha256": contract_sha,
        "summary_sha256": bound["controls"]["summary"]["sha256"],
        "reconciliation_sha256": pins.reconciliation_sha256,
        "archive_audit_sha256": pins.archive_audit_sha256,
        "materializer_sha256": pins.materializer_sha256,
        "source_archive_declared_hashes": archive_hashes(pins),
        "source_archive_rehashed_by_this_audit": False,
        "record_count": len(records), "raw_file_count": len(raw_paths),
        "receipt_count": len(receipt_paths),
        "raw_total_bytes": sum(row["bytes"] for row in raw_manifest),
        "raw_manifest_sha256": object_sha256(raw_manifest),
        "receipt_manifest_sha256": object_sha256(receipt_manifest),
        "all_raw_md5_sha256_rehashed_twice_and_matched": True,
        "all_receipt_record_seals_verified": True,
        "all_samples_accounted_finite_in_receipts": True,
        "duration_at_least_60_seconds_count": duration60,
        "duration_total_seconds": math.fsum(durations),
        "duration_min_seconds": min(durations), "duration_max_seconds": max(durations),
        "header_frame_mismatch_count": header_mismatches,
        "native_format_counts": dict(sorted(Counter(
            f"{r['measurement']['sample_rate']}Hz/{r['measurement']['channels']}ch/"
            f"{r['measurement']['decoder_format']}/{r['measurement']['decoder_subtype']}"
            for r in records).items())),
        "speech_title_review_flags": speech_flags,
        "missing_performer_credits": missing_performers,
        "speech_and_missing_performer_overlap_count":
            len(speech_mbids & missing_performer_mbids),
        "neither_speech_nor_missing_performer_review_flag_count":
            pins.count - len(speech_mbids | missing_performer_mbids),
        "second_full_decode_performed": False,
        "decode_scope": "Decoded PCM receipts and accounting checked; no second audio decode performed.",
        "classifier_admission": False, "cohort_selected": False,
        "annotation_alignment_verified": False, "human_music_usability_claim": False,
        "records": [{"mbid": item["mbid"], "raw_path": item["raw_path"],
                     **bound["raw"][Path(item["raw_path"]).name],
                     "actual_duration_seconds": record["measurement"]["actual_duration_seconds"],
                     "duration_at_least_60_seconds": record["measurement"]["duration_at_least_60_seconds"],
                     "header_matches_actual_eof": record["measurement"]["header_matches_actual_eof"],
                     "nonfinite_sample_count": record["measurement"]["nonfinite_sample_count"],
                     "pcm_sha256": record["measurement"]["pcm_sha256"]}
                    for item, record in zip(items, records)],
    }


def audit(physical, reconciliation, archive_audit, materializer):
    """Run only the fixed production audit; tests exercise _audit with fixtures."""
    return _audit(physical, reconciliation, archive_audit, materializer, PRODUCTION_PINS)


def write_exclusive(path, value):
    path = Path(path)
    with path.open("xb") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                sort_keys=True).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-dir", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--archive-audit", type=Path, required=True)
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output is not None:
            physical = args.physical_dir.resolve()
            report = args.output.resolve()
            require(report != physical and physical not in report.parents,
                    "Audit report must be outside the immutable physical output")
        result = audit(args.physical_dir, args.reconciliation, args.archive_audit,
                       args.materializer)
        if args.output is not None:
            write_exclusive(args.output, result)
    except Exception as exc:
        print(json.dumps({"status": "failed_independent_audit", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({key: value for key, value in result.items() if key != "records"},
                     sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
