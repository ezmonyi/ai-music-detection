#!/usr/bin/env python3
"""Resolve the single duplicated VocalSet ZIP basename without rewriting v1.

The v1 acquisition summary, annotation audit, 243 audio files and 243 receipts
are immutable inputs.  Both duplicate central-directory members are fully
retrieved into a scoped quarantine.  Selection is permitted only when their
full SHA-256 values are identical or exactly one decoded duration matches the
source annotation.  The whole archive MD5 remains unverified.
"""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import wave
import zipfile

import acquire_vocalset_breath as v1


AMBIGUOUS_NAME = "m9_caro_vibrato.wav"
EXPECTED_V1_CODE_SHA256 = "673d177446047a723f18e283621136f671219970f769333e555b1eb3d428e81c"
EXPECTED_V1_SUMMARY_SHA256 = "9ecafdb9870d028f58ff4fe170da501d435f09e39c19d837c2f1e142f2e91a0f"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(path):
    def reject(value):
        raise ValueError(f"Non-standard/nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def wav_metadata(raw):
    with wave.open(io.BytesIO(raw), "rb") as audio:
        sample_rate = audio.getframerate()
        frames = audio.getnframes()
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        if audio.getcomptype() != "NONE":
            raise ValueError("Compressed WAV is outside the acquisition contract")
        pcm = audio.readframes(frames)
        if len(pcm) != frames * channels * sample_width:
            raise ValueError("Truncated PCM payload")
    return {
        "sample_rate": sample_rate,
        "frames": frames,
        "channels": channels,
        "sample_width": sample_width,
        "duration_sec": frames / sample_rate,
    }


def choose_candidate(records, annotation_duration, tolerance=0.01):
    if len(records) < 2:
        raise ValueError("Ambiguity resolution requires at least two candidates")
    hashes = {record["audio_sha256"] for record in records}
    if len(hashes) == 1:
        return min(records, key=lambda record: record["archive_member"]), "byte_identical_full_sha256"
    matching = [record for record in records if abs(record["duration_sec"] - annotation_duration) <= tolerance]
    if len(matching) == 1:
        return matching[0], "unique_annotation_duration_match"
    raise ValueError(
        "Duplicate members differ and annotation duration does not select exactly one: "
        + json.dumps([
            {
                "archive_member": record["archive_member"],
                "audio_sha256": record["audio_sha256"],
                "duration_sec": record["duration_sec"],
                "duration_residual_sec": record["duration_sec"] - annotation_duration,
            }
            for record in records
        ], sort_keys=True)
    )


def atomic_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(f"Existing materialization differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)


def atomic_json(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    atomic_bytes(path, raw)


def candidate_slug(archive_member):
    parent = Path(archive_member).parent
    singer_dir = parent.parent.parent.name
    if not re.fullmatch(r"(?:male|female)\d+", singer_dir):
        raise ValueError(f"Unexpected candidate archive path: {archive_member}")
    return singer_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--acquisition-dir", type=Path, required=True)
    args = parser.parse_args()
    acquisition_dir = args.acquisition_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    v1_summary_path = acquisition_dir / "acquisition_summary.json"
    annotation_audit_path = acquisition_dir / "annotation_audit.json"
    v2_summary_path = acquisition_dir / "acquisition_summary_v2.json"
    resolution_path = acquisition_dir / "ambiguity_resolution_v2" / "resolution.json"
    if v2_summary_path.exists() or resolution_path.exists():
        raise FileExistsError("Refusing to overwrite an existing v2 resolution")

    actual_v1_code_sha = v1.digest(Path(v1.__file__).read_bytes())
    actual_v1_summary_sha = v1.digest(v1_summary_path.read_bytes())
    if actual_v1_code_sha != EXPECTED_V1_CODE_SHA256:
        raise ValueError(f"Frozen v1 code changed: {actual_v1_code_sha}")
    if actual_v1_summary_sha != EXPECTED_V1_SUMMARY_SHA256:
        raise ValueError(f"Frozen v1 summary changed: {actual_v1_summary_sha}")
    v1_summary = strict_json(v1_summary_path)
    expected_error = [{"filename": AMBIGUOUS_NAME, "error": "ZIP filename is absent or ambiguous", "matches": 2}]
    if not (
        v1_summary.get("status") == "complete_accounting"
        and v1_summary.get("selected") == 244
        and v1_summary.get("downloaded_verified") == 243
        and v1_summary.get("errors") == expected_error
        and len(v1_summary.get("manifest", [])) == 243
        and v1_summary.get("whole_archive_hash_verified") is False
    ):
        raise ValueError("Frozen v1 summary does not have the expected sole duplicate-basename failure")
    old_names = [row["filename"] for row in v1_summary["manifest"]]
    if len(set(old_names)) != 243 or AMBIGUOUS_NAME in old_names:
        raise ValueError("Frozen v1 manifest does not contain exactly 243 distinct resolved names")

    annotation_audit = strict_json(annotation_audit_path)
    annotation_rows = {row["filename"]: row for row in annotation_audit["labels"]}
    if set(old_names) | {AMBIGUOUS_NAME} != set(annotation_rows):
        raise ValueError("Annotation audit and v1 manifest do not form the expected 244-name set")
    annotation_row = annotation_rows[AMBIGUOUS_NAME]
    label_path = labels_dir / "labels" / "vocalset" / (Path(AMBIGUOUS_NAME).stem + ".breath.json")
    label_raw = label_path.read_bytes()
    label = json.loads(label_raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    annotation_duration = label.get("duration_sec")
    if not isinstance(annotation_duration, (int, float)) or isinstance(annotation_duration, bool) or not math.isfinite(annotation_duration) or annotation_duration <= 0:
        raise ValueError("Ambiguous label has invalid duration")
    if label.get("audio_file") != AMBIGUOUS_NAME or sha256(label_raw) != annotation_row["annotation_sha256"]:
        raise ValueError("Ambiguous source annotation identity/SHA mismatch")

    with zipfile.ZipFile(v1.RemoteZip()) as archive:
        candidates = [info for info in archive.infolist() if Path(info.filename).name == AMBIGUOUS_NAME]
    if len(candidates) != 2:
        raise ValueError(f"Expected exactly two central-directory candidates, found {len(candidates)}")

    records = []
    raw_by_member = {}
    for info in sorted(candidates, key=lambda item: item.filename):
        raw = v1.read_member(info)
        metadata = wav_metadata(raw)
        record = {
            "archive_member": info.filename,
            "header_offset": info.header_offset,
            "compression_type": info.compress_type,
            "compressed_bytes": info.compress_size,
            "member_bytes": info.file_size,
            "member_crc32": f"{info.CRC:08x}",
            "audio_sha256": sha256(raw),
            **metadata,
            "annotation_duration_sec": annotation_duration,
            "duration_residual_sec": metadata["duration_sec"] - annotation_duration,
        }
        slug = candidate_slug(info.filename)
        quarantine_path = acquisition_dir / "ambiguity_resolution_v2" / "quarantine" / slug / AMBIGUOUS_NAME
        candidate_receipt = acquisition_dir / "ambiguity_resolution_v2" / "quarantine" / slug / "receipt.json"
        atomic_bytes(quarantine_path, raw)
        record["quarantine_path"] = str(quarantine_path.resolve())
        atomic_json(candidate_receipt, record)
        records.append(record)
        raw_by_member[info.filename] = raw

    selected, method = choose_candidate(records, float(annotation_duration))
    canonical_path = acquisition_dir / "audio" / AMBIGUOUS_NAME
    atomic_bytes(canonical_path, raw_by_member[selected["archive_member"]])
    final_row = {
        **annotation_row,
        "audio_path": str(canonical_path.resolve()),
        "audio_sha256": selected["audio_sha256"],
        "archive_member": selected["archive_member"],
        "member_crc32": selected["member_crc32"],
        "member_bytes": selected["member_bytes"],
        "sample_rate": selected["sample_rate"],
        "frames": selected["frames"],
        "channels": selected["channels"],
        "duration_verified": selected["duration_sec"],
        "resolution_method": method,
        "resolution_record_path": str(resolution_path.resolve()),
    }
    receipt_path = acquisition_dir / "receipts_v2" / (AMBIGUOUS_NAME + ".json")
    final_row["receipt_path"] = str(receipt_path.resolve())
    atomic_json(receipt_path, final_row)
    resolution = {
        "schema": "vocalset-breath-duplicate-resolution-v2",
        "status": "resolved",
        "ambiguous_filename": AMBIGUOUS_NAME,
        "selection_method": method,
        "selected_archive_member": selected["archive_member"],
        "selected_audio_sha256": selected["audio_sha256"],
        "annotation_path": str(label_path.resolve()),
        "annotation_sha256": sha256(label_raw),
        "annotation_duration_sec": annotation_duration,
        "candidates": records,
        "canonical_audio_path": str(canonical_path.resolve()),
        "canonical_receipt_path": str(receipt_path.resolve()),
        "frozen_v1_code_path": str(Path(v1.__file__).resolve()),
        "frozen_v1_code_sha256": actual_v1_code_sha,
        "frozen_v1_summary_path": str(v1_summary_path.resolve()),
        "frozen_v1_summary_sha256": actual_v1_summary_sha,
        "annotation_audit_path": str(annotation_audit_path.resolve()),
        "annotation_audit_sha256": v1.digest(annotation_audit_path.read_bytes()),
        "whole_archive_hash_verified": False,
        "archive_reported_md5": v1.MD5,
        "whole_archive_hash_note": "Both duplicate members were verified individually; selected byte ranges do not verify the full archive MD5.",
        "script_sha256": v1.digest(Path(__file__).read_bytes()),
        "runtime": {"python": sys.version.split()[0]},
    }
    atomic_json(resolution_path, resolution)

    manifest = sorted([*v1_summary["manifest"], final_row], key=lambda row: row["filename"])
    v2_summary = {
        "schema": "vocalset-breath-acquisition-v2",
        "status": "complete_accounting",
        "selected": 244,
        "downloaded_verified": 244,
        "errors": [],
        "manifest": manifest,
        "whole_archive_hash_verified": False,
        "annotation_audit_sha256": v1.digest(annotation_audit_path.read_bytes()),
        "frozen_v1_summary_sha256": actual_v1_summary_sha,
        "frozen_v1_code_sha256": actual_v1_code_sha,
        "ambiguity_resolution_path": str(resolution_path.resolve()),
        "ambiguity_resolution_sha256": v1.digest(resolution_path.read_bytes()),
    }
    atomic_json(v2_summary_path, v2_summary)
    print(json.dumps({
        "status": "complete_accounting",
        "selected": 244,
        "downloaded_verified": 244,
        "selection_method": method,
        "selected_archive_member": selected["archive_member"],
        "summary": str(v2_summary_path),
    }))


if __name__ == "__main__":
    main()
