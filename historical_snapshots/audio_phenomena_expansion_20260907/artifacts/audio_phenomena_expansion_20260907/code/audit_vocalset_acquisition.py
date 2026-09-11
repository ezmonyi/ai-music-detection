#!/usr/bin/env python3
"""Independent, read-only audit of the completed VocalSet breath acquisition.

This script does not access the network and does not modify source annotations,
audio, receipts, or the acquisition summary.  It writes the requested audit only
after the acquisition reports complete 244/244 accounting without errors.
"""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import wave
import zlib


EVENT_FIELDS = (
    "breath_events",
    "silent_breaths",
    "uncertain",
    "hard_negatives",
    "exhales",
)
SINGER_PATTERN = re.compile(r"^([fm]\d+)_")
EXPECTED = {
    "selected": 244,
    "singers": 20,
    "named_reviewer_primary_clips": 114,
    "primary_raw_high_medium_events": 146,
    "primary_merged_high_medium_events": 144,
    "primary_zero_event_clips": 38,
}


class AuditPrerequisiteError(RuntimeError):
    """Raised before output when the acquisition is absent or incomplete."""


def _reject_json_constant(value):
    raise ValueError(f"Non-standard/nonfinite JSON constant: {value}")


def load_json(path):
    return json.loads(Path(path).read_text(), parse_constant=_reject_json_constant)


def sha256_file(path, chunk_bytes=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_length_crc32(path, chunk_bytes=1024 * 1024):
    length = 0
    checksum = 0
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            length += len(chunk)
            checksum = zlib.crc32(chunk, checksum)
    return length, f"{checksum & 0xffffffff:08x}"


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def wav_metadata(path):
    with wave.open(str(path), "rb") as audio:
        if audio.getcomptype() != "NONE":
            raise ValueError(f"Compressed WAV is outside the acquisition contract: {path}")
        sample_rate = audio.getframerate()
        frames = audio.getnframes()
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        decoded_bytes = 0
        while True:
            raw = audio.readframes(65536)
            if not raw:
                break
            decoded_bytes += len(raw)
    expected_decoded_bytes = frames * channels * sample_width
    if decoded_bytes != expected_decoded_bytes:
        raise ValueError(
            f"Truncated/oversized PCM payload: decoded={decoded_bytes}, expected={expected_decoded_bytes}"
        )
    return {
        "sample_rate": sample_rate,
        "frames": frames,
        "channels": channels,
        "sample_width": sample_width,
        "decoded_pcm_bytes": decoded_bytes,
        "duration_sec": frames / sample_rate,
    }


def _error(errors, filename, check, detail):
    errors.append({"filename": filename, "check": check, "detail": str(detail)})


def _expect_equal(errors, filename, check, actual, expected):
    if actual != expected:
        _error(errors, filename, check, f"observed={actual!r}, expected={expected!r}")


def require_completed_acquisition(acquisition_dir, expected_selected=244, summary_name="acquisition_summary.json"):
    path = Path(summary_name)
    if not path.is_absolute():
        path = Path(acquisition_dir) / path
    if not path.is_file():
        raise AuditPrerequisiteError(f"Missing completed acquisition summary: {path}")
    summary = load_json(path)
    selected = summary.get("selected")
    downloaded = summary.get("downloaded_verified")
    errors = summary.get("errors")
    manifest = summary.get("manifest")
    if (
        summary.get("status") != "complete_accounting"
        or selected != expected_selected
        or downloaded != expected_selected
        or errors != []
        or not isinstance(manifest, list)
        or len(manifest) != expected_selected
    ):
        raise AuditPrerequisiteError(
            "Acquisition is not complete and clean: "
            f"status={summary.get('status')!r}, selected={selected!r}, "
            f"downloaded_verified={downloaded!r}, errors={len(errors) if isinstance(errors, list) else 'invalid'}, "
            f"manifest={len(manifest) if isinstance(manifest, list) else 'invalid'}"
        )
    return path, summary


def audit_acquisition(acquisition_dir, labels_dir, expected=None, summary_name="acquisition_summary.json"):
    acquisition_dir = Path(acquisition_dir).resolve()
    labels_dir = Path(labels_dir).resolve()
    expected = dict(EXPECTED if expected is None else expected)
    summary_path, acquisition = require_completed_acquisition(
        acquisition_dir, expected["selected"], summary_name=summary_name
    )
    annotation_audit_path = acquisition_dir / "annotation_audit.json"
    if not annotation_audit_path.is_file():
        raise AuditPrerequisiteError(f"Missing annotation audit: {annotation_audit_path}")
    annotation_audit = load_json(annotation_audit_path)

    errors = []
    manifest = acquisition["manifest"]
    names = [row.get("filename") for row in manifest]
    duplicate_names = sorted(name for name, count in collections.Counter(names).items() if count > 1)
    if duplicate_names:
        _error(errors, "<manifest>", "unique_filenames", duplicate_names)
    _expect_equal(errors, "<manifest>", "manifest_name_count", len(set(names)), expected["selected"])

    annotation_rows = annotation_audit.get("labels")
    if not isinstance(annotation_rows, list):
        annotation_rows = []
        _error(errors, "<annotation_audit>", "labels_list", "missing or invalid")
    annotation_by_name = {row.get("filename"): row for row in annotation_rows}
    if len(annotation_by_name) != len(annotation_rows):
        _error(errors, "<annotation_audit>", "unique_filenames", "duplicate annotation audit rows")
    _expect_equal(errors, "<annotation_audit>", "selected_labels", annotation_audit.get("selected_labels"), expected["selected"])
    _expect_equal(errors, "<annotation_audit>", "label_row_count", len(annotation_rows), expected["selected"])
    _expect_equal(errors, "<annotation_audit>", "name_set", set(annotation_by_name), set(names))
    _expect_equal(
        errors,
        "<acquisition_summary>",
        "annotation_audit_sha256",
        acquisition.get("annotation_audit_sha256"),
        sha256_file(annotation_audit_path),
    )

    singer_counts = collections.Counter()
    empty_reviewer_material = collections.Counter()
    empty_reviewer_singer = collections.Counter()
    reviewer_strata = collections.Counter()
    confidence_counts = collections.Counter()
    primary_raw = 0
    primary_merged = 0
    primary_zero = 0
    primary_clips = 0
    overlapping_raw_primary_clips = 0
    merged_overlap_violations = 0
    archive_members = []
    total_member_bytes = 0

    for row in manifest:
        name = row.get("filename")
        if not isinstance(name, str):
            _error(errors, "<manifest>", "filename_type", repr(name))
            continue
        match = SINGER_PATTERN.match(name)
        if not match:
            _error(errors, name, "singer_prefix", "expected ^([fm]\\d+)_")
            singer = None
        else:
            singer = match.group(1)
            singer_counts[singer] += 1
            _expect_equal(errors, name, "manifest_singer", row.get("singer"), singer)

        expected_label_path = labels_dir / "labels" / "vocalset" / (Path(name).stem + ".breath.json")
        if not expected_label_path.is_file():
            _error(errors, name, "source_annotation_exists", expected_label_path)
            continue
        try:
            label = load_json(expected_label_path)
        except Exception as error:
            _error(errors, name, "source_annotation_json", repr(error))
            continue
        annotation_sha = sha256_file(expected_label_path)
        annotation_row = annotation_by_name.get(name)
        if annotation_row is None:
            _error(errors, name, "annotation_audit_row", "missing")
        else:
            _expect_equal(errors, name, "annotation_audit_sha", annotation_row.get("annotation_sha256"), annotation_sha)
            _expect_equal(errors, name, "annotation_audit_singer", annotation_row.get("singer"), singer)
            _expect_equal(errors, name, "annotation_audit_path", annotation_row.get("annotation_path"), str(expected_label_path.resolve()))
        _expect_equal(errors, name, "manifest_annotation_sha", row.get("annotation_sha256"), annotation_sha)
        _expect_equal(errors, name, "manifest_annotation_path", row.get("annotation_path"), str(expected_label_path.resolve()))
        _expect_equal(errors, name, "label_audio_file", label.get("audio_file"), name)

        duration = label.get("duration_sec")
        if not finite_number(duration) or duration <= 0:
            _error(errors, name, "annotation_duration", repr(duration))
            duration_valid = False
        else:
            duration = float(duration)
            duration_valid = True
            _expect_equal(errors, name, "manifest_annotation_duration", row.get("duration_sec"), label.get("duration_sec"))

        event_intervals = {}
        for field in EVENT_FIELDS:
            events = label.get(field)
            if not isinstance(events, list):
                _error(errors, name, f"{field}_type", "expected list")
                events = []
            valid_intervals = []
            for index, event in enumerate(events):
                if not isinstance(event, dict):
                    _error(errors, name, f"{field}[{index}]", "expected object")
                    continue
                start, end = event.get("start_sec"), event.get("end_sec")
                confidence_counts[f"{field}:{event.get('confidence', 'missing')}"] += 1
                if not (
                    duration_valid
                    and finite_number(start)
                    and finite_number(end)
                    and 0 <= start < end <= duration + 0.001
                ):
                    _error(errors, name, f"{field}[{index}]_bounds", f"start={start!r}, end={end!r}, duration={duration!r}")
                    continue
                valid_intervals.append((float(start), float(end), event.get("confidence")))
            event_intervals[field] = valid_intervals

        labeler = label.get("labeler")
        labeler = labeler.strip() if isinstance(labeler, str) else ""
        review_time = label.get("review_time_sec")
        review_positive = finite_number(review_time) and review_time > 0
        if not finite_number(review_time) or review_time < 0:
            _error(errors, name, "review_time_sec", repr(review_time))
        reviewer_strata[("named" if labeler else "empty") + ("_positive_time" if review_positive else "_nonpositive_time")] += 1
        if not labeler:
            material = name.split("_", 2)[1] if "_" in name else "unknown"
            empty_reviewer_material[material] += 1
            if singer:
                empty_reviewer_singer[singer] += 1

        hm = [(start, end) for start, end, confidence in event_intervals["breath_events"] if confidence in ("high", "medium")]
        merged = merge_intervals(hm)
        if len(merged) < len(hm):
            overlapping_raw_primary_clips += int(bool(labeler and review_positive))
        if any(right_start < left_end for (_, left_end), (right_start, _) in zip(merged[:-1], merged[1:])):
            merged_overlap_violations += 1
            _error(errors, name, "merged_positive_overlap", merged)
        if labeler and review_positive:
            primary_clips += 1
            primary_raw += len(hm)
            primary_merged += len(merged)
            primary_zero += int(not merged)

        receipt_path = Path(row.get("receipt_path", acquisition_dir / "receipts" / (name + ".json")))
        if not receipt_path.is_file():
            _error(errors, name, "receipt_exists", receipt_path)
        else:
            try:
                receipt = load_json(receipt_path)
                _expect_equal(errors, name, "receipt_matches_manifest", receipt, row)
            except Exception as error:
                _error(errors, name, "receipt_json", repr(error))

        expected_audio_path = acquisition_dir / "audio" / name
        _expect_equal(errors, name, "manifest_audio_path", row.get("audio_path"), str(expected_audio_path.resolve()))
        if not expected_audio_path.is_file():
            _error(errors, name, "audio_exists", expected_audio_path)
            continue
        try:
            length, crc32 = file_length_crc32(expected_audio_path)
            audio_sha = sha256_file(expected_audio_path)
            metadata = wav_metadata(expected_audio_path)
        except Exception as error:
            _error(errors, name, "audio_decode", repr(error))
            continue
        _expect_equal(errors, name, "audio_sha256", audio_sha, row.get("audio_sha256"))
        _expect_equal(errors, name, "member_bytes", length, row.get("member_bytes"))
        _expect_equal(errors, name, "member_crc32", crc32, str(row.get("member_crc32", "")).lower())
        _expect_equal(errors, name, "sample_rate", metadata["sample_rate"], row.get("sample_rate"))
        _expect_equal(errors, name, "frames", metadata["frames"], row.get("frames"))
        _expect_equal(errors, name, "channels", metadata["channels"], row.get("channels"))
        recorded_duration = row.get("duration_verified")
        if not finite_number(recorded_duration) or abs(metadata["duration_sec"] - recorded_duration) > 1e-12:
            _error(errors, name, "duration_vs_receipt", f"decoded={metadata['duration_sec']}, receipt={recorded_duration!r}")
        if duration_valid and abs(metadata["duration_sec"] - duration) > 0.01:
            _error(errors, name, "duration_vs_annotation", f"decoded={metadata['duration_sec']}, annotation={duration}")
        archive_member = row.get("archive_member")
        if not isinstance(archive_member, str) or Path(archive_member).name != name:
            _error(errors, name, "archive_member", repr(archive_member))
        else:
            archive_members.append(archive_member)
        total_member_bytes += length

    observed = {
        "selected": len(manifest),
        "unique_names": len(set(names)),
        "singers": len(singer_counts),
        "named_reviewer_primary_clips": primary_clips,
        "primary_raw_high_medium_events": primary_raw,
        "primary_merged_high_medium_events": primary_merged,
        "primary_zero_event_clips": primary_zero,
        "empty_reviewer_clips": reviewer_strata["empty_positive_time"] + reviewer_strata["empty_nonpositive_time"],
        "overlapping_raw_primary_clips": overlapping_raw_primary_clips,
        "merged_overlap_violations": merged_overlap_violations,
    }
    for key, expected_value in expected.items():
        _expect_equal(errors, "<cohort>", key, observed.get(key), expected_value)
    _expect_equal(errors, "<cohort>", "unique_archive_members", len(set(archive_members)), expected["selected"])
    _expect_equal(errors, "<cohort>", "acquisition_whole_archive_hash_verified", acquisition.get("whole_archive_hash_verified"), False)
    _expect_equal(errors, "<cohort>", "annotation_whole_archive_hash_verified", annotation_audit.get("whole_archive_hash_verified"), False)

    result = {
        "schema": "vocalset-breath-independent-acquisition-audit-v1",
        "status": "passed" if not errors else "failed",
        "gate_passed": not errors,
        "no_model_fit": True,
        "no_ai_human_labels": True,
        "code_sha256": sha256_file(__file__),
        "acquisition_summary_path": str(summary_path),
        "acquisition_summary_sha256": sha256_file(summary_path),
        "annotation_audit_path": str(annotation_audit_path),
        "annotation_audit_sha256": sha256_file(annotation_audit_path),
        "labels_dir": str(labels_dir),
        "expected": expected,
        "observed": observed,
        "singer_counts": dict(sorted(singer_counts.items())),
        "reviewer_strata": dict(sorted(reviewer_strata.items())),
        "empty_reviewer_material_counts": dict(sorted(empty_reviewer_material.items())),
        "empty_reviewer_singer_counts": dict(sorted(empty_reviewer_singer.items())),
        "event_confidence_counts": dict(sorted(confidence_counts.items())),
        "total_verified_member_bytes": total_member_bytes,
        "whole_archive_hash_verified": False,
        "archive_reported_md5": annotation_audit.get("archive_reported_md5"),
        "whole_archive_hash_note": "Selected member CRC/length/SHA and decoded PCM were verified; byte-range retrieval does not verify the whole archive MD5.",
        "checks": {
            "all_names_unique": len(set(names)) == expected["selected"],
            "all_singer_prefixes_valid": sum(singer_counts.values()) == expected["selected"],
            "all_source_annotation_sha256_match": not any(e["check"].endswith("annotation_sha") for e in errors),
            "all_audio_sha256_crc_length_match": not any(e["check"] in {"audio_sha256", "member_crc32", "member_bytes"} for e in errors),
            "all_pcm_metadata_and_duration_match": not any(e["check"] in {"audio_decode", "sample_rate", "frames", "channels", "duration_vs_receipt", "duration_vs_annotation"} for e in errors),
            "all_annotation_durations_and_event_bounds_valid": not any("duration" in e["check"] or "_bounds" in e["check"] for e in errors),
            "no_overlap_after_positive_merge": merged_overlap_violations == 0,
        },
        "errors": errors,
        "runtime": {"python": sys.version.split()[0]},
    }
    return result


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--summary-name", default="acquisition_summary.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.acquisition_dir / "independent_audit.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite independent audit: {output}")
    result = audit_acquisition(
        args.acquisition_dir, args.labels_dir, summary_name=args.summary_name
    )
    atomic_write_json(output, result)
    print(json.dumps({"output": str(output), "status": result["status"], "errors": len(result["errors"])}))
    if not result["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
