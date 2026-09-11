#!/usr/bin/env python3
"""Materialize all reconciled Saraga Hindustani audio; acquisition, not admission.

The production CLI has no checksum override, download, cohort, or model options.
Only ZIP payloads named by the pinned reconciliation may become raw/{MBID}.mp3.
Tests replace pins in memory for explicitly synthetic, temporary fixtures.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import zipfile

VERSION = "saraga_hindustani_physical_materialization_v1"
COUNT = 108
ARCHIVE_BYTES = 4_109_172_493
ARCHIVE_MD5 = "ea9ed2885ea37a1b10e42f60cf299702"
ARCHIVE_SHA256 = "cd3abd54288efd95e85ae3bf8b13e770e0914bfac12fa07dcba04c6ddc641fb7"
RECON_SHA256 = "d92da602fc6e5dc74469c3fdf1c39d61b4086a4be3a7c4c9ed7f565de6aff2a3"
AUDIT_SHA256 = "1b22ef5a089996af1d45be775b0ef6fbb7d7fc67072b60d1a576ee05f52d3ec1"
ROOT = "saraga1.5_hindustani"
BLOCK_FRAMES = 65536
PCM_ENCODING = "IEEE754 float64 little-endian; C order [frame, channel]; no header"
MEASUREMENT_KEYS = {
    "sample_rate", "channels", "decoder_format", "decoder_subtype", "header_frames",
    "actual_frames", "header_minus_actual_frames", "header_matches_actual_eof",
    "actual_duration_seconds", "duration_at_least_60_seconds", "read_calls_including_empty_eof",
    "real_empty_read_observed", "read_block_frames", "pcm_sha256", "pcm_encoding",
    "sample_count", "finite_sample_count", "nonfinite_sample_count", "sample_min", "sample_max",
    "peak_absolute", "rms_all_samples", "raw_hashes_before_decode", "raw_hashes_after_decode",
}
MBID_RE = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
CODE_PATH = Path(__file__).absolute()
LOADED_CODE_SHA256 = hashlib.sha256(CODE_PATH.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def digest_string(value, length=64):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value)


def safe_path(path):
    """Reject symlinks in every existing component, including broken symlinks."""
    path = Path(os.path.abspath(path))
    for component in [*reversed(path.parents), path]:
        if component.is_symlink():
            raise ValueError(f"Symlink forbidden: {component}")
    return path


def regular_file(path):
    path = safe_path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
            f"Expected regular, singly linked file: {path}")
    return path


def hashes(path):
    path = regular_file(path)
    before = path.stat()
    md5, sha, count = hashlib.md5(), hashlib.sha256(), 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            count += len(block)
            md5.update(block)
            sha.update(block)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
             before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size,
                                    after.st_mtime_ns, after.st_ctime_ns),
            f"File changed while hashing: {path}")
    return {"bytes": count, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def no_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(regular_file(path).read_text(), object_pairs_hook=no_duplicates,
                      parse_constant=no_constant)


def safe_member(name):
    require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name,
            "Invalid ZIP member name")
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and "." not in name.rstrip("/").split("/")
            and str(p) == name.rstrip("/"), f"Unsafe ZIP member: {name!r}")


def is_resource(name):
    return "__MACOSX" in PurePosixPath(name).parts or any(
        part.startswith("._") for part in PurePosixPath(name).parts)


def unique(rows, key, label):
    require(isinstance(rows, list), f"Invalid {label} list")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and key in row, f"Invalid {label} row")
        value = row[key]
        require(isinstance(value, str) and value not in result,
                f"Duplicate/invalid {label} {key}: {value!r}")
        result[value] = row
    return result


def validate_map(recon, audit):
    """Validate all identities, checksum joins, and exact audio inventory."""
    require(recon.get("status") ==
            "passed_archive_catalog_checksum_path_reconciliation_v2_not_audio_admission"
            and recon.get("failure_reasons") == []
            and recon.get("physical_audio_decoded") is False
            and recon.get("cohort_selected") is False
            and recon.get("classifier_admission") is False,
            "Reconciliation did not pass the acquisition-only boundary")
    require(audit.get("status") == "passed_archive_integrity_not_audio_admission"
            and audit.get("source_record") == 4301737
            and audit.get("classifier_admission") is False
            and audit.get("extracted_audio_files") == 0
            and audit.get("physically_decoded_audio_files") == 0,
            "Invalid source archive audit")
    expected = {"md5": ARCHIVE_MD5, "sha256": ARCHIVE_SHA256}
    require(audit.get("archive_bytes") == ARCHIVE_BYTES
            and audit.get("archive_hashes") == expected
            and recon["archive_integrity_evidence"]["archive_bytes"] == ARCHIVE_BYTES
            and recon["archive_integrity_evidence"]["archive_hashes"] == expected
            and recon["inputs"]["archive_audit"]["sha256"] == AUDIT_SHA256,
            "Source archive identity disagreement")
    records = unique(audit.get("records"), "path", "archive")
    require(len(records) == audit.get("archive_members_verified"), "Audit inventory mismatch")
    for name, row in records.items():
        safe_member(name)
        require(integer(row.get("bytes")) and integer(row.get("compressed_bytes"))
                and row.get("crc_verified") is True
                and digest_string(row.get("md5"), 32)
                and digest_string(row.get("sha256"))
                and digest_string(row.get("crc32"), 8), "Invalid archive member evidence")
    tracks = unique(recon.get("tracks"), "mbid", "metadata")
    matches = unique(recon.get("matched_audio"), "mbid", "matched audio")
    require(len(tracks) == len(matches) == COUNT and tracks.keys() == matches.keys(),
            "Expected exactly all reconciled recording IDs")
    audio = {n for n in records if n.lower().endswith(".mp3") and not is_resource(n)}
    resources = {n for n in records if n.lower().endswith(".mp3") and is_resource(n)}
    require(len(audio) == len(resources) == COUNT and audit.get("mp3_members") == 2 * COUNT,
            "Expected true MP3s and separate AppleDouble MP3 inventory")
    paths, md5s, shas, items = set(), set(), set(), []
    for mbid, track in sorted(tracks.items()):
        require(MBID_RE.fullmatch(mbid), "Unsafe/noncanonical recording MBID")
        match = matches[mbid]
        name = track.get("archive_audio_path")
        require(isinstance(name, str) and name in audio and name not in paths
                and name.startswith(ROOT + "/") and name.endswith(".mp3.mp3")
                and name == match.get("archive_audio_path"), "Ambiguous archive audio member")
        record = records[name]
        require(track.get("path_resolution") == match.get("path_resolution") ==
                "exact_unique_mp3_md5" and match.get("catalog_md5") ==
                match.get("archive_md5") == record["md5"], "Checksum join failed")
        require(record["md5"] not in md5s and record["sha256"] not in shas,
                "Duplicate audio checksum")
        require(type(track.get("speech_title_review_flag")) is bool
                and isinstance(track.get("performer_credits"), list), "Invalid metadata flags")
        paths.add(name)
        md5s.add(record["md5"])
        shas.add(record["sha256"])
        items.append({"mbid": mbid, "raw_path": f"raw/{mbid}.mp3",
                      "archive_member": record, "reconciled_metadata": track,
                      "checksum_match": match})
    require(paths == audio, "Missing/extra audio member mapping")
    return items, records


def validate_zip(archive_path, records):
    """Inspect directory only; extraction subsequently reads selected audio to CRC EOF."""
    with zipfile.ZipFile(archive_path) as source:
        infos = source.infolist()
        require(len(infos) <= 20000 and len({i.filename for i in infos}) == len(infos),
                "Oversized/duplicate ZIP inventory")
        actual = {}
        for info in infos:
            safe_member(info.filename)
            mode = info.external_attr >> 16
            require(not stat.S_ISLNK(mode) and not (info.flag_bits & 1)
                    and stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR),
                    "Symlink/special/encrypted ZIP member forbidden")
            require(0 <= info.file_size <= 4 * 1024**3, "Oversized ZIP member")
            if not info.is_dir():
                actual[info.filename] = info
        require(actual.keys() == records.keys(), "ZIP file inventory differs from audit")
        for name, info in actual.items():
            row = records[name]
            require(info.file_size == row["bytes"]
                    and info.compress_size == row["compressed_bytes"]
                    and f"{info.CRC:08x}" == row["crc32"], "ZIP member directory mismatch")


def runtime_info():
    import numpy as np
    import soundfile as sf
    require(sys.version_info >= (3, 9), "Python 3.9 or newer required")
    return {"python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "numpy": np.__version__,
            "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
            "numpy_module_sha256": hashes(Path(np.__file__))["sha256"],
            "soundfile_module_sha256": hashes(Path(sf.__file__))["sha256"],
            "decoder_claim": "SoundFile/libsndfile only; actual per-item reads prove decoding"}


def decode_to_eof(path):
    """Read sequentially until a real empty read, independently of header frames."""
    import numpy as np
    import soundfile as sf
    before = hashes(path)
    pcm = hashlib.sha256()
    frames = calls = 0
    peak = scaled_squares = 0.0
    minimum, maximum = math.inf, -math.inf
    with sf.SoundFile(str(path), mode="r") as decoder:
        rate, channels, header_frames = int(decoder.samplerate), int(decoder.channels), int(decoder.frames)
        require(rate > 0 and 0 < channels <= 256 and header_frames >= 0,
                "Invalid decoder header")
        format_name, subtype = decoder.format, decoder.subtype
        while True:
            block = decoder.read(BLOCK_FRAMES, dtype="float64", always_2d=True)
            calls += 1
            require(isinstance(block, np.ndarray) and block.dtype == np.dtype("float64")
                    and block.ndim == 2 and block.shape[1] == channels
                    and block.shape[0] <= BLOCK_FRAMES, "Unexpected decoder output schema")
            if block.shape[0] == 0:
                break
            require(np.isfinite(block).all(), "Nonfinite decoded PCM")
            frames += block.shape[0]
            pcm.update(np.asarray(block, dtype="<f8", order="C").tobytes(order="C"))
            block_peak = float(np.max(np.abs(block)))
            minimum = min(minimum, float(np.min(block)))
            maximum = max(maximum, float(np.max(block)))
            new_peak = max(peak, block_peak)
            if new_peak:
                scaled_squares = (scaled_squares * (peak / new_peak)**2
                                  + float(np.sum((block / new_peak)**2, dtype=np.float64)))
            peak = new_peak
        require(frames > 0, "Empty decoded recording")
    require(hashes(path) == before, "Raw source changed during decoding")
    samples = frames * channels
    rms = peak * math.sqrt(min(1.0, scaled_squares / samples))
    require(all(math.isfinite(x) for x in (minimum, maximum, peak, rms)), "Nonfinite PCM statistics")
    return {"sample_rate": rate, "channels": channels, "decoder_format": format_name,
            "decoder_subtype": subtype, "header_frames": header_frames,
            "actual_frames": frames, "header_minus_actual_frames": header_frames - frames,
            "header_matches_actual_eof": header_frames == frames,
            "actual_duration_seconds": frames / rate, "duration_at_least_60_seconds": frames >= 60 * rate,
            "read_calls_including_empty_eof": calls, "real_empty_read_observed": True,
            "read_block_frames": BLOCK_FRAMES, "pcm_sha256": pcm.hexdigest(),
            "pcm_encoding": PCM_ENCODING, "sample_count": samples,
            "finite_sample_count": samples, "nonfinite_sample_count": 0,
            "sample_min": minimum, "sample_max": maximum, "peak_absolute": peak,
            "rms_all_samples": rms, "raw_hashes_before_decode": before,
            "raw_hashes_after_decode": before}


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_pending(pending, destination):
    """Hard-link publication is atomic and fails if the destination exists."""
    os.link(pending, destination, follow_symlinks=False)
    pending.unlink()  # Only this invocation's completed staging link is removed.
    fsync_directory(destination.parent)


def write_new_json(path, value):
    safe_path(path)
    pending = path.with_name("." + path.name + ".pending")
    with pending.open("xb") as handle:
        handle.write(canonical(value))
        handle.flush()
        os.fsync(handle.fileno())
    publish_pending(pending, path)


@contextmanager
def writer_lock(output):
    lock = output / "writer.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    identity = os.fstat(fd)
    try:
        os.write(fd, canonical({"pid": os.getpid(), "hostname": platform.node()}))
        os.fsync(fd)
        yield
    finally:
        os.close(fd)
        now = lock.lstat()
        require((now.st_dev, now.st_ino) == (identity.st_dev, identity.st_ino), "Writer lock replaced")
        lock.unlink()
        fsync_directory(output)


def inventory(output, items, complete=False):
    allowed = {"writer.lock", "contract.json", "raw", "receipts", "summary.json"}
    require({p.name for p in output.iterdir()} <= allowed, "Unexpected output-root files")
    for path in output.iterdir():
        safe_path(path)
        require(path.is_dir() if path.name in ("raw", "receipts") else path.is_file(),
                "Unexpected output entry type")
    ids = {item["mbid"] for item in items}
    for directory, suffix in (("raw", ".mp3"), ("receipts", ".json")):
        child = output / directory
        if child.exists():
            names = {p.name for p in child.iterdir()}
            expected = {mbid + suffix for mbid in ids}
            require(names == expected if complete else names <= expected,
                    f"Unexpected/incomplete {directory} inventory (staging files require review)")
            for path in child.iterdir():
                regular_file(path)
        else:
            require(not complete, f"Missing {directory} directory")


def extract_audio(archive_path, item, destination):
    """Stream one approved MP3 member only, checking ZIP CRC and both hashes."""
    require(not destination.exists(), "Raw destination conflict")
    pending = destination.with_name("." + destination.name + ".pending")
    md5, sha, count = hashlib.md5(), hashlib.sha256(), 0
    expected = item["archive_member"]
    with zipfile.ZipFile(archive_path) as source:
        with source.open(expected["path"]) as incoming, pending.open("xb") as outgoing:
            for block in iter(lambda: incoming.read(1024 * 1024), b""):
                count += len(block)
                require(count <= expected["bytes"], "Archive member exceeded expected size")
                md5.update(block)
                sha.update(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    require({"bytes": count, "md5": md5.hexdigest(), "sha256": sha.hexdigest()} ==
            {k: expected[k] for k in ("bytes", "md5", "sha256")}, "Extracted member hash mismatch")
    publish_pending(pending, destination)


def validate_receipt(receipt, item, contract_hash, current_raw):
    require(set(receipt) == {"record", "record_sha256"}, "Invalid receipt envelope")
    record = receipt["record"]
    require(receipt["record_sha256"] == object_sha(record), "Receipt digest mismatch")
    require(set(record) == {"status", "contract_sha256", "item", "measurement",
                            "classifier_admission", "cohort_selected"}, "Invalid receipt schema")
    require(record["status"] == "passed_physical_acquisition_not_audio_admission"
            and record["contract_sha256"] == contract_hash and record["item"] == item
            and record["classifier_admission"] is False and record["cohort_selected"] is False,
            "Receipt contract/identity mismatch")
    m = record["measurement"]
    require(isinstance(m, dict) and set(m) == MEASUREMENT_KEYS,
            "Invalid measurement schema")
    expected = {k: item["archive_member"][k] for k in ("bytes", "md5", "sha256")}
    require(current_raw == expected == m.get("raw_hashes_before_decode") ==
            m.get("raw_hashes_after_decode"), "Receipt/raw content mismatch")
    for name in ("sample_rate", "channels", "actual_frames", "sample_count", "finite_sample_count"):
        require(integer(m.get(name), 1), f"Invalid receipt measurement: {name}")
    require(integer(m.get("header_frames")) and integer(m.get("read_calls_including_empty_eof"), 2)
            and m["read_calls_including_empty_eof"] >= math.ceil(m["actual_frames"] / BLOCK_FRAMES) + 1
            and m.get("real_empty_read_observed") is True and m.get("read_block_frames") == BLOCK_FRAMES
            and m.get("pcm_encoding") == PCM_ENCODING and digest_string(m.get("pcm_sha256"))
            and isinstance(m.get("decoder_format"), str) and bool(m["decoder_format"])
            and isinstance(m.get("decoder_subtype"), str),
            "Invalid EOF/PCM evidence")
    require(m["channels"] <= 256
            and m["sample_count"] == m["finite_sample_count"] == m["actual_frames"] * m["channels"]
            and type(m.get("nonfinite_sample_count")) is int and m["nonfinite_sample_count"] == 0
            and type(m.get("header_minus_actual_frames")) is int
            and m.get("header_minus_actual_frames") == m["header_frames"] - m["actual_frames"]
            and m.get("header_matches_actual_eof") is (m["header_frames"] == m["actual_frames"])
            and m.get("actual_duration_seconds") == m["actual_frames"] / m["sample_rate"]
            and m.get("duration_at_least_60_seconds") is (m["actual_frames"] >= 60 * m["sample_rate"]),
            "Inconsistent receipt measurements")
    for name in ("sample_min", "sample_max", "peak_absolute", "rms_all_samples"):
        require(type(m.get(name)) in (int, float) and math.isfinite(m[name]), "Invalid finite statistics")
    require(m["sample_min"] <= m["sample_max"]
            and 0 <= m["rms_all_samples"] <= m["peak_absolute"]
            and m["peak_absolute"] == max(abs(m["sample_min"]), abs(m["sample_max"])),
            "Inconsistent amplitude statistics")
    return record


def run(archive, reconciliation, archive_audit, output, workers=1):
    require(integer(workers, 1) and workers <= 4, "Workers must be 1 through 4")
    archive, reconciliation, archive_audit, output = map(
        safe_path, (archive, reconciliation, archive_audit, output))
    require(not output.exists() or output.is_dir(), "Output is not a directory")
    require(all(output != p and output not in p.parents for p in
                (archive, reconciliation, archive_audit, CODE_PATH)), "Inputs must be outside output")
    named = {"code": CODE_PATH, "reconciliation": reconciliation, "archive_audit": archive_audit}
    snapshots = {key: hashes(path) for key, path in named.items()}
    require(snapshots["code"]["sha256"] == LOADED_CODE_SHA256, "Loaded code changed")
    require(snapshots["reconciliation"]["sha256"] == RECON_SHA256
            and snapshots["archive_audit"]["sha256"] == AUDIT_SHA256, "Pinned input hash mismatch")
    recon, audit = read_json(reconciliation), read_json(archive_audit)
    items, records = validate_map(recon, audit)
    expected_archive = {"bytes": ARCHIVE_BYTES, "md5": ARCHIVE_MD5, "sha256": ARCHIVE_SHA256}
    require(archive.stat().st_size == ARCHIVE_BYTES and hashes(archive) == expected_archive,
            "Whole archive size/hash mismatch")
    validate_zip(archive, records)
    runtime = runtime_info()
    contract = {"version": VERSION, "expected_recordings": COUNT,
                "archive_path": str(archive), "archive_hashes": expected_archive,
                "inputs": {key: {"path": str(named[key]), **value} for key, value in snapshots.items()},
                "runtime": runtime, "decode_block_frames": BLOCK_FRAMES, "pcm_encoding": PCM_ENCODING,
                "cohort_selected": False, "classifier_admission": False, "items": items}
    contract_hash = object_sha(contract)

    def unchanged():
        require(all(hashes(path) == snapshots[key] for key, path in named.items()),
                "Input/code contract changed during execution")

    unchanged()
    output.mkdir(parents=True, exist_ok=True)
    with writer_lock(output):
        inventory(output, items)
        contract_path = output / "contract.json"
        if contract_path.exists():
            require(read_json(contract_path) == contract, "Existing output contract conflict")
        else:
            require({p.name for p in output.iterdir()} == {"writer.lock"},
                    "Existing output without contract is an explicit conflict")
            write_new_json(contract_path, contract)
        for directory in ("raw", "receipts"):
            (output / directory).mkdir(exist_ok=True)

        def process(item):
            unchanged()
            raw = output / item["raw_path"]
            receipt_path = output / "receipts" / (item["mbid"] + ".json")
            if receipt_path.exists():
                record = validate_receipt(read_json(receipt_path), item, contract_hash, hashes(raw))
                unchanged()
                return record
            require(not raw.exists(), f"Raw file without receipt is an explicit conflict: {raw}")
            extract_audio(archive, item, raw)
            measurement = decode_to_eof(raw)
            record = {"status": "passed_physical_acquisition_not_audio_admission",
                      "contract_sha256": contract_hash, "item": item, "measurement": measurement,
                      "classifier_admission": False, "cohort_selected": False}
            receipt = {"record": record, "record_sha256": object_sha(record)}
            validate_receipt(receipt, item, contract_hash, hashes(raw))
            unchanged()
            write_new_json(receipt_path, receipt)
            print(json.dumps({"mbid": item["mbid"], "status": record["status"],
                              "actual_frames": measurement["actual_frames"]}), flush=True)
            return record

        # Each worker has its own ZIP handle; each audio stream remains sequential.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            completed = list(pool.map(process, items))
        require(len(completed) == COUNT, "Incomplete physical materialization")
        unchanged()
        require(runtime_info() == runtime, "Runtime changed during execution")
        require(hashes(archive) == expected_archive, "Source archive changed during materialization")
        inventory(output, items, complete=True)
        require(read_json(contract_path) == contract, "Output contract changed during execution")
        # Re-read every persisted receipt and hash every raw file before final publication.
        final = [validate_receipt(read_json(output / "receipts" / (item["mbid"] + ".json")),
                                  item, contract_hash, hashes(output / item["raw_path"])) for item in items]
        require(final == completed, "Persisted records changed before completion")
        unchanged()
        summary = {"status": "passed_all_physical_acquisition_not_audio_admission",
                   "version": VERSION, "contract_sha256": contract_hash, "record_count": COUNT,
                   "source_archive_hashes_before": expected_archive,
                   "source_archive_hashes_after": expected_archive,
                   "input_code_contract_unchanged": True, "runtime_unchanged": True,
                   "expected_file_inventory_verified": True,
                   "classifier_admission": False, "cohort_selected": False,
                   "annotation_alignment_verified": False, "human_music_usability_claim": False,
                   "speech_title_review_flags": sum(i["reconciled_metadata"]["speech_title_review_flag"] for i in items),
                   "missing_performer_credits": sum(not i["reconciled_metadata"]["performer_credits"] for i in items),
                   "duration_at_least_60_seconds_count": sum(r["measurement"]["duration_at_least_60_seconds"] for r in final),
                   "header_frame_mismatch_count": sum(not r["measurement"]["header_matches_actual_eof"] for r in final),
                   "records": final}
        summary_path = output / "summary.json"
        if summary_path.exists():
            require(read_json(summary_path) == summary, "Existing summary conflict")
        else:
            unchanged()
            write_new_json(summary_path, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--archive-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args(argv)
    try:
        result = run(args.archive, args.reconciliation, args.archive_audit, args.output_dir, args.workers)
    except Exception as exc:
        print(json.dumps({"status": "failed_or_incomplete_no_new_success_summary",
                          "error_type": type(exc).__name__, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
