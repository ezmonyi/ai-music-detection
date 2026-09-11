"""Independent FFmpeg duration cross-check for the frozen 39 short-input diagnosis.

This diagnostic does not resample, write audio, alter cohort membership, or treat
either decoder as ground truth.  FFmpeg emits native-rate/native-channel f64le to
stdout; the stream is counted and hashed without retaining decoded audio.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys


VERSION = "diagnose_native30_short_ffmpeg_v1"
DIAGNOSIS_SHA256 = "86df6c6773797d7d2511e10857589f49917db644f9a64b21867bf642084cf08b"
REMOTE_COPY_ROOT = PurePosixPath(
    "/mnt/nfs-data/users/yi/audio_phenomena_expansion_20260907/"
    "native30_new1695_original_copies_v1"
)
LOCAL_ORIGINAL_ROOT = Path("/Users/yi/Documents/code/music")
FFMPEG = Path("/opt/homebrew/bin/ffmpeg")
FFPROBE = Path("/opt/homebrew/bin/ffprobe")
EXPECTED_RECORDS = 39
READ_BYTES = 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(READ_BYTES), b""):
            value.update(block)
    return value.hexdigest()


def strict_json(path):
    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_binding(path, *, allow_symlink=False):
    requested = Path(path)
    require(requested.exists(), "missing file: " + str(requested))
    require(allow_symlink or not requested.is_symlink(), "symlink forbidden: " + str(requested))
    resolved = requested.resolve(strict=True)
    require(resolved.is_file(), "regular file required: " + str(requested))
    stat = resolved.stat()
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "bytes": stat.st_size,
        "sha256": digest(resolved),
    }


def command_version(binary):
    command = [str(binary), "-version"]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
    require(result.returncode == 0, f"version command failed ({result.returncode}): {binary}")
    lines = result.stdout.splitlines()
    require(lines, "empty version output: " + str(binary))
    return {"command": command, "first_line": lines[0], "stdout": result.stdout.rstrip("\n")}


def local_original(remote_path):
    remote = PurePosixPath(remote_path)
    try:
        relative = remote.relative_to(REMOTE_COPY_ROOT)
    except ValueError as exc:
        raise ValueError("native path is outside frozen remote copy root: " + remote_path) from exc
    require(relative.parts and ".." not in relative.parts, "unsafe relative native path")
    local = LOCAL_ORIGINAL_ROOT.joinpath(*relative.parts)
    require(local.is_relative_to(LOCAL_ORIGINAL_ROOT), "mapped path escaped local original root")
    return local


def probe(path):
    command = [
        str(FFPROBE), "-v", "error", "-select_streams", "a:0",
        "-show_entries",
        "stream=index,codec_name,sample_rate,channels,start_time,duration,duration_ts,time_base:format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
    require(result.returncode == 0, f"ffprobe failed ({result.returncode}): {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("ffprobe returned invalid JSON") from exc
    require(len(payload.get("streams", [])) == 1, "ffprobe did not select exactly one first audio stream")
    return command, payload, result.stderr


def decode_f64le(path, native_rate, native_channels):
    command = [
        str(FFMPEG), "-v", "error", "-nostdin", "-i", str(path),
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-ar", str(native_rate),
        "-ac", str(native_channels), "-c:a", "pcm_f64le", "-f", "f64le", "pipe:1",
    ]
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    require(process.stdout is not None and process.stderr is not None, "FFmpeg pipes unavailable")
    stream_hash = hashlib.sha256()
    byte_count = 0
    try:
        while True:
            block = process.stdout.read(READ_BYTES)
            if not block:
                break
            stream_hash.update(block)
            byte_count += len(block)
        stderr_bytes = process.stderr.read()
        returncode = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    require(returncode == 0, f"ffmpeg failed ({returncode}): {stderr.strip()}")
    bytes_per_frame = native_channels * 8
    require(byte_count % bytes_per_frame == 0, "FFmpeg f64le byte count is not whole native-channel frames")
    return command, {
        "decoded_bytes": byte_count,
        "decoded_frames": byte_count // bytes_per_frame,
        "f64le_stream_sha256": stream_hash.hexdigest(),
        "bytes_per_sample": 8,
        "channels": native_channels,
        "sample_rate_hz": native_rate,
        "stderr": stderr,
    }


def diagnose_record(record):
    ident = record["id"]
    native_path = record["native_path"]
    require(record["source_binding"] == {"path": native_path, "sha256": record["native_sha256"]},
            "source binding disagreement: " + ident)
    require(record["source_group"] == "FMA" and record["native_channels"] == 2,
            "unexpected source/channels: " + ident)
    require(type(record["native_rate_hz"]) is int and record["native_rate_hz"] > 0,
            "invalid native rate: " + ident)
    local = local_original(native_path)
    require(local.is_file() and not local.is_symlink(), "regular non-symlink local original required: " + str(local))
    before_stat = local.stat()
    before_sha = digest(local)
    require(before_sha == record["native_sha256"], "local original SHA mismatch: " + ident)

    probe_command, probe_result, probe_stderr = probe(local)
    ffmpeg_command, decoded = decode_f64le(local, record["native_rate_hz"], record["native_channels"])

    after_stat = local.stat()
    after_sha = digest(local)
    require(after_sha == record["native_sha256"], "local original SHA changed/mismatched after decode: " + ident)
    require(
        (before_stat.st_dev, before_stat.st_ino, before_stat.st_size, before_stat.st_mtime_ns, before_stat.st_ctime_ns)
        == (after_stat.st_dev, after_stat.st_ino, after_stat.st_size, after_stat.st_mtime_ns, after_stat.st_ctime_ns),
        "local original stat signature changed during diagnostic: " + ident,
    )

    frames = decoded["decoded_frames"]
    rate = record["native_rate_hz"]
    libsndfile_frames = record["actual_frames"]
    required_frames = record["required_frames"]
    return {
        "status": "ffmpeg_stream_decode_completed_no_ground_truth_inference",
        "id": ident,
        "source_group": record["source_group"],
        "label": record["label"],
        "group_id": record["group_id"],
        "component_id": record["component_id"],
        "remote_copy_path_not_read": native_path,
        "local_original_path": str(local),
        "native_sha256_expected": record["native_sha256"],
        "native_sha256_before": before_sha,
        "native_sha256_after": after_sha,
        "source_bytes": before_stat.st_size,
        "source_stat_signature_before": [before_stat.st_dev, before_stat.st_ino, before_stat.st_size,
                                          before_stat.st_mtime_ns, before_stat.st_ctime_ns],
        "source_stat_signature_after": [after_stat.st_dev, after_stat.st_ino, after_stat.st_size,
                                         after_stat.st_mtime_ns, after_stat.st_ctime_ns],
        "ffprobe_command": probe_command,
        "ffprobe_result_informational_not_frame_authority": probe_result,
        "ffprobe_stderr": probe_stderr,
        "ffmpeg_command": ffmpeg_command,
        "ffmpeg_decode": decoded,
        "libsndfile_diagnosis_frames": libsndfile_frames,
        "required_30s_frames": required_frames,
        "ffmpeg_duration_seconds": frames / rate,
        "libsndfile_diagnosis_duration_seconds": libsndfile_frames / rate,
        "ffmpeg_minus_libsndfile_frames": frames - libsndfile_frames,
        "ffmpeg_minus_libsndfile_seconds": (frames - libsndfile_frames) / rate,
        "ffmpeg_minus_required_frames": frames - required_frames,
        "ffmpeg_minus_required_seconds": (frames - required_frames) / rate,
        "pcm_hash_comparison": "not_performed_decoder_PCM_bit_equality_not_required",
        "ground_truth_claim": False,
        "cohort_change_authorized": False,
    }


def main():
    here = Path(__file__).resolve()
    root = here.parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", type=Path, default=root / "audit/native30_short_failure_diagnosis_v2.json")
    parser.add_argument("--output", type=Path, default=root / "audit/native30_short_ffmpeg_v1.json")
    args = parser.parse_args()

    diagnosis = args.diagnosis.resolve(strict=True)
    output = args.output.absolute()
    require(not output.exists() and output.parent.is_dir() and not output.parent.is_symlink(),
            "new output in a regular existing parent required")
    code_before = digest(here)
    diagnosis_before = digest(diagnosis)
    require(diagnosis_before == DIAGNOSIS_SHA256, "frozen diagnosis SHA mismatch")
    source = strict_json(diagnosis)
    require(source["status"] == "confirmed_39_native_short_inputs_source_blind_policy",
            "unexpected diagnosis status")
    records = source["records"]
    require(len(records) == EXPECTED_RECORDS and len({row["id"] for row in records}) == EXPECTED_RECORDS,
            "expected 39 unique diagnosis records")

    tools = {
        "ffmpeg": {**file_binding(FFMPEG, allow_symlink=True), "version": command_version(FFMPEG)},
        "ffprobe": {**file_binding(FFPROBE, allow_symlink=True), "version": command_version(FFPROBE)},
    }
    completed, failures = [], []
    for index, record in enumerate(records, 1):
        try:
            completed.append(diagnose_record(record))
        except Exception as exc:
            failures.append({"id": record.get("id"), "exception": type(exc).__name__, "message": str(exc)})
        print(json.dumps({"event": "ffmpeg_duration_crosscheck", "reviewed": index,
                          "planned": len(records), "completed": len(completed), "failed": len(failures)},
                         sort_keys=True), file=sys.stderr, flush=True)

    require(digest(diagnosis) == diagnosis_before, "frozen diagnosis changed during diagnostic")
    require(digest(here) == code_before, "diagnostic code changed during execution")
    differences = Counter(row["ffmpeg_minus_libsndfile_frames"] for row in completed)
    requirement_differences = Counter(row["ffmpeg_minus_required_frames"] for row in completed)
    receipt = {
        "version": VERSION,
        "status": "complete_ffmpeg_crosscheck_no_ground_truth_or_cohort_change" if not failures
                  else "partial_ffmpeg_crosscheck_with_failures",
        "scope": "FFmpeg_first_audio_stream_native_rate_native_channels_f64le_stream_count_and_hash_only",
        "input_binding": {"path": str(diagnosis), "sha256_before": diagnosis_before,
                          "sha256_after": digest(diagnosis), "bytes": diagnosis.stat().st_size},
        "implementation_binding": {"path": str(here), "sha256_before": code_before,
                                   "sha256_after": digest(here), "bytes": here.stat().st_size},
        "path_mapping": {"remote_copy_root_not_read": str(REMOTE_COPY_ROOT),
                         "local_original_root": str(LOCAL_ORIGINAL_ROOT),
                         "rule": "append relative path below remote copy root to local original root"},
        "tools": tools,
        "command_policy": "native-rate/native-channel pcm_f64le to stdout; no resampling and no audio output file",
        "records_expected": EXPECTED_RECORDS,
        "records_completed": len(completed),
        "records_failed": len(failures),
        "ffmpeg_minus_libsndfile_frame_counts": {str(key): value for key, value in sorted(differences.items())},
        "ffmpeg_minus_required_frame_counts": {str(key): value for key, value in sorted(requirement_differences.items())},
        "records": completed,
        "failures": failures,
        "interpretation_limit": "decoder outputs are observations, not universal duration ground truth; PCM equality is not required",
        "audio_files_written": 0,
        "resampling_performed": False,
        "classifier_fits": 0,
        "cohort_admitted": False,
        "feature_extraction_authorized": False,
    }
    payload = canonical(receipt)
    temporary = output.with_name("." + output.name + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"status": receipt["status"], "completed": len(completed), "failed": len(failures),
                      "output": str(output), "output_sha256": digest(output)}, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
