#!/usr/bin/env python3
"""Validate and materialize official GuitarSet mic audio and JAMS annotations.

This is a CPU-only provenance and physical-decode validator.  It does not
extract BC features, assign class labels, audit dataset overlap, fit a model,
or pass an external measurement gate.
"""

import argparse
from collections import Counter
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


VERSION = "1.0"
METADATA_SHA256 = "1ebc52f572801b6c3834bf8007217274d6036de7ae16a00d0f29fd505876a8e8"
EXPECTED_ARCHIVES = {
    "audio_mono-mic.zip": {"bytes": 656_927_981, "md5": "275966d6610ac34999b58426beb119c3"},
    "annotation.zip": {"bytes": 39_132_574, "md5": "b39b78e63d3446f2e54ddb7a54df9b10"},
}
REQUIRED_SOURCE_PRODUCTS = {
    "upstream_metadata.json",
    "acquisition_receipt.json",
    "audio_mono-mic.zip",
    "annotation.zip",
}
EXPECTED_PLAYERS = tuple(f"{value:02d}" for value in range(6))
EXPECTED_SCORE_COUNT = 30
PERFORMANCES = ("comp", "solo")
MAX_MEMBERS_PER_ARCHIVE = 20_000
MAX_MEMBER_BYTES = 4 * 1024**3
MAX_TOTAL_UNCOMPRESSED_BYTES = 128 * 1024**3
READ_BYTES = 1024 * 1024
DECODE_FRAMES = 65_536
PCM_ENCODING = (
    "SHA-256 of decoded samples in frame-major/channel-minor C order; each sample is "
    "IEEE-754 float64 encoded explicitly little-endian (<f8); no resampling, mixing, "
    "normalization, or feature extraction"
)
IDENTITY_RE = re.compile(
    r"^(?P<player>[0-9]{2})_(?P<score>.+)_(?P<performance>comp|solo)$"
)
KNOWN_ANNOTATION_WARNINGS = {
    "04_BN3-154-E_comp": [{
        "kind": "official_repository_issue_not_verified_correction",
        "reported_alignment_offset_seconds": 0.409,
        "issue": "https://github.com/marl/GuitarSet/issues/5",
        "action": "retained unchanged; not silently corrected or excluded",
    }],
    "04_Jazz1-200-B_comp": [{
        "kind": "official_repository_issue_not_verified_correction",
        "reported_alignment_offset_seconds": 0.309,
        "issue": "https://github.com/marl/GuitarSet/issues/5",
        "action": "retained unchanged; not silently corrected or excluded",
    }],
    "02_Funk2-119-G_comp": [{
        "kind": "official_repository_issue_not_verified_correction",
        "reported_problem": "possible duplicated note",
        "issue": "https://github.com/marl/GuitarSet/issues/4",
        "action": "retained unchanged; not silently corrected or excluded",
    }],
}
CODE_PATH = Path(__file__).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def strict_json_bytes(data):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(data.decode("utf-8"), object_pairs_hook=unique_pairs,
                      parse_constant=reject_constant)


def safe_filesystem_path(path):
    """Return a canonical absolute path, rejecting the target and resolved symlinks."""
    raw = Path(os.path.abspath(os.fspath(path)))
    if raw.is_symlink():
        raise ValueError(f"Symlink filesystem path forbidden: {raw}")
    # Canonicalization permits platform aliases such as macOS /var -> /private/var,
    # while every task-created descendant is still checked below.
    path = raw.resolve(strict=False)
    chain = list(reversed(path.parents)) + [path]
    for component in chain:
        if component.is_symlink():
            raise ValueError(f"Symlink filesystem path forbidden: {component}")
    return path


def regular_file(path):
    path = safe_filesystem_path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), f"Expected regular file: {path}")
    return path


def file_fingerprint(path, include_md5=True):
    path = regular_file(path)
    before = path.stat()
    sha = hashlib.sha256()
    md5 = hashlib.md5() if include_md5 else None
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * READ_BYTES), b""):
            count += len(block)
            sha.update(block)
            if md5 is not None:
                md5.update(block)
    after = path.stat()
    signature = lambda value: (value.st_dev, value.st_ino, value.st_size,
                               value.st_mtime_ns, value.st_ctime_ns)
    require(signature(before) == signature(after), f"File changed while hashing: {path}")
    result = {"bytes": count, "sha256": sha.hexdigest()}
    if md5 is not None:
        result["md5"] = md5.hexdigest()
    return result


def safe_member_name(name):
    require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name,
            "Invalid ZIP member name")
    path = PurePosixPath(name)
    parts = name.rstrip("/").split("/")
    require(not path.is_absolute() and ".." not in path.parts and "." not in parts
            and "" not in parts and str(path) == name.rstrip("/")
            and not re.fullmatch(r"[A-Za-z]:", path.parts[0]),
            f"Unsafe ZIP member path: {name!r}")
    return path


def is_apple_resource(name):
    parts = PurePosixPath(name).parts
    return "__MACOSX" in parts or any(part.startswith("._") for part in parts)


def _source_inventory(source):
    result = {}
    for path in source.rglob("*"):
        safe_filesystem_path(path)
        require(path.is_dir() or stat.S_ISREG(path.lstat().st_mode),
                f"Non-regular source entry forbidden: {path}")
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            safe_member_name(relative)
            result[relative] = path
    return result


def verify_source(source, commit_sha256):
    """Independently bind the closed acquisition publication and official identities."""
    require(re.fullmatch(r"[0-9a-f]{64}", commit_sha256 or ""),
            "--commit-sha256 must be 64 lowercase hexadecimal characters")
    source = safe_filesystem_path(source)
    require(source.is_dir(), "Source must be an existing archive folder")
    inventory = _source_inventory(source)
    require("COMMIT.json" in inventory, "Source folder has no COMMIT.json")
    commit_fingerprint = file_fingerprint(inventory["COMMIT.json"], include_md5=False)
    require(commit_fingerprint["sha256"] == commit_sha256, "Source COMMIT SHA-256 mismatch")
    commit_bytes = inventory["COMMIT.json"].read_bytes()
    commit = strict_json_bytes(commit_bytes)
    require(isinstance(commit, dict) and set(commit) == {"status", "kind", "products"}
            and commit["status"] == "committed"
            and commit["kind"] == "archive_acquisition_only"
            and isinstance(commit["products"], dict), "Invalid acquisition COMMIT schema")
    products = commit["products"]
    require("COMMIT.json" not in products, "Acquisition COMMIT cannot list itself as a product")
    require(REQUIRED_SOURCE_PRODUCTS <= set(products),
            "Acquisition COMMIT is missing required scientific products")
    require(set(inventory) == set(products) | {"COMMIT.json"},
            "Source folder/COMMIT exact product inventory mismatch")
    product_fingerprints = {}
    source_stats = {}
    for name in sorted(products):
        safe_member_name(name)
        record = products[name]
        require(isinstance(record, dict) and set(record) == {"bytes", "sha256"}
                and type(record["bytes"]) is int and record["bytes"] >= 0
                and isinstance(record["sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]),
                f"Invalid acquisition product record: {name}")
        actual = file_fingerprint(inventory[name], include_md5=False)
        require(actual == record, f"Acquisition product bytes/hash mismatch: {name}")
        product_fingerprints[name] = actual
        info = inventory[name].stat()
        source_stats[name] = [info.st_dev, info.st_ino, info.st_size,
                              info.st_mtime_ns, info.st_ctime_ns]

    metadata_path = inventory["upstream_metadata.json"]
    metadata_fingerprint = product_fingerprints["upstream_metadata.json"]
    require(metadata_fingerprint["sha256"] == METADATA_SHA256,
            "Official metadata SHA-256 mismatch")
    metadata_bytes = metadata_path.read_bytes()
    metadata = strict_json_bytes(metadata_bytes)
    require(metadata.get("doi") == "10.5281/zenodo.3371780"
            and metadata.get("metadata", {}).get("license", {}).get("id") == "cc-by-4.0"
            and metadata.get("metadata", {}).get("access_right") == "open"
            and metadata.get("metadata", {}).get("version") == "1.1.0",
            "Unexpected GuitarSet provenance, license, or version")
    files = metadata.get("files")
    require(isinstance(files, list), "Invalid official metadata file inventory")
    entries = {}
    for item in files:
        require(isinstance(item, dict) and isinstance(item.get("key"), str)
                and item["key"] not in entries, "Duplicate/invalid official metadata file")
        entries[item["key"]] = item

    receipt = strict_json_bytes(inventory["acquisition_receipt.json"].read_bytes())
    require(receipt.get("status") == "archives_verified_not_decoded"
            and receipt.get("metadata_sha256") == METADATA_SHA256
            and receipt.get("doi") == "10.5281/zenodo.3371780"
            and receipt.get("license") == "CC-BY-4.0"
            and receipt.get("zip_crc_checked") is False
            and receipt.get("audio_decoded") is False
            and receipt.get("bc_measured") is False
            and receipt.get("classifier_fits") == 0
            and receipt.get("external_gate_passed") is False,
            "Invalid acquisition-only receipt boundary")
    receipt_archives = receipt.get("archives")
    require(isinstance(receipt_archives, list), "Invalid acquisition receipt archive list")
    receipt_by_name = {}
    for item in receipt_archives:
        require(isinstance(item, dict) and isinstance(item.get("name"), str)
                and item["name"] not in receipt_by_name, "Duplicate/invalid receipt archive")
        receipt_by_name[item["name"]] = item
    require(set(receipt_by_name) == set(EXPECTED_ARCHIVES),
            "Acquisition receipt archive inventory mismatch")

    archive_fingerprints = {}
    for name, expected in EXPECTED_ARCHIVES.items():
        entry = entries.get(name)
        require(isinstance(entry, dict)
                and entry.get("size") == expected["bytes"]
                and entry.get("checksum") == "md5:" + expected["md5"]
                and entry.get("links", {}).get("self") ==
                f"https://zenodo.org/api/records/3371780/files/{name}/content",
                f"Official archive metadata identity mismatch: {name}")
        actual = file_fingerprint(inventory[name], include_md5=True)
        require(actual["bytes"] == expected["bytes"] and actual["md5"] == expected["md5"],
                f"Official archive bytes/MD5 mismatch: {name}")
        committed = product_fingerprints[name]
        require({key: actual[key] for key in ("bytes", "sha256")} == committed,
                f"Archive disagrees with acquisition COMMIT: {name}")
        row = receipt_by_name[name]
        require(set(row) == {"name", "bytes", "md5", "sha256"}
                and row == {"name": name, "bytes": actual["bytes"],
                            "md5": actual["md5"], "sha256": actual["sha256"]},
                f"Archive disagrees with acquisition receipt: {name}")
        archive_fingerprints[name] = actual

    return {
        "source": source,
        "inventory": inventory,
        "source_stats": source_stats,
        "commit_bytes": commit_bytes,
        "metadata_bytes": metadata_bytes,
        "commit_fingerprint": commit_fingerprint,
        "metadata_fingerprint": metadata_fingerprint,
        "product_fingerprints": product_fingerprints,
        "archive_fingerprints": archive_fingerprints,
    }


def verify_source_unchanged(evidence):
    current = _source_inventory(evidence["source"])
    require(set(current) == set(evidence["product_fingerprints"]) | {"COMMIT.json"},
            "Source inventory changed during validation")
    require(file_fingerprint(current["COMMIT.json"], False) == evidence["commit_fingerprint"],
            "Source COMMIT changed during validation")
    for name, expected in evidence["product_fingerprints"].items():
        info = current[name].stat()
        signature = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]
        require(signature == evidence["source_stats"][name],
                f"Source product changed during validation: {name}")
        # Re-hash the two scientific inputs; selected member extraction also checks member hashes.
        if name in EXPECTED_ARCHIVES or name == "upstream_metadata.json":
            require(file_fingerprint(current[name], False) == expected,
                    f"Source product content changed during validation: {name}")


def scan_zip(path, archive_name=None, max_members=MAX_MEMBERS_PER_ARCHIVE,
             max_member_bytes=MAX_MEMBER_BYTES,
             max_total_bytes=MAX_TOTAL_UNCOMPRESSED_BYTES):
    """Read every ZIP member to EOF, thereby checking CRC, and hash its bytes."""
    path = regular_file(path)
    archive_name = archive_name or path.name
    rows = []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        require(len(infos) <= max_members, "ZIP member-count safety bound exceeded")
        require(len({info.filename for info in infos}) == len(infos),
                "Duplicate ZIP member name")
        total = 0
        for info in infos:
            require(info.orig_filename == info.filename,
                    "NUL or noncanonical ZIP member name forbidden")
            safe_member_name(info.orig_filename)
            require(not (info.flag_bits & 1), "Encrypted ZIP member forbidden")
            mode = info.external_attr >> 16
            kind = stat.S_IFMT(mode)
            require(not stat.S_ISLNK(mode)
                    and kind in (0, stat.S_IFREG, stat.S_IFDIR),
                    "Symlink or special ZIP member forbidden")
            require(type(info.file_size) is int and 0 <= info.file_size <= max_member_bytes,
                    "ZIP member uncompressed-size safety bound exceeded")
            total += info.file_size
            require(total <= max_total_bytes, "ZIP total uncompressed-size safety bound exceeded")
            sha, md5, count = hashlib.sha256(), hashlib.md5(), 0
            with archive.open(info, "r") as handle:
                for block in iter(lambda: handle.read(READ_BYTES), b""):
                    count += len(block)
                    require(count <= info.file_size, "ZIP member expanded beyond declared size")
                    sha.update(block)
                    md5.update(block)
            require(count == info.file_size, "Short ZIP member read")
            rows.append({
                "archive": archive_name,
                "path": info.filename,
                "basename": PurePosixPath(info.filename).name,
                "is_directory": info.is_dir(),
                "apple_metadata": is_apple_resource(info.filename),
                "bytes": count,
                "compressed_bytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
                "crc_verified_by_full_read": True,
                "md5": md5.hexdigest(),
                "sha256": sha.hexdigest(),
            })
    return rows


def parse_media_identity(member_name, media_kind):
    safe_member_name(member_name)
    require(media_kind in ("audio", "annotation"), "Unknown media kind")
    basename = PurePosixPath(member_name).name
    suffix = ".wav" if media_kind == "audio" else ".jams"
    require(basename.lower().endswith(suffix), f"Unexpected {media_kind} suffix")
    stem = basename[:-len(suffix)]
    if media_kind == "audio" and stem.endswith("_mic"):
        stem = stem[:-4]
    match = IDENTITY_RE.fullmatch(stem)
    require(match is not None, f"Unparseable GuitarSet {media_kind} identity: {basename}")
    score = match.group("score")
    require(score and not score.startswith("_") and not score.endswith("_"),
            f"Invalid GuitarSet score identity: {basename}")
    return {
        "item_id": stem,
        "player_id": match.group("player"),
        "score_id": score,
        "performance": match.group("performance"),
    }


def validate_media_join(audio_rows, annotation_rows, expected_players=EXPECTED_PLAYERS,
                        expected_score_count=EXPECTED_SCORE_COUNT):
    """Require the exact player x score x performance cross-product and one-to-one join."""
    expected_players = tuple(expected_players)
    require(expected_players and len(set(expected_players)) == len(expected_players),
            "Invalid expected player set")
    require(type(expected_score_count) is int and expected_score_count > 0,
            "Invalid expected score count")

    def index(rows, kind):
        result = {}
        for row in rows:
            identity = parse_media_identity(row["path"], kind)
            item_id = identity["item_id"]
            require(item_id not in result, f"Duplicate {kind} identity: {item_id}")
            result[item_id] = {"identity": identity, "member": row}
        return result

    audio = index(audio_rows, "audio")
    annotations = index(annotation_rows, "annotation")
    expected_count = len(expected_players) * expected_score_count * len(PERFORMANCES)
    require(len(audio) == expected_count, f"Expected exactly {expected_count} real microphone WAV files")
    require(len(annotations) == expected_count, f"Expected exactly {expected_count} real JAMS files")
    require(set(audio) == set(annotations), "Missing audio/JAMS identity or non-exact join")
    identities = [audio[key]["identity"] for key in sorted(audio)]
    require({row["player_id"] for row in identities} == set(expected_players),
            "Unexpected GuitarSet player set")
    scores = {row["score_id"] for row in identities}
    require(len(scores) == expected_score_count, "Unexpected GuitarSet score count")
    actual_crossproduct = {
        (row["player_id"], row["score_id"], row["performance"]) for row in identities
    }
    expected_crossproduct = {
        (player, score, performance)
        for player in expected_players for score in scores for performance in PERFORMANCES
    }
    require(actual_crossproduct == expected_crossproduct,
            "Incomplete player x score x comp/solo cross-product")
    player_counts = Counter(row["player_id"] for row in identities)
    score_counts = Counter(row["score_id"] for row in identities)
    per_player = expected_score_count * len(PERFORMANCES)
    per_score = len(expected_players) * len(PERFORMANCES)
    require(set(player_counts.values()) == {per_player},
            f"Each player must have exactly {per_player} rows")
    require(set(score_counts.values()) == {per_score},
            f"Each score must have exactly {per_score} rows")
    pairs = []
    for item_id in sorted(audio):
        require(audio[item_id]["identity"] == annotations[item_id]["identity"],
                f"Audio/JAMS parsed identity mismatch: {item_id}")
        pairs.append({**audio[item_id]["identity"],
                      "audio_member": audio[item_id]["member"],
                      "annotation_member": annotations[item_id]["member"]})
    return pairs


def selected_media(inventories):
    audio_rows = [row for row in inventories["audio_mono-mic.zip"]
                  if not row["is_directory"] and not row["apple_metadata"]
                  and row["path"].lower().endswith(".wav")]
    annotation_rows = [row for row in inventories["annotation.zip"]
                       if not row["is_directory"] and not row["apple_metadata"]
                       and row["path"].lower().endswith(".jams")]
    return audio_rows, annotation_rows


def materialize_member(archive, member, destination):
    """Materialize one already-inventoried regular member without extractall()."""
    destination = safe_filesystem_path(destination)
    require(not destination.exists(), f"Materialized destination already exists: {destination}")
    pending = destination.with_name("." + destination.name + ".pending")
    require(not pending.exists(), f"Materialization staging conflict: {pending}")
    sha, count = hashlib.sha256(), 0
    with archive.open(member["path"], "r") as incoming, pending.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(READ_BYTES), b""):
            count += len(block)
            require(count <= member["bytes"], "Materialized member exceeded inventoried size")
            sha.update(block)
            outgoing.write(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    require(count == member["bytes"] and sha.hexdigest() == member["sha256"],
            "Materialized member bytes/SHA-256 mismatch")
    pending.rename(destination)
    return {"bytes": count, "sha256": sha.hexdigest()}


def parse_jams(path):
    document = strict_json_bytes(regular_file(path).read_bytes())
    duration = document.get("file_metadata", {}).get("duration")
    require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0,
            "Invalid JAMS file_metadata.duration")
    annotations = document.get("annotations")
    require(isinstance(annotations, list), "Invalid JAMS annotations list")
    annotation_counts = Counter()
    observation_counts = Counter()
    for annotation in annotations:
        require(isinstance(annotation, dict) and isinstance(annotation.get("namespace"), str)
                and annotation["namespace"], "Invalid JAMS namespace")
        data = annotation.get("data")
        require(isinstance(data, list), "Invalid JAMS annotation data")
        annotation_counts[annotation["namespace"]] += 1
        observation_counts[annotation["namespace"]] += len(data)
    return {
        "file_metadata_duration_seconds": float(duration),
        "namespace_annotation_counts": dict(sorted(annotation_counts.items())),
        "namespace_observation_counts": dict(sorted(observation_counts.items())),
    }


def decode_audio(path):
    """Decode float64 blocks until an actual empty read and hash canonical PCM bytes."""
    import numpy as np
    import soundfile as sf

    before = file_fingerprint(path, include_md5=False)
    pcm_sha = hashlib.sha256()
    actual_frames = calls = 0
    with sf.SoundFile(str(path), mode="r") as decoder:
        sample_rate = int(decoder.samplerate)
        channels = int(decoder.channels)
        header_frames = int(decoder.frames)
        require(sample_rate > 0 and header_frames >= 0, "Invalid audio decoder header")
        require(channels == 1, f"Unexpected non-mono microphone audio: {channels} channels")
        decoder_format = str(decoder.format)
        decoder_subtype = str(decoder.subtype)
        while True:
            block = decoder.read(DECODE_FRAMES, dtype="float64", always_2d=True)
            calls += 1
            require(isinstance(block, np.ndarray) and block.dtype == np.dtype("float64")
                    and block.ndim == 2 and block.shape[1] == channels
                    and 0 <= block.shape[0] <= DECODE_FRAMES,
                    "Unexpected SoundFile float64 decoder output")
            if block.shape[0] == 0:
                break
            require(np.isfinite(block).all(), "Nonfinite decoded PCM sample")
            actual_frames += int(block.shape[0])
            pcm_sha.update(np.asarray(block, dtype="<f8", order="C").tobytes(order="C"))
    after = file_fingerprint(path, include_md5=False)
    require(after == before, "Materialized audio changed during decode")
    require(actual_frames > 0, "Empty decoded audio")
    require(actual_frames == header_frames,
            "Decoded frame count differs from SoundFile header frame count")
    return {
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "subtype": decoder_subtype,
        "format": decoder_format,
        "header_frames": header_frames,
        "decoded_frames": actual_frames,
        "duration_seconds": actual_frames / sample_rate,
        "float64_samples_checked_finite": actual_frames * channels,
        "nonfinite_samples": 0,
        "read_calls_including_empty_eof": calls,
        "empty_eof_observed": True,
        "decode_block_frames": DECODE_FRAMES,
        "decoded_pcm_sha256": pcm_sha.hexdigest(),
        "decoded_pcm_canonical_encoding": PCM_ENCODING,
        "materialized_file": before,
    }


def runtime_info():
    import numpy as np
    import soundfile as sf
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "soundfile": sf.__version__,
        "libsndfile": sf.__libsndfile_version__,
        "decoder": "SoundFile/libsndfile float64 sequential reads to real EOF",
    }


def write_new(path, data):
    path = safe_filesystem_path(path)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def output_products(output):
    products = {}
    for path in sorted(output.rglob("*")):
        safe_filesystem_path(path)
        if path.is_dir():
            continue
        relative = path.relative_to(output).as_posix()
        if relative == "COMMIT.json":
            continue
        products[relative] = file_fingerprint(path, include_md5=False)
    return products


def _counter_add(total, values):
    for key, value in values.items():
        total[key] += value


def run(source, commit_sha256, output):
    source = safe_filesystem_path(source)
    output = safe_filesystem_path(output)
    require(not output.exists(), "Output must be a new exclusive folder")
    require(output.parent.is_dir(), "Output parent directory must already exist")
    require(output != source and output not in source.parents and source not in output.parents,
            "Source and output folders must be disjoint")
    output.mkdir(mode=0o755)
    try:
        validator_code = file_fingerprint(CODE_PATH, include_md5=False)
        evidence = verify_source(source, commit_sha256)
        inventories = {}
        for archive_name in EXPECTED_ARCHIVES:
            print(json.dumps({"event": "zip_full_crc_scan_started", "archive": archive_name}),
                  flush=True)
            inventories[archive_name] = scan_zip(source / archive_name, archive_name)
            print(json.dumps({"event": "zip_full_crc_scan_completed", "archive": archive_name,
                              "members": len(inventories[archive_name])}), flush=True)
        audio_rows, annotation_rows = selected_media(inventories)
        pairs = validate_media_join(audio_rows, annotation_rows,
                                    EXPECTED_PLAYERS, EXPECTED_SCORE_COUNT)

        (output / "audio").mkdir()
        (output / "annotations").mkdir()
        (output / "source").mkdir()
        write_new(output / "source" / "upstream_metadata.json", evidence["metadata_bytes"])
        write_new(output / "source" / "acquisition_COMMIT.json", evidence["commit_bytes"])
        inventory_document = {
            "version": VERSION,
            "all_members_read_fully": True,
            "all_members_crc_verified": True,
            "safety_limits": {
                "max_members_per_archive": MAX_MEMBERS_PER_ARCHIVE,
                "max_member_uncompressed_bytes": MAX_MEMBER_BYTES,
                "max_total_uncompressed_bytes_per_archive": MAX_TOTAL_UNCOMPRESSED_BYTES,
            },
            "archives": inventories,
        }
        write_new(output / "archive_member_inventory.json", canonical_json(inventory_document))

        records = []
        namespace_annotations, namespace_observations = Counter(), Counter()
        sample_rates, channels, subtypes = Counter(), Counter(), Counter()
        total_audio_duration = total_jams_duration = 0.0
        total_frames = 0
        audio_archive_path = source / "audio_mono-mic.zip"
        annotation_archive_path = source / "annotation.zip"
        with zipfile.ZipFile(audio_archive_path) as audio_archive, \
                zipfile.ZipFile(annotation_archive_path) as annotation_archive:
            for index, pair in enumerate(pairs, 1):
                audio_name = pair["audio_member"]["basename"]
                annotation_name = pair["annotation_member"]["basename"]
                audio_relative = f"audio/{audio_name}"
                annotation_relative = f"annotations/{annotation_name}"
                annotation_file = output / annotation_relative
                audio_file = output / audio_relative
                annotation_hash = materialize_member(
                    annotation_archive, pair["annotation_member"], annotation_file)
                audio_hash = materialize_member(audio_archive, pair["audio_member"], audio_file)
                jams = parse_jams(annotation_file)
                decoded = decode_audio(audio_file)
                require(audio_hash == decoded["materialized_file"],
                        "Audio materialization/decode fingerprint disagreement")
                _counter_add(namespace_annotations, jams["namespace_annotation_counts"])
                _counter_add(namespace_observations, jams["namespace_observation_counts"])
                sample_rates[str(decoded["sample_rate_hz"])] += 1
                channels[str(decoded["channels"])] += 1
                subtypes[decoded["subtype"]] += 1
                total_frames += decoded["decoded_frames"]
                total_audio_duration += decoded["duration_seconds"]
                total_jams_duration += jams["file_metadata_duration_seconds"]
                records.append({
                    "version": VERSION,
                    "item_id": pair["item_id"],
                    "player_id": pair["player_id"],
                    "score_id": pair["score_id"],
                    "performance": pair["performance"],
                    "class_label": None,
                    "role": "external_measurement_control_only",
                    "annotation_is_ground_truth_for_nonlinear_coupling": False,
                    "known_annotation_warnings": KNOWN_ANNOTATION_WARNINGS.get(pair["item_id"], []),
                    "audio": {
                        "archive": "audio_mono-mic.zip",
                        "archive_member": pair["audio_member"]["path"],
                        "archive_member_sha256": pair["audio_member"]["sha256"],
                        "materialized_path": audio_relative,
                        "materialized": audio_hash,
                        "decoded": decoded,
                    },
                    "annotation": {
                        "archive": "annotation.zip",
                        "archive_member": pair["annotation_member"]["path"],
                        "archive_member_sha256": pair["annotation_member"]["sha256"],
                        "materialized_path": annotation_relative,
                        "materialized": annotation_hash,
                        "jams": jams,
                    },
                })
                if index % 30 == 0 or index == len(pairs):
                    print(json.dumps({"event": "media_pairs_decoded", "completed": index,
                                      "total": len(pairs)}), flush=True)

        manifest_bytes = b"".join(canonical_json(record) for record in records)
        write_new(output / "source_manifest.jsonl", manifest_bytes)
        copied_metadata = file_fingerprint(output / "source" / "upstream_metadata.json", False)
        copied_commit = file_fingerprint(output / "source" / "acquisition_COMMIT.json", False)
        require(copied_metadata == evidence["metadata_fingerprint"]
                and copied_commit == evidence["commit_fingerprint"],
                "Copied source evidence bytes changed")
        verify_source_unchanged(evidence)
        require(file_fingerprint(CODE_PATH, include_md5=False) == validator_code,
                "Validator code changed during execution")
        player_file_counts = Counter(record["player_id"] for record in records)
        score_file_counts = Counter(record["score_id"] for record in records)
        summary = {
            "status": "passed_archive_crc_materialization_and_physical_decode_not_measurement_admission",
            "version": VERSION,
            "role": "external_measurement_control_only",
            "class_label": None,
            "source": {
                "folder": str(source),
                "acquisition_commit_sha256": evidence["commit_fingerprint"]["sha256"],
                "official_metadata_sha256": evidence["metadata_fingerprint"]["sha256"],
                "doi": "10.5281/zenodo.3371780",
                "version": "1.1.0",
                "license": "CC-BY-4.0",
                "copied_acquisition_commit": copied_commit,
                "copied_official_metadata": copied_metadata,
                "archives": evidence["archive_fingerprints"],
            },
            "counts": {
                "archive_members_by_archive": {key: len(value) for key, value in inventories.items()},
                "apple_metadata_members_by_archive": {
                    key: sum(row["apple_metadata"] for row in value)
                    for key, value in inventories.items()
                },
                "real_microphone_wav": len(audio_rows),
                "real_jams": len(annotation_rows),
                "exact_audio_annotation_pairs": len(records),
                "players": len({record["player_id"] for record in records}),
                "scores": len({record["score_id"] for record in records}),
                "files_per_player": dict(sorted(player_file_counts.items())),
                "files_per_score": dict(sorted(score_file_counts.items())),
                "performances": dict(sorted(Counter(record["performance"] for record in records).items())),
            },
            "audio": {
                "total_decoded_frames": total_frames,
                "total_duration_seconds": total_audio_duration,
                "sample_rate_file_counts": dict(sorted(sample_rates.items())),
                "channel_file_counts": dict(sorted(channels.items())),
                "subtype_file_counts": dict(sorted(subtypes.items())),
                "canonical_pcm_encoding": PCM_ENCODING,
                "all_files_fully_decoded_float64_finite_to_empty_eof": True,
                "resampled": False,
                "mixed": False,
                "features_extracted": False,
            },
            "annotations": {
                "total_file_metadata_duration_seconds": total_jams_duration,
                "namespace_annotation_counts": dict(sorted(namespace_annotations.items())),
                "namespace_observation_counts": dict(sorted(namespace_observations.items())),
                "namespace_count_not_assumed": True,
                "known_warning_items": KNOWN_ANNOTATION_WARNINGS,
                "silently_corrected_or_excluded_warning_items": 0,
                "ground_truth_for_nonlinear_coupling": False,
            },
            "runtime": runtime_info(),
            "validator_code": validator_code,
            "claims": {
                "overlap_clean": False,
                "overlap_audited": False,
                "bc_extracted": False,
                "partition_frozen": False,
                "classifier_fits": 0,
                "external_measurement_gate_passed": False,
                "audio_annotations_are_exact_identity_matched": True,
            },
            "source_manifest": "source_manifest.jsonl",
            "archive_member_inventory": "archive_member_inventory.json",
        }
        write_new(output / "validation_summary.json", canonical_json(summary))
        products = output_products(output)
        commit = {
            "status": "committed",
            "kind": "guitarset_archive_validation_materialization_v1",
            "products": products,
            "source_commit_sha256": commit_sha256,
            "classifier_fits": 0,
            "external_measurement_gate_passed": False,
        }
        write_new(output / "COMMIT.json", canonical_json(commit))
        print(json.dumps({"event": "validation_committed", "output": str(output),
                          "pairs": len(records), "products": len(products)}), flush=True)
        return summary
    except BaseException as error:
        failure = {
            "status": "failed_not_committed",
            "error_type": type(error).__name__,
            "error": str(error),
            "source": str(source),
            "requested_source_commit_sha256": commit_sha256,
            "output": str(output),
            "commit_published": False,
            "partial_files_preserved_for_audit": True,
        }
        failure_path = output / "EXECUTION_FAILURE.json"
        if not failure_path.exists():
            try:
                write_new(failure_path, canonical_json(failure))
            except BaseException:
                pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="Successful committed GuitarSet archive-acquisition folder")
    parser.add_argument("--commit-sha256", required=True,
                        help="Expected SHA-256 of the acquisition COMMIT.json bytes")
    parser.add_argument("--output", required=True, type=Path,
                        help="New exclusive materialization folder; existing paths are refused")
    args = parser.parse_args()
    run(args.source, args.commit_sha256, args.output)


if __name__ == "__main__":
    main()
