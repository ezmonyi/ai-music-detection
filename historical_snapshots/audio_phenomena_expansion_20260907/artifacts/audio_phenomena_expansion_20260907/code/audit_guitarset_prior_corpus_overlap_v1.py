#!/usr/bin/env python3
"""Scoped, read-only GuitarSet overlap evidence audit.

This program compares a trusted GuitarSet v1 materialization with an explicit
set of prior classifier and external-control manifests.  It does not decode
prior audio, perform approximate matching, make an admission decision, or
establish global non-overlap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


VERSION = "guitarset_prior_corpus_overlap_v1"
F64_REPRESENTATION = "ieee754_float64_le_c_frame_channel_no_header"
F32_REPRESENTATION = "ieee754_float32_le_c_frame_channel_no_header"
PCM16_REPRESENTATION = "signed_pcm16_wav_frame_bytes"
TRUSTED_GUITARSET_KINDS = {
    "guitarset_archive_validation_materialization_v1",
    "guitarset_archive_validation_materialization_v2",
}
MAX_MANIFEST_BYTES = 256 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 8 * 1024 * 1024
MAX_GUITARSET_PRODUCTS = 2000
HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_COUNTS = {
    "master_10s": 10141,
    "master_30s": 4497,
    "materialization_2641": 2641,
    "v6_metadata": 2174,
    "equal60_lineage": 1604,
    "mureka": 500,
    "saraga": 103,
    "nsynth": 4096,
    "vocalset_pairs": 90,
    "mir1k": 100,
    "musdb": 24,
    "timbre_pairs": 52,
    "timbre_chunks": 153,
}

CSV_REQUIRED = {
    "master_10s": ("id", "track", "source_id", "source_group", "group_id",
                   "artist_or_creator", "raw_sha256", "source_audio_path", "source_locator"),
    "master_30s": ("id", "track", "source_id", "source_group", "group_id",
                   "artist_or_creator", "raw_sha256", "source_audio_path", "source_locator"),
    "v6_metadata": ("id", "source_group", "group_id", "native_sample_rate_hz"),
    "equal60_lineage": ("item_id", "source_id", "group_id", "standardized_path",
                        "standardized_sr", "standardized_channels", "standardized_frames",
                        "standardized_file_sha256", "source_sample_rate", "source_channels",
                        "source_total_frames", "source_audio_path", "source_audio_sha256",
                        "standardized_waveform_sha256"),
    "mureka": ("item_id", "source_id", "source_group", "group_id", "source_audio_path",
               "standardized_path",
               "source_audio_sha256", "source_sample_rate", "source_channels",
               "sf_actual_read_frames", "sf_sequential_float64_sha256",
               "sf_sequential_float32_sha256", "crop_start_frame", "crop_frames",
               "native_crop_float64_sha256", "native_crop_float32_sha256",
               "standardized_sr", "standardized_channels", "standardized_frames",
               "standardized_waveform_sha256", "standardized_file_sha256"),
    "saraga": ("id", "source_id", "source_group", "group_id", "audio_path",
               "registered_raw_sha256", "source_audio_sha256", "physical_frames",
               "physical_sample_rate_hz", "physical_channels", "crop_start_frame",
               "crop_frames", "native_crop_float64_sha256", "native_crop_float32_sha256"),
    "vocalset_pairs": ("pair_id", "singer", "content_id", "straight_filename",
                       "straight_audio_path", "straight_audio_sha256", "straight_archive_member",
                       "straight_sample_rate_hz", "straight_frames", "vibrato_filename",
                       "vibrato_audio_path", "vibrato_audio_sha256", "vibrato_archive_member",
                       "vibrato_sample_rate_hz", "vibrato_frames"),
    "mir1k": ("stem", "singer", "song_id", "clip_id", "wav_remote_path", "wav_local_path",
              "wav_sha256", "sample_rate_hz", "channels", "audio_frames"),
    "timbre_pairs": ("pair_id", "corpus", "instrument_group", "anchor_source_id",
                     "peer_source_id", "anchor_source_path", "peer_source_path",
                     "anchor_source_sha256", "peer_source_sha256"),
    "timbre_chunks": ("corpus", "source_id", "source_path", "track_name",
                      "instrument_group", "source_sha256", "native_sr_hz",
                      "native_duration_sec"),
}

JSONL_REQUIRED = {
    "materialization_2641": ("item_id", "source_id", "group_id", "title", "artist_or_creator",
                             "container_member", "source_locator", "source_path", "native_path",
                             "native_sha256", "view_10s_sha256", "view_max60s_sha256"),
    "nsynth": ("id", "path", "file_sha256", "pcm_sha256", "native_sample_rate",
               "native_channels", "frames"),
    "musdb": ("track_id", "track_name", "dataset", "source_container", "mixture",
              "mixture_sha256", "sample_rate", "channels"),
}

INPUT_ARGUMENTS = tuple(EXPECTED_COUNTS)


class AuditError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n").encode("utf-8")


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(data, label):
    try:
        return json.loads(data, object_pairs_hook=strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"Invalid JSON in {label}: {error}") from error


def regular_file(path):
    path = Path(path)
    require(path.exists(), f"Missing required path: {path}")
    require(not path.is_symlink(), f"Symlink is not accepted: {path}")
    mode = path.stat().st_mode
    require(stat.S_ISREG(mode), f"Not a regular file: {path}")
    require(path.stat().st_size <= MAX_MANIFEST_BYTES, f"Input file exceeds bound: {path}")
    return path


def fingerprint(path):
    path = regular_file(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"bytes": size, "sha256": digest.hexdigest()}


def sha256_value(value, label, allow_empty=False):
    value = str(value or "").strip().lower()
    if allow_empty and not value:
        return None
    require(bool(HEX64.fullmatch(value)), f"Invalid SHA-256 in {label}")
    return value


def integer(value, label, minimum=0):
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise AuditError(f"Invalid integer in {label}: {value!r}") from error
    require(result >= minimum, f"Out-of-range integer in {label}: {result}")
    return result


def read_csv(path, adapter):
    path = regular_file(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames is not None, f"Missing CSV header: {path}")
            require(len(reader.fieldnames) == len(set(reader.fieldnames)),
                    f"Duplicate CSV header: {path}")
            missing = sorted(set(CSV_REQUIRED[adapter]) - set(reader.fieldnames))
            require(not missing, f"{adapter} missing required fields: {missing}")
            rows = list(reader)
            for line_number, row in enumerate(rows, 2):
                require(None not in row, f"Malformed extra CSV fields: {path}:{line_number}")
                require(all(row.get(key) is not None for key in CSV_REQUIRED[adapter]),
                        f"Missing CSV values: {path}:{line_number}")
    except UnicodeDecodeError as error:
        raise AuditError(f"Invalid UTF-8 CSV: {path}") from error
    return rows, list(reader.fieldnames)


def read_jsonl(path, adapter):
    path = regular_file(path)
    rows = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            require(len(raw) <= MAX_JSONL_LINE_BYTES,
                    f"Oversized JSONL line {line_number}: {path}")
            require(raw.strip(), f"Blank JSONL line {line_number}: {path}")
            row = strict_json(raw.decode("utf-8"), f"{path}:{line_number}")
            require(isinstance(row, dict), f"JSONL row is not an object: {path}:{line_number}")
            missing = sorted(set(JSONL_REQUIRED[adapter]) - set(row))
            require(not missing, f"{adapter} missing required fields at row {line_number}: {missing}")
            rows.append(row)
    return rows, sorted(set().union(*(row.keys() for row in rows))) if rows else []


def normalized_aliases(value):
    if value is None:
        return set()
    raw = unicodedata.normalize("NFKC", str(value)).strip().lower().replace("\\", "/")
    if not raw:
        return set()
    candidates = {raw, PurePosixPath(raw).name}
    name = PurePosixPath(raw).name
    if "." in name:
        candidates.add(name.rsplit(".", 1)[0])
    result = set()
    for candidate in candidates:
        normalized = re.sub(r"[^a-z0-9]+", "_", candidate).strip("_")
        normalized = re.sub(r"_mic$", "", normalized)
        if len(normalized) >= 6:
            result.add(normalized)
    return result


def path_entity_keys(value):
    value = unicodedata.normalize("NFKC", str(value or "")).strip().replace("\\", "/")
    return {value} if value else set()


def identifier_records(values):
    records = []
    for field, value in values:
        for alias in sorted(normalized_aliases(value)):
            records.append({"field": field, "value": str(value), "alias": alias})
    return records


def encoded(field, digest, semantic):
    digest = sha256_value(digest, field)
    return {"field": field, "sha256": digest, "semantic": semantic}


def pcm(field, digest, representation, semantic, sample_rate, channels, frames):
    return {
        "field": field,
        "sha256": sha256_value(digest, field),
        "representation": representation,
        "semantic": semantic,
        "sample_rate_hz": integer(sample_rate, field + ".sample_rate", 1),
        "channels": integer(channels, field + ".channels", 1),
        "frames": integer(frames, field + ".frames", 1),
    }


def logical(adapter, row_number, item_id, identifiers, encoded_hashes=(), pcm_hashes=(),
            entity_keys=()):
    require(str(item_id).strip(), f"{adapter} row {row_number} has empty item identifier")
    # Only the row's stable item identity participates in entity accounting.
    # Descriptive fields such as source_id, artist, or corpus must not collapse
    # every recording in a collection into one entity.
    keys = {"id:" + alias for alias in normalized_aliases(item_id)}
    keys.update("sha256:" + value["sha256"] for value in encoded_hashes
                if value["semantic"] in ("native_source_file", "materialized_source_file"))
    keys.update("path:" + alias for alias in entity_keys if alias)
    return {
        "adapter": adapter,
        "row_number": row_number,
        "item_id": str(item_id),
        "identifiers": identifier_records(identifiers),
        "encoded_hashes": list(encoded_hashes),
        "pcm_hashes": list(pcm_hashes),
        "entity_keys": sorted(keys),
    }


def adapt_csv(adapter, rows):
    out = []
    for n, row in enumerate(rows, 2):
        if adapter in ("master_10s", "master_30s"):
            ids = [(key, row[key]) for key in ("id", "track", "source_id", "source_group",
                                                "group_id", "artist_or_creator",
                                                "source_audio_path", "source_locator")]
            out.append(logical(adapter, n, row["id"], ids,
                               [encoded("raw_sha256", row["raw_sha256"], "native_source_file")],
                               entity_keys=path_entity_keys(row["source_audio_path"])))
        elif adapter == "v6_metadata":
            ids = [(key, row[key]) for key in ("id", "source_group", "group_id")]
            out.append(logical(adapter, n, row["id"], ids))
        elif adapter == "equal60_lineage":
            ids = [(key, row[key]) for key in ("item_id", "source_id", "group_id",
                                                "source_audio_path", "standardized_path")]
            enc = [encoded("source_audio_sha256", row["source_audio_sha256"], "native_source_file"),
                   encoded("standardized_file_sha256", row["standardized_file_sha256"],
                           "standardized_file")]
            pcms = [pcm("standardized_waveform_sha256", row["standardized_waveform_sha256"],
                        F32_REPRESENTATION, "standardized_60s_crop",
                        row["standardized_sr"], row["standardized_channels"],
                        row["standardized_frames"])]
            out.append(logical(adapter, n, row["item_id"], ids, enc, pcms,
                               path_entity_keys(row["source_audio_path"])))
        elif adapter == "mureka":
            ids = [(key, row[key]) for key in ("item_id", "source_id", "source_group", "group_id",
                                                "source_audio_path", "standardized_path")]
            enc = [encoded("source_audio_sha256", row["source_audio_sha256"], "native_source_file"),
                   encoded("standardized_file_sha256", row["standardized_file_sha256"],
                           "standardized_file")]
            pcms = [
                pcm("sf_sequential_float64_sha256", row["sf_sequential_float64_sha256"],
                    F64_REPRESENTATION, "full_source", row["source_sample_rate"],
                    row["source_channels"], row["sf_actual_read_frames"]),
                pcm("sf_sequential_float32_sha256", row["sf_sequential_float32_sha256"],
                    F32_REPRESENTATION, "full_source", row["source_sample_rate"],
                    row["source_channels"], row["sf_actual_read_frames"]),
                pcm("native_crop_float64_sha256", row["native_crop_float64_sha256"],
                    F64_REPRESENTATION, "native_crop", row["source_sample_rate"],
                    row["source_channels"], row["crop_frames"]),
                pcm("native_crop_float32_sha256", row["native_crop_float32_sha256"],
                    F32_REPRESENTATION, "native_crop", row["source_sample_rate"],
                    row["source_channels"], row["crop_frames"]),
                pcm("standardized_waveform_sha256", row["standardized_waveform_sha256"],
                    F32_REPRESENTATION, "standardized_60s_crop", row["standardized_sr"],
                    row["standardized_channels"], row["standardized_frames"]),
            ]
            out.append(logical(adapter, n, row["item_id"], ids, enc, pcms,
                               path_entity_keys(row["source_audio_path"])))
        elif adapter == "saraga":
            ids = [(key, row[key]) for key in ("id", "source_id", "source_group", "group_id",
                                                "audio_path")]
            require(row["registered_raw_sha256"] == row["source_audio_sha256"],
                    f"saraga row {n} registered/source SHA mismatch")
            enc = [encoded("source_audio_sha256", row["source_audio_sha256"], "native_source_file")]
            pcms = [
                pcm("native_crop_float64_sha256", row["native_crop_float64_sha256"],
                    F64_REPRESENTATION, "native_crop", row["physical_sample_rate_hz"],
                    row["physical_channels"], row["crop_frames"]),
                pcm("native_crop_float32_sha256", row["native_crop_float32_sha256"],
                    F32_REPRESENTATION, "native_crop", row["physical_sample_rate_hz"],
                    row["physical_channels"], row["crop_frames"]),
            ]
            out.append(logical(adapter, n, row["id"], ids, enc, pcms,
                               path_entity_keys(row["audio_path"])))
        elif adapter == "vocalset_pairs":
            for side in ("straight", "vibrato"):
                ids = [("pair_id", row["pair_id"]), ("singer", row["singer"]),
                       ("content_id", row["content_id"]), (side + "_filename", row[side + "_filename"]),
                       (side + "_audio_path", row[side + "_audio_path"]),
                       (side + "_archive_member", row[side + "_archive_member"])]
                out.append(logical(adapter, n, row["pair_id"] + ":" + side, ids,
                                   [encoded(side + "_audio_sha256", row[side + "_audio_sha256"],
                                            "native_source_file")],
                                   entity_keys=path_entity_keys(row[side + "_audio_path"])))
        elif adapter == "mir1k":
            ids = [(key, row[key]) for key in ("stem", "singer", "song_id", "clip_id",
                                                "wav_remote_path", "wav_local_path")]
            out.append(logical(adapter, n, row["stem"], ids,
                               [encoded("wav_sha256", row["wav_sha256"], "native_source_file")],
                               entity_keys=path_entity_keys(row["wav_local_path"])))
        elif adapter == "timbre_pairs":
            for side in ("anchor", "peer"):
                ids = [("pair_id", row["pair_id"]), ("corpus", row["corpus"]),
                       ("instrument_group", row["instrument_group"]),
                       (side + "_source_id", row[side + "_source_id"]),
                       (side + "_source_path", row[side + "_source_path"])]
                out.append(logical(adapter, n, row["pair_id"] + ":" + side, ids,
                                   [encoded(side + "_source_sha256", row[side + "_source_sha256"],
                                            "native_source_file")],
                                   entity_keys=path_entity_keys(row[side + "_source_path"])))
        elif adapter == "timbre_chunks":
            ids = [(key, row[key]) for key in ("source_id", "source_path", "track_name",
                                                "corpus", "instrument_group")]
            out.append(logical(adapter, n, row["source_id"], ids,
                               [encoded("source_sha256", row["source_sha256"], "native_source_file")],
                               entity_keys=path_entity_keys(row["source_path"])))
        else:
            raise AuditError(f"Unsupported CSV adapter: {adapter}")
    return out


def adapt_jsonl(adapter, rows):
    out = []
    for n, row in enumerate(rows, 1):
        if adapter == "materialization_2641":
            ids = [(key, row[key]) for key in ("item_id", "source_id", "group_id", "title",
                                                "artist_or_creator", "container_member",
                                                "source_locator", "source_path", "native_path")]
            enc = [encoded("native_sha256", row["native_sha256"], "native_source_file"),
                   encoded("view_10s_sha256", row["view_10s_sha256"], "derived_view_file"),
                   encoded("view_max60s_sha256", row["view_max60s_sha256"], "derived_view_file")]
            out.append(logical(adapter, n, row["item_id"], ids, enc,
                               entity_keys=path_entity_keys(row["native_path"])))
        elif adapter == "nsynth":
            ids = [("id", row["id"]), ("path", row["path"])]
            out.append(logical(adapter, n, row["id"], ids,
                               [encoded("file_sha256", row["file_sha256"], "native_source_file")],
                               [pcm("pcm_sha256", row["pcm_sha256"], PCM16_REPRESENTATION,
                                    "full_source", row["native_sample_rate"],
                                    row["native_channels"], row["frames"])],
                               path_entity_keys(row["path"])))
        elif adapter == "musdb":
            ids = [(key, row[key]) for key in ("track_id", "track_name", "dataset",
                                                "source_container", "mixture")]
            out.append(logical(adapter, n, row["track_id"], ids,
                               [encoded("mixture_sha256", row["mixture_sha256"],
                                        "materialized_source_file")],
                               entity_keys=path_entity_keys(row["mixture"])))
        else:
            raise AuditError(f"Unsupported JSONL adapter: {adapter}")
    return out


def adapt_saraga_proofs(root, saraga_rows):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), f"Invalid Saraga proof root: {root}")
    evidence = []
    inventory = {}
    expected_paths = set()
    for n, row in enumerate(saraga_rows, 2):
        item_id = row["id"]
        proof_path = root / "items" / item_id / "proof.json"
        expected_paths.add(proof_path.resolve())
        fp = fingerprint(proof_path)
        inventory[str(proof_path)] = fp
        proof = strict_json(proof_path.read_text(encoding="utf-8"), str(proof_path))
        require(isinstance(proof, dict) and proof.get("item_id") == item_id,
                f"Saraga proof item mismatch: {proof_path}")
        interval = proof.get("interval_proof")
        require(isinstance(interval, dict), f"Missing interval_proof: {proof_path}")
        required = ("whole_pcm_sha256", "whole_pcm_encoding", "observed_actual_frames",
                    "native_sample_rate_hz", "native_channels", "native_float64_sha256")
        missing = sorted(set(required) - set(interval))
        require(not missing, f"Saraga proof missing fields {missing}: {proof_path}")
        require(interval["whole_pcm_encoding"] ==
                "IEEE754 float64 little-endian; C order [frame, channel]; no header",
                f"Unexpected Saraga whole PCM encoding: {proof_path}")
        require(sha256_value(interval["native_float64_sha256"], "native_float64_sha256") ==
                sha256_value(row["native_crop_float64_sha256"], "native_crop_float64_sha256"),
                f"Saraga proof/metadata crop SHA mismatch: {proof_path}")
        require(integer(interval["native_sample_rate_hz"], "proof sample rate", 1) ==
                integer(row["physical_sample_rate_hz"], "metadata sample rate", 1),
                f"Saraga proof/metadata sample-rate mismatch: {proof_path}")
        require(integer(interval["native_channels"], "proof channels", 1) ==
                integer(row["physical_channels"], "metadata channels", 1),
                f"Saraga proof/metadata channel mismatch: {proof_path}")
        evidence.append({
            "adapter": "saraga_proof",
            "row_number": n,
            "item_id": item_id,
            "identifiers": identifier_records([("item_id", item_id)]),
            "encoded_hashes": [],
            "pcm_hashes": [pcm("whole_pcm_sha256", interval["whole_pcm_sha256"],
                                F64_REPRESENTATION, "full_source",
                                interval["native_sample_rate_hz"], interval["native_channels"],
                                interval["observed_actual_frames"])],
            "entity_keys": ["id:" + alias for alias in normalized_aliases(item_id)],
        })
    actual = {p.resolve() for p in root.glob("items/*/proof.json") if p.is_file()}
    require(actual == expected_paths,
            f"Saraga proof inventory mismatch: expected {len(expected_paths)}, found {len(actual)}")
    return evidence, inventory


def verify_guitarset(folder, expected_commit_sha256, expected_count=360,
                     expected_players=6, expected_scores=30):
    folder = Path(folder)
    require(folder.is_dir() and not folder.is_symlink(), f"Invalid GuitarSet materialization: {folder}")
    commit_path = regular_file(folder / "COMMIT.json")
    commit_fp = fingerprint(commit_path)
    require(commit_fp["sha256"] == sha256_value(expected_commit_sha256, "GuitarSet COMMIT SHA"),
            "GuitarSet COMMIT SHA-256 mismatch")
    commit = strict_json(commit_path.read_text(encoding="utf-8"), str(commit_path))
    require(commit.get("status") == "committed", "GuitarSet source is not committed")
    require(commit.get("kind") in TRUSTED_GUITARSET_KINDS,
            "Unexpected GuitarSet COMMIT kind")
    products = commit.get("products")
    require(isinstance(products, dict) and products, "Invalid GuitarSet product inventory")
    require(len(products) <= MAX_GUITARSET_PRODUCTS, "Too many GuitarSet products")
    actual_paths = {}
    for path in folder.rglob("*"):
        require(not path.is_symlink(), f"Symlink in GuitarSet products: {path}")
        if path.is_file() and path.name != "COMMIT.json":
            relative = path.relative_to(folder).as_posix()
            actual_paths[relative] = path
    require(set(actual_paths) == set(products), "GuitarSet COMMIT product inventory mismatch")
    product_fingerprints = {}
    for relative, path in sorted(actual_paths.items()):
        require(not PurePosixPath(relative).is_absolute() and ".." not in PurePosixPath(relative).parts,
                f"Unsafe GuitarSet product path: {relative}")
        fp = fingerprint(path)
        expected = products[relative]
        require(isinstance(expected, dict) and expected.get("bytes") == fp["bytes"]
                and expected.get("sha256") == fp["sha256"],
                f"GuitarSet product mismatch: {relative}")
        product_fingerprints[relative] = fp
    require("source_manifest.jsonl" in products and "validation_summary.json" in products,
            "GuitarSet required products absent")
    summary = strict_json((folder / "validation_summary.json").read_text(encoding="utf-8"),
                          str(folder / "validation_summary.json"))
    require(isinstance(summary, dict)
            and str(summary.get("status", "")).startswith("passed_archive_crc_materialization")
            and summary.get("counts", {}).get("exact_audio_annotation_pairs") == expected_count,
            "GuitarSet validation summary does not bind the expected passed pair count")
    manifest_path = folder / "source_manifest.jsonl"
    rows = []
    with manifest_path.open("rb") as handle:
        for n, raw in enumerate(handle, 1):
            require(raw.strip() and len(raw) <= MAX_JSONL_LINE_BYTES,
                    f"Invalid GuitarSet manifest line {n}")
            row = strict_json(raw.decode("utf-8"), f"{manifest_path}:{n}")
            require(isinstance(row, dict), f"Invalid GuitarSet row {n}")
            rows.append(row)
    require(len(rows) == expected_count, f"Expected {expected_count} GuitarSet rows, found {len(rows)}")
    tuples = set()
    item_ids, audio_paths, annotation_paths = set(), set(), set()
    items = []
    players, scores = set(), set()
    for n, row in enumerate(rows, 1):
        required = ("item_id", "player_id", "score_id", "performance", "audio", "annotation")
        missing = sorted(set(required) - set(row))
        require(not missing, f"GuitarSet row {n} missing fields: {missing}")
        item_id = str(row["item_id"])
        player = str(row["player_id"])
        score = str(row["score_id"])
        performance = str(row["performance"])
        require(performance in ("comp", "solo"), f"Invalid GuitarSet performance: {performance}")
        require(item_id == f"{player}_{score}_{performance}",
                f"GuitarSet item/tuple identity mismatch row {n}")
        require(item_id not in item_ids, f"Duplicate GuitarSet item_id: {item_id}")
        item_ids.add(item_id)
        key = (player, score, performance)
        require(key not in tuples, f"Duplicate GuitarSet tuple: {key}")
        tuples.add(key); players.add(player); scores.add(score)
        audio = row["audio"]
        decoded = audio.get("decoded") if isinstance(audio, dict) else None
        materialized = audio.get("materialized") if isinstance(audio, dict) else None
        require(isinstance(decoded, dict) and isinstance(materialized, dict),
                f"Invalid GuitarSet audio evidence row {n}")
        require(decoded.get("decoded_pcm_canonical_encoding") ==
                "IEEE754 float64 little-endian; C order [frame, channel]; no header",
                f"Unexpected GuitarSet PCM encoding row {n}")
        require(integer(decoded.get("channels"), "GuitarSet channels", 1) == 1,
                f"Non-mono GuitarSet row {n}")
        materialized_path = str(audio.get("materialized_path", ""))
        require(materialized_path in products, f"Uncommitted GuitarSet audio path row {n}")
        require(materialized_path not in audio_paths,
                f"Duplicate GuitarSet materialized audio path row {n}")
        audio_paths.add(materialized_path)
        file_sha = sha256_value(materialized.get("sha256"), "GuitarSet materialized SHA")
        require(products[materialized_path]["sha256"] == file_sha
                and products[materialized_path]["bytes"] ==
                integer(materialized.get("bytes"), "GuitarSet materialized bytes"),
                f"GuitarSet row/product audio SHA mismatch row {n}")
        require(sha256_value(audio.get("archive_member_sha256"),
                             "GuitarSet archive member SHA") == file_sha,
                f"GuitarSet archive/materialized audio SHA mismatch row {n}")
        annotation = row["annotation"]
        require(isinstance(annotation, dict) and isinstance(annotation.get("materialized"), dict),
                f"Invalid GuitarSet annotation evidence row {n}")
        annotation_path = str(annotation.get("materialized_path", ""))
        require(annotation_path in products, f"Uncommitted GuitarSet annotation path row {n}")
        require(annotation_path not in annotation_paths,
                f"Duplicate GuitarSet materialized annotation path row {n}")
        annotation_paths.add(annotation_path)
        annotation_sha = sha256_value(annotation["materialized"].get("sha256"),
                                      "GuitarSet annotation SHA")
        require(products[annotation_path]["sha256"] == annotation_sha
                and products[annotation_path]["bytes"] ==
                integer(annotation["materialized"].get("bytes"), "GuitarSet annotation bytes"),
                f"GuitarSet row/product annotation SHA mismatch row {n}")
        require(sha256_value(annotation.get("archive_member_sha256"),
                             "GuitarSet annotation archive member SHA") == annotation_sha,
                f"GuitarSet archive/materialized annotation SHA mismatch row {n}")
        identifiers = [("item_id", item_id), ("score_id", score),
                       ("archive_member", audio.get("archive_member", "")),
                       ("materialized_path", materialized_path)]
        items.append(logical(
            "guitarset", n, item_id, identifiers,
            [encoded("materialized.sha256", file_sha, "materialized_source_file"),
             encoded("archive_member_sha256", audio.get("archive_member_sha256"),
                     "archive_member_file")],
            [pcm("decoded_pcm_sha256", decoded.get("decoded_pcm_sha256"),
                 F64_REPRESENTATION, "full_source", decoded.get("sample_rate_hz"),
                 decoded.get("channels"), decoded.get("decoded_frames"))],
            path_entity_keys(materialized_path)))
    require(len(players) == expected_players and len(scores) == expected_scores
            and len(tuples) == expected_count,
            "GuitarSet player x score x 2-performance tuple inventory mismatch")
    require(tuples == {(p, s, perf) for p in players for s in scores for perf in ("comp", "solo")},
            "GuitarSet tuple inventory is not the complete cross-product")
    declared_source_identifiers = [{"field": "dataset_name", "value": "GuitarSet"}]
    source_summary = summary.get("source", {})
    if isinstance(source_summary, dict) and source_summary.get("doi"):
        declared_source_identifiers.append({"field": "doi", "value": str(source_summary["doi"])})
    return items, {"folder": str(folder), "commit": commit_fp, "commit_kind": commit["kind"],
                   "product_count": len(products), "products_verified": True,
                   "manifest_rows": len(rows), "tuple_cross_product_verified": True,
                   "declared_source_identifiers": declared_source_identifiers}, product_fingerprints


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def deduplicated_entity_count(items):
    uf = UnionFind(len(items))
    owner = {}
    for index, item in enumerate(items):
        keys = item["entity_keys"] or [f"fallback:{item['adapter']}:{item['row_number']}:{item['item_id']}"]
        for key in keys:
            if key in owner:
                uf.union(index, owner[key])
            else:
                owner[key] = index
    return len({uf.find(index) for index in range(len(items))})


def compact_evidence(item, evidence):
    return {"adapter": item["adapter"], "row_number": item["row_number"],
            "item_id": item["item_id"], **evidence}


def compare(guitarset, prior, declared_source_identifiers=()):
    alias_index = defaultdict(list)
    encoded_index = defaultdict(list)
    pcm_digest_index = defaultdict(list)
    for item in prior:
        for evidence in item["identifiers"]:
            alias_index[evidence["alias"]].append(compact_evidence(item, evidence))
        for evidence in item["encoded_hashes"]:
            encoded_index[evidence["sha256"]].append(compact_evidence(item, evidence))
        for evidence in item["pcm_hashes"]:
            pcm_digest_index[evidence["sha256"]].append(compact_evidence(item, evidence))

    identifier_candidates, encoded_matches, pcm_matches, incompatible = [], [], [], []
    for target_source in declared_source_identifiers:
        for alias in normalized_aliases(target_source["value"]):
            for candidate in alias_index.get(alias, []):
                identifier_candidates.append({
                    "guitarset_item_id": None,
                    "target": {**target_source, "alias": alias},
                    "prior": candidate,
                    "candidate_scope": "declared_dataset_or_upstream_source",
                })
    for target in guitarset:
        for target_id in target["identifiers"]:
            for candidate in alias_index.get(target_id["alias"], []):
                identifier_candidates.append({"guitarset_item_id": target["item_id"],
                                              "target": target_id, "prior": candidate})
        for target_hash in target["encoded_hashes"]:
            for candidate in encoded_index.get(target_hash["sha256"], []):
                encoded_matches.append({"guitarset_item_id": target["item_id"],
                                        "target": target_hash, "prior": candidate})
        for target_pcm in target["pcm_hashes"]:
            for candidate in pcm_digest_index.get(target_pcm["sha256"], []):
                same_rep = candidate["representation"] == target_pcm["representation"]
                same_tuple = all(candidate[key] == target_pcm[key]
                                 for key in ("sample_rate_hz", "channels", "frames"))
                record = {"guitarset_item_id": target["item_id"], "target": target_pcm,
                          "prior": candidate}
                if same_rep and same_tuple:
                    pcm_matches.append(record)
                else:
                    record["reason"] = "digest_equal_but_representation_or_shape_tuple_incompatible"
                    incompatible.append(record)
    key = lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"))
    return {
        "identifier_candidates": sorted(identifier_candidates, key=key),
        "exact_encoded_file_matches": sorted(encoded_matches, key=key),
        "exact_compatible_float64_pcm_matches": sorted(pcm_matches, key=key),
        "incompatible_pcm_digest_coincidences": sorted(incompatible, key=key),
    }


def load_inputs(paths, expected_counts):
    prior = []
    inventory = {}
    rows_by_adapter = {}
    schemas = {}
    for adapter in INPUT_ARGUMENTS:
        path = Path(paths[adapter])
        fp = fingerprint(path)
        if adapter in CSV_REQUIRED:
            rows, schema = read_csv(path, adapter)
            adapted = adapt_csv(adapter, rows)
        elif adapter in JSONL_REQUIRED:
            rows, schema = read_jsonl(path, adapter)
            adapted = adapt_jsonl(adapter, rows)
        else:
            raise AuditError(f"No versioned adapter for {adapter}")
        require(len(rows) == expected_counts[adapter],
                f"{adapter} expected {expected_counts[adapter]} rows, found {len(rows)}")
        rows_by_adapter[adapter] = rows
        prior.extend(adapted)
        schemas[adapter] = schema
        inventory[adapter] = {"path": str(path), **fp, "rows": len(rows),
                              "logical_audio_entries": len(adapted),
                              "adapter_version": adapter + "_schema_v1",
                              "observed_fields": schema}
    proof_evidence, proof_inventory = adapt_saraga_proofs(paths["saraga_proofs"],
                                                         rows_by_adapter["saraga"])
    prior.extend(proof_evidence)
    return prior, inventory, proof_inventory


def evidence_summary(items):
    by_adapter = defaultdict(lambda: Counter(logical_audio_entries=0, encoded_hash_entries=0,
                                             compatible_f64_pcm_entries=0,
                                             full_source_f64_pcm_entries=0,
                                             native_crop_f64_pcm_entries=0,
                                             logical_entries_with_compatible_f64=0,
                                             incompatible_pcm_entries=0))
    for item in items:
        row = by_adapter[item["adapter"]]
        row["logical_audio_entries"] += 1
        row["encoded_hash_entries"] += len(item["encoded_hashes"])
        has_f64 = False
        for entry in item["pcm_hashes"]:
            if entry["representation"] == F64_REPRESENTATION:
                row["compatible_f64_pcm_entries"] += 1
                row[entry["semantic"] + "_f64_pcm_entries"] += 1
                has_f64 = True
            else:
                row["incompatible_pcm_entries"] += 1
        if has_f64:
            row["logical_entries_with_compatible_f64"] += 1
    result = {}
    for key, value in sorted(by_adapter.items()):
        value["logical_entries_without_compatible_f64"] = (
            value["logical_audio_entries"] - value["logical_entries_with_compatible_f64"])
        result[key] = dict(value)
    return result


def write_new(path, data):
    with Path(path).open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def run(guitarset, guitarset_commit_sha256, inputs, output, expected_counts=None,
        _expected_guitarset_shape=(6, 30)):
    expected_counts = dict(EXPECTED_COUNTS if expected_counts is None else expected_counts)
    require(set(expected_counts) == set(EXPECTED_COUNTS), "Expected-count override keys mismatch")
    require(set(inputs) == set(INPUT_ARGUMENTS) | {"saraga_proofs"}, "Input path keys mismatch")
    output = Path(output)
    require(not output.exists(), "Output must be a new exclusive folder")
    require(output.parent.is_dir(), "Output parent must exist")
    output.mkdir()
    try:
        code_fp = fingerprint(Path(__file__).resolve())
        expected_players, expected_scores = _expected_guitarset_shape
        guitarset_items, guitarset_binding, guitarset_products = verify_guitarset(
            guitarset, guitarset_commit_sha256,
            expected_count=expected_players * expected_scores * 2,
            expected_players=expected_players, expected_scores=expected_scores)
        prior, input_inventory, proof_inventory = load_inputs(inputs, expected_counts)
        comparisons = compare(guitarset_items, prior,
                              guitarset_binding["declared_source_identifiers"])
        initial_fingerprints = {key: {"bytes": value["bytes"], "sha256": value["sha256"]}
                                for key, value in input_inventory.items()}
        report = {
            "version": VERSION,
            "status": "scoped_evidence_report_requires_parent_adjudication",
            "scope": {
                "declared_identifier_candidates": True,
                "exact_encoded_file_hashes": True,
                "available_canonical_float64_full_and_crop_hashes": True,
                "decoded_other_prior_sources": False,
                "transformed_overlap_coverage": 0,
                "approximate_or_perceptual_matching": False,
                "composition_level_overlap": False,
            },
            "guitarset": guitarset_binding,
            "inputs": input_inventory,
            "saraga_proofs": {"root": str(inputs["saraga_proofs"]),
                               "count": len(proof_inventory), "files": proof_inventory},
            "accounting": {
                "manifest_row_views": sum(value["rows"] for value in input_inventory.values()),
                "logical_audio_evidence_entries": len(prior),
                "deduplicated_connected_source_entities": deduplicated_entity_count(prior),
                "multiplicity_preserved_in_match_lists": True,
                "row_views_are_not_claimed_as_unique_recordings": True,
                "by_adapter": evidence_summary(prior),
            },
            "comparisons": comparisons,
            "candidate_counts": {key: len(value) for key, value in comparisons.items()},
            "limitations": [
                "No prior-corpus audio was decoded in this scoped stage.",
                "Full native decoded-float64 coverage is limited to fields explicitly supplied by Mureka and Saraga proofs/metadata.",
                "NSynth PCM hashes are signed PCM16 frame bytes and Equal60 hashes are transformed float32; neither is comparable to GuitarSet float64 hashes.",
                "Encoded-file hash inequality cannot exclude equal audio across containers, headers, codecs, gains, crops, resampling, mixing, or channel selection.",
                "Identifier nonmatches cannot establish audio, performance, composition, or dataset non-overlap.",
                "Transformed, cropped, codec-robust, approximate, and perceptual overlap coverage is zero.",
                "Prior external-control use is disclosed reuse, not classifier train/test leakage unless the same or related performance entered a fitted cohort.",
            ],
            "claims": {
                "global_non_overlap_established": False,
                "transformed_novelty_established": False,
                "independent_reviewer_acceptance": False,
                "admission_decision_made": False,
                "bc_extracted": False,
                "classifier_fits": 0,
                "parent_adjudication_required": True,
            },
            "auditor_code": code_fp,
        }
        write_new(output / "overlap_report.json", canonical_json(report))

        # Rebind every input and source product immediately before COMMIT publication.
        for key, value in input_inventory.items():
            require(fingerprint(value["path"]) == initial_fingerprints[key],
                    f"Input changed during audit: {key}")
        for path, fp in proof_inventory.items():
            require(fingerprint(path) == fp, f"Saraga proof changed during audit: {path}")
        for relative, fp in guitarset_products.items():
            require(fingerprint(Path(guitarset) / relative) == fp,
                    f"GuitarSet product changed during audit: {relative}")
        require(fingerprint(Path(guitarset) / "COMMIT.json") == guitarset_binding["commit"],
                "GuitarSet COMMIT changed during audit")
        require(fingerprint(Path(__file__).resolve()) == code_fp, "Auditor code changed during audit")
        report_fp = fingerprint(output / "overlap_report.json")
        commit = {
            "status": "committed",
            "kind": VERSION,
            "products": {"overlap_report.json": report_fp},
            "guitarset_commit_sha256": guitarset_binding["commit"]["sha256"],
            "independent_reviewer_acceptance": False,
            "admission_decision_made": False,
            "global_non_overlap_established": False,
            "transformed_overlap_coverage": 0,
        }
        write_new(output / "COMMIT.json", canonical_json(commit))
        return report
    except BaseException as error:
        failure = {"status": "failed_not_committed", "error_type": type(error).__name__,
                   "error": str(error), "commit_published": False,
                   "partial_output_preserved": True}
        failure_path = output / "EXECUTION_FAILURE.json"
        if not failure_path.exists():
            try:
                write_new(failure_path, canonical_json(failure))
            except BaseException:
                pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guitarset", required=True, type=Path)
    parser.add_argument("--guitarset-commit-sha256", required=True)
    for name in INPUT_ARGUMENTS:
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--saraga-proofs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {name: getattr(args, name) for name in INPUT_ARGUMENTS}
    paths["saraga_proofs"] = args.saraga_proofs
    run(args.guitarset, args.guitarset_commit_sha256, paths, args.output)


if __name__ == "__main__":
    main()
