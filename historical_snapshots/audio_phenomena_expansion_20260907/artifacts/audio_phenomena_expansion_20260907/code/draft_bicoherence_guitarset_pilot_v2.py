#!/usr/bin/env python3
"""Draft-only GuitarSet partition/preprocessing provenance; never extracts BC.

An independently accepted overlap JSON is mandatory. Its exact schema is:
  schema: "bc_guitarset_overlap_acceptance_v1"
  status: "accepted_for_development_split_only"
  source_materialization_commit_sha256, source_manifest_sha256: bound digests
  report: {path, bytes, sha256} of the reviewed evidence document
  independent_review: {accepted: true, reviewer_id: nonempty string,
                       reviewed_report_sha256: same digest as report}
  corpora: nonempty list of {corpus_id, manifest: {path, bytes, sha256},
           rows_checked: positive integer, representations: nonempty string list,
           checks: {source_lineage: true, identifiers: true,
                    exact_file_hashes: bool, canonical_pcm_hashes: bool},
           limitations: nonempty string list}
  unresolved_matches: []
  global_non_overlap_proven: false

The caller supplies the trusted acceptance SHA. This tool checks that assertion
and all referenced file bindings; it cannot independently establish the reviewer's
identity or the completeness of their overlap investigation. Unsupported exact
hash comparisons must be described in limitations. No global non-overlap claim.

The only CLI action is drafting. It verifies source integrity, decodes and
standardizes the 90 development recordings, and binds crop SHA-256 values. It
does not save derived audio, build injections, import BC extractors, or measure
reserved/unused BC. Future measurement still requires a separate reviewed tool.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import platform
import re
import sys

import numpy as np
import scipy
from scipy.signal import resample_poly

import validate_guitarset_archives_v2 as source_validator

VERSION = "draft_bicoherence_guitarset_pilot_v2"
DESIGN_SHA = "3626b35d6d6de1b445e42fbb0c9bca33a3bee213995584b4d857eba26af15f08"
EXPECTED_PLAYERS = tuple(f"{i:02d}" for i in range(6))
VALIDATOR_SHA = "cda25ca63e382a1363ade813cd114838099bee55ee60f346888f2b1cd2ff6d6c"
SCORE_COUNT = 30
TARGET_RATE, CROP_SAMPLES = 16000, 128000
PCM_ENCODING = source_validator.PCM_ENCODING
require = source_validator.require
canonical = source_validator.canonical_json
strict_json = source_validator.strict_json_bytes
write_new = source_validator.write_new


def fp(path):
    return source_validator.file_fingerprint(path, include_md5=False)


def binding(path):
    path = source_validator.regular_file(path)
    return {"path": str(path), **fp(path)}


def check_sha(value, label):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"Invalid {label} SHA-256")


def read_json(path):
    return strict_json(source_validator.regular_file(path).read_bytes())


def child(root, relative):
    source_validator.safe_member_name(relative)
    path = source_validator.regular_file(root / relative)
    require(path.is_relative_to(root), "Materialization path escapes root")
    return path


def validate_decoded_metadata(row):
    d = row["audio"]["decoded"]
    for name in ("sample_rate_hz", "decoded_frames", "header_frames"):
        require(type(d.get(name)) is int and d[name] > 0, f"Invalid positive decoded {name}")
    require(d.get("channels") == 1 and type(d.get("channels")) is int,
            "Native microphone recording must be mono")
    require(d["decoded_frames"] == d["header_frames"]
            and d.get("float64_samples_checked_finite") == d["decoded_frames"]
            and d.get("nonfinite_samples") == 0 and d.get("empty_eof_observed") is True
            and d.get("decoded_pcm_canonical_encoding") == PCM_ENCODING,
            "Invalid full finite decode scope")
    duration = d.get("duration_seconds")
    require(type(duration) in (float, int) and math.isfinite(duration)
            and math.isclose(duration, d["decoded_frames"] / d["sample_rate_hz"], rel_tol=0, abs_tol=1e-12),
            "Decoded duration disagrees with native frames/rate")
    check_sha(d.get("decoded_pcm_sha256"), "canonical PCM")
    up = TARGET_RATE // math.gcd(d["sample_rate_hz"], TARGET_RATE)
    down = d["sample_rate_hz"] // math.gcd(d["sample_rate_hz"], TARGET_RATE)
    predicted = (d["decoded_frames"] * up + down - 1) // down
    require(predicted >= CROP_SAMPLES, f"Short source fails draft: {row['item_id']}")
    return predicted


def partition(rows):
    """Pure fixed crossed split with all roles and cross-role PCM checks."""
    require(isinstance(rows, list) and len(rows) == len(EXPECTED_PLAYERS) * SCORE_COUNT * 2,
            "Source must contain the complete player/score/performance grid")
    ids, combinations = set(), set()
    for row in rows:
        item = row["item_id"]
        require(item not in ids, "Duplicate item ID")
        ids.add(item)
        player, score, performance = row["player_id"], row["score_id"], row["performance"]
        require(item == f"{player}_{score}_{performance}", "Item identity disagrees with source metadata")
        require(performance in ("comp", "solo"), "Invalid performance label")
        triple = (player, score, performance)
        require(triple not in combinations, "Duplicate player/score/performance")
        combinations.add(triple)
        validate_decoded_metadata(row)
    players, scores = {r["player_id"] for r in rows}, {r["score_id"] for r in rows}
    require(players == set(EXPECTED_PLAYERS) and len(scores) == SCORE_COUNT,
            "Unexpected player or score inventory")
    require(combinations == {(p, s, k) for p in players for s in scores for k in ("comp", "solo")},
            "Incomplete crossed source inventory")
    ordered_players = sorted(players, key=lambda s: (hashlib.sha256(
        ("BC-GuitarSet-player-20260907|" + s).encode()).hexdigest(), s))
    ordered_scores = sorted(scores, key=lambda s: (hashlib.sha256(
        ("BC-GuitarSet-score-20260907|" + s).encode()).hexdigest(), s))
    dev_players, dev_scores = set(ordered_players[:3]), set(ordered_scores[:15])
    roles, counts = [], Counter()
    pcm = {"development": set(), "reserved": set(), "unused": set()}
    styles = {role: Counter() for role in pcm}
    for row in sorted(rows, key=lambda r: r["item_id"]):
        pd, sd = row["player_id"] in dev_players, row["score_id"] in dev_scores
        role = "development" if pd and sd else "reserved" if not pd and not sd else "unused"
        counts[role] += 1
        pcm[role].add(row["audio"]["decoded"]["decoded_pcm_sha256"])
        match = re.match(r"[A-Za-z]+", row["score_id"])
        style = match.group() if match else row["score_id"].split("-")[0]
        styles[role][style] += 1
        roles.append({"item_id": row["item_id"], "player_id": row["player_id"],
                      "score_id": row["score_id"], "performance": row["performance"],
                      "style_from_score_prefix": style, "split_role": role, "source_record": row})
    require(dict(counts) == {"development": 90, "reserved": 90, "unused": 180},
            "Crossed split count mismatch")
    require(not pcm["development"] & pcm["reserved"],
            "Canonical PCM duplicate crosses development/reserved; explicit review required")
    return {"ordered_player_ids": ordered_players, "ordered_score_ids": ordered_scores,
            "development_player_ids": ordered_players[:3], "reserved_player_ids": ordered_players[3:],
            "development_score_ids": ordered_scores[:15], "reserved_score_ids": ordered_scores[15:],
            "role_counts": dict(counts), "style_recording_counts_by_role": {
                role: dict(sorted(counter.items())) for role, counter in styles.items()},
            "canonical_pcm_disjoint_development_reserved": True,
            "independent_performers_claimed": False, "rows": roles}


def verify_materialization(root, trusted_commit_sha, validator_path):
    """Verify the real validator publication; no archive-acquisition-only fallback."""
    check_sha(trusted_commit_sha, "trusted materialization COMMIT")
    root = source_validator.safe_filesystem_path(root)
    require(root.is_dir(), "Materialization source directory missing")
    inventory = source_validator._source_inventory(root)
    require("COMMIT.json" in inventory and fp(inventory["COMMIT.json"])["sha256"] == trusted_commit_sha,
            "Trusted materialization COMMIT mismatch")
    commit = read_json(inventory["COMMIT.json"])
    require(set(commit) == {"status", "kind", "products", "source_commit_sha256", "classifier_fits",
                           "external_measurement_gate_passed"}
            and commit["status"] == "committed"
            and commit["kind"] == "guitarset_archive_validation_materialization_v1"
            and commit["classifier_fits"] == 0 and commit["external_measurement_gate_passed"] is False,
            "Wrong materialization COMMIT schema/scope")
    products = commit["products"]
    require(isinstance(products, dict) and set(inventory) == set(products) | {"COMMIT.json"},
            "Materialization exact product inventory mismatch")
    actual_products = {}
    for name, expected in sorted(products.items()):
        require(isinstance(expected, dict) and set(expected) == {"bytes", "sha256"}
                and type(expected["bytes"]) is int and expected["bytes"] >= 0,
                "Invalid materialization product binding")
        check_sha(expected["sha256"], "product")
        actual_products[name] = fp(child(root, name))
        require(actual_products[name] == expected, f"Materialization product hash mismatch: {name}")
    required = {"source_manifest.jsonl", "validation_summary.json", "archive_member_inventory.json",
                "source/upstream_metadata.json", "source/acquisition_COMMIT.json"}
    require(required <= set(products), "Missing required materialization provenance products")
    summary = read_json(inventory["validation_summary.json"])
    require(summary.get("status") == "passed_archive_crc_materialization_and_physical_decode_not_measurement_admission"
            and summary.get("role") == "external_measurement_control_only" and summary.get("class_label") is None,
            "Wrong source validation scope")
    require(summary.get("validator_code") == fp(validator_path), "Source validator code binding mismatch")
    require(summary.get("source_manifest") == "source_manifest.jsonl"
            and summary.get("archive_member_inventory") == "archive_member_inventory.json",
            "Unexpected source evidence paths")
    claims = summary.get("claims", {})
    require(claims == {"overlap_clean": False, "overlap_audited": False, "bc_extracted": False,
                      "partition_frozen": False, "classifier_fits": 0,
                      "external_measurement_gate_passed": False,
                      "audio_annotations_are_exact_identity_matched": True}, "Invalid source claims boundary")
    s = summary.get("source", {})
    require(s.get("doi") == "10.5281/zenodo.3371780" and s.get("version") == "1.1.0"
            and s.get("license") == "CC-BY-4.0", "Wrong GuitarSet release or license")
    require(s.get("official_metadata_sha256") == source_validator.METADATA_SHA256
            and actual_products["source/upstream_metadata.json"]["sha256"] == source_validator.METADATA_SHA256
            and s.get("copied_official_metadata") == actual_products["source/upstream_metadata.json"]
            and s.get("copied_acquisition_commit") == actual_products["source/acquisition_COMMIT.json"]
            and s.get("acquisition_commit_sha256") == commit["source_commit_sha256"]
            == actual_products["source/acquisition_COMMIT.json"]["sha256"], "Source lineage binding mismatch")
    acquisition = read_json(inventory["source/acquisition_COMMIT.json"])
    require(acquisition.get("status") == "committed" and acquisition.get("kind") == "archive_acquisition_only",
            "Invalid copied acquisition COMMIT")
    for name, expected in source_validator.EXPECTED_ARCHIVES.items():
        archived = s.get("archives", {}).get(name, {})
        require(archived.get("bytes") == expected["bytes"] and archived.get("md5") == expected["md5"],
                "Official archive size/MD5 not established by source validator")
        check_sha(archived.get("sha256"), "source archive")
        require(acquisition.get("products", {}).get(name) == {k: archived[k] for k in ("bytes", "sha256")},
                "Archive SHA binding differs from acquisition")
    archive_inventory = read_json(inventory["archive_member_inventory.json"])
    require(archive_inventory.get("all_members_read_fully") is True
            and archive_inventory.get("all_members_crc_verified") is True
            and set(archive_inventory.get("archives", {})) == set(source_validator.EXPECTED_ARCHIVES),
            "Full sequential archive CRC evidence missing")
    rows = [strict_json(line) for line in inventory["source_manifest.jsonl"].read_bytes().splitlines() if line.strip()]
    split = partition(rows)
    audio_names, annotation_names = set(), set()
    for row in rows:
        require(row.get("role") == "external_measurement_control_only" and row.get("class_label") is None
                and row.get("annotation_is_ground_truth_for_nonlinear_coupling") is False,
                "Source row scope mismatch")
        for kind, archive in (("audio", "audio_mono-mic.zip"), ("annotation", "annotation.zip")):
            entry = row[kind]
            name = entry["materialized_path"]
            identity = {key: row[key] for key in ("item_id", "player_id", "score_id", "performance")}
            require(source_validator.parse_media_identity(name, kind) == identity
                    and source_validator.parse_media_identity(entry["archive_member"], kind) == identity,
                    "Materialized/archive filename identity disagrees with source row")
            (audio_names if kind == "audio" else annotation_names).add(name)
            require(name in products and entry["materialized"] == actual_products[name]
                    and entry["archive"] == archive
                    and entry["archive_member_sha256"] == entry["materialized"]["sha256"],
                    "Source row materialized/archive hash mismatch")
            matching = [m for m in archive_inventory["archives"][archive]
                        if m["path"] == entry["archive_member"]]
            require(len(matching) == 1 and matching[0]["sha256"] == entry["archive_member_sha256"]
                    and matching[0]["bytes"] == entry["materialized"]["bytes"]
                    and matching[0].get("crc_verified_by_full_read") is True
                    and matching[0]["is_directory"] is False and matching[0]["apple_metadata"] is False,
                    "Source row does not match CRC-scanned archive member")
        require(row["audio"]["decoded"]["materialized_file"] == row["audio"]["materialized"],
                "Decoded file binding mismatch")
    require(len(audio_names) == len(annotation_names) == 360 and
            set(products) == required | audio_names | annotation_names,
            "Materialized media inventory is not exact 360 audio/JAMS pairs")
    counts = summary.get("counts", {})
    for key, value in (("real_microphone_wav", 360), ("real_jams", 360),
                       ("exact_audio_annotation_pairs", 360), ("players", 6), ("scores", 30)):
        require(counts.get(key) == value, "Validation summary source count mismatch")
    require(counts.get("files_per_player") == dict(Counter(r["player_id"] for r in rows))
            and counts.get("files_per_score") == dict(Counter(r["score_id"] for r in rows))
            and counts.get("performances") == {"comp": 180, "solo": 180}, "Source crossed counts mismatch")
    audio_summary = summary.get("audio", {})
    require(audio_summary.get("all_files_fully_decoded_float64_finite_to_empty_eof") is True
            and audio_summary.get("canonical_pcm_encoding") == PCM_ENCODING
            and all(audio_summary.get(k) is False for k in ("resampled", "mixed", "features_extracted"))
            and audio_summary.get("channel_file_counts") == {"1": 360}
            and audio_summary.get("total_decoded_frames") == sum(r["audio"]["decoded"]["decoded_frames"] for r in rows),
            "Incomplete native finite decode summary")
    require(audio_summary.get("sample_rate_file_counts") == dict(Counter(
        str(r["audio"]["decoded"]["sample_rate_hz"]) for r in rows)), "Native-rate summary mismatch")
    annotations = summary.get("annotations", {})
    require(annotations.get("silently_corrected_or_excluded_warning_items") == 0
            and annotations.get("ground_truth_for_nonlinear_coupling") is False,
            "Annotation warning/ground-truth boundary mismatch")
    return {"root": str(root), "commit": binding(inventory["COMMIT.json"]), "products": actual_products,
            "source_validator": binding(validator_path)}, split


def verify_overlap(path, trusted_sha, source):
    check_sha(trusted_sha, "trusted overlap acceptance")
    evidence_binding = binding(path)
    require(evidence_binding["sha256"] == trusted_sha, "Trusted overlap acceptance SHA mismatch")
    evidence = read_json(path)
    require(set(evidence) == {"schema", "status", "source_materialization_commit_sha256", "source_manifest_sha256",
            "report", "independent_review", "corpora", "unresolved_matches", "global_non_overlap_proven"},
            "Invalid overlap acceptance schema")
    require(evidence["schema"] == "bc_guitarset_overlap_acceptance_v1"
            and evidence["status"] == "accepted_for_development_split_only"
            and evidence["source_materialization_commit_sha256"] == source["commit"]["sha256"]
            and evidence["source_manifest_sha256"] == source["products"]["source_manifest.jsonl"]["sha256"],
            "Overlap acceptance source or scope mismatch")
    require(evidence["unresolved_matches"] == [] and evidence["global_non_overlap_proven"] is False,
            "Unresolved matches or unsupported global overlap claim")
    report = evidence["report"]
    require(binding(report["path"]) == report, "Overlap report file binding mismatch")
    review = evidence["independent_review"]
    require(set(review) == {"accepted", "reviewer_id", "reviewed_report_sha256"}
            and review["accepted"] is True and isinstance(review["reviewer_id"], str)
            and review["reviewer_id"].strip() and review["reviewed_report_sha256"] == report["sha256"],
            "Independent overlap acceptance is required")
    corpora = evidence["corpora"]
    require(isinstance(corpora, list) and corpora, "Covered overlap corpora must be enumerated")
    names = set()
    for corpus in corpora:
        require(set(corpus) == {"corpus_id", "manifest", "rows_checked", "representations", "checks", "limitations"},
                "Invalid covered-corpus schema")
        name = corpus["corpus_id"]
        require(isinstance(name, str) and name.strip() and name not in names, "Duplicate/invalid covered corpus")
        names.add(name)
        require(binding(corpus["manifest"]["path"]) == corpus["manifest"], "Covered manifest hash mismatch")
        require(type(corpus["rows_checked"]) is int and corpus["rows_checked"] > 0, "Invalid covered row count")
        for key in ("representations", "limitations"):
            require(isinstance(corpus[key], list) and corpus[key] and
                    all(isinstance(s, str) and s.strip() for s in corpus[key]), "Overlap scope/limitations missing")
        checks = corpus["checks"]
        require(set(checks) == {"source_lineage", "identifiers", "exact_file_hashes", "canonical_pcm_hashes"}
                and checks["source_lineage"] is True and checks["identifiers"] is True
                and all(type(value) is bool for value in checks.values()), "Overlap comparison scope invalid")
    return {"acceptance": evidence_binding, "evidence": evidence,
            "independent_review_identity_verified_by_this_tool": False}


def standardize(samples, native_rate):
    """Pure full-record resample then floor-centered crop; no amplitude changes."""
    require(isinstance(samples, np.ndarray) and samples.dtype == np.float64 and samples.ndim == 1
            and samples.size > 0 and np.isfinite(samples).all(), "Expected nonempty finite mono float64 samples")
    require(type(native_rate) is int and native_rate > 0, "Native rate must be a positive integer")
    factor = math.gcd(native_rate, TARGET_RATE)
    up, down = TARGET_RATE // factor, native_rate // factor
    full = samples if native_rate == TARGET_RATE else resample_poly(
        samples, up, down, window=("kaiser", 5.0), padtype="constant", cval=0.0)
    expected_length = (len(samples) * up + down - 1) // down
    require(len(full) == expected_length and np.isfinite(full).all(), "Unexpected resampled length or values")
    require(len(full) >= CROP_SAMPLES, "Short resampled source fails draft; no padding")
    start = (len(full) - CROP_SAMPLES) // 2
    crop = full[start:start + CROP_SAMPLES].copy()
    crop_bytes = np.asarray(crop, dtype="<f8", order="C").tobytes()
    provenance = {"native_sample_rate_hz": native_rate, "native_frames": int(len(samples)),
                  "target_sample_rate_hz": TARGET_RATE, "resampled_frames": int(len(full)),
                  "resampling_applied": native_rate != TARGET_RATE, "up": up, "down": down,
                  "window": ["kaiser", 5.0], "padtype": "constant", "cval": 0.0,
                  "resample_scope": "entire_native_recording_before_crop",
                  "native_16khz_policy": "unchanged_array", "crop_start": start,
                  "crop_stop_exclusive": start + CROP_SAMPLES, "crop_frames": CROP_SAMPLES,
                  "crop_center_rule": "floor((resampled_frames - 128000)/2)",
                  "crop_pcm_encoding": PCM_ENCODING, "crop_pcm_sha256": hashlib.sha256(crop_bytes).hexdigest(),
                  "crop_zero_amplitude": not np.any(crop),
                  "measurement_support": "unsupported_zero_background_rms" if not np.any(crop) else "not_measured",
                  "derived_audio_saved": False, "bc_extracted": False}
    return crop, provenance


def decode_selected(path, decoded, decoder_factory=None):
    """Full selected PCM decode/hash to actual EOF; injectable factory is for tests."""
    if decoder_factory is None:
        import soundfile as sf
        decoder_factory = sf.SoundFile
    before = fp(path)
    chunks = []
    with decoder_factory(str(path), mode="r") as decoder:
        require(int(decoder.channels) == 1 and int(decoder.samplerate) == decoded["sample_rate_hz"]
                and int(decoder.frames) == decoded["decoded_frames"]
                and str(decoder.format) == decoded["format"] and str(decoder.subtype) == decoded["subtype"],
                "Selected decoder header differs from verified source metadata")
        while True:
            block = decoder.read(65536, dtype="float64", always_2d=True)
            require(isinstance(block, np.ndarray) and block.dtype == np.float64
                    and block.ndim == 2 and block.shape[1] == 1 and len(block) <= 65536
                    and np.isfinite(block).all(), "Invalid selected float64 decode block")
            if not len(block):
                break
            chunks.append(block[:, 0].copy())
    samples = np.concatenate(chunks) if chunks else np.empty(0, np.float64)
    require(len(samples) == decoded["decoded_frames"] and len(samples) > 0, "Selected decoded length mismatch")
    digest = hashlib.sha256(np.asarray(samples, dtype="<f8", order="C").tobytes()).hexdigest()
    require(digest == decoded["decoded_pcm_sha256"], "Selected canonical PCM SHA mismatch")
    require(fp(path) == before == decoded["materialized_file"], "Selected source changed while decoding")
    return samples


def libsndfile_from_maps(maps_text):
    """Select one actually executable mapped libsndfile file, never a guessed path."""
    libraries = set()
    for line in maps_text.splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) != 6 or "x" not in columns[1]:
            continue
        mapped_path = Path(columns[5])
        if "libsndfile" in mapped_path.name and ".so" in mapped_path.name:
            require(mapped_path.is_absolute() and mapped_path.is_file(), "Loaded libsndfile mapping is unavailable")
            libraries.add(mapped_path.resolve())
    require(len(libraries) == 1, "Cannot uniquely bind actually loaded libsndfile binary")
    return next(iter(libraries))


def runtime_executable_binding(invocation_path):
    """Preserve venv invocation identity while hashing its resolved interpreter.

    Interpreter symlinks are conventional in virtual environments. This explicit
    runtime-only resolution does not relax source/product symlink rejection.
    """
    invoked = Path(invocation_path).absolute()
    resolved = invoked.resolve(strict=True)
    return {"invocation_path": str(invoked), "invocation_is_symlink": invoked.is_symlink(),
            "invocation_differs_from_resolved": invoked != resolved,
            "resolved_binary": binding(resolved)}


def runtime_bindings():
    import soundfile as sf
    import scipy.signal._signaltools as signaltools
    import scipy.signal._upfirdn as upfirdn
    import scipy.signal._upfirdn_apply as upfirdn_apply
    import scipy.signal._fir_filter_design as fir_design
    import numpy.core._multiarray_umath as numpy_core
    import numpy.random._pcg64 as pcg64
    require(sys.version_info[:2] == (3, 11) and np.__version__ == "1.26.4"
            and scipy.__version__ == "1.17.1" and sf.__version__ == "0.14.0",
            "Draft requires Python3.11 NumPy1.26.4 SciPy1.17.1 soundfile0.14.0")
    modules = {"numpy": np, "scipy": scipy, "soundfile": sf, "signaltools": signaltools,
               "upfirdn": upfirdn, "upfirdn_apply": upfirdn_apply, "pcg64": pcg64,
               "fir_filter_design": fir_design, "numpy_core": numpy_core}
    # SoundFile 0.14 need not expose _libname. Linux's actual process mappings
    # identify the loaded library, including wheel-specific hashed filenames.
    maps_path = Path("/proc/self/maps")
    require(maps_path.is_file(), "Loaded libsndfile binding requires Linux /proc/self/maps")
    library_name = libsndfile_from_maps(maps_path.read_text())
    return {"python": platform.python_version(), "executable": runtime_executable_binding(sys.executable),
            "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
            "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
            "libsndfile_binary": binding(library_name),
            "modules": {name: binding(module.__file__) for name, module in modules.items()}}


def code_bindings(design, validator):
    here = Path(__file__).resolve()
    require(fp(design)["sha256"] == DESIGN_SHA, "Prospective design hash mismatch")
    require(fp(validator)["sha256"] == VALIDATOR_SHA
            and fp(source_validator.__file__)["sha256"] == VALIDATOR_SHA,
            "Validator v2 immutable hash mismatch")
    require(fp(validator) == fp(source_validator.__file__), "Requested/imported source validator differs")
    return {"design": binding(design), "draft_tool": binding(here),
            "draft_tests": binding(here.with_name("test_draft_bicoherence_guitarset_pilot_v2.py")),
            "source_validator": binding(validator), "runtime": runtime_bindings()}


def draft(source, source_commit_sha256, overlap_evidence, overlap_sha256, design, validator, output):
    output = source_validator.safe_filesystem_path(output)
    source = source_validator.safe_filesystem_path(source)
    require(not output.exists() and output.parent.is_dir(), "Output must be new; parent directory must exist")
    require(output != source and not output.is_relative_to(source) and not source.is_relative_to(output),
            "Draft output and source must be disjoint")
    codes = code_bindings(design, validator)
    bound_source, split = verify_materialization(source, source_commit_sha256, validator)
    overlap = verify_overlap(overlap_evidence, overlap_sha256, bound_source)
    output.mkdir(exist_ok=False)
    try:
        standardized = []
        for row in split["rows"]:
            if row["split_role"] != "development":
                continue
            original = row["source_record"]
            path = child(source, original["audio"]["materialized_path"])
            samples = decode_selected(path, original["audio"]["decoded"])
            _, provenance = standardize(samples, original["audio"]["decoded"]["sample_rate_hz"])
            standardized.append({"item_id": row["item_id"], "original_audio": binding(path),
                                 "original_annotation": binding(child(source, original["annotation"]["materialized_path"])),
                                 "source_pcm_sha256": original["audio"]["decoded"]["decoded_pcm_sha256"],
                                 "preprocessing": provenance})
        require(len(standardized) == 90, "Exactly 90 development records must be standardized")
        # Repeat complete product/code/evidence hashes. Reserved sources are
        # byte-verified here but not decoded or transformed by this draft tool.
        require(code_bindings(design, validator) == codes, "Code/runtime changed during draft")
        end_source, end_split = verify_materialization(source, source_commit_sha256, validator)
        require(end_source == bound_source and end_split == split, "Source or split changed during draft")
        require(verify_overlap(overlap_evidence, overlap_sha256, end_source) == overlap,
                "Accepted overlap evidence changed during draft")
        document = {"version": VERSION, "stage": "partition_preprocessing_draft_only_requires_parent_review",
                    "bindings": codes, "source": bound_source, "overlap": overlap, "split": split,
                    "development_preprocessing": standardized, "bc_extracted": False,
                    "injections_constructed": False, "derived_audio_saved": False,
                    "reserved_or_unused_decoded_by_this_tool": False, "classifier_fits": 0,
                    "external_gate_passed": False, "future_measurement_producer_review_required": True}
        write_new(output / "draft.json", canonical(document))
        draft_binding = fp(output / "draft.json")
        write_new(output / "COMMIT.json", canonical({"status": "committed",
                  "kind": "guitarset_bc_partition_preprocessing_draft_only_v2",
                  "products": {"draft.json": draft_binding}, "bc_extracted": False,
                  "source_materialization_commit_sha256": source_commit_sha256}))
        return {"draft": str(output / "draft.json"), "draft_sha256_for_parent_review": draft_binding["sha256"],
                "development": 90, "reserved": 90, "unused": 180, "bc_extracted": False}
    except BaseException as exc:
        failure = output / "FAILED.json"
        if not failure.exists():
            write_new(failure, canonical({"status": "failed_no_commit", "exception": type(exc).__name__,
                       "message": str(exc), "partial_outputs_preserved": True, "bc_extracted": False}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-commit-sha256", "overlap-evidence", "overlap-sha256", "design", "validator", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = draft(args.source, args.source_commit_sha256, args.overlap_evidence, args.overlap_sha256,
                   args.design, args.validator, args.output)
    print(canonical(result).decode(), end="")


if __name__ == "__main__":
    main()

