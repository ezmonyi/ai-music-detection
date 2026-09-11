#!/usr/bin/env python3
"""Scoped GuitarSet overlap evidence audit, with validator-v2 PCM binding.

This derives the manifest adapters and comparison rules from overlap auditor v1,
while pinning the exact GuitarSet validator-v2 decoded-PCM description.  It does
not decode source audio, perform transformed matching, or make an acceptance or
admission decision.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath


_V1_PATH = Path(__file__).with_name("audit_guitarset_prior_corpus_overlap_v1.py")
_SPEC = importlib.util.spec_from_file_location("_guitarset_overlap_audit_v1", _V1_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise RuntimeError(f"Cannot load v1 overlap auditor: {_V1_PATH}")
_v1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_v1)

VERSION = "guitarset_prior_corpus_overlap_v2"
EXPECTED_V1_SHA256 = "bea01a5988c93f81dc51ed7cb25296fa73b79c2ee41cbc6cacb6aad31cbcd6ec"
F64_REPRESENTATION = _v1.F64_REPRESENTATION
GUITARSET_PCM_ENCODING_V2 = (
    "SHA-256 of decoded samples in frame-major/channel-minor C order; each sample is "
    "IEEE-754 float64 encoded explicitly little-endian (<f8); no resampling, mixing, "
    "normalization, or feature extraction"
)
SARAGA_WHOLE_PCM_ENCODING = (
    "IEEE754 float64 little-endian; C order [frame, channel]; no header"
)

AuditError = _v1.AuditError
EXPECTED_COUNTS = _v1.EXPECTED_COUNTS
INPUT_ARGUMENTS = _v1.INPUT_ARGUMENTS
TRUSTED_GUITARSET_KINDS = _v1.TRUSTED_GUITARSET_KINDS

_v1.require(_v1.fingerprint(_V1_PATH)["sha256"] == EXPECTED_V1_SHA256,
            "Derived overlap-auditor v1 dependency SHA-256 mismatch")


def validate_guitarset_decoded_descriptor(decoded, label="GuitarSet decoded audio"):
    """Validate the real validator-v2 descriptor and its full-decode invariants."""
    _v1.require(isinstance(decoded, dict), f"Invalid {label} object")
    _v1.require(decoded.get("decoded_pcm_canonical_encoding") == GUITARSET_PCM_ENCODING_V2,
                f"Unexpected {label} PCM encoding")
    sample_rate = _v1.integer(decoded.get("sample_rate_hz"), label + ".sample_rate_hz", 1)
    channels = _v1.integer(decoded.get("channels"), label + ".channels", 1)
    frames = _v1.integer(decoded.get("decoded_frames"), label + ".decoded_frames", 1)
    header_frames = _v1.integer(decoded.get("header_frames"), label + ".header_frames", 1)
    finite_samples = _v1.integer(decoded.get("float64_samples_checked_finite"),
                                 label + ".float64_samples_checked_finite", 1)
    nonfinite = _v1.integer(decoded.get("nonfinite_samples"), label + ".nonfinite_samples", 0)
    _v1.require(header_frames == frames, f"{label} header/decoded frame mismatch")
    _v1.require(finite_samples == frames * channels, f"{label} finite sample count mismatch")
    _v1.require(nonfinite == 0, f"{label} contains nonfinite samples")
    _v1.require(decoded.get("empty_eof_observed") is True,
                f"{label} did not record a real empty EOF read")
    digest = _v1.sha256_value(decoded.get("decoded_pcm_sha256"), label + ".decoded_pcm_sha256")
    return _v1.pcm("decoded_pcm_sha256", digest, F64_REPRESENTATION, "full_source",
                   sample_rate, channels, frames)


def verify_guitarset(folder, expected_commit_sha256, expected_count=360,
                     expected_players=6, expected_scores=30):
    """Verify every committed GuitarSet product and all manifest tuple evidence."""
    folder = Path(folder)
    _v1.require(folder.is_dir() and not folder.is_symlink(),
                f"Invalid GuitarSet materialization: {folder}")
    commit_path = _v1.regular_file(folder / "COMMIT.json")
    commit_fp = _v1.fingerprint(commit_path)
    _v1.require(commit_fp["sha256"] ==
                _v1.sha256_value(expected_commit_sha256, "GuitarSet COMMIT SHA"),
                "GuitarSet COMMIT SHA-256 mismatch")
    commit = _v1.strict_json(commit_path.read_text(encoding="utf-8"), str(commit_path))
    _v1.require(commit.get("status") == "committed", "GuitarSet source is not committed")
    _v1.require(commit.get("kind") in TRUSTED_GUITARSET_KINDS,
                "Unexpected GuitarSet COMMIT kind")
    products = commit.get("products")
    _v1.require(isinstance(products, dict) and products, "Invalid GuitarSet product inventory")
    _v1.require(len(products) <= _v1.MAX_GUITARSET_PRODUCTS, "Too many GuitarSet products")

    actual_paths = {}
    for path in folder.rglob("*"):
        _v1.require(not path.is_symlink(), f"Symlink in GuitarSet products: {path}")
        if path.is_file() and path.name != "COMMIT.json":
            actual_paths[path.relative_to(folder).as_posix()] = path
    _v1.require(set(actual_paths) == set(products),
                "GuitarSet COMMIT product inventory mismatch")
    product_fingerprints = {}
    for relative, path in sorted(actual_paths.items()):
        _v1.require(not PurePosixPath(relative).is_absolute()
                    and ".." not in PurePosixPath(relative).parts,
                    f"Unsafe GuitarSet product path: {relative}")
        fp = _v1.fingerprint(path)
        expected = products[relative]
        _v1.require(isinstance(expected, dict) and expected.get("bytes") == fp["bytes"]
                    and expected.get("sha256") == fp["sha256"],
                    f"GuitarSet product mismatch: {relative}")
        product_fingerprints[relative] = fp

    _v1.require("source_manifest.jsonl" in products and "validation_summary.json" in products,
                "GuitarSet required products absent")
    summary_path = folder / "validation_summary.json"
    summary = _v1.strict_json(summary_path.read_text(encoding="utf-8"), str(summary_path))
    _v1.require(isinstance(summary, dict)
                and str(summary.get("status", "")).startswith(
                    "passed_archive_crc_materialization")
                and summary.get("counts", {}).get("exact_audio_annotation_pairs") == expected_count,
                "GuitarSet validation summary does not bind the expected passed pair count")
    _v1.require(summary.get("audio", {}).get("canonical_pcm_encoding") ==
                GUITARSET_PCM_ENCODING_V2,
                "GuitarSet summary PCM encoding does not match validator-v2 contract")
    validator_code = summary.get("validator_code")
    _v1.require(isinstance(validator_code, dict),
                "GuitarSet summary lacks validator-code binding")
    validator_code_binding = {
        "bytes": _v1.integer(validator_code.get("bytes"), "validator_code.bytes", 1),
        "sha256": _v1.sha256_value(validator_code.get("sha256"), "validator_code.sha256"),
    }

    manifest_path = folder / "source_manifest.jsonl"
    rows = []
    with manifest_path.open("rb") as handle:
        for n, raw in enumerate(handle, 1):
            _v1.require(raw.strip() and len(raw) <= _v1.MAX_JSONL_LINE_BYTES,
                        f"Invalid GuitarSet manifest line {n}")
            row = _v1.strict_json(raw.decode("utf-8"), f"{manifest_path}:{n}")
            _v1.require(isinstance(row, dict), f"Invalid GuitarSet row {n}")
            rows.append(row)
    _v1.require(len(rows) == expected_count,
                f"Expected {expected_count} GuitarSet rows, found {len(rows)}")

    tuples = set()
    item_ids, audio_paths, annotation_paths = set(), set(), set()
    players, scores, items = set(), set(), []
    for n, row in enumerate(rows, 1):
        required = ("item_id", "player_id", "score_id", "performance", "audio", "annotation")
        missing = sorted(set(required) - set(row))
        _v1.require(not missing, f"GuitarSet row {n} missing fields: {missing}")
        item_id = str(row["item_id"])
        player, score, performance = (str(row[key]) for key in
                                      ("player_id", "score_id", "performance"))
        _v1.require(performance in ("comp", "solo"),
                    f"Invalid GuitarSet performance: {performance}")
        _v1.require(item_id == f"{player}_{score}_{performance}",
                    f"GuitarSet item/tuple identity mismatch row {n}")
        _v1.require(item_id not in item_ids, f"Duplicate GuitarSet item_id: {item_id}")
        item_ids.add(item_id)
        key = (player, score, performance)
        _v1.require(key not in tuples, f"Duplicate GuitarSet tuple: {key}")
        tuples.add(key); players.add(player); scores.add(score)

        audio = row["audio"]
        decoded = audio.get("decoded") if isinstance(audio, dict) else None
        materialized = audio.get("materialized") if isinstance(audio, dict) else None
        _v1.require(isinstance(materialized, dict), f"Invalid GuitarSet audio evidence row {n}")
        pcm_evidence = validate_guitarset_decoded_descriptor(decoded, f"GuitarSet row {n}")
        _v1.require(pcm_evidence["channels"] == 1, f"Non-mono GuitarSet row {n}")
        decoded_materialized = decoded.get("materialized_file")
        _v1.require(isinstance(decoded_materialized, dict),
                    f"Missing decoded materialized-file binding row {n}")

        materialized_path = str(audio.get("materialized_path", ""))
        _v1.require(materialized_path in products, f"Uncommitted GuitarSet audio path row {n}")
        _v1.require(materialized_path not in audio_paths,
                    f"Duplicate GuitarSet materialized audio path row {n}")
        audio_paths.add(materialized_path)
        file_sha = _v1.sha256_value(materialized.get("sha256"),
                                    "GuitarSet materialized SHA")
        file_bytes = _v1.integer(materialized.get("bytes"), "GuitarSet materialized bytes")
        _v1.require(products[materialized_path] == {"bytes": file_bytes, "sha256": file_sha},
                    f"GuitarSet row/product audio fingerprint mismatch row {n}")
        _v1.require(decoded_materialized == materialized,
                    f"GuitarSet decoder/materialized fingerprint mismatch row {n}")
        _v1.require(_v1.sha256_value(audio.get("archive_member_sha256"),
                                     "GuitarSet archive member SHA") == file_sha,
                    f"GuitarSet archive/materialized audio SHA mismatch row {n}")

        annotation = row["annotation"]
        _v1.require(isinstance(annotation, dict)
                    and isinstance(annotation.get("materialized"), dict),
                    f"Invalid GuitarSet annotation evidence row {n}")
        annotation_path = str(annotation.get("materialized_path", ""))
        _v1.require(annotation_path in products,
                    f"Uncommitted GuitarSet annotation path row {n}")
        _v1.require(annotation_path not in annotation_paths,
                    f"Duplicate GuitarSet materialized annotation path row {n}")
        annotation_paths.add(annotation_path)
        annotation_sha = _v1.sha256_value(annotation["materialized"].get("sha256"),
                                          "GuitarSet annotation SHA")
        annotation_bytes = _v1.integer(annotation["materialized"].get("bytes"),
                                       "GuitarSet annotation bytes")
        _v1.require(products[annotation_path] ==
                    {"bytes": annotation_bytes, "sha256": annotation_sha},
                    f"GuitarSet row/product annotation fingerprint mismatch row {n}")
        _v1.require(_v1.sha256_value(annotation.get("archive_member_sha256"),
                                     "GuitarSet annotation archive member SHA") == annotation_sha,
                    f"GuitarSet archive/materialized annotation SHA mismatch row {n}")

        identifiers = [("item_id", item_id), ("score_id", score),
                       ("archive_member", audio.get("archive_member", "")),
                       ("materialized_path", materialized_path)]
        items.append(_v1.logical(
            "guitarset", n, item_id, identifiers,
            [_v1.encoded("materialized.sha256", file_sha, "materialized_source_file"),
             _v1.encoded("archive_member_sha256", audio.get("archive_member_sha256"),
                         "archive_member_file")],
            [pcm_evidence], _v1.path_entity_keys(materialized_path)))

    _v1.require(len(players) == expected_players and len(scores) == expected_scores
                and len(tuples) == expected_count,
                "GuitarSet player x score x 2-performance tuple inventory mismatch")
    _v1.require(tuples == {(p, s, perf) for p in players for s in scores
                           for perf in ("comp", "solo")},
                "GuitarSet tuple inventory is not the complete cross-product")
    declared = [{"field": "dataset_name", "value": "GuitarSet"}]
    source_summary = summary.get("source", {})
    if isinstance(source_summary, dict) and source_summary.get("doi"):
        declared.append({"field": "doi", "value": str(source_summary["doi"])})
    binding = {
        "folder": str(folder), "commit": commit_fp, "commit_kind": commit["kind"],
        "product_count": len(products), "products_verified": True,
        "manifest_rows": len(rows), "tuple_cross_product_verified": True,
        "declared_source_identifiers": declared,
        "decoded_pcm_contract": {
            "observed_description": GUITARSET_PCM_ENCODING_V2,
            "semantic_namespace": F64_REPRESENTATION,
            "description_exactly_pinned": True,
            "all_rows_full_decode_invariants_verified": True,
        },
        "source_validator_code": validator_code_binding,
    }
    return items, binding, product_fingerprints


def write_new(path, data):
    with Path(path).open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def run(guitarset, guitarset_commit_sha256, inputs, output, expected_counts=None,
        _expected_guitarset_shape=(6, 30)):
    expected_counts = dict(EXPECTED_COUNTS if expected_counts is None else expected_counts)
    _v1.require(set(expected_counts) == set(EXPECTED_COUNTS),
                "Expected-count override keys mismatch")
    _v1.require(set(inputs) == set(INPUT_ARGUMENTS) | {"saraga_proofs"},
                "Input path keys mismatch")
    output = Path(output)
    _v1.require(not output.exists(), "Output must be a new exclusive folder")
    _v1.require(output.parent.is_dir(), "Output parent must exist")
    output.mkdir()
    try:
        code_fp = _v1.fingerprint(Path(__file__).resolve())
        base_code_fp = _v1.fingerprint(_V1_PATH)
        _v1.require(base_code_fp["sha256"] == EXPECTED_V1_SHA256,
                    "Derived overlap-auditor v1 dependency SHA-256 mismatch")
        expected_players, expected_scores = _expected_guitarset_shape
        guitarset_items, guitarset_binding, guitarset_products = verify_guitarset(
            guitarset, guitarset_commit_sha256,
            expected_count=expected_players * expected_scores * 2,
            expected_players=expected_players, expected_scores=expected_scores)
        prior, input_inventory, proof_inventory = _v1.load_inputs(inputs, expected_counts)
        comparisons = _v1.compare(guitarset_items, prior,
                                  guitarset_binding["declared_source_identifiers"])
        initial_fingerprints = {
            key: {"bytes": value["bytes"], "sha256": value["sha256"]}
            for key, value in input_inventory.items()
        }
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
            "encoding_contracts": {
                "guitarset_validator_v2": {
                    "exact_description": GUITARSET_PCM_ENCODING_V2,
                    "semantic_namespace": F64_REPRESENTATION,
                },
                "saraga_whole_pcm": {
                    "exact_description": SARAGA_WHOLE_PCM_ENCODING,
                    "semantic_namespace": F64_REPRESENTATION,
                },
                "compatibility_basis": (
                    "Both explicitly denote headerless little-endian float64 samples in "
                    "C frame-major/channel-minor order; equality additionally requires "
                    "sample rate, channels, and frames."
                ),
                "descriptions_are_distinct_and_independently_pinned": True,
            },
            "guitarset": guitarset_binding,
            "inputs": input_inventory,
            "saraga_proofs": {"root": str(inputs["saraga_proofs"]),
                               "count": len(proof_inventory), "files": proof_inventory},
            "accounting": {
                "manifest_row_views": sum(value["rows"] for value in input_inventory.values()),
                "logical_audio_evidence_entries": len(prior),
                "deduplicated_connected_source_entities": _v1.deduplicated_entity_count(prior),
                "multiplicity_preserved_in_match_lists": True,
                "row_views_are_not_claimed_as_unique_recordings": True,
                "by_adapter": _v1.evidence_summary(prior),
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
            "derived_adapter_implementation": {
                **base_code_fp,
                "path": str(_V1_PATH),
                "expected_sha256": EXPECTED_V1_SHA256,
                "exact_expected_sha256_verified": True,
            },
        }
        write_new(output / "overlap_report.json", _v1.canonical_json(report))

        for key, value in input_inventory.items():
            _v1.require(_v1.fingerprint(value["path"]) == initial_fingerprints[key],
                        f"Input changed during audit: {key}")
        for path, fp in proof_inventory.items():
            _v1.require(_v1.fingerprint(path) == fp,
                        f"Saraga proof changed during audit: {path}")
        for relative, fp in guitarset_products.items():
            _v1.require(_v1.fingerprint(Path(guitarset) / relative) == fp,
                        f"GuitarSet product changed during audit: {relative}")
        _v1.require(_v1.fingerprint(Path(guitarset) / "COMMIT.json") ==
                    guitarset_binding["commit"], "GuitarSet COMMIT changed during audit")
        _v1.require(_v1.fingerprint(Path(__file__).resolve()) == code_fp,
                    "Auditor code changed during audit")
        _v1.require(_v1.fingerprint(_V1_PATH) == base_code_fp,
                    "Derived v1 adapter implementation changed during audit")
        _v1.require(base_code_fp["sha256"] == EXPECTED_V1_SHA256,
                    "Derived v1 adapter implementation is not the pinned revision")
        report_fp = _v1.fingerprint(output / "overlap_report.json")
        commit = {
            "status": "committed", "kind": VERSION,
            "products": {"overlap_report.json": report_fp},
            "guitarset_commit_sha256": guitarset_binding["commit"]["sha256"],
            "independent_reviewer_acceptance": False,
            "admission_decision_made": False,
            "global_non_overlap_established": False,
            "transformed_overlap_coverage": 0,
        }
        write_new(output / "COMMIT.json", _v1.canonical_json(commit))
        return report
    except BaseException as error:
        failure = {"status": "failed_not_committed", "error_type": type(error).__name__,
                   "error": str(error), "commit_published": False,
                   "partial_output_preserved": True}
        failure_path = output / "EXECUTION_FAILURE.json"
        if not failure_path.exists():
            try:
                write_new(failure_path, _v1.canonical_json(failure))
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
