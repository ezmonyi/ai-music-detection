"""Close physical intake separately from the immutable partial producer run.

Checks all successful output PCM, receipts and source bytes; does NOT replay
native DSP for 1656 inputs. The eight pinned pilot outputs carry independent
replay evidence. No audio is copied, padded, replaced, relabeled or classified.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib
import importlib.util
import math
import os
from pathlib import Path
import platform
import re
import sys

import numpy as np
import soundfile as sf

CONTRACT = "7cf9b3d2436958c8e7d0d4e5121ed317b04a1126abaafe6ba97022e88dc82e4a"
AUDITOR = "b04153059a8f7e81f7d11ca865914d9f341129025d6b606af18e1b617a2985b0"
RUN = "9746294937094572a41241fe33d48ae7"
TOTAL, ELIGIBLE, EXCLUDED, MONO = 1695, 1656, 39, 51
LOCAL_ARTIFACT_ROOT = Path("/Users/yi/Documents/code/music/artifacts/audio_phenomena_expansion_20260907")
REMOTE_ARTIFACT_ROOT = Path("/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907")
HISTORICAL_METADATA_COPIES = {
    "/Users/yi/Documents/code/music/artifacts/open_models_spectral_500_20260901/state/acestep_standardization.jsonl": {
        "path": "/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/audit/native30_historical_metadata_copies_v2/open_models_spectral_500_20260901/state/acestep_standardization.jsonl",
        "sha256": "036d9f3cc5792589929975f5266d236a9aaf9d335bb26d1c2d06b776e3d527dd"},
    "/Users/yi/Documents/code/music/artifacts/source_diversity_expansion_20260905/state/aime_long_manifest.jsonl": {
        "path": "/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/audit/native30_historical_metadata_copies_v2/source_diversity_expansion_20260905/state/aime_long_manifest.jsonl",
        "sha256": "b7fae0c67d147d02a0cd93f399787973c4b9a31e8271165347fe4e3baa1ff8ba"},
    "/Users/yi/Documents/code/music/artifacts/external_generator_500_testset_20260904/testset/manifest.jsonl": {
        "path": "/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/audit/native30_historical_metadata_copies_v2/external_generator_500_testset_20260904/testset/manifest.jsonl",
        "sha256": "49dbe01b8be98857361b3a437db44ee937698e014bf70ccc7d443c24ad342fb4"},
    "/Users/yi/Documents/code/music/artifacts/open_models_spectral_500_20260901/state/heartmula_standardization.jsonl": {
        "path": "/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/audit/native30_historical_metadata_copies_v2/open_models_spectral_500_20260901/state/heartmula_standardization.jsonl",
        "sha256": "f84aa91ba0dccf8a84b5890735c0e9dac3fb02225af42bd4d15af55453eef522"},
    "/Users/yi/Documents/code/music/artifacts/source_diversity_expansion_20260905/audit/legacy_native_1000.json": {
        "path": "/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/audit/native30_historical_metadata_copies_v2/source_diversity_expansion_20260905/audit/legacy_native_1000.json",
        "sha256": "a2b9cf9b3c1eb8ffa06acd7cb405b7f553d4cc1f4f3d3b10bb92e3511008577c"},
    "/Users/yi/Documents/code/music/artifacts/source_diversity_expansion_20260905/manifests/final/metadata_30s.csv": {
        "path": "/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/audit/native30_historical_metadata_copies_v2/source_diversity_expansion_20260905/manifests/final/metadata_30s.csv",
        "sha256": "eb313f97e2d69d743423efc170eecb24f1f00ee77ad789461772daecd058c242"},
    "/Users/yi/Documents/code/music/artifacts/source_diversity_expansion_20260905/state/materialization_manifest.jsonl": {
        "path": "/mnt/nfs-code/users/yi/audio_phenomena_expansion_20260907/audit/native30_historical_metadata_copies_v2/source_diversity_expansion_20260905/state/materialization_manifest.jsonl",
        "sha256": "ae51c2c3a5602e9825369b4f59ccd96f9182b9c7219b5364cda34164f557a0ec"},
}
STATUS = "physical_intake_closed_with_explicit_exclusions"
POLICY = {"eligibility": "sequential_actual_frames >= 30*native_rate_hz",
          "source_or_label_used": False, "padding": False, "replacement": False, "relabel": False}
SCOPE = {"classifier_fits": 0, "cohort_admitted": False, "feature_extraction_authorized": False}


def load_auditor():
    path = Path(__file__).resolve().with_name("audit_native30_new1695_inventory_v1.py")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != AUDITOR:
        raise ValueError("pinned independent auditor code SHA mismatch")
    spec = importlib.util.spec_from_file_location("_native30_pinned_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.bind_file(path, AUDITOR)
    return module


a = load_auditor()
require = a.require


def safe_id(ident):
    require(isinstance(ident, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", ident), "unsafe record ID")
    return ident


def indexed(records, name):
    require(isinstance(records, list), name + " must be a list")
    result = {safe_id(row["id"]): row for row in records}
    require(len(result) == len(records), "duplicate IDs in " + name)
    return result


def validate_documents(contract, run, diagnosis, run_sha):
    require(contract.get("status") == "frozen_before_any_audio_processing" and
            contract.get("scope") == "new1695_native_stereo_DSP_only" and
            contract.get("expected_count") == TOTAL, "contract status/scope/count")
    require(all(contract.get(k) == v for k, v in SCOPE.items()), "contract no-admission scope")
    require(run.get("status") == "partial_no_COMMIT" and run.get("run_id") == RUN and
            run.get("contract_sha256") == CONTRACT and run.get("expected") == TOTAL and
            run.get("completed") == ELIGIBLE and run.get("failed") == EXCLUDED and
            run.get("classifier_fits") == 0 and run.get("cohort_admitted") is False, "terminal run binding/status/count")
    require(diagnosis.get("status") == "confirmed_39_native_short_inputs_source_blind_policy" and
            diagnosis.get("contract_sha256") == CONTRACT and diagnosis.get("run_summary_sha256") == run_sha and
            diagnosis.get("policy") == POLICY and all(diagnosis.get(k) == v for k, v in SCOPE.items()),
            "diagnosis status/bindings/policy/scope")
    require(diagnosis.get("decoder_scope") == "pinned_libsndfile_sequential_EOF_not_universal_audio_duration",
            "diagnosis decoder scope")
    for key in ("soundfile", "libsndfile", "numpy"):
        require(diagnosis.get("runtime", {}).get(key) == contract["runtime"][key], "diagnosis runtime mismatch")
    rows = indexed(contract["rows"], "contract")
    failures = indexed(run["failures"], "run failures")
    excluded = indexed(diagnosis["records"], "diagnosis")
    mono = indexed(contract["excluded_native_mono"], "mono metadata")
    require(len(rows) == TOTAL and len(failures) == len(excluded) == EXCLUDED and len(mono) == MONO,
            "accountability counts")
    require(set(failures) == set(excluded) and set(failures) <= set(rows) and not (set(mono) & set(rows)),
            "accountability disjoint/complete IDs")
    require(all(r.get("reason") == "native_mono" and r.get("native_evidence", {}).get("channels") == 1
                for r in mono.values()), "mono exclusion metadata")
    return rows, failures, excluded


def inventory(root, success_ids, failure_ids):
    require(root.is_dir() and not root.is_symlink(), "regular original root required")
    require({p.name for p in root.iterdir()} == {"writer.lock", "contract.json", "items", "audio", "failures", "runs"},
            "original partial inventory changed; no original COMMIT/manifest allowed")
    for name in ("writer.lock", "contract.json"):
        require((root / name).is_file() and not (root / name).is_symlink(), "nonregular root file")
    expected = {"items": {i + ".json" for i in success_ids}, "audio": {i + ".wav" for i in success_ids},
                "failures": {i + "." + RUN + ".json" for i in failure_ids}, "runs": {RUN + ".json"}}
    paths = []
    for name, names in expected.items():
        directory = root / name
        require(directory.is_dir() and not directory.is_symlink(), "nonregular original directory")
        require({p.name for p in directory.iterdir()} == names, "extra/missing original " + name)
        for path in sorted(directory.iterdir()):
            require(path.is_file() and not path.is_symlink(), "nested/nonregular original product")
            paths.append(path)
    return paths


def validate_failure(record, failure, row, root, bindings):
    ident = row["id"]
    require(failure.get("id") == ident and failure.get("contract_sha256") == CONTRACT and
            failure.get("row_sha256") == a.vh(row) and failure.get("run_id") == RUN and
            failure.get("exception") == "ValueError" and
            failure.get("message") == "native region shorter than30s: padding forbidden", "failure receipt lineage")
    for key in ("id", "source_group", "label", "group_id", "component_id", "role"):
        require(record.get(key) == row.get(key), "diagnosis row lineage: " + key)
    native, rate = row["native_evidence"], row["native_evidence"]["sample_rate_hz"]
    require(a.integer(rate, 1) and native.get("channels") == record.get("native_channels") == 2 and
            record.get("native_rate_hz") == rate and record.get("native_path") == row["execution_native_path"] and
            record.get("native_sha256") == native["sha256"], "diagnosis native lineage")
    actual, required = record.get("actual_frames"), 30 * rate
    require(a.integer(actual) and actual < required and record.get("primary_second_pass_frames") == actual and
            record.get("required_frames") == required and record.get("deficit_frames") == required - actual,
            "diagnosis two-pass frame count/deficit")
    seconds = record.get("deficit_seconds")
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0 and
            math.isclose(seconds, (required - actual) / rate, rel_tol=1e-12, abs_tol=1e-12), "positive numeric duration deficit")
    require(record.get("empty_eof_observed") is True and a.integer(record.get("header_frames")) and
            record.get("diagnosis_method") == "two_fresh_libsndfile_sequential_passes_different_blocks_to_empty_EOF",
            "diagnosis sequential EOF method")
    first, second = record.get("primary_pcm_sha256"), record.get("primary_second_pcm_sha256")
    require(a.is_hash(first) and a.is_hash(second) and record.get("different_block_pcm_bit_exact") is (first == second),
            "diagnosis PCM observation consistency")
    require(isinstance(record.get("ffmpeg_diagnostic"), dict) and record["ffmpeg_diagnostic"].get("status") in
            {"unavailable", "decoded_without_resampling_or_channel_conversion"}, "FFmpeg diagnostic status")
    path = root / "failures" / (ident + "." + RUN + ".json")
    expected = record.get("producer_failure", {})
    require(expected.get("path") == str(path) and a.is_hash(expected.get("sha256")), "diagnosis failure path/hash")
    bindings[str(path)] = a.bind_file(path, expected["sha256"])
    require(a.load(path) == failure, "failure file/run summary join")
    source = Path(row["execution_native_path"])
    require(record.get("source_binding") == {"path": str(source), "sha256": native["sha256"]}, "diagnosis source binding")
    binding = a.bind_file(source, native["sha256"])
    require(record.get("source_stat_signature") == binding["signature"][:5] and
            record.get("source_byte_count") == binding["signature"][2], "diagnosis source signature/bytes")
    bindings[str(source)] = binding


def validate_output(wav, receipt):
    audit = receipt["audit"]
    binding = a.bind_file(wav, receipt["file_sha256"])
    require(str(wav) == receipt.get("standardized_path") and a.integer(receipt.get("file_bytes"), 1) and
            binding["signature"][2] == receipt["file_bytes"], "output path/bytes")
    with sf.SoundFile(wav) as stream:
        require((stream.samplerate, stream.channels, stream.frames, stream.format, stream.subtype) ==
                (a.RATE, 2, a.FRAMES, "WAV", "FLOAT"), "output WAV format")
        values = stream.read(dtype="float32", always_2d=True)
        require(len(stream.read(1, dtype="float32", always_2d=True)) == 0, "output empty EOF")
    require(values.shape == (a.FRAMES, 2) and np.isfinite(values).all(), "output shape/finite")
    pcm = hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()
    require(pcm == receipt.get("waveform_float32_sha256") == audit.get("output_waveform_float32_sha256"), "output PCM hash")
    require(float(np.max(np.abs(values))) == audit.get("peak_absolute") and
            int(np.count_nonzero(np.abs(values) > 1)) == audit.get("samples_abs_above_one"), "output quality recount")
    return binding


def resolve_declared_path(declared_path):
    """Relocate only this artifact tree; missing/outside paths are never skipped."""
    require(isinstance(declared_path, str), "declared path must be a string")
    path = Path(declared_path)
    require(path.is_absolute() and ".." not in path.parts, "unsafe declared absolute path")
    if declared_path in HISTORICAL_METADATA_COPIES:
        return Path(HISTORICAL_METADATA_COPIES[declared_path]["path"])
    if path.is_relative_to(LOCAL_ARTIFACT_ROOT):
        path = REMOTE_ARTIFACT_ROOT / path.relative_to(LOCAL_ARTIFACT_ROOT)
        # Reject redirected copies outside the sole authorized artifact tree.
        require(path.resolve().is_relative_to(REMOTE_ARTIFACT_ROOT.resolve()), "relocated path escapes artifact tree")
    return path


def bind_declared_file(declared_path, expected_sha):
    if declared_path in HISTORICAL_METADATA_COPIES:
        require(expected_sha == HISTORICAL_METADATA_COPIES[declared_path]["sha256"], "historical metadata allowlist SHA mismatch")
    return a.bind_file(resolve_declared_path(declared_path), expected_sha)


def bind_declared_files(declared, bindings):
    """Keys preserve original declared paths; binding.path is the resolved path.

    Relocation never alters the declared digest. Every copy must hash identically.
    """
    if isinstance(declared, dict):
        if "path" in declared and "sha256" in declared:
            require(a.is_hash(declared["sha256"]), "invalid upstream digest")
            binding = bind_declared_file(declared["path"], declared["sha256"])
            if "bytes" in declared:
                require(declared["bytes"] == binding["signature"][2], "upstream byte count")
            old = bindings.get(declared["path"])
            require(old is None or old == binding, "conflicting upstream bindings")
            bindings[declared["path"]] = binding
        else:
            for value in declared.values():
                bind_declared_files(value, bindings)
    elif isinstance(declared, list):
        for value in declared:
            bind_declared_files(value, bindings)


def validate_runtime(contract, bindings):
    runtime = contract["runtime"]
    versions = {"python": platform.python_version(), "numpy": np.__version__, "soundfile": sf.__version__,
                "libsndfile": sf.__libsndfile_version__, "scipy": importlib.import_module("scipy").__version__}
    require(all(runtime[k] == v for k, v in versions.items()), "current runtime version mismatch")
    require(Path(sys.executable).resolve() == Path(runtime["executable"]["path"]).resolve(), "current Python executable mismatch")
    for name, bound in runtime["module_files"].items():
        require(Path(importlib.import_module(name).__file__).resolve() == Path(bound["path"]).resolve(), "loaded module path mismatch")
    bind_declared_files(runtime, bindings)
    return versions


def attrition(contract, successes, failures):
    result = {}
    for key in ("source_group", "label"):
        candidates = contract["rows"] + contract["excluded_native_mono"]
        selected = Counter(r[key] for r in contract["rows"])
        eligible = Counter(r["row"][key] for r in successes)
        short = Counter(r[key] for r in failures)
        mono = Counter(r[key] for r in contract["excluded_native_mono"])
        result[key] = {value: {"metadata_candidates": sum(r[key] == value for r in candidates),
                              "native_stereo_attempted": selected[value], "eligible": eligible[value],
                              "excluded_short": short[value], "excluded_mono_metadata": mono[value],
                              "short_attrition_numerator": short[value], "short_attrition_denominator": selected[value],
                              "short_attrition_fraction": short[value] / selected[value] if selected[value] else None}
                       for value in sorted({r[key] for r in candidates})}
    return result


def verify(root, diagnosis_path, diagnosis_sha, run_path, run_sha, pilot, pilot_audit):
    require(a.is_hash(diagnosis_sha) and a.is_hash(run_sha), "mandatory valid input SHAs")
    require(run_path == root / "runs" / (RUN + ".json"), "run path must be original terminal run")
    inputs = {"contract": a.bind_file(root / "contract.json", CONTRACT),
              "run_summary": a.bind_file(run_path, run_sha), "diagnosis": a.bind_file(diagnosis_path, diagnosis_sha),
              "pilot_commit": a.bind_file(pilot / "COMMIT.json", a.PILOT_COMMIT),
              "pilot_independent_audit": a.bind_file(pilot_audit, a.PILOT_AUDIT),
              "finalizer": a.bind_file(Path(__file__).resolve()),
              "finalizer_tests": a.bind_file(Path(__file__).resolve().with_name("test_finalize_native30_partial_intake_v2.py")),
              "auditor_helpers": a.bind_file(Path(a.__file__), AUDITOR)}
    contract, run, diagnosis = a.load(root / "contract.json"), a.load(run_path), a.load(diagnosis_path)
    require(contract.get("output_root") == str(root), "contract original root")
    rows, failures, exclusions = validate_documents(contract, run, diagnosis, run_sha)
    success_ids = set(rows) - set(failures)
    require(len(success_ids) == ELIGIBLE, "success count")
    inventory(root, success_ids, failures)
    upstream = {}
    for field in ("bindings", "source_provenance_bindings"):
        bind_declared_files(contract.get(field, {}), upstream)
    runtime = validate_runtime(contract, upstream)
    declared = diagnosis.get("input_bindings", {})
    require(declared.get(str(root / "contract.json")) == CONTRACT and declared.get(str(run_path)) == run_sha,
            "diagnosis input bindings join")
    for path, digest in declared.items():
        require(a.is_hash(digest), "diagnosis declared digest")
        upstream[path] = bind_declared_file(path, digest)
    pc, pa = a.load(pilot / "COMMIT.json"), a.load(pilot_audit)
    pilot_wavs = {}
    for name, value in pc.get("products", {}).items():
        if name.startswith("audio/") and name.endswith(".wav"):
            ident = safe_id(Path(name).stem)
            require(name == "audio/" + ident + ".wav" and ident not in pilot_wavs, "unsafe/duplicate pilot product")
            pilot_wavs[ident] = (name, value)
    replay = a.validate_pilot_replay(pc, pa, pilot_wavs)
    require(set(pilot_wavs) <= success_ids, "all eight pilot IDs must be eligible")
    pilot_bindings = {}
    for ident, (name, value) in pilot_wavs.items():
        binding = a.bind_file(pilot / name, value["sha256"])
        require(binding["signature"][2] == value["bytes"], "pilot WAV bytes")
        pilot_bindings[ident] = binding
    products, sources, records, matches = {}, {}, [], []
    for index, ident in enumerate(sorted(success_ids), 1):
        row = rows[ident]
        item = root / "items" / (ident + ".json")
        products[str(item)] = a.bind_file(item)
        envelope = a.load(item)
        require(set(envelope) == {"payload", "receipt_sha256"}, "receipt envelope fields")
        receipt = envelope["payload"]
        require(envelope["receipt_sha256"] == a.vh(receipt), "receipt envelope hash")
        audit = receipt["audit"]
        a.validate_receipt_metadata(receipt, row, audit, contract, CONTRACT)
        source = Path(row["execution_native_path"])
        source_binding = a.bind_file(source, row["native_evidence"]["sha256"])
        require(audit.get("source_sha256_before") == audit.get("source_sha256_after") == source_binding["sha256"] and
                audit.get("source_stat_signature") == source_binding["signature"][:5], "success native source binding")
        old = sources.get(str(source))
        require(old is None or old == source_binding, "conflicting repeated source")
        sources[str(source)] = source_binding
        wav = root / "audio" / (ident + ".wav")
        products[str(wav)] = validate_output(wav, receipt)
        if ident in pilot_wavs:
            require(products[str(wav)]["sha256"] == pilot_bindings[ident]["sha256"] and
                    replay[ident]["float32_bit_exact"] is True, "eight pilot byte-exact match")
            matches.append(ident)
        records.append(receipt)
        if index % 100 == 0:
            print(a.canonical({"event": "partial_intake_output_verification", "verified": index}).decode().strip(), flush=True)
    failure_bindings = {}
    for ident in sorted(failures):
        validate_failure(exclusions[ident], failures[ident], rows[ident], root, failure_bindings)
    require(set(matches) == set(pilot_wavs), "missing pilot match")
    bindings = {"inputs": inputs, "upstream_code_runtime_provenance": upstream, "original_success_products": products,
                "success_sources": sources, "failure_products_and_sources": failure_bindings, "pilot_wavs": pilot_bindings}
    manifest = {"version": "finalize_native30_partial_intake_v2", "status": STATUS, "contract_sha256": CONTRACT,
                "original_root": str(root), "original_run_status": "partial_no_COMMIT", "run_summary_sha256": run_sha,
                "diagnosis_sha256": diagnosis_sha, "policy": POLICY, "attempted": TOTAL, "eligible_count": ELIGIBLE,
                "excluded_short_count": EXCLUDED, "records": records,
                "excluded_short_records": [exclusions[i] for i in sorted(exclusions)],
                "original_failure_receipts": [failures[i] for i in sorted(failures)],
                "excluded_native_mono": contract["excluded_native_mono"],
                "all_attempted_ids": sorted(rows), "eligible_ids": sorted(success_ids), "excluded_short_ids": sorted(failures), **SCOPE}
    summary = {"status": STATUS, "metadata_candidates": TOTAL + MONO, "attempted": TOTAL, "eligible": ELIGIBLE,
               "excluded_short": EXCLUDED, "excluded_mono_metadata": MONO, "attrition": attrition(contract, records, list(exclusions.values())),
               "original_run_status": "partial_no_COMMIT", "policy": POLICY, "runtime": runtime,
               "pilot_bit_exact_matches": sorted(matches), "pilot_independent_audit_sha256": a.PILOT_AUDIT,
               "verification_scope": "all1656_output_PCM_and_receipt_source_hash_verification_plus_eight_pilot_replay_bindings",
               "full1656_native_DSP_replay_performed": False,
               "limitations": ["The original 1695 producer run remains partial and has no original COMMIT or manifest.",
                               "Duration exclusions use the pinned libsndfile decoder; FFmpeg observations are diagnostic only.",
                               "Only eight outputs carry independent native-to-output DSP replay evidence; numerical libraries are shared.",
                               "Physical intake closure is not cohort admission, feature extraction, fitting, or a classification claim."], **SCOPE}
    return manifest, summary, bindings, success_ids, set(failures)


def finalize(root, diagnosis_path, diagnosis_sha, run_path, run_sha, pilot, pilot_audit, output):
    root, diagnosis_path, run_path, pilot, pilot_audit, output = [Path(p).absolute() for p in
        (root, diagnosis_path, run_path, pilot, pilot_audit, output)]
    require(not output.exists() and not output.is_symlink(), "closure output already exists; no overwrite/resume")
    require(output.parent.is_dir() and not output.parent.is_symlink(), "regular existing output parent required")
    require(not output.resolve().is_relative_to(root.resolve()) and not root.resolve().is_relative_to(output.resolve()),
            "closure must be separate from original root")
    lock = root / "writer.lock"
    require(lock.is_file() and not lock.is_symlink(), "original writer lock required")
    with lock.open("rb") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        manifest, summary, bindings, successes, failures = verify(root, diagnosis_path, diagnosis_sha, run_path, run_sha, pilot, pilot_audit)
        for group in bindings.values():
            a.recheck(group)
        inventory(root, successes, failures)
        output.mkdir(exist_ok=False)
        # Publication is append-only. An interrupted output without COMMIT is
        # intentionally incomplete and cannot be reused by this CLI.
        a.write_exclusive(output / "manifest.json", manifest)
        a.write_exclusive(output / "summary.json", summary)
        a.write_exclusive(output / "upstream_bindings.json", bindings)
        closure_products = {name: a.bind_file(output / name) for name in
                            ("manifest.json", "summary.json", "upstream_bindings.json")}
        for group in bindings.values():
            a.recheck(group)
        a.recheck(closure_products)
        inventory(root, successes, failures)
        commit = {"status": STATUS, "version": "finalize_native30_partial_intake_v2", "contract_sha256": CONTRACT,
                  "original_root": str(root), "original_run_id": RUN, "original_run_status": "partial_no_COMMIT",
                  "attempted": TOTAL, "eligible": ELIGIBLE, "excluded_short": EXCLUDED, "excluded_mono_metadata": MONO,
                  "run_summary_sha256": run_sha, "diagnosis_sha256": diagnosis_sha,
                  "pilot_commit_sha256": a.PILOT_COMMIT, "pilot_independent_audit_sha256": a.PILOT_AUDIT,
                  "products": {name: {"sha256": b["sha256"], "bytes": b["signature"][2]} for name, b in closure_products.items()},
                  "all_bound_inputs_and_products_end_rehashed": True, **SCOPE}
        a.write_exclusive(output / "COMMIT.json", commit)
    return {"status": STATUS, "output": str(output), "commit_sha256": a.sha(output / "COMMIT.json"),
            "eligible": ELIGIBLE, "excluded_short": EXCLUDED}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "diagnosis", "diagnosis-sha256", "run-summary", "run-summary-sha256", "pilot", "pilot-audit", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = finalize(args.root, args.diagnosis, args.diagnosis_sha256, args.run_summary, args.run_summary_sha256,
                      args.pilot, args.pilot_audit, args.output)
    print(a.canonical(result).decode().strip())


if __name__ == "__main__":
    main()
