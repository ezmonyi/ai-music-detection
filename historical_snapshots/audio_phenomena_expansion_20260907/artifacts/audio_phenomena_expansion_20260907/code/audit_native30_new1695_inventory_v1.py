"""Independent batch inventory/PCM audit; not full native-DSP replay for 1695.

Does not import producer code. Verifies every committed product, source byte
hash, receipt, output PCM and crop metadata. Eight batch WAVs are compared with
pilot WAVs whose separate independent-replay receipt is immutably bound.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import uuid

import numpy as np


PILOT_COMMIT = "a008a59ec808aaf09a33e1147e72250ec4a35240c8c5a087ce2b6f622cad5498"
PILOT_AUDIT = "957c3936eb375c351e7e885a6af4cef76d4c49e058f1daffc4e6cf09c5a9aebb"
COUNT = 1695
MONO = 51
RATE = 44100
FRAMES = 1323000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(path):
    value = Path(path).stat(follow_symlinks=False)
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, stat.S_IFMT(value.st_mode)]


def bind_file(path, expected=None):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "regular non-symlink file required: " + str(path))
    before = signature(path)
    digest = sha(path)
    after = signature(path)
    require(before == after, "file changed while hashing: " + str(path))
    if expected is not None:
        require(digest == expected, "file SHA mismatch: " + str(path))
    return {"path": str(path), "sha256": digest, "signature": after}


def recheck(bindings):
    for value in bindings.values():
        require(bind_file(value["path"], value["sha256"]) == value, "bound file changed: " + value["path"])


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def vh(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def load(path):
    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=_pairs, parse_constant=invalid)


def validate_batch_documents(commit, contract, manifest, expected_contract):
    require(commit.get("status") == "committed_new1695_DSP_only" and commit.get("completed") == COUNT,
            "batch COMMIT status/count")
    require(commit.get("contract_sha256") == manifest.get("contract_sha256") == expected_contract,
            "contract join")
    require(contract.get("status") == "frozen_before_any_audio_processing" and
            contract.get("scope") == "new1695_native_stereo_DSP_only" and
            contract.get("expected_count") == COUNT, "contract status/scope/count")
    require(contract.get("classifier_fits") == 0 and contract.get("cohort_admitted") is False and
            contract.get("feature_extraction_authorized") is False, "contract no-admission scope")
    require(manifest.get("version") == "materialize_native30_new1695_v1" and
            manifest.get("status") == "all1695_DSP_materialized_not_cohort_admitted" and
            manifest.get("count") == COUNT, "manifest status/count")
    require(manifest.get("classifier_fits") == 0 and manifest.get("cohort_admitted") is False and
            manifest.get("feature_extraction_authorized") is False, "manifest no-admission scope")
    require(manifest.get("excluded_native_mono") == contract.get("excluded_native_mono") and
            len(contract.get("excluded_native_mono", [])) == MONO, "mono exclusion lineage")
    require(isinstance(commit.get("products"), dict) and commit["products"], "missing COMMIT products")


def validate_full_inventory(root, products):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "regular batch root required")
    required_root = {"COMMIT.json", "writer.lock", "contract.json", "manifest.json", "items", "audio", "failures", "runs"}
    require({p.name for p in root.iterdir()} == required_root, "unexpected batch root inventory")
    for name in ("COMMIT.json", "writer.lock", "contract.json", "manifest.json"):
        path = root / name
        require(path.is_file() and not path.is_symlink(), "unsafe batch root file: " + name)
    observed = {"contract.json", "manifest.json"}
    for name in ("items", "audio", "failures", "runs"):
        directory = root / name
        require(directory.is_dir() and not directory.is_symlink(), "unsafe batch directory: " + name)
        for path in directory.iterdir():
            require(path.is_file() and not path.is_symlink(), "nested/nonregular batch product: " + str(path))
            observed.add(str(path.relative_to(root)))
    require(observed == set(products), "extra or missing committed product")
    for name, value in products.items():
        require(isinstance(value, dict) and set(value) == {"bytes", "sha256"} and
                integer(value["bytes"], 0) and is_hash(value["sha256"]), "invalid product binding: " + name)
    return observed


def validate_pilot_replay(pilot_commit, pilot_audit, pilot_wavs):
    require(pilot_commit.get("status") == "committed_DSP_pilot_only", "pilot COMMIT status")
    require(len(pilot_wavs) == 8, "pilot WAV inventory")
    require(pilot_audit.get("status") == "passed_independent_pilot_DSP_replay" and
            pilot_audit.get("producer_imported") is False and pilot_audit.get("cohort_admitted") is False and
            pilot_audit.get("classifier_fits") == 0, "independent pilot audit status/scope")
    require(PILOT_COMMIT in pilot_audit.get("bindings", {}).values(), "independent audit/pilot COMMIT binding")
    records = pilot_audit.get("records", [])
    require(len(records) == 8 and len({r.get("id") for r in records}) == 8 and
            all(r.get("float32_bit_exact") is True and r.get("output_frames") == FRAMES for r in records),
            "independent pilot replay records")
    require({r["id"] for r in records} == set(pilot_wavs), "pilot replay/WAV identity mismatch")
    return {r["id"]: r for r in records}


def validate_receipt_metadata(receipt, row, audit, contract, expected_contract):
    ident = row["id"]
    require(receipt.get("status") == "materialized_DSP_only" and receipt.get("contract_sha256") == expected_contract,
            "receipt status/contract: " + ident)
    require(receipt.get("row") == row and receipt.get("row_sha256") == vh(row), "manifest row binding: " + ident)
    for key in ("id", "source_group", "group_id", "component_id", "label", "role"):
        require(receipt.get(key) == row.get(key), "receipt top-level lineage: " + ident + ":" + key)
    require(receipt.get("waveform_bit_exact_roundtrip") is True, "receipt roundtrip flag: " + ident)
    native = row["native_evidence"]
    rate = native["sample_rate_hz"]
    require(integer(rate, 1) and native.get("channels") == 2 and is_hash(native.get("sha256")), "native metadata: " + ident)
    require(audit.get("status") == "verified_DSP_only_not_cohort_admission" and
            audit.get("classifier_admitted") is False and audit.get("configuration") == contract["configuration"],
            "DSP audit status/configuration: " + ident)
    require(audit.get("source_path") == row["execution_native_path"] and
            audit.get("native_rate_hz") == rate and audit.get("native_channels") == 2,
            "DSP audit native lineage: " + ident)
    sequential = audit.get("sequential_decode", {})
    actual = sequential.get("actual_frames")
    require(integer(actual, 30 * rate) and actual <= 60 * rate and sequential.get("empty_eof_observed") is True and
            integer(sequential.get("read_calls_including_empty_eof"), 1) and is_hash(sequential.get("float64_pcm_sha256")),
            "native sequential decode metadata: " + ident)
    header = sequential.get("header", {})
    require(header.get("sample_rate_hz") == rate and header.get("channels") == 2 and integer(header.get("frames"), 0) and
            isinstance(header.get("format"), str) and header.get("format") and isinstance(header.get("subtype"), str) and header.get("subtype"),
            "native header metadata: " + ident)
    require(audit.get("header_minus_actual_frames") == header["frames"] - actual, "header/actual difference: " + ident)
    start = (actual - 30 * rate) // 2
    coordinates = {"region_start_frame": 0, "region_frames": actual, "crop_start_frame": start,
                   "crop_frames": 30 * rate, "crop_end_frame_exclusive": start + 30 * rate}
    require(audit.get("region_kind") == "full_native_sequential_decode" and
            audit.get("approved_region_float64_sha256") is None and audit.get("coordinates") == coordinates and
            audit.get("sequential_passes_identical") is True and is_hash(audit.get("native_crop_float64_sha256")),
            "native crop metadata: " + ident)
    require(audit.get("output_rate_hz") == RATE and audit.get("output_channels") == 2 and
            audit.get("output_frames") == FRAMES and audit.get("sample_count") == FRAMES * 2 and
            is_hash(audit.get("output_waveform_float32_sha256")), "DSP output metadata: " + ident)
    for key in ("python", "numpy", "scipy", "soundfile", "libsndfile"):
        require(audit.get("runtime", {}).get(key) == contract.get("runtime", {}).get(key),
                "DSP runtime lineage: " + ident + ":" + key)
    return rate, actual, coordinates


def audit(root, expected_commit, expected_contract, pilot, pilot_audit_path):
    import soundfile as sf

    root, pilot, pilot_audit_path = Path(root).absolute(), Path(pilot).absolute(), Path(pilot_audit_path).absolute()
    inputs = {
        "batch_commit": bind_file(root / "COMMIT.json", expected_commit),
        "batch_contract": bind_file(root / "contract.json", expected_contract),
        "batch_manifest": bind_file(root / "manifest.json"),
        "pilot_commit": bind_file(pilot / "COMMIT.json", PILOT_COMMIT),
        "pilot_audit": bind_file(pilot_audit_path, PILOT_AUDIT),
    }
    commit, contract, manifest = (load(root / name) for name in ("COMMIT.json", "contract.json", "manifest.json"))
    pc, pa = load(pilot / "COMMIT.json"), load(pilot_audit_path)
    validate_batch_documents(commit, contract, manifest, expected_contract)
    validate_full_inventory(root, commit["products"])
    product_bindings = {}
    total_bytes = 0
    for name, value in commit["products"].items():
        binding = bind_file(root / name, value["sha256"])
        require(binding["signature"][2] == value["bytes"], "product byte count: " + name)
        product_bindings[name] = binding
        total_bytes += value["bytes"]
    pilot_products = pc.get("products", {})
    pilot_wavs = {Path(name).stem: (name, value) for name, value in pilot_products.items() if name.startswith("audio/") and name.endswith(".wav")}
    replay = validate_pilot_replay(pc, pa, pilot_wavs)
    pilot_bindings = {ident: bind_file(pilot / name, value["sha256"]) for ident, (name, value) in pilot_wavs.items()}
    rows = {row["id"]: row for row in contract["rows"]}
    records = manifest["records"]
    require(len(contract["rows"]) == len(rows) == len(records) == COUNT, "cohort counts/duplicate contract rows")
    require(len({record.get("row", {}).get("id") for record in records}) == COUNT, "duplicate manifest rows")
    compared, deficits, overshoot = [], [], []
    rates, sources, labels = Counter(), Counter(), Counter()
    source_bindings = {}
    samples = 0
    for number, receipt in enumerate(records, 1):
        ident = receipt.get("row", {}).get("id")
        require(ident in rows, "manifest ID absent from contract")
        row, payload_audit = rows[ident], receipt["audit"]
        rate, actual, coordinates = validate_receipt_metadata(receipt, row, payload_audit, contract, expected_contract)
        item_path = root / "items" / (ident + ".json")
        envelope = load(item_path)
        require(set(envelope) == {"payload", "receipt_sha256"} and envelope["payload"] == receipt and
                envelope["receipt_sha256"] == vh(receipt), "receipt envelope: " + ident)
        source = Path(row["execution_native_path"])
        source_binding = bind_file(source, row["native_evidence"]["sha256"])
        old = source_bindings.get(str(source.absolute()))
        require(old is None or old == source_binding, "conflicting repeated native source: " + str(source))
        source_bindings[str(source.absolute())] = source_binding
        require(payload_audit["source_sha256_before"] == payload_audit["source_sha256_after"] == source_binding["sha256"],
                "native byte identity: " + ident)
        require(payload_audit.get("source_stat_signature") == source_binding["signature"][:5],
                "native stat-signature lineage: " + ident)
        wav = root / "audio" / (ident + ".wav")
        require(str(wav) == receipt["standardized_path"] and is_hash(receipt.get("file_sha256")) and
                integer(receipt.get("file_bytes"), 1) and wav.stat().st_size == receipt["file_bytes"] and
                sha(wav) == receipt["file_sha256"], "WAV lineage: " + ident)
        with sf.SoundFile(wav) as stream:
            require((stream.samplerate, stream.channels, stream.frames, stream.format, stream.subtype) ==
                    (RATE, 2, FRAMES, "WAV", "FLOAT"), "WAV format: " + ident)
            values = stream.read(dtype="float32", always_2d=True)
            require(len(stream.read(1, dtype="float32", always_2d=True)) == 0, "output EOF: " + ident)
        require(values.shape == (FRAMES, 2) and np.isfinite(values).all(), "output shape/finite: " + ident)
        pcm_sha = hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()
        require(pcm_sha == receipt["waveform_float32_sha256"] == payload_audit["output_waveform_float32_sha256"],
                "output PCM SHA: " + ident)
        peak, above = float(np.max(np.abs(values))), int(np.count_nonzero(np.abs(values) > 1))
        require(peak == payload_audit["peak_absolute"] and above == payload_audit["samples_abs_above_one"],
                "quality recount: " + ident)
        if above:
            overshoot.append({"id": ident, "source": row["source_group"], "peak": peak, "channel_samples": above})
        if payload_audit["header_minus_actual_frames"]:
            deficits.append({"id": ident, "difference": payload_audit["header_minus_actual_frames"]})
        if ident in pilot_wavs:
            require(receipt["file_sha256"] == pilot_bindings[ident]["sha256"] and replay[ident]["float32_bit_exact"] is True,
                    "pilot/batch/replay mismatch: " + ident)
            compared.append(ident)
        samples += values.size; rates[rate] += 1; sources[row["source_group"]] += 1; labels[row["label"]] += 1
        if number % 100 == 0:
            print(json.dumps({"event": "inventory_audit_progress", "verified": number}), flush=True)
    require(set(compared) == set(pilot_wavs) == set(replay), "missing pilot overlap")
    recheck(inputs); recheck(product_bindings); recheck(source_bindings); recheck(pilot_bindings)
    require(load(root / "COMMIT.json") == commit and load(root / "contract.json") == contract and
            load(root / "manifest.json") == manifest and load(pilot_audit_path) == pa, "bound JSON changed")
    return {"status": "all1695_inventory_and_pcm_verified_not_full1695_native_DSP_replay",
            "commit_sha256": expected_commit, "contract_sha256": expected_contract,
            "pilot_independent_audit_sha256": PILOT_AUDIT, "products": len(commit["products"]),
            "product_bytes": total_bytes, "records": COUNT, "channel_samples_checked": samples,
            "input_bindings": inputs,
            "end_recheck": {"committed_products": len(product_bindings),
                            "unique_native_sources": len(source_bindings), "pilot_wavs": len(pilot_bindings),
                            "all_signatures_and_sha256_unchanged": True},
            "source_counts": dict(sources), "labels": dict(labels), "native_rates": dict(rates),
            "pilot_bit_exact_matches": sorted(compared), "header_frame_differences": deficits,
            "float_overshoot_records": overshoot, "classifier_fits": 0, "cohort_admitted": False,
            "limitations": ["Only eight outputs have an independently replayed native-to-output DSP reference; remaining outputs were checked against immutable receipts, native byte hashes, output PCM hashes, and internally consistent crop metadata.",
                            "Numerical libraries are shared; this is not an independent 1695-source decoder or DSP replay."]}


def write_exclusive(path, value):
    path = Path(path)
    require(path.parent.is_dir() and not path.parent.is_symlink(), "regular output parent required")
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    for key in ("root", "commit-sha256", "contract-sha256", "pilot", "pilot-audit", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    script = Path(__file__).absolute()
    script_binding = bind_file(script)
    result = audit(Path(args.root), args.commit_sha256, args.contract_sha256, Path(args.pilot), Path(args.pilot_audit))
    require(bind_file(script) == script_binding, "auditor changed")
    result["auditor_binding"] = script_binding
    write_exclusive(Path(args.output), result)
    print(json.dumps({"status": result["status"], "receipt_sha256": sha(args.output), "records": result["records"]}))


if __name__ == "__main__":
    main()
