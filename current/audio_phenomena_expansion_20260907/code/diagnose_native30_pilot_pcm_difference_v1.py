"""Read-only eight-pilot container/PCM comparison; no DSP generation or writes."""
import argparse
import hashlib
from pathlib import Path
import struct

import numpy as np
import soundfile as sf
import audit_native30_new1695_inventory_v1 as a

CONTRACT = "7cf9b3d2436958c8e7d0d4e5121ed317b04a1126abaafe6ba97022e88dc82e4a"
AUDITOR = "b04153059a8f7e81f7d11ca865914d9f341129025d6b606af18e1b617a2985b0"


def chunks(raw):
    a.require(raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", "RIFF/WAVE header")
    a.require(struct.unpack_from("<I", raw, 4)[0] + 8 == len(raw), "RIFF declared size")
    result, offset = [], 12
    while offset < len(raw):
        a.require(offset + 8 <= len(raw), "truncated RIFF chunk header")
        name = raw[offset:offset + 4].decode("ascii")
        size = struct.unpack_from("<I", raw, offset + 4)[0]
        end = offset + 8 + size
        a.require(end <= len(raw), "truncated RIFF payload")
        record = {"id": name, "header_offset": offset, "payload_offset": offset + 8, "payload_bytes": size,
                  "payload_sha256": hashlib.sha256(raw[offset + 8:end]).hexdigest()}
        if name != "data" and size <= 256:
            record["payload_hex"] = raw[offset + 8:end].hex()
        if name == "PEAK":
            a.require(size >= 8, "short PEAK chunk")
            record.update(version=struct.unpack_from("<I", raw, offset + 8)[0],
                          timestamp_unix=struct.unpack_from("<I", raw, offset + 12)[0],
                          timestamp_byte_offsets=list(range(offset + 12, offset + 16)))
        result.append(record)
        offset = end + (size & 1)
    a.require(offset == len(raw), "RIFF final alignment")
    return result


def pcm(path):
    with sf.SoundFile(path) as stream:
        a.require((stream.frames, stream.samplerate, stream.channels, stream.format, stream.subtype) ==
                  (a.FRAMES, a.RATE, 2, "WAV", "FLOAT"), "float32 WAV format")
        values = stream.read(dtype="float32", always_2d=True)
        a.require(len(stream.read(1, dtype="float32", always_2d=True)) == 0, "WAV empty EOF")
    a.require(np.isfinite(values).all(), "nonfinite WAV")
    values = np.ascontiguousarray(values, dtype="<f4")
    return values, hashlib.sha256(values.tobytes()).hexdigest()


def run(root, pilot, audit_path, output):
    root, pilot, audit_path, output = [Path(p).absolute() for p in (root, pilot, audit_path, output)]
    bindings = {"code": a.bind_file(Path(__file__).absolute()), "auditor": a.bind_file(Path(a.__file__).absolute(), AUDITOR),
                "contract": a.bind_file(root / "contract.json", CONTRACT),
                "pilot_commit": a.bind_file(pilot / "COMMIT.json", a.PILOT_COMMIT),
                "pilot_independent_audit": a.bind_file(audit_path, a.PILOT_AUDIT)}
    contract = a.load(root / "contract.json"); rows = {r["id"]: r for r in contract["rows"]}
    commit, independent = a.load(pilot / "COMMIT.json"), a.load(audit_path)
    for name, declared in commit["products"].items():
        bindings[str(pilot / name)] = a.bind_file(pilot / name, declared["sha256"])
        a.require(bindings[str(pilot / name)]["signature"][2] == declared["bytes"], "pilot product byte count")
    pilot_contract = a.load(pilot / "contract.json")
    pilot_rows = {r["id"]: r for r in pilot_contract["rows"]}
    wavs = {Path(name).stem: name for name in commit["products"] if name.startswith("audio/") and name.endswith(".wav")}
    replay = a.validate_pilot_replay(commit, independent, wavs)
    records = []
    for ident in sorted(wavs):
        wav, reference = root / "audio" / (ident + ".wav"), pilot / wavs[ident]
        item = root / "items" / (ident + ".json")
        bindings[str(item)] = a.bind_file(item)
        envelope = a.load(item); receipt = envelope["payload"]
        a.require(set(envelope) == {"payload", "receipt_sha256"} and envelope["receipt_sha256"] == a.vh(receipt), "batch envelope")
        row = rows[ident]
        a.validate_receipt_metadata(receipt, row, receipt["audit"], contract, CONTRACT)
        prior = a.load(pilot / "items" / (ident + ".json"))
        bindings[str(wav)] = a.bind_file(wav, receipt["file_sha256"])
        a.require(a.sha(reference) == prior["file_sha256"], "pilot receipt/WAV file hash")
        current_pcm, current_sha = pcm(wav); prior_pcm, prior_sha = pcm(reference)
        a.require(current_sha == receipt["waveform_float32_sha256"] == receipt["audit"]["output_waveform_float32_sha256"], "batch PCM receipt binding")
        a.require(prior_sha == prior["audit"]["output_waveform_float32_sha256"], "pilot PCM receipt binding")
        before, after = reference.read_bytes(), wav.read_bytes()
        a.require(len(before) == len(after), "container length mismatch")
        differences = np.flatnonzero(np.frombuffer(before, dtype=np.uint8) != np.frombuffer(after, dtype=np.uint8)).tolist()
        prior_chunks, current_chunks = chunks(before), chunks(after)
        offset_map = []
        for offset in differences:
            matches = [ch for ch in current_chunks if ch["header_offset"] <= offset < ch["payload_offset"] + ch["payload_bytes"]]
            a.require(len(matches) == 1, "difference outside a unique RIFF chunk")
            chunk = matches[0]
            offset_map.append({"file_offset": offset, "chunk": chunk["id"], "chunk_payload_offset": offset - chunk["payload_offset"],
                               "pilot_byte": before[offset], "batch_byte": after[offset],
                               "is_peak_timestamp_byte": offset in chunk.get("timestamp_byte_offsets", [])})
        native_bindings = {}
        for label, path, digest in (("batch", row["execution_native_path"], row["native_evidence"]["sha256"]),
                                     ("pilot", prior["audit"]["source_path"], pilot_rows[ident]["native_evidence"]["sha256"])):
            if path not in bindings: bindings[path] = a.bind_file(path, digest)
            a.require(bindings[path]["sha256"] == digest, "native source hash binding")
            native_bindings[label] = bindings[path]
        ca, pa = receipt["audit"], prior["audit"]
        fields = ("configuration", "native_rate_hz", "native_channels", "coordinates", "sequential_decode",
                  "native_crop_float64_sha256", "region_kind", "approved_region_float64_sha256",
                  "output_waveform_float32_sha256", "runtime")
        field_comparison = {key: {"equal": ca.get(key) == pa.get(key), "batch": ca.get(key), "pilot": pa.get(key)} for key in fields}
        records.append({"id": ident, "source_group": row["source_group"], "batch_wav": bindings[str(wav)],
                        "pilot_wav": bindings[str(reference)], "file_bytes_equal": before == after,
                        "file_byte_difference_count": len(differences), "differing_bytes": offset_map,
                        "batch_chunks": current_chunks, "pilot_chunks": prior_chunks,
                        "batch_float32_pcm_sha256": current_sha, "pilot_float32_pcm_sha256": prior_sha,
                        "pcm_bit_exact": current_pcm.tobytes() == prior_pcm.tobytes(),
                        "max_abs_sample_error": float(np.max(np.abs(current_pcm.astype(np.float64) - prior_pcm.astype(np.float64)))),
                        "only_peak_timestamp_bytes_differ": bool(differences) and all(v["is_peak_timestamp_byte"] for v in offset_map),
                        "original_native_evidence_equal": row["native_evidence"] == pilot_rows[ident]["native_evidence"],
                        "native_file_sha256_equal": native_bindings["batch"]["sha256"] == native_bindings["pilot"]["sha256"],
                        "native_bindings": native_bindings, "audit_field_comparison": field_comparison,
                        "independent_replay_record": replay[ident]})
    a.recheck(bindings)
    result = {"status": "eight_pilot_container_and_pcm_comparison_completed", "records": records,
              "count": len(records), "pcm_bit_exact_count": sum(r["pcm_bit_exact"] for r in records),
              "whole_file_equal_count": sum(r["file_bytes_equal"] for r in records),
              "peak_timestamp_only_difference_count": sum(r["only_peak_timestamp_bytes_differ"] for r in records),
              "input_bindings": bindings, "all_bound_files_end_rehashed": True,
              "native_DSP_replayed": False, "cohort_admitted": False, "classifier_fits": 0,
              "feature_extraction_authorized": False}
    a.write_exclusive(output, result)
    print(a.canonical({"output": str(output), "sha256": a.sha(output), **{k:result[k] for k in
          ("count", "pcm_bit_exact_count", "whole_file_equal_count", "peak_timestamp_only_difference_count")}}).decode().strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "pilot", "pilot-audit", "output"): parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(args.root, args.pilot, args.pilot_audit, args.output)
