#!/usr/bin/env python3
"""Bind completed exact60 recovery provenance into a NEW extraction receipt.

Offline, no model imports/fitting/inference. Audio/stem/spectrogram bytes are
not reopened: their stage-receipt hashes are compared to the full strict audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import prepare_evaluation_inputs_v4 as P

EMPTY_SHA = hashlib.sha256(b"").hexdigest()
CHECKPOINT = "8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331"
RECOVERY_CODES = {
    "recover_equal60_empty_beats.py": "253bb11247e766eb737e7204dfbe87ae1d029f5db9a7f6bd723c04b6704846d0",
    "recover_equal60_empty_beats_v2.py": "323dcd8c0f5440b88e67acb784b98833929ce6e6c1a9b86ea75309fe5a995169",
}
UPSTREAM_CODES = {
    "inference.py": "ccccff4665399b931ebf181b73d264a6ad64fc1fe8acb4d3b950df825c47ce2d",
    "postprocessor.py": "431d6aeef41a8e9e9e63ee2577e895a5e214e0dbbcd2d505458cdfec02a611ee",
    "utils.py": "779d6102ea28bdf94c8227157790191f5abbbb4564d0c36ce5ce0859e067cfab",
}
R_COLUMNS = ["r__ibi_cv", "r__tempo_tv", "r__tempo_entropy"]
BAR_COLUMNS = ["p__section_bars_cv", "p__section_bars_offmode_fraction", "p__section_bars_median"]
DURATION_COLUMNS = ["p__section_duration_cv", "p__section_duration_entropy", "p__section_duration_median"]
REPLAY_API = "beat_this.inference.File2Beats(checkpoint, device=cuda:gpu, float16=True, dbn=False)(input_path)"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return P.sha(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def parse_json(text):
    def pairs(items):
        require(len({k for k, _ in items}) == len(items), "Duplicate JSON key")
        return dict(items)
    def constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    out = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    require(isinstance(out, dict), "Expected JSON object")
    return out


class Snapshot:
    """Hash exactly the bytes read, then verify all small evidence again at exit."""
    def __init__(self):
        self.files = {}
        self.directories = {}

    def inventory(self, path):
        path = Path(path).resolve()
        require(path.is_dir(), "Missing evidence directory: " + str(path))
        names = {p.name for p in path.iterdir()}
        require(str(path) not in self.directories or self.directories[str(path)] == names, "Evidence inventory changed")
        self.directories[str(path)] = names
        return names

    def read(self, path):
        path = Path(path).resolve()
        require(path.is_file(), "Missing evidence: " + str(path))
        raw = path.read_bytes()
        record = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        require(str(path) not in self.files or self.files[str(path)] == record, "Evidence changed between reads")
        self.files[str(path)] = record
        return raw

    def json(self, path, embed=True):
        raw = self.read(path)
        text = raw.decode("utf-8")
        value = parse_json(text)
        envelope = {"path": str(Path(path).resolve()), **self.files[str(Path(path).resolve())]}
        if embed:
            envelope.update(content=value, original_utf8=text)
        return value, envelope

    def check(self):
        for path, expected in self.files.items():
            require(Path(path).is_file() and Path(path).stat().st_size == expected["bytes"] and sha(path) == expected["sha256"], "Evidence changed during binding: " + path)
        for path, expected in self.directories.items():
            require({p.name for p in Path(path).iterdir()} == expected, "Evidence inventory changed during binding: " + path)


def rows_from_bytes(raw):
    import io
    reader = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""))
    header = next(reader, [])
    require(bool(header) and len(header) == len(set(header)), "Malformed/duplicate feature header")
    values = list(reader)
    require(all(len(row) == len(header) for row in values), "Malformed feature row")
    rows = [dict(zip(header, row)) for row in values]
    require(rows and all(row.get("item_id") for row in rows) and len({r["item_id"] for r in rows}) == len(rows), "Missing/duplicate feature IDs")
    return rows


def missing(value):
    return str(value).strip().lower() in {"", "nan", "na", "null"}


def validate_arrays(sidecar):
    beats, downbeats = sidecar["beats"], sidecar["downbeats"]
    require(beats == [] and isinstance(downbeats, list) and bool(downbeats), "Not empty-beat/nonempty-downbeat failure")
    require(all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 60 for x in downbeats), "Invalid orphan downbeats")
    require(all(a < b for a, b in zip(downbeats, downbeats[1:])), "Unordered/duplicate orphan downbeats")


def product_hashes(root, item, stage):
    item_id = item["item_id"]
    if stage == "beats":
        return {str(root / "beats" / (item_id + ".beats")): item["beats"]["sha256"]}
    return {**{str(root / "demix/htdemucs" / item_id / (stem + ".wav")): item["stems"][stem]["sha256"] for stem in ("bass", "drums", "other", "vocals")},
            str(root / "structure" / (item_id + ".json")): item["structure"]["sha256"],
            str(root / "spec" / (item_id + ".npy")): item["spectrogram"]["sha256"]}


def original_command(contract, items):
    runtime = Path(contract["runtime_root"])
    return [str(runtime / "venv/bin/beat_this"), *[r["input"]["path"] for r in items], "-o", str(Path(contract["output_root"]) / "beats"),
            "--model", str(runtime / "checkpoints/hub/checkpoints/beat_this-final0.ckpt"), "--no-dbn", "--gpu", "0", "--float16", "--skip-existing"]


def validate_invocation(invocation, sidecar, root):
    require(isinstance(invocation, list) and len(invocation) == 8 and all(isinstance(x, str) for x in invocation), "Malformed recovery invocation")
    require(Path(invocation[0]).name.startswith("python") and Path(invocation[1]).name == "recover_equal60_empty_beats_v2.py", "Wrong recovery executable/script")
    require(invocation[2:] == ["--prepared-dir", str(root.parent), "--output-root", str(root), "--gpu", str(sidecar["gpu"])], "Changed recovery invocation arguments")


def validate_sidecar(sidecar, item, feature, receipt, contract_sha, root):
    validate_arrays(sidecar)
    require(sidecar["item_id"] == item["item_id"] and sidecar["input_path"] == item["input"]["path"] and sidecar["input_sha256"] == item["input"]["sha256"], "Recovery item/input mismatch")
    require(sidecar["run_contract_sha256"] == contract_sha and sidecar["recovery_code_sha256"] == receipt["recovery_code_sha256"], "Recovery contract/code mismatch")
    require(sidecar["checkpoint_sha256"] == CHECKPOINT and sidecar["float16"] is True and sidecar["dbn"] is False, "Recovery model/precision/DBN mismatch")
    require(type(sidecar["gpu"]) is int and sidecar["gpu"] == receipt["gpu"] and "RTX 5090" in sidecar["gpu_name"], "Recovery GPU mismatch")
    require(sidecar["error"] == "Not all downbeats are beats." and sidecar["status"] == "verified_unavailable_no_beat_events", "Wrong recovery failure/status")
    require(sidecar["original_log_sha256"] == receipt["log_sha256"], "Recovery log mismatch")
    dependencies = sidecar["dependency_sha256"]
    require(len(dependencies) == 3 and {Path(p).name: value for p, value in dependencies.items()} == UPSTREAM_CODES, "Changed upstream recovery code")
    require(item["beats"] == {**item["beats"], "sha256": EMPTY_SHA, "status": "empty_unavailable", "beat_count": 0}, "Recovered strict-audit beats are not empty/unavailable")
    require(all(missing(feature[c]) for c in R_COLUMNS + BAR_COLUMNS), "Recovered R/P-bar metric must be NaN, never zero")
    require(feature["r_eligible"] == "0" and feature["beat_output_status"] == "empty" and feature["n_beats"] == "0" and feature["n_downbeats"] == "0", "Recovered feature availability/count mismatch")
    require(all(missing(feature[c]) or math.isfinite(float(feature[c])) for c in DURATION_COLUMNS), "Invalid P duration feature")
    if "recovery_invocation" in receipt:
        require(sidecar.get("recovery_invocation") == receipt["recovery_invocation"] and sidecar.get("recovery_api") == REPLAY_API, "Sidecar/receipt replay mismatch")
        validate_invocation(receipt["recovery_invocation"], sidecar, root)


def bind(strict_audit_dir, inference_root, output_dir, recovery_protocol, code_dir=None, synthetic=False):
    audit_dir, root, output = Path(strict_audit_dir).resolve(), Path(inference_root).resolve(), Path(output_dir).resolve()
    code_dir = Path(code_dir or Path(__file__).parent).resolve()
    require(not output.exists(), "Refusing existing output directory")
    snap = Snapshot()
    receipt, original_envelope = snap.json(audit_dir / "extraction_receipt.json")
    require("recovery_evidence_binding" not in receipt, "Receipt is already derivative/bound")
    audit, audit_envelope = snap.json(audit_dir / "strict_inference_audit.json", False)
    contract, contract_envelope = snap.json(root / "run_contract.json")
    completion, completion_envelope = snap.json(root / "completion.json")
    require(receipt.get("status") == audit.get("status") == completion.get("status") == "passed", "Full completed audit/extraction/inference required")
    require(receipt.get("classifier_fitted") is False and receipt.get("no_short_padding_possible") is True, "Unsafe extraction receipt")
    require(receipt["strict_inference_audit_sha256"] == audit_envelope["sha256"], "Original strict-audit hash mismatch")
    features_path = audit_dir / "features/expanded_features_60s.csv"
    features = rows_from_bytes(snap.read(features_path))
    require(receipt["features_sha256"] == snap.files[str(features_path)]["sha256"], "Full features SHA mismatch")
    items = audit["items"]
    require(isinstance(items, list) and len({r["item_id"] for r in items}) == len(items), "Duplicate strict-audit item")
    by_id = {r["item_id"]: r for r in items}
    feature_by_id = {r["item_id"]: r for r in features}
    count = len(items)
    require(set(by_id) == set(feature_by_id) and receipt["rows"] == audit["rows"] == contract["rows"] == completion["rows"] == count, "Full row count/ID mismatch")
    if synthetic:
        require(0 < count < 500 and all(i.startswith("synthetic_v4_") for i in by_id), "Synthetic mode requires small synthetic_v4_ corpus")
    else:
        require(count == 1604 and audit["manifest_sha256"] == P.PREPARED_SHA and audit["materialization_contract_sha256"] == P.MATERIALIZATION_CONTRACT, "Frozen full1604 manifest/context mismatch")
    require(audit["manifest_sha256"] == contract["manifest_sha256"] and completion["run_contract_sha256"] == contract_envelope["sha256"], "Inference completion/manifest contract mismatch")
    require(contract["output_root"] == str(root), "Inference output root mismatch")
    require(contract["runtime_code_checkpoint_bias_checks"] == audit["runtime_code_checkpoint_bias_checks"], "Runtime/checkpoint audit mismatch")
    require(CHECKPOINT in contract["runtime_code_checkpoint_bias_checks"].values(), "Frozen Beat This checkpoint absent")
    require(audit["code_sha256"] == hashlib.sha256(snap.read(code_dir / "verify_extract_equal60.py")).hexdigest(), "Strict auditor code changed")
    require(contract["code_sha256"] == hashlib.sha256(snap.read(code_dir / "run_equal60_inference_batches.py")).hexdigest(), "Frozen runner code changed")
    require(set(contract["dependencies"]) == {"verify_extract_equal60.py", "materialize_equal60_inputs_v2.py"}, "Incomplete runner dependency graph")
    for name, expected in contract["dependencies"].items():
        require(hashlib.sha256(snap.read(code_dir / name)).hexdigest() == expected, "Runner dependency changed: " + name)
    for item in items:
        require(set(item["stems"]) == {"bass", "drums", "other", "vocals"}, "Incomplete strict stem audit")
        for audio in [item["input"], *item["stems"].values()]:
            require(audio["frames"] == 2646000 and audio["sample_rate"] == 44100 and audio["channels"] == 2 and audio["finite"] is True, "Strict exact60 audio mismatch")
        require(item["input"]["subtype"] == "FLOAT" and item["structure"]["input_path"] == item["input"]["path"] and item["spectrogram"]["shape"] == [4, 6000, 81], "Strict representation mismatch")
        feature = feature_by_id[item["item_id"]]
        require(feature["status"] == "complete" and float(feature["duration_sec"]) == 60, "Incomplete/wrong-duration feature row")
        hashes = parse_json(feature["input_hashes"])
        expected_hashes = {"source_audio_sha256": item["input"]["sha256"], **{s + "_sha256": item["stems"][s]["sha256"] for s in item["stems"]}, "beats_sha256": item["beats"]["sha256"], "structure_sha256": item["structure"]["sha256"]}
        require(hashes == expected_hashes and all(re.fullmatch("[0-9a-f]{64}", h) for h in hashes.values()), "Feature/strict-audit input hash mismatch")
    shards = contract["shards"]
    require(shards and [s["index"] for s in shards] == list(range(len(shards))) and sum(s["rows"] for s in shards) == count, "Invalid shard counts/indexes")
    require(completion["completed_stage_shards"] == 2 * len(shards) and (synthetic or len(shards) == 67), "Completion must cover all134 stage-shards")
    names = {f"{stage}_{s['index']:02d}.json" for s in shards for stage in ("beats", "allinone")}
    require(snap.inventory(root / "receipts") == names, "Missing/extra stage receipts")
    protocol_raw = snap.read(recovery_protocol)
    protocol_text = protocol_raw.decode("utf-8")
    require("v1 receipt's `command`" in protocol_text and "NOT the" in protocol_text and "--gpu 5" in protocol_text and "recover_equal60_empty_beats.py --prepared-dir" in protocol_text, "Missing explicit v1 protocol correction")
    recovery_codes = {}
    for name, expected in RECOVERY_CODES.items():
        body = snap.read(code_dir / name)
        require(hashlib.sha256(body).hexdigest() == expected, "Unreviewed recovery code version: " + name)
        recovery_codes[name] = {"sha256": expected, "original_utf8": body.decode("utf-8")}
    all_receipts, beat_receipts, sidecars, corrections = {}, {}, {}, []
    stage_ids = {"beats": [], "allinone": []}
    shard_ids = {}
    for shard in shards:
        for stage in ("beats", "allinone"):
            name = f"{stage}_{shard['index']:02d}.json"
            current, envelope = snap.json(root / "receipts" / name)
            all_receipts[name] = {k: v for k, v in envelope.items() if k not in {"content", "original_utf8"}}
            if stage == "beats":
                beat_receipts[name] = envelope
            require(current["status"] == "passed" and current["stage"] == stage and current["shard_index"] == shard["index"] and current["run_contract_sha256"] == contract_envelope["sha256"], "Stage receipt identity/contract mismatch")
            ids = current["item_ids"]
            require(len(ids) == shard["rows"] and len(set(ids)) == len(ids) and set(ids) <= set(by_id), "Stage receipt row/ID mismatch")
            reconstructed_shard = "".join(by_id[i]["input"]["path"] + "\n" for i in ids).encode()
            require(hashlib.sha256(reconstructed_shard).hexdigest() == shard["sha256"], "Stage IDs/order differ from frozen shard SHA")
            require(shard["index"] not in shard_ids or shard_ids[shard["index"]] == ids, "Stage shard ID order mismatch")
            shard_ids[shard["index"]] = ids
            stage_ids[stage].extend(ids)
            products = {p: h for item_id in ids for p, h in product_hashes(root, by_id[item_id], stage).items()}
            refs = current.get("recovery_sidecars", [])
            recovering = current.get("recovery_status") is not None
            require(recovering == bool(refs) and len(refs) == len(set(refs)), "Missing/duplicate recovery references")
            require(not recovering or (stage == "beats" and current["recovery_status"] == "same_model_replay_verified_empty_unavailable"), "Recovery attached to invalid stage/status")
            require(set(current["outputs"]) == set(products) | set(refs), "Missing/extra receipt products/recovery records")
            for path, expected in products.items():
                value = current["outputs"][path]
                require(set(value) == {"sha256", "bytes"} and value["sha256"] == expected and type(value["bytes"]) is int and value["bytes"] >= 0, "Product receipt differs from strict audit")
                if stage == "beats":
                    require((value["bytes"] == 0) == (expected == EMPTY_SHA), "Beat size/empty hash contradiction")
            if not recovering:
                require(not {"recovery_sidecars", "recovery_code_sha256", "recovery_invocation", "preserved_existing_sha256"} & set(current), "Unclassified recovery receipt fields")
                continue
            require(current["recovery_code_sha256"] in RECOVERY_CODES.values(), "Unknown recovery adapter SHA")
            recovered_ids = []
            log_path = root / "logs" / f"beats_{shard['index']:02d}.log"
            log_raw = snap.read(log_path)
            require(hashlib.sha256(log_raw).hexdigest() == current["log_sha256"], "Original recovery log hash mismatch")
            for ref in refs:
                path = Path(ref)
                require(path.parent == root / "recovery_empty_beats" and path.suffix == ".json" and str(path) not in sidecars, "Duplicate/out-of-scope recovery sidecar")
                sidecar, side_envelope = snap.json(path)
                require(sidecar["item_id"] in ids and path.name == sidecar["item_id"] + ".json", "Sidecar item/shard mismatch")
                require(current["outputs"][ref] == {k: side_envelope[k] for k in ("sha256", "bytes")}, "Sidecar receipt hash mismatch")
                item = by_id[sidecar["item_id"]]
                validate_sidecar(sidecar, item, feature_by_id[sidecar["item_id"]], current, contract_envelope["sha256"], root)
                target = root / "beats" / (sidecar["item_id"] + ".beats")
                require(snap.read(target) == b"", "Recovered beat file must be truly zero bytes")
                sidecars[str(path)] = side_envelope
                recovered_ids.append(sidecar["item_id"])
            require(len(recovered_ids) == len(set(recovered_ids)), "Repeated recovered item")
            failed = re.findall(r'Could not process "([^"]+)"\. Rerun with this file alone for details\.', log_raw.decode("utf-8"))
            require(len(failed) == len(set(failed)) and set(failed) == {by_id[i]["input"]["path"] for i in recovered_ids}, "Recovered items differ from logged failures")
            expected_preserved = {p: h for i in ids if i not in recovered_ids for p, h in product_hashes(root, by_id[i], "beats").items()}
            require(current["preserved_existing_sha256"] == expected_preserved, "Preserved successful outputs mismatch")
            original = original_command(contract, [by_id[i] for i in ids])
            if current["recovery_code_sha256"] == RECOVERY_CODES["recover_equal60_empty_beats.py"]:
                require("recovery_invocation" not in current and current.get("command") == original, "Malformed v1 command provenance")
                require(synthetic or (shard["index"] == 19 and recovered_ids == ["aime_mtg_jamendo_06005"]), "Undocumented v1 recovery")
                corrections.append({"stage_receipt": name, "item_ids": recovered_ids, "original_field": "command",
                                    "corrected_semantics": "original failed shard CLI; not the executed recovery call",
                                    "actual_recovery_invocation_evidence": "documented in embedded reviewed protocol, not recorded in original v1 receipt",
                                    "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(), "original_receipt_preserved": True})
            else:
                require("command" not in current and current.get("original_shard_command") == original and "recovery_invocation" in current, "Malformed explicit v2 provenance")
    for stage, ids in stage_ids.items():
        require(len(ids) == count and len(set(ids)) == count and set(ids) == set(by_id), "Stage cohort omission/duplication: " + stage)
    require(sidecars, "No recovery evidence to bind")
    require(snap.inventory(root / "recovery_empty_beats") == {Path(p).name for p in sidecars}, "Missing/extra recovery sidecar inventory")
    if not synthetic:
        require(any(c["stage_receipt"] == "beats_19.json" for c in corrections), "Known v1 recovery missing")
    snap.read(Path(__file__))
    snap.check()
    binding = {"schema_version": 1, "status": "bound_completed_recovery_evidence", "synthetic_test_only": synthetic,
               "binder_sha256": sha(Path(__file__)), "original_extraction_receipt": original_envelope,
               "strict_inference_audit": audit_envelope, "inference_run_contract": contract_envelope,
               "inference_completion": completion_envelope, "all_stage_receipts_sha256": all_receipts,
               "beat_stage_receipts": beat_receipts, "recovery_sidecars": sidecars, "v1_provenance_corrections": corrections,
               "recovery_codes": recovery_codes, "reviewed_recovery_protocol": {"sha256": hashlib.sha256(protocol_raw).hexdigest(), "original_utf8": protocol_text},
               "source_files": dict(snap.files), "recovered_items": sorted(e["content"]["item_id"] for e in sidecars.values()),
               "recovered_feature_rows": {e["content"]["item_id"]: {k: v for k, v in feature_by_id[e["content"]["item_id"]].items()
                   if k in {"item_id", "label", "source_id", "beat_output_status", "n_beats", "n_downbeats", "r_eligible", "r_feature_status", "p_eligible", "p_feature_status", *R_COLUMNS, *BAR_COLUMNS, *DURATION_COLUMNS}}
                   for e in sidecars.values()},
               "validated_rows": count, "validated_stage_shards": len(names), "all_original_fields_preserved": True,
               "unavailable_features": R_COLUMNS + BAR_COLUMNS, "P_duration_features_retained": True,
               "audio_stems_spectrograms_reopened": False, "model_fitting_performed": False,
               "neural_inference_performed": False, "source_transfer_J": None}
    binding["binding_payload_sha256"] = digest(binding)
    derivative = {**receipt, "recovery_evidence_binding": binding}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=output.name + ".tmp.", dir=output.parent))
    try:
        target = temp / "extraction_receipt.json"
        target.write_text(json.dumps(derivative, indent=2, sort_keys=True, allow_nan=False) + "\n")
        snap.check()
        require(not output.exists(), "Output appeared during binding")
        os.rename(temp, output)
    except BaseException:
        shutil.rmtree(temp)
        raise
    return {"status": "bound_completed_recovery_evidence", "rows": count, "stage_shards": len(names), "recovered_items": binding["recovered_items"],
            "original_receipt_sha256": original_envelope["sha256"], "derivative_receipt_sha256": sha(output / "extraction_receipt.json"),
            "derivative_receipt": str(output / "extraction_receipt.json"), "synthetic_test_only": synthetic, "model_fitting_performed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-audit-dir", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recovery-protocol", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--synthetic-test-only", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(bind(args.strict_audit_dir, args.inference_root, args.output_dir, args.recovery_protocol, args.code_dir, args.synthetic_test_only), indent=2))


if __name__ == "__main__":
    main()
