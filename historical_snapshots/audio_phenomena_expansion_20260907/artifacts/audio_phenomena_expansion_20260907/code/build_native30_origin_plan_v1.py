#!/usr/bin/env python3
"""Build an evidence-only native-origin exact-30 plan; never open audio files."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path


EXPECTED = {
    "screen": "ee21f8a1c28fa6a847f2fc2893f9acf41f30baabee72682c0ac69a80ed38ec87",
    "native30_evidence": "a3dcf87f703b7709b01d963e1a1c2e96f9a223d159180d1b793021337a3b160a",
    "prior60_freeze": "06dc798c0551d7f9b49c67e226e2633609b4e434fec06ad16ef28aa08340f979",
    "equal60_manifest": "f52b93bf4ff820bd35845749b91a2c7139b901588ebde2ebf62f4442f583d3bf",
    "equal60_contract": "3c2849a50d63a55ffae579ed0a90ab358c6e546ad8297f2503c56fed3f5926d5",
    "aime_long": "b7fae0c67d147d02a0cd93f399787973c4b9a31e8271165347fe4e3baa1ff8ba",
    "materialization": "ae51c2c3a5602e9825369b4f59ccd96f9182b9c7219b5364cda34164f557a0ec",
    "legacy_native": "a2b9cf9b3c1eb8ffa06acd7cb405b7f553d4cc1f4f3d3b10bb92e3511008577c",
    "mureka_manifest": "3de2deeb6fe8447c6e1683cdfbeaff36c2dd3fde0e639298bad2fe46e2cad159",
    "mureka_native60": "341ac9e98fc5156109508ce08c8142587bfa98e7e753cde4387438905833507d",
    "mureka_contract": "6e327fc8eb5b041f89afd9e903a082a428f78331dae051b072c110c2fc7ba51e",
    "mureka_acquisition_contract": "e985adc956116851a7358f3ed9279241a6f92b65d28f22b0eb40e94023de8d24",
    "mureka_acquisition_summary": "c682d5c9ebbcd291be57141bd558dc691b7c1a2f1505dcb160849e9e558a9b50",
    "saraga_manifest": "f52ec01d834df7bf73e33e0133adf4d5f93ed81c2a7c913bbdfdd86610642fc9",
    "saraga_native60": "49b4fa45237b83ac98d4e249d4c85b2729706061cf35fae60957f32d36ed5e45",
    "saraga_proof_inventory": "b0dba7785eb626bd06a6f9531907418946cca32291395c7c4f03f677df33712c",
    "saraga_contract": "0cf1e78ccdb29e63c421a46bae48e2da76ef0f3116aad442a3a3d91d6e044440",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def strict_json_bytes(data: bytes):
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)


def strict_csv_bytes(data: bytes, id_field: str | None = None):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("missing or duplicate CSV header")
    rows = list(reader)
    if id_field:
        seen = set()
        for row in rows:
            ident = row.get(id_field, "")
            if not ident or ident in seen:
                raise ValueError(f"missing or duplicate {id_field}: {ident!r}")
            seen.add(ident)
    return rows


class Binder:
    def __init__(self):
        self.files = {}

    def read(self, path: Path, expected: str | None = None) -> bytes:
        if path.is_symlink():
            raise ValueError(f"input must not be a symlink: {path}")
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"input must be a regular non-symlink file: {path}")
        data = path.read_bytes()
        actual = digest(data)
        if expected is not None and actual != expected:
            raise ValueError(f"SHA mismatch for {path}: expected {expected}, got {actual}")
        old = self.files.get(str(path))
        if old is not None and old != actual:
            raise ValueError(f"input changed between reads: {path}")
        self.files[str(path)] = actual
        return data

    def json(self, path: Path, expected: str | None = None):
        return strict_json_bytes(self.read(path, expected))

    def csv(self, path: Path, expected: str | None = None, id_field: str | None = None):
        return strict_csv_bytes(self.read(path, expected), id_field)

    def jsonl(self, path: Path, expected: str | None = None, id_field: str | None = None):
        data = self.read(path, expected)
        rows, seen = [], set()
        for number, line in enumerate(data.splitlines(), 1):
            if not line.strip():
                continue
            row = strict_json_bytes(line)
            if id_field:
                ident = row.get(id_field)
                if not ident or ident in seen:
                    raise ValueError(f"missing/duplicate {id_field} at {path}:{number}")
                seen.add(ident)
            rows.append((number, row))
        return rows

    def recheck(self):
        for name, old in self.files.items():
            if digest(Path(name).read_bytes()) != old:
                raise ValueError(f"input changed during build: {name}")


def index(rows, field):
    return {row[field]: row for row in rows}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def full_native_center30(duration_s, sample_rate, actual_frames=None):
    rate = int(sample_rate)
    duration = float(duration_s)
    require(30.0 <= duration <= 60.0, "full-native branch requires 30 <= duration <= 60")
    frames = 30 * rate
    start = None if actual_frames is None else (int(actual_frames) - frames) // 2
    if start is not None:
        require(start >= 0 and start + frames <= int(actual_frames), "invalid full-native center crop")
    return {
        "policy": "full_native_le60_center_exact30",
        "start_frame": start,
        "frames": frames,
        "start_formula_if_unresolved": "(validated_actual_native_frames - 30 * native_sample_rate_hz) // 2",
        "physical_frame_validation_required": start is None,
    }


def approved_region_center30(region_start, region_frames, sample_rate, native_float64_sha256=None, mandatory_hash=False):
    rate = int(sample_rate)
    require(int(region_frames) == 60 * rate, "approved native region is not exact 60 seconds")
    if mandatory_hash:
        require(hex64(native_float64_sha256), "mandatory native float64 interval hash missing")
    return {
        "policy": "center_exact30_within_approved_native60_region",
        "start_frame": int(region_start) + 15 * rate,
        "frames": 30 * rate,
        "approved_region_start_frame": int(region_start),
        "approved_region_frames": int(region_frames),
        "approved_region_native_float64_sha256": native_float64_sha256,
        "must_reproduce_approved_region_before_subcrop": True,
    }


def atomic_exclusive_json(path: Path, value):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return digest(body)


def build(args, binder: Binder, script_path: Path):
    script_sha_start = digest(script_path.read_bytes())
    root = args.root.resolve()
    audio = root / "artifacts/audio_phenomena_expansion_20260907"
    diversity = root / "artifacts/source_diversity_expansion_20260905"
    paths = {
        "screen": audio / "audit/equal30_candidates_screen_v1.json",
        "native30_evidence": audio / "audit/native30_evidence_review_v2.json",
        "prior60_freeze": audio / "preregistration/sc_native2174_measurement_frozen_v6.json",
        "equal60_manifest": audio / "manifests/equal60_development_v2/inference_manifest.csv",
        "equal60_contract": audio / "manifests/equal60_development_v2/materialization_contract.json",
        "aime_long": diversity / "state/aime_long_manifest.jsonl",
        "materialization": diversity / "state/materialization_manifest.jsonl",
        "legacy_native": diversity / "audit/legacy_native_1000.json",
        "mureka_manifest": audio / "manifests/mureka60_inputs_v2/inference_manifest.csv",
        "mureka_native60": audio / "manifests/mureka60_inputs_v2/native_metadata_60s.csv",
        "mureka_contract": audio / "manifests/mureka60_inputs_v2/materialization_contract.json",
        "mureka_acquisition_contract": audio / "external_validation/music8k_mureka500_v1/contract.json",
        "mureka_acquisition_summary": audio / "external_validation/music8k_mureka500_v1/summary.json",
        "saraga_manifest": audio / "manifests/saraga_external103_intervals_v1/final/inference_manifest.csv",
        "saraga_native60": audio / "manifests/saraga_external103_intervals_v1/final/native_metadata_60s.csv",
        "saraga_proof_inventory": audio / "manifests/saraga_external103_intervals_v1/final/item_proof_inventory.json",
        "saraga_contract": audio / "manifests/saraga_external103_intervals_v1/frozen_measurement_contract.json",
    }
    screen_doc = binder.json(paths["screen"], EXPECTED["screen"])
    evidence_doc = binder.json(paths["native30_evidence"], EXPECTED["native30_evidence"])
    freeze = binder.json(paths["prior60_freeze"], EXPECTED["prior60_freeze"])
    screen = index(screen_doc["rows"], "id")
    evidence = index(evidence_doc["candidates"], "id")
    freeze_rows = index([x["metadata"] | {"freeze_evidence": x} for x in freeze["selected"]], "id")
    require(len(screen) == 5100 and len(evidence) == 1746 and len(freeze_rows) == 2174, "unexpected primary input count")

    equal_rows = index(binder.csv(paths["equal60_manifest"], EXPECTED["equal60_manifest"], "item_id"), "item_id")
    equal_contract = binder.json(paths["equal60_contract"], EXPECTED["equal60_contract"])
    config = equal_contract["configuration"]
    require(config["coordinate_system"] == "exact_frozen_native_crop_then_resample" and
            config["normalization"] is False and config["dc_removal"] is False and
            config["limiting"] is False and config["short_input_padding"] is False,
            "equal60 processing contract changed")
    aime = index([x for _, x in binder.jsonl(paths["aime_long"], EXPECTED["aime_long"], "id")], "id")
    material = index([x for _, x in binder.jsonl(paths["materialization"], EXPECTED["materialization"], "item_id")], "item_id")
    legacy = index(binder.json(paths["legacy_native"], EXPECTED["legacy_native"])["details"], "id")

    mureka_rows = index(binder.csv(paths["mureka_manifest"], EXPECTED["mureka_manifest"], "item_id"), "item_id")
    mureka_native = index(binder.csv(paths["mureka_native60"], EXPECTED["mureka_native60"], "id"), "id")
    mureka_contract = binder.json(paths["mureka_contract"], EXPECTED["mureka_contract"])
    binder.json(paths["mureka_acquisition_contract"], EXPECTED["mureka_acquisition_contract"])
    mureka_summary = binder.json(paths["mureka_acquisition_summary"], EXPECTED["mureka_acquisition_summary"])
    require(mureka_contract["decoder_admission_claim"] == "acquisition_center_interval_verified_not_full_stream_decoder_agreement", "Mureka decoder limit changed")

    saraga_rows = index(binder.csv(paths["saraga_manifest"], EXPECTED["saraga_manifest"], "item_id"), "item_id")
    saraga_native = index(binder.csv(paths["saraga_native60"], EXPECTED["saraga_native60"], "id"), "id")
    saraga_inventory = binder.json(paths["saraga_proof_inventory"], EXPECTED["saraga_proof_inventory"])
    saraga_contract = binder.json(paths["saraga_contract"], EXPECTED["saraga_contract"])

    def common(ident, origin_set):
        s = screen[ident]
        if origin_set == "prior60":
            f = freeze_rows[ident]
            for field in ("label", "source_group", "role", "group_id"):
                require(str(f[field]) == str(s[field]), f"freeze/screen conflict {ident}:{field}")
        return {"id": ident, "label": s["label"], "source_group": s["source_group"],
                "role": s["role"], "group_id": s["group_id"], "component_id": s["component_id"],
                "origin_set": origin_set}

    plan_rows = []
    prior_counts = Counter()
    for ident in sorted(freeze_rows):
        f, s = freeze_rows[ident], screen[ident]
        source = f["source_group"]
        row = common(ident, "prior60")
        if source in {"MTG-Jamendo", "Suno", "human_maestro_v3", "human_medleydb", "human_moisesdb"}:
            receipt = equal_rows[ident]
            require(receipt["source_channels"] == "2" and receipt["native_crop_shared_with_fhm"] == "True", f"invalid equal60 source {ident}")
            if source == "MTG-Jamendo":
                origin, receipt_name, sha_field, path_field = aime[ident], "aime_long", "raw_sha256", "native_path"
            elif source == "Suno":
                origin, receipt_name, sha_field, path_field = legacy[ident], "legacy_native", "native_sha256", "native_path"
            else:
                origin, receipt_name, sha_field, path_field = material[ident], "materialization", "native_sha256", "native_path"
            require(receipt["source_audio_sha256"] == origin[sha_field], f"origin SHA mismatch {ident}")
            require(receipt["standardized_file_sha256"] == f["freeze_evidence"]["standardized_file_sha256"], f"freeze output mismatch {ident}")
            rate = int(receipt["source_sample_rate"])
            region = approved_region_center30(receipt["crop_start_frame"], receipt["crop_frames"], rate)
            row.update({
                "source_origin": {"path": receipt["source_audio_path"], "sha256": receipt["source_audio_sha256"],
                                  "sample_rate_hz": rate, "channels": 2, "scope": "full_original_file",
                                  "provenance_receipt": {"path": str(paths[receipt_name].resolve()),
                                                             "sha256": EXPECTED[receipt_name], "row_key": ident},
                                  "original_receipt_path": origin[path_field],
                                  "path_relation": "same_path" if receipt["source_audio_path"] == origin[path_field] else "byte_identity_by_sha_not_path"},
                "approved_native60_region": {**region, "evidence_strength": "frozen_native_frame_bounds_from_receipt",
                                             "region_receipt": {"path": str(paths["equal60_manifest"].resolve()),
                                                                "sha256": EXPECTED["equal60_manifest"], "row_key": ident},
                                             "native_float64_sha256": None,
                                             "decoder_limit": "none_recorded_for_full_original_equal60_source"},
                "planned_native30": region,
                "prior_standardized60_evidence_not_origin": {"path": f["freeze_evidence"]["standardized_path"],
                                                               "sha256": f["freeze_evidence"]["standardized_file_sha256"]},
            })
        elif source == "Mureka_v9":
            receipt, native = mureka_rows[ident], mureka_native[ident]
            raw_id = ident.removeprefix("music8k_mureka_v9_")
            require(raw_id != ident, f"invalid Mureka id {ident}")
            item_path = audio / f"external_validation/music8k_mureka500_v1/items/{raw_id}.json"
            expected_item_sha = mureka_summary["receipts_sha256"][raw_id + ".json"]
            item = binder.json(item_path, expected_item_sha)
            require(receipt["source_audio_sha256"] == native["source_audio_sha256"] == item["sha256"], f"Mureka SHA mismatch {ident}")
            require(receipt["group_id"] == item["reference_group_id"] == f["group_id"], f"Mureka group mismatch {ident}")
            require(receipt["decoder_admission_claim"] == mureka_contract["decoder_admission_claim"] and receipt["center_seek_equals_sequential_float64"] == "True", f"Mureka interval proof missing {ident}")
            region = approved_region_center30(native["crop_start_frame"], native["crop_frames"], native["native_sample_rate_hz"], native["native_crop_float64_sha256"], True)
            row.update({
                "source_origin": {"path": receipt["source_audio_path"], "sha256": receipt["source_audio_sha256"],
                                  "sample_rate_hz": int(receipt["source_sample_rate"]), "channels": 2,
                                  "scope": "accepted_native60_interval_from_full_original",
                                  "provenance_receipt": {"path": str(item_path.resolve()),
                                                             "sha256": expected_item_sha, "row_key": raw_id},
                                  "conditioning": {"reference_group_id": item["reference_group_id"], "reference_artist_hash": item["reference_artist_hash"],
                                                   "reference_lyrics_hash": item["reference_lyrics_hash"], "generation_lyrics_hash": item["generation_lyrics_hash"], "caption_hash": item["caption_hash"]}},
                "approved_native60_region": {**region, "source_sha256": native["source_audio_sha256"],
                                             "evidence_strength": "verified_native_interval_hash",
                                             "region_receipt": {"path": str(paths["mureka_native60"].resolve()),
                                                                "sha256": EXPECTED["mureka_native60"], "row_key": ident},
                                             "decoder_limit": mureka_contract["decoder_admission_claim"],
                                             "seek_equals_sequential_float64": True,
                                             "decoder_limit_proofs": [
                                                 {"path": str(paths["mureka_contract"].resolve()), "sha256": EXPECTED["mureka_contract"]},
                                                 {"path": str(paths["mureka_acquisition_contract"].resolve()), "sha256": EXPECTED["mureka_acquisition_contract"]}]},
                "planned_native30": region,
                "prior_standardized60_evidence_not_origin": {"path": f["freeze_evidence"]["standardized_path"], "sha256": f["freeze_evidence"]["standardized_file_sha256"]},
            })
        elif source == "human_saraga_hindustani_v1":
            receipt, native = saraga_rows[ident], saraga_native[ident]
            proof_path = audio / f"manifests/saraga_external103_intervals_v1/items/{ident}/proof.json"
            proof = binder.json(proof_path, saraga_inventory[ident])
            interval = proof["interval_proof"]
            mbid = ident.removeprefix("saraga_hindustani_")
            physical_key = f"manifests/saraga_hindustani_physical_v1/receipts/{mbid}.json"
            physical_path = audio / physical_key
            physical_sha = saraga_contract["contract"]["inputs_sha256"][physical_key]
            physical = binder.json(physical_path, physical_sha)
            raw_sha = interval["raw_hashes_after"]["sha256"]
            require(raw_sha == receipt["original_source_audio_sha256"] == native["source_audio_sha256"], f"Saraga SHA mismatch {ident}")
            require(physical["record"]["measurement"]["raw_hashes_after_decode"]["sha256"] == raw_sha, f"Saraga physical SHA mismatch {ident}")
            require(proof["wav"]["file_sha256"] == f["freeze_evidence"]["standardized_file_sha256"], f"Saraga output mismatch {ident}")
            require(interval["seek_proof"]["exact_array_equal"] is True, f"Saraga seek proof missing {ident}")
            region = approved_region_center30(native["crop_start_frame"], native["crop_frames"], native["native_sample_rate_hz"], native["native_crop_float64_sha256"], True)
            row.update({
                "source_origin": {"path": interval["source_audio_path"], "sha256": raw_sha,
                                  "sample_rate_hz": int(native["native_sample_rate_hz"]), "channels": 2,
                                  "scope": "accepted_native60_interval_from_archive_member_original",
                                  "provenance_receipt": {"path": str(physical_path.resolve()),
                                                             "sha256": physical_sha, "row_key": mbid},
                                  "archive_member": interval["archive_member"]},
                "approved_native60_region": {**region, "source_sha256": raw_sha,
                                             "evidence_strength": "verified_native_interval_hash",
                                             "region_receipt": {"path": str(paths["saraga_native60"].resolve()),
                                                                "sha256": EXPECTED["saraga_native60"], "row_key": ident},
                                             "decoder_limit": "exact_center_interval_seek_equals_sequential; header/actual EOF differences retained",
                                             "seek_equals_sequential_float64": True,
                                             "interval_proof_path": str(proof_path.resolve()), "interval_proof_sha256": saraga_inventory[ident]},
                "planned_native30": region,
                "prior_standardized60_evidence_not_origin": {"path": f["freeze_evidence"]["standardized_path"], "sha256": f["freeze_evidence"]["standardized_file_sha256"]},
            })
        else:
            raise ValueError(f"unsupported prior60 source: {source}")
        prior_counts[source] += 1
        plan_rows.append(row)

    excluded = []
    new_counts = Counter()
    for ident in sorted(evidence):
        ev = evidence[ident]
        row = common(ident, "native30_evidence_v2")
        native = ev["native_evidence"]
        receipt_name = ev["receipt_binding"]
        receipt_binding = evidence_doc["input_bindings"].get(receipt_name)
        require(receipt_binding is not None, f"missing native30 evidence receipt binding {receipt_name}")
        row["source_origin"] = {"path": native["path"], "sha256": native["sha256"],
                                "sample_rate_hz": int(native["sample_rate_hz"]), "channels": int(native["channels"]),
                                "duration_s": native["duration_s"], "scope": "full_original_file",
                                "provenance_receipt": {"path": receipt_binding["path"],
                                                       "sha256": receipt_binding["sha256"],
                                                       "row_key": ev["receipt_row"]}}
        if int(native["channels"]) != 2:
            row["exclusion_reasons"] = ["native_channels_not_2"]
            row["planned_native30"] = None
            excluded.append(row)
            continue
        probe = ev.get("native_header_probe") or {}
        region = full_native_center30(native["duration_s"], native["sample_rate_hz"], probe.get("frames"))
        row["planned_native30"] = region
        row["approved_native60_region"] = None
        new_counts[row["source_group"]] += 1
        plan_rows.append(row)

    require(len(plan_rows) == 3869 and len(excluded) == 51, "unexpected plan/exclusion counts")
    require(len({x["id"] for x in plan_rows}) == len(plan_rows), "duplicate planned IDs")
    require(not ({x["id"] for x in plan_rows} & {x["id"] for x in excluded}), "planned/excluded overlap")
    binder.recheck()
    script_sha_end = digest(script_path.read_bytes())
    require(script_sha_start == script_sha_end, "builder changed during run")
    bindings = [{"path": path, "sha256": sha} for path, sha in sorted(binder.files.items())]
    return {
        "schema_version": "native30-origin-plan-v1",
        "status": "draft_not_admitted_not_frozen_for_execution",
        "scope": "Evidence-ready metadata plan only. No audio opened, decoded, hashed, transformed, extracted, or fitted by this build.",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "implementation_binding": {"path": str(script_path.resolve()), "sha256_at_start": script_sha_start,
                                   "sha256_at_end": script_sha_end, "unchanged_during_run": True},
        "input_bindings": bindings,
        "counts": {"planned_rows": len(plan_rows), "prior60_rows": len(freeze_rows), "new_native30_stereo_rows": sum(new_counts.values()),
                   "native_mono_exclusions": len(excluded), "input_evidence_rows": len(freeze_rows) + len(evidence),
                   "classifier_fits": 0, "audio_files_opened": 0},
        "prior60_source_counts": dict(sorted(prior_counts.items())),
        "new_native30_source_counts": dict(sorted(new_counts.items())),
        "uniform_execution_rule_not_authorized": {
            "selection": "center exact30 inside the recorded native region",
            "order": ["verify bound source and approved-region evidence", "crop exact native frames", "resample to 44100 only when needed", "write stereo FLOAT"],
            "forbidden": ["standardized60_as_native_substitute", "padding", "normalization", "gain", "dc_removal", "limiting", "source_specific_correction"],
        },
        "limitations": ["Plan is not source validation or physical admission.", "Container stereo is not acoustic channel-independence proof.",
                        "Mureka and Saraga must reuse and reproduce their approved native60 regions under the bound decoder proofs.",
                        "New exact30 output paths and hashes do not exist yet."],
        "rows": sorted(plan_rows, key=lambda x: x["id"]),
        "excluded_rows": sorted(excluded, key=lambda x: x["id"]),
        "input_recheck_before_publication": "all_bound_inputs_unchanged",
    }


def parse_args():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root / "artifacts/audio_phenomena_expansion_20260907/audit/native30_origin_plan_v1.json")
    return parser.parse_args()


def main():
    args = parse_args()
    script = Path(__file__).resolve()
    plan = build(args, Binder(), script)
    output_sha = atomic_exclusive_json(args.output, plan)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": output_sha, "counts": plan["counts"],
                      "prior60_source_counts": plan["prior60_source_counts"], "new_native30_source_counts": plan["new_native30_source_counts"]}, indent=2))


if __name__ == "__main__":
    main()
