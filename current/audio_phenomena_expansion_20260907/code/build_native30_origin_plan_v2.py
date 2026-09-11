#!/usr/bin/env python3
"""Build a draft native-origin exact-30 metadata plan without opening audio."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from collections import Counter
from pathlib import Path, PurePosixPath


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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_bytes(data: bytes):
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)


def strict_csv_bytes(data: bytes, id_field=None):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), "missing or duplicate CSV header")
    rows = list(reader)
    if id_field:
        seen = set()
        for row in rows:
            ident = row.get(id_field, "")
            require(ident and ident not in seen, f"missing or duplicate {id_field}: {ident!r}")
            seen.add(ident)
    return rows


def index_unique(rows, field):
    result = {}
    for row in rows:
        key = row.get(field)
        require(key is not None and key != "" and key not in result, f"missing or duplicate {field}: {key!r}")
        result[key] = row
    return result


def strict_int(value, name, minimum=None):
    require(not isinstance(value, bool), f"{name}: bool is not an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float):
        require(math.isfinite(value) and value.is_integer(), f"{name}: non-integral or non-finite")
        result = int(value)
    elif isinstance(value, str):
        text = value.strip()
        require(text and (text.isdigit() or (text[0] == "-" and text[1:].isdigit())), f"{name}: invalid integer")
        result = int(text)
    else:
        raise ValueError(f"{name}: invalid integer type")
    if minimum is not None:
        require(result >= minimum, f"{name}: below minimum {minimum}")
    return result


def strict_float(value, name, minimum=None, maximum=None):
    require(not isinstance(value, bool), f"{name}: bool is not numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: invalid number") from error
    require(math.isfinite(result), f"{name}: non-finite")
    if minimum is not None:
        require(result >= minimum, f"{name}: below minimum {minimum}")
    if maximum is not None:
        require(result <= maximum, f"{name}: above maximum {maximum}")
    return result


def hex64(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def safe_recorded_path(value, name):
    require(isinstance(value, str) and value, f"{name}: missing path")
    path = PurePosixPath(value)
    require(path.is_absolute() and ".." not in path.parts, f"{name}: path must be absolute without traversal")
    return value


class Binder:
    def __init__(self):
        self.files = {}

    def read(self, path: Path, expected=None):
        original = path.absolute()
        for candidate in (original, *original.parents):
            require(not candidate.is_symlink(), f"symlink input/ancestor refused: {candidate}")
        resolved = original.resolve()
        require(resolved.is_file(), f"input is not a regular file: {resolved}")
        data = resolved.read_bytes()
        actual = digest(data)
        if expected is not None:
            require(actual == expected, f"SHA mismatch for {resolved}: expected {expected}, got {actual}")
        old = self.files.get(str(resolved))
        require(old is None or old == actual, f"input changed between reads: {resolved}")
        self.files[str(resolved)] = actual
        return data

    def json(self, path, expected=None):
        return strict_json_bytes(self.read(path, expected))

    def csv(self, path, expected=None, id_field=None):
        return strict_csv_bytes(self.read(path, expected), id_field)

    def jsonl(self, path, expected=None, id_field=None):
        data = self.read(path, expected)
        rows, seen = [], set()
        for number, line in enumerate(data.splitlines(), 1):
            if not line.strip():
                continue
            row = strict_json_bytes(line)
            if id_field:
                ident = row.get(id_field)
                require(ident and ident not in seen, f"missing or duplicate {id_field} at {path}:{number}")
                seen.add(ident)
            rows.append(row)
        return rows

    def recheck(self):
        for path, old in self.files.items():
            require(digest(Path(path).read_bytes()) == old, f"input changed during build: {path}")


def full_native_center30(duration_s, sample_rate, header_frames=None):
    rate = strict_int(sample_rate, "sample_rate", 1)
    duration = strict_float(duration_s, "duration_s", 30.0, 60.0)
    exact30_frames = 30 * rate
    suggestion = None
    if header_frames is not None:
        header = strict_int(header_frames, "header_frames", exact30_frames)
        suggestion = {
            "header_frames": header,
            "header_suggested_start_frame": (header - exact30_frames) // 2,
            "authority": "container_header_only_not_actual_decoded_eof",
        }
    return {
        "policy": "full_native_le60_center_exact30_after_sequential_decode",
        "duration_receipt_s": duration,
        "start_frame": None,
        "frames": exact30_frames,
        "actual_decoded_frames": None,
        "start_formula_after_validation": "(sequentially_decoded_actual_frames - 30 * native_sample_rate_hz) // 2",
        "header_suggestion_not_execution_coordinate": suggestion,
        "sequential_decode_required": True,
        "physical_frame_validation_required": True,
    }


def approved_region_center30(region_start, region_frames, sample_rate, float64_sha256, evidence_strength, mandatory_hash):
    rate = strict_int(sample_rate, "sample_rate", 1)
    start = strict_int(region_start, "native_region.start_frame", 0)
    frames = strict_int(region_frames, "native_region.frames", 1)
    require(frames == 60 * rate, "native region is not exactly 60 seconds")
    require(float64_sha256 is None or hex64(float64_sha256), "invalid native region float64 SHA")
    if mandatory_hash:
        require(hex64(float64_sha256), "mandatory native region float64 SHA missing")
    native_region = {
        "start_frame": start,
        "frames": frames,
        "float64_sha256": float64_sha256,
        "evidence_strength": evidence_strength,
    }
    planned = {
        "policy": "center_exact30_within_approved_native60_region",
        "coordinate_space": "native_source_frames",
        "start_frame": start + 15 * rate,
        "frames": 30 * rate,
        "derived_from_native_region": True,
        "must_reproduce_native_region_before_subcrop": mandatory_hash,
    }
    return native_region, planned


def validate_common(screen, origin_set, freeze=None, evidence=None):
    require(screen.get("duration_exposure_candidate") is True, f"{screen.get('id')}: not a duration candidate")
    require(screen.get("exclusion_reasons") == [], f"{screen.get('id')}: screen exclusions present")
    require(screen.get("role") == "development", f"{screen.get('id')}: expected development role")
    required = ("id", "label", "source_group", "group_id", "component_id")
    for field in required:
        require(field in screen and screen[field] is not None, f"screen missing {field}")
    if freeze is not None:
        for field in ("id", "label", "source_group", "group_id", "role"):
            require(str(freeze.get(field)) == str(screen.get(field)), f"freeze/screen conflict {screen['id']}:{field}")
    if evidence is not None:
        for field in ("id", "label", "source_group", "component_id"):
            require(str(evidence.get(field)) == str(screen.get(field)), f"evidence/screen conflict {screen['id']}:{field}")
        native = evidence["native_evidence"]
        joins = evidence["join_fields"]
        require(joins.get("canonical_id") == screen["id"], f"{screen['id']}: evidence canonical ID conflict")
        require(joins.get("raw_sha256") == native.get("sha256") and hex64(native.get("sha256")), f"{screen['id']}: evidence raw SHA conflict")
        require(strict_float(joins.get("native_duration_s"), "join duration") == strict_float(native.get("duration_s"), "native duration"), f"{screen['id']}: evidence duration conflict")
        channels = strict_int(native.get("channels"), "native channels", 1)
        require(evidence.get("native_stereo_metadata_eligible") is (channels == 2), f"{screen['id']}: stereo eligibility conflict")
    return {
        "id": screen["id"], "label": screen["label"], "source_group": screen["source_group"],
        "role": screen["role"], "group_id": screen["group_id"], "component_id": screen["component_id"],
        "input_occurrences": screen.get("input_occurrences"), "selected_metadata_input": screen.get("selected_metadata_input"),
        "origin_set": origin_set,
    }


def validate_mureka_join(ident, receipt, native, item, freeze, contract, acquisition_contract):
    rate = strict_int(native["native_sample_rate_hz"], "Mureka native rate", 1)
    channels = strict_int(native["physical_channels"], "Mureka native channels", 1)
    start = strict_int(native["crop_start_frame"], "Mureka crop start", 0)
    frames = strict_int(native["crop_frames"], "Mureka crop frames", 1)
    end = strict_int(native["crop_end_frame_exclusive"], "Mureka crop end", 1)
    f64 = native["native_crop_float64_sha256"]
    raw_id = ident.removeprefix("music8k_mureka_v9_")
    require(raw_id != ident and item["id"] == raw_id, f"{ident}: invalid Mureka ID join")
    for value in (receipt["source_audio_sha256"], item["sha256"], native["source_audio_sha256"]):
        require(value == native["source_audio_sha256"] and hex64(value), f"{ident}: Mureka raw SHA conflict")
    require(receipt["source_audio_path"] == native["audio_path"], f"{ident}: Mureka source path conflict")
    require(receipt["standardized_path"] == freeze["freeze_evidence"]["standardized_path"], f"{ident}: Mureka standardized path conflict")
    require(receipt["standardized_file_sha256"] == freeze["freeze_evidence"]["standardized_file_sha256"], f"{ident}: Mureka standardized hash conflict")
    for value in (receipt["crop_start_frame"],): require(strict_int(value, "Mureka receipt crop start", 0) == start, f"{ident}: Mureka crop start conflict")
    require(strict_int(receipt["crop_frames"], "Mureka receipt crop frames", 1) == frames, f"{ident}: Mureka crop frames conflict")
    require(strict_int(receipt["crop_end_frame_exclusive"], "Mureka receipt crop end", 1) == end == start + frames, f"{ident}: Mureka crop end conflict")
    require(receipt["native_crop_float64_sha256"] == f64 and hex64(f64), f"{ident}: Mureka native crop hash conflict")
    config = contract["configuration"]
    for value in (receipt["source_sample_rate"], item["sample_rate"], config["sample_rate_hz"], freeze["native_sample_rate_hz"]):
        require(strict_int(value, "Mureka joined rate", 1) == rate, f"{ident}: Mureka rate conflict")
    for value in (receipt["source_channels"], item["channels"], config["channels"], freeze["freeze_evidence"]["native_channels"]):
        require(strict_int(value, "Mureka joined channels", 1) == channels == 2, f"{ident}: Mureka channels conflict")
    require(frames == strict_int(config["crop_frames"], "Mureka contract crop frames", 1) == 60 * rate, f"{ident}: Mureka contract crop conflict")
    claim = "acquisition_center_interval_verified_not_full_stream_decoder_agreement"
    require(contract["decoder_admission_claim"] == receipt["decoder_admission_claim"] == native["decoder_admission_claim"] == claim, f"{ident}: Mureka decoder limit conflict")
    require(config["full_stream_decoder_equality_required"] is False and config["seek_agreement"] == "exact_float64_array_equal", f"{ident}: Mureka acquisition interval limit changed")
    require(receipt["center_seek_equals_sequential_float64"] == native["center_seek_equals_sequential_float64"] == "True", f"{ident}: Mureka seek proof missing")
    require(item["contract_sha256"] == contract["acquisition_contract_sha256"] == EXPECTED["mureka_acquisition_contract"], f"{ident}: Mureka acquisition contract binding conflict")
    require(acquisition_contract["status"] == "frozen_for_acquisition_only" and acquisition_contract["classifier_authorized"] is False, f"{ident}: Mureka acquisition scope changed")
    require(item["reference_group_id"] == receipt["group_id"] == freeze["group_id"], f"{ident}: Mureka group conflict")
    require(receipt["standardized_frames"] == str(freeze["freeze_evidence"]["standardized_frames"]), f"{ident}: Mureka standardized frames conflict")
    require(strict_int(receipt["standardized_sr"], "Mureka standardized rate", 1) == freeze["freeze_evidence"]["standardized_sr"], f"{ident}: Mureka standardized rate conflict")
    return {"rate": rate, "channels": channels, "start": start, "frames": frames, "float64_sha256": f64, "claim": claim}


def validate_saraga_join(ident, receipt, native, proof, physical, freeze):
    interval = proof["interval_proof"]
    measurement = physical["record"]["measurement"]
    rate = strict_int(native["native_sample_rate_hz"], "Saraga native rate", 1)
    channels = strict_int(native["physical_channels"], "Saraga native channels", 1)
    start = strict_int(native["crop_start_frame"], "Saraga crop start", 0)
    frames = strict_int(native["crop_frames"], "Saraga crop frames", 1)
    end = strict_int(native["crop_end_frame_exclusive"], "Saraga crop end", 1)
    f64 = native["native_crop_float64_sha256"]
    require(interval["source_audio_path"] == native["audio_path"] == receipt["original_source_audio_path"], f"{ident}: Saraga source path conflict")
    raw = interval["raw_hashes_after"]["sha256"]
    for value in (native["source_audio_sha256"], native["registered_raw_sha256"], receipt["original_source_audio_sha256"], measurement["raw_hashes_after_decode"]["sha256"]):
        require(value == raw and hex64(value), f"{ident}: Saraga raw SHA conflict")
    for value in (interval["crop_start_frame"], interval["seek_proof"]["requested_start_frame"], interval["seek_proof"]["returned_start_frame"]):
        require(strict_int(value, "Saraga proof crop start", 0) == start, f"{ident}: Saraga crop start conflict")
    for value in (interval["crop_frames"], interval["retained_crop_frames"], interval["seek_proof"]["requested_frames"], interval["seek_proof"]["observed_frames"]):
        require(strict_int(value, "Saraga proof crop frames", 1) == frames, f"{ident}: Saraga crop frames conflict")
    require(strict_int(interval["crop_end_frame_exclusive"], "Saraga proof crop end", 1) == end == start + frames, f"{ident}: Saraga crop end conflict")
    for value in (interval["native_float64_sha256"], interval["seek_proof"]["float64_sha256"]):
        require(value == f64 and hex64(value), f"{ident}: Saraga native crop hash conflict")
    for value in (receipt["native_sample_rate_hz"], interval["native_sample_rate_hz"], interval["observed_header"]["sample_rate"], measurement["sample_rate"], freeze["native_sample_rate_hz"]):
        require(strict_int(value, "Saraga joined rate", 1) == rate, f"{ident}: Saraga rate conflict")
    for value in (interval["native_channels"], interval["observed_header"]["channels"], measurement["channels"], freeze["freeze_evidence"]["native_channels"]):
        require(strict_int(value, "Saraga joined channels", 1) == channels == 2, f"{ident}: Saraga channels conflict")
    require(measurement["decoder_format"] == interval["observed_header"]["decoder_format"] and measurement["decoder_subtype"] == interval["observed_header"]["decoder_subtype"], f"{ident}: Saraga source format conflict")
    require(interval["seek_proof"]["exact_array_equal"] is True and interval["seek_proof"]["exact_bytes_equal"] is True, f"{ident}: Saraga seek proof missing")
    out_hash = proof["wav"]["file_sha256"]
    require(out_hash == receipt["standardized_file_sha256"] == freeze["freeze_evidence"]["standardized_file_sha256"], f"{ident}: Saraga standardized hash conflict")
    require(receipt["standardized_path"] == receipt["audio_path"] == freeze["freeze_evidence"]["standardized_path"], f"{ident}: Saraga standardized path conflict")
    require(strict_int(receipt["standardized_frames"], "Saraga standardized frames", 1) == freeze["freeze_evidence"]["standardized_frames"] == 60 * 44100, f"{ident}: Saraga standardized frames conflict")
    require(receipt["group_id"] == native["group_id"] == freeze["group_id"], f"{ident}: Saraga group conflict")
    return {"rate": rate, "channels": channels, "start": start, "frames": frames, "float64_sha256": f64, "decoder_format": measurement["decoder_format"], "decoder_subtype": measurement["decoder_subtype"]}


def atomic_exclusive_json(path: Path, value):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return digest(body)


def build(args, binder, script_path):
    script_sha_start = digest(script_path.read_bytes())
    root = args.root.resolve()
    audio = root / "artifacts/audio_phenomena_expansion_20260907"
    diversity = root / "artifacts/source_diversity_expansion_20260905"
    paths = {
        "screen": audio / "audit/equal30_candidates_screen_v1.json", "native30_evidence": audio / "audit/native30_evidence_review_v2.json",
        "prior60_freeze": audio / "preregistration/sc_native2174_measurement_frozen_v6.json",
        "equal60_manifest": audio / "manifests/equal60_development_v2/inference_manifest.csv", "equal60_contract": audio / "manifests/equal60_development_v2/materialization_contract.json",
        "aime_long": diversity / "state/aime_long_manifest.jsonl", "materialization": diversity / "state/materialization_manifest.jsonl", "legacy_native": diversity / "audit/legacy_native_1000.json",
        "mureka_manifest": audio / "manifests/mureka60_inputs_v2/inference_manifest.csv", "mureka_native60": audio / "manifests/mureka60_inputs_v2/native_metadata_60s.csv",
        "mureka_contract": audio / "manifests/mureka60_inputs_v2/materialization_contract.json", "mureka_acquisition_contract": audio / "external_validation/music8k_mureka500_v1/contract.json",
        "mureka_acquisition_summary": audio / "external_validation/music8k_mureka500_v1/summary.json",
        "saraga_manifest": audio / "manifests/saraga_external103_intervals_v1/final/inference_manifest.csv", "saraga_native60": audio / "manifests/saraga_external103_intervals_v1/final/native_metadata_60s.csv",
        "saraga_proof_inventory": audio / "manifests/saraga_external103_intervals_v1/final/item_proof_inventory.json", "saraga_contract": audio / "manifests/saraga_external103_intervals_v1/frozen_measurement_contract.json",
    }
    screen_doc = binder.json(paths["screen"], EXPECTED["screen"]); evidence_doc = binder.json(paths["native30_evidence"], EXPECTED["native30_evidence"]); freeze_doc = binder.json(paths["prior60_freeze"], EXPECTED["prior60_freeze"])
    screen = index_unique(screen_doc["rows"], "id"); evidence = index_unique(evidence_doc["candidates"], "id"); freeze = index_unique([x["metadata"] | {"freeze_evidence": x} for x in freeze_doc["selected"]], "id")
    require((len(screen), len(evidence), len(freeze)) == (5100, 1746, 2174), "unexpected primary input count")
    equal = index_unique(binder.csv(paths["equal60_manifest"], EXPECTED["equal60_manifest"], "item_id"), "item_id")
    equal_contract = binder.json(paths["equal60_contract"], EXPECTED["equal60_contract"]); cfg = equal_contract["configuration"]
    require(cfg["coordinate_system"] == "exact_frozen_native_crop_then_resample" and all(cfg[x] is False for x in ("normalization", "dc_removal", "limiting", "short_input_padding")), "equal60 contract changed")
    aime = index_unique(binder.jsonl(paths["aime_long"], EXPECTED["aime_long"], "id"), "id"); material = index_unique(binder.jsonl(paths["materialization"], EXPECTED["materialization"], "item_id"), "item_id"); legacy = index_unique(binder.json(paths["legacy_native"], EXPECTED["legacy_native"])["details"], "id")
    mr = index_unique(binder.csv(paths["mureka_manifest"], EXPECTED["mureka_manifest"], "item_id"), "item_id"); mn = index_unique(binder.csv(paths["mureka_native60"], EXPECTED["mureka_native60"], "id"), "id")
    mc = binder.json(paths["mureka_contract"], EXPECTED["mureka_contract"]); mac = binder.json(paths["mureka_acquisition_contract"], EXPECTED["mureka_acquisition_contract"]); ms = binder.json(paths["mureka_acquisition_summary"], EXPECTED["mureka_acquisition_summary"])
    sr = index_unique(binder.csv(paths["saraga_manifest"], EXPECTED["saraga_manifest"], "item_id"), "item_id"); sn = index_unique(binder.csv(paths["saraga_native60"], EXPECTED["saraga_native60"], "id"), "id")
    inventory = binder.json(paths["saraga_proof_inventory"], EXPECTED["saraga_proof_inventory"]); sc = binder.json(paths["saraga_contract"], EXPECTED["saraga_contract"])
    rows, excluded, prior_counts, new_counts = [], [], Counter(), Counter()
    for ident in sorted(freeze):
        f = freeze[ident]; source = f["source_group"]; row = validate_common(screen[ident], "prior60", freeze=f)
        if source in {"MTG-Jamendo", "Suno", "human_maestro_v3", "human_medleydb", "human_moisesdb"}:
            receipt = equal[ident]
            require(receipt["label"] == f["label"] and receipt["group_id"] == f["group_id"] and receipt["role"] == "development" and receipt["source_id"] == source, f"{ident}: equal60 metadata conflict")
            require(receipt["source_channels"] == "2" and receipt["native_crop_shared_with_fhm"] == "True", f"{ident}: equal60 native evidence conflict")
            if source == "MTG-Jamendo": origin, receipt_name, sha_field, path_field = aime[ident], "aime_long", "raw_sha256", "native_path"
            elif source == "Suno": origin, receipt_name, sha_field, path_field = legacy[ident], "legacy_native", "native_sha256", "native_path"
            else: origin, receipt_name, sha_field, path_field = material[ident], "materialization", "native_sha256", "native_path"
            require(receipt["source_audio_sha256"] == origin[sha_field] and hex64(receipt["source_audio_sha256"]), f"{ident}: original SHA conflict")
            require(receipt["standardized_path"] == f["freeze_evidence"]["standardized_path"] and receipt["standardized_file_sha256"] == f["freeze_evidence"]["standardized_file_sha256"], f"{ident}: freeze output conflict")
            rate = strict_int(receipt["source_sample_rate"], "equal60 native rate", 1); require(rate == strict_int(f["native_sample_rate_hz"], "freeze native rate", 1), f"{ident}: native rate conflict")
            native_region, planned = approved_region_center30(receipt["crop_start_frame"], receipt["crop_frames"], rate, None, "frozen_native_frame_bounds_from_receipt", False)
            row.update({"source_origin": {"path": safe_recorded_path(receipt["source_audio_path"], "source path"), "sha256": receipt["source_audio_sha256"], "sample_rate_hz": rate, "channels": 2, "scope": "full_original_file", "provenance_receipt": {"path": str(paths[receipt_name].resolve()), "sha256": EXPECTED[receipt_name], "row_key": ident}, "original_receipt_path": origin[path_field], "path_relation": "same_path" if receipt["source_audio_path"] == origin[path_field] else "byte_identity_by_sha_not_path"},
                        "native_region": {**native_region, "source_sha256": receipt["source_audio_sha256"], "region_receipt": {"path": str(paths["equal60_manifest"].resolve()), "sha256": EXPECTED["equal60_manifest"], "row_key": ident}, "decoder_limit": "none_recorded_for_full_original_equal60_source"}, "planned_native30": planned})
        elif source == "Mureka_v9":
            raw_id = ident.removeprefix("music8k_mureka_v9_"); item_path = audio / f"external_validation/music8k_mureka500_v1/items/{raw_id}.json"; item_sha = ms["receipts_sha256"][raw_id + ".json"]; item = binder.json(item_path, item_sha)
            joined = validate_mureka_join(ident, mr[ident], mn[ident], item, f, mc, mac); native_region, planned = approved_region_center30(joined["start"], joined["frames"], joined["rate"], joined["float64_sha256"], "verified_native_interval_hash", True)
            row.update({"source_origin": {"path": safe_recorded_path(mr[ident]["source_audio_path"], "Mureka source path"), "sha256": mr[ident]["source_audio_sha256"], "sample_rate_hz": joined["rate"], "channels": joined["channels"], "scope": "accepted_native60_interval_from_full_original", "provenance_receipt": {"path": str(item_path.resolve()), "sha256": item_sha, "row_key": raw_id}, "conditioning": {k: item[k] for k in ("reference_group_id", "reference_artist_hash", "reference_lyrics_hash", "generation_lyrics_hash", "caption_hash")}},
                        "native_region": {**native_region, "source_sha256": mr[ident]["source_audio_sha256"], "region_receipt": {"path": str(paths["mureka_native60"].resolve()), "sha256": EXPECTED["mureka_native60"], "row_key": ident}, "decoder_limit": joined["claim"], "seek_equals_sequential_float64": True, "decoder_limit_proofs": [{"path": str(paths["mureka_contract"].resolve()), "sha256": EXPECTED["mureka_contract"]}, {"path": str(paths["mureka_acquisition_contract"].resolve()), "sha256": EXPECTED["mureka_acquisition_contract"]}]}, "planned_native30": planned})
        elif source == "human_saraga_hindustani_v1":
            proof_path = audio / f"manifests/saraga_external103_intervals_v1/items/{ident}/proof.json"; proof_sha = inventory[ident]; proof = binder.json(proof_path, proof_sha); mbid = ident.removeprefix("saraga_hindustani_")
            physical_key = f"manifests/saraga_hindustani_physical_v1/receipts/{mbid}.json"; physical_path = audio / physical_key; physical_sha = sc["contract"]["inputs_sha256"][physical_key]; physical = binder.json(physical_path, physical_sha)
            joined = validate_saraga_join(ident, sr[ident], sn[ident], proof, physical, f); native_region, planned = approved_region_center30(joined["start"], joined["frames"], joined["rate"], joined["float64_sha256"], "verified_native_interval_hash", True)
            row.update({"source_origin": {"path": safe_recorded_path(sn[ident]["audio_path"], "Saraga source path"), "sha256": sn[ident]["source_audio_sha256"], "sample_rate_hz": joined["rate"], "channels": joined["channels"], "source_format": joined["decoder_format"], "source_subtype": joined["decoder_subtype"], "scope": "accepted_native60_interval_from_archive_member_original", "archive_member": proof["interval_proof"]["archive_member"], "provenance_receipt": {"path": str(physical_path.resolve()), "sha256": physical_sha, "row_key": mbid}},
                        "native_region": {**native_region, "source_sha256": sn[ident]["source_audio_sha256"], "region_receipt": {"path": str(paths["saraga_native60"].resolve()), "sha256": EXPECTED["saraga_native60"], "row_key": ident}, "decoder_limit": "exact_interval_seek_equals_sequential; header/actual EOF differences retained", "seek_equals_sequential_float64": True, "interval_proof": {"path": str(proof_path.resolve()), "sha256": proof_sha}}, "planned_native30": planned})
        else: raise ValueError(f"unsupported prior60 source: {source}")
        row["prior_standardized60_evidence_not_origin"] = {"path": f["freeze_evidence"]["standardized_path"], "sha256": f["freeze_evidence"]["standardized_file_sha256"]}
        rows.append(row); prior_counts[source] += 1
    for ident in sorted(evidence):
        ev = evidence[ident]; row = validate_common(screen[ident], "native30_evidence_v2", evidence=ev); native = ev["native_evidence"]; receipt_binding = evidence_doc["input_bindings"].get(ev["receipt_binding"]); require(receipt_binding, f"{ident}: missing evidence receipt binding")
        rate = strict_int(native["sample_rate_hz"], "native rate", 1); channels = strict_int(native["channels"], "native channels", 1)
        row["source_origin"] = {"path": safe_recorded_path(native["path"], "native source path"), "sha256": native["sha256"], "sample_rate_hz": rate, "channels": channels, "duration_s": strict_float(native["duration_s"], "native duration", 30), "scope": "full_original_file", "provenance_receipt": {"path": receipt_binding["path"], "sha256": receipt_binding["sha256"], "row_key": ev["receipt_row"]}}
        if channels != 2:
            row.update({"exclusion_reasons": ["native_channels_not_2"], "native_region": None, "planned_native30": None}); excluded.append(row); continue
        probe = ev.get("native_header_probe") or {}; row["native_region"] = None; row["planned_native30"] = full_native_center30(native["duration_s"], rate, probe.get("frames")); rows.append(row); new_counts[row["source_group"]] += 1
    require(len(rows) == 3869 and len(excluded) == 51 and len({r["id"] for r in rows}) == 3869, "unexpected output counts or duplicates")
    require(not ({r["id"] for r in rows} & {r["id"] for r in excluded}), "planned/excluded overlap")
    require(all(r["planned_native30"]["start_frame"] is None and r["planned_native30"]["physical_frame_validation_required"] is True for r in rows if r["origin_set"] == "native30_evidence_v2"), "new native starts must remain unresolved")
    binder.recheck(); script_sha_end = digest(script_path.read_bytes()); require(script_sha_start == script_sha_end, "builder changed during run")
    return {"schema_version": "native30-origin-plan-v2", "status": "draft_not_admitted_not_frozen_for_execution", "scope": "Evidence-ready metadata plan only. No audio opened, decoded, hashed, transformed, extracted, or fitted by this build.", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "implementation_binding": {"path": str(script_path.resolve()), "sha256_at_start": script_sha_start, "sha256_at_end": script_sha_end, "unchanged_during_run": True},
            "input_bindings": [{"path": p, "sha256": s} for p, s in sorted(binder.files.items())],
            "counts": {"planned_rows": len(rows), "prior60_rows": len(freeze), "new_native30_stereo_rows": sum(new_counts.values()), "new_native30_unresolved_actual_starts": sum(new_counts.values()), "native_mono_exclusions": len(excluded), "input_evidence_rows": len(freeze) + len(evidence), "classifier_fits": 0, "audio_files_opened": 0},
            "prior60_source_counts": dict(sorted(prior_counts.items())), "new_native30_source_counts": dict(sorted(new_counts.items())),
            "uniform_execution_rule_not_authorized": {"selection": "center exact30 in native frames; new full-native rows require sequential actual-EOF decode before coordinates exist", "forbidden": ["header_frames_as_actual_eof", "standardized60_as_native_substitute", "padding", "normalization", "gain", "dc_removal", "limiting", "source_specific_correction"]},
            "limitations": ["Plan is not source validation, admission, or an execution freeze.", "All 1695 new full-native coordinates remain unresolved until bounded sequential decode.", "Mureka and Saraga must reproduce the bound approved native60 interval before subcrop.", "Container stereo is not acoustic channel-independence proof."],
            "rows": sorted(rows, key=lambda r: r["id"]), "excluded_rows": sorted(excluded, key=lambda r: r["id"]), "input_recheck_before_publication": "all_bound_inputs_unchanged"}


def parse_args():
    root = Path(__file__).resolve().parents[3]; parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=root); parser.add_argument("--output", type=Path, default=root / "artifacts/audio_phenomena_expansion_20260907/audit/native30_origin_plan_v2.json"); return parser.parse_args()


def main():
    args = parse_args(); script = Path(__file__).resolve(); binder = Binder(); plan = build(args, binder, script); binder.recheck(); require(digest(script.read_bytes()) == plan["implementation_binding"]["sha256_at_end"], "builder changed before publication"); output_sha = atomic_exclusive_json(args.output, plan)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": output_sha, "counts": plan["counts"], "prior60_source_counts": plan["prior60_source_counts"], "new_native30_source_counts": plan["new_native30_source_counts"]}, indent=2))


if __name__ == "__main__":
    main()
