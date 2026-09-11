#!/usr/bin/env python3
"""Reconcile equal30 candidates with native receipts and optionally probe headers.

This utility never decodes samples or extracts features.  Its probes use
soundfile.info (remote NFS paths) and ffprobe stream metadata (local MP3s).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from decimal import Decimal


EXPECTED_SHA256 = {
    "screen": "ee21f8a1c28fa6a847f2fc2893f9acf41f30baabee72682c0ac69a80ed38ec87",
    "master30": "eb313f97e2d69d743423efc170eecb24f1f00ee77ad789461772daecd058c242",
    "accepted60_freeze": "06dc798c0551d7f9b49c67e226e2633609b4e434fec06ad16ef28aa08340f979",
    "acestep": "036d9f3cc5792589929975f5266d236a9aaf9d335bb26d1c2d06b776e3d527dd",
    "heartmula": "f84aa91ba0dccf8a84b5890735c0e9dac3fb02225af42bd4d15af55453eef522",
    "aime_long": "b7fae0c67d147d02a0cd93f399787973c4b9a31e8271165347fe4e3baa1ff8ba",
    "materialization": "ae51c2c3a5602e9825369b4f59ccd96f9182b9c7219b5364cda34164f557a0ec",
    "legacy_native": "a2b9cf9b3c1eb8ffa06acd7cb405b7f553d4cc1f4f3d3b10bb92e3511008577c",
    "external500": "49dbe01b8be98857361b3a437db44ee937698e014bf70ccc7d443c24ad342fb4",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def no_dupes(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def load_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=no_dupes,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                yield line_number, json.loads(
                    line,
                    object_pairs_hook=no_dupes,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite JSON constant: {value}")
                    ),
                )


def decimal_equal(left, right) -> bool:
    return Decimal(str(left)) == Decimal(str(right))


def read_csv_unique(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"missing or duplicate CSV headers: {path}")
        rows = {}
        for line_number, row in enumerate(reader, 2):
            ident = row.get("id", "")
            if not ident or ident in rows:
                raise ValueError(f"missing/duplicate id at {path}:{line_number}: {ident!r}")
            rows[ident] = (line_number, row)
        return rows


def header_probe_remote(records, host: str, python_path: str):
    remote = r'''import json,os,sys
import soundfile as sf
out=[]
for x in json.load(sys.stdin):
 r={"id":x["id"],"role":x["role"],"path":x["path"],"present":False,"header_ok":False}
 if os.path.isfile(x["path"]):
  r["present"]=True
  try:
   h=sf.info(x["path"]); r.update(header_ok=True,frames=int(h.frames),sample_rate_hz=int(h.samplerate),channels=int(h.channels),duration_s=float(h.frames)/float(h.samplerate))
  except Exception as e:r["error"]=str(e)
 out.append(r)
print(json.dumps(out,separators=(",",":")))'''
    command = python_path + " -c " + shlex.quote(remote)
    result = subprocess.run(
        ["ssh", host, command], input=json.dumps(records), text=True,
        capture_output=True, timeout=55, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"remote probe failed rc={result.returncode}: {result.stderr[:1000]}")
    return json.loads(result.stdout, object_pairs_hook=no_dupes)


def header_probe_local(records):
    out = []
    for x in records:
        row = {"id": x["id"], "role": x["role"], "path": x["path"],
               "present": os.path.isfile(x["path"]), "header_ok": False}
        if row["present"]:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=channels,sample_rate,duration",
                 "-of", "json", x["path"]], text=True, capture_output=True,
                timeout=15, check=False,
            )
            try:
                stream = json.loads(result.stdout)["streams"][0]
                row.update(
                    header_ok=True,
                    channels=int(stream["channels"]),
                    sample_rate_hz=int(stream["sample_rate"]),
                    duration_s=float(stream["duration"]),
                )
            except Exception as exc:
                row["error"] = f"{exc}; stderr={result.stderr[:300]}"
        out.append(row)
    return out


def summarize_probes(candidates):
    summary = defaultdict(Counter)
    for row in candidates:
        for key in ("native_header_probe", "view_header_probe"):
            probe = row.get(key)
            if not probe:
                continue
            role = "native_origin" if key.startswith("native") else row["view_evidence"]["role"]
            bucket = summary[(row["source_group"], role)]
            bucket["total"] += 1
            bucket["present"] += bool(probe.get("present"))
            bucket["header_ok"] += bool(probe.get("header_ok"))
            if probe.get("header_ok"):
                bucket["supports_any_exact30"] += probe["duration_s"] + 1e-12 >= 30.0
                expected = row["native_evidence"]["channels"] if key.startswith("native") else row["view_evidence"].get("channels")
                if expected is None:
                    bucket["channel_comparison_not_applicable"] += 1
                elif probe["channels"] == expected:
                    bucket["channel_matches_recorded"] += 1
                needed = 30.0 + (row["registry"]["audio_offset_s"] if key.startswith("native") else 0.0)
                bucket["registered_offset_window_fits"] += probe["duration_s"] + 1e-12 >= needed
    return [dict(source_group=k[0], path_role=k[1], **dict(v)) for k, v in sorted(summary.items())]


def main():
    repo = Path(__file__).resolve().parents[3]
    art = repo / "artifacts/audio_phenomena_expansion_20260907"
    div = repo / "artifacts/source_diversity_expansion_20260905"
    opn = repo / "artifacts/open_models_spectral_500_20260901"
    ext = repo / "artifacts/external_generator_500_testset_20260904"
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=Path, default=art / "audit/equal30_candidates_screen_v1.json")
    ap.add_argument("--master30", type=Path, default=div / "manifests/final/metadata_30s.csv")
    ap.add_argument("--accepted60-freeze", type=Path, default=art / "preregistration/sc_native2174_measurement_frozen_v6.json")
    ap.add_argument("--acestep", type=Path, default=opn / "state/acestep_standardization.jsonl")
    ap.add_argument("--heartmula", type=Path, default=opn / "state/heartmula_standardization.jsonl")
    ap.add_argument("--aime-long", type=Path, default=div / "state/aime_long_manifest.jsonl")
    ap.add_argument("--materialization", type=Path, default=div / "state/materialization_manifest.jsonl")
    ap.add_argument("--legacy-native", type=Path, default=div / "audit/legacy_native_1000.json")
    ap.add_argument("--external500", type=Path, default=ext / "testset/manifest.jsonl")
    ap.add_argument("--output", type=Path, default=art / "audit/native30_evidence_review_v2.json")
    ap.add_argument("--probe-headers", action="store_true")
    ap.add_argument("--remote-host", default="5090-5")
    ap.add_argument("--remote-python", default="/mnt/nfs-code/users/yi/dynamics_rhythm_external_benchmark_20260903/venv/bin/python")
    args = ap.parse_args()
    script_sha_at_start = sha256(Path(__file__).resolve())
    inputs = {k: getattr(args, k) for k in EXPECTED_SHA256}
    bindings = {}
    for name, path in inputs.items():
        actual = sha256(path)
        if actual != EXPECTED_SHA256[name]:
            raise SystemExit(f"SHA mismatch for {name}: expected {EXPECTED_SHA256[name]}, got {actual}")
        bindings[name] = {"path": str(path.resolve()), "sha256": actual}

    screen = load_json(args.screen)
    selected = {
        r["id"]: r for r in screen["rows"]
        if r["duration_exposure_candidate"] and r["recorded_native_channel_status"] == "unknown"
    }
    if len(selected) != 1746:
        raise SystemExit(f"expected 1746 candidates, found {len(selected)}")
    master = read_csv_unique(args.master30)
    evidence = {}

    def register(ident, receipt_name, receipt_line, native, view, id_rule):
        if ident not in selected:
            return
        if ident in evidence:
            raise ValueError(f"multiple native receipts for {ident}")
        _, registry = master[ident]
        if registry["raw_sha256"] != native["sha256"]:
            raise ValueError(f"raw_sha256 conflict for {ident}")
        if not decimal_equal(registry["native_duration_s"], native["duration_s"]):
            raise ValueError(f"native duration conflict for {ident}")
        evidence[ident] = {
            "receipt_binding": receipt_name,
            "receipt_row": receipt_line,
            "id_join_rule": id_rule,
            "join_fields": {
                "canonical_id": ident,
                "raw_sha256": native["sha256"],
                "native_duration_s": native["duration_s"],
            },
            "native_evidence": native,
            "view_evidence": view,
        }

    for generator, receipt_name, path in (
        ("acestep", "acestep", args.acestep),
        ("heartmula", "heartmula", args.heartmula),
    ):
        for line, x in load_jsonl(path):
            ident = f"ai_{generator}_{x['id']}"
            register(ident, receipt_name, line,
                     {"path": x["source"], "sha256": x["source_sha256"], "channels": x["source_channels"], "duration_s": x["source_duration_s"], "sample_rate_hz": x["source_sample_rate"], "channel_field": "source_channels"},
                     {"path": x["output"], "sha256": x["output_sha256"], "channels": x["output_channels"], "duration_s": x["output_duration_s"], "padding_frames": x["padding_frames"], "role": "receipt_hashed_exact30_view"},
                     f"ai_{generator}_ + receipt.id")

    for line, x in load_jsonl(args.aime_long):
        register(x["id"], "aime_long", line,
                 {"path": x["native_path"], "sha256": x["raw_sha256"], "channels": x["original_channels"], "duration_s": x["original_duration_s"], "sample_rate_hz": x["original_sample_rate"], "channel_field": "original_channels"},
                 {"path": x["view_30s_path"], "sha256": x["view_30s_sha256"], "channels": x["standardized_channels"], "duration_s": 30.0, "crop_start_s": x["view_30s_crop_start_s"], "role": "receipt_hashed_exact30_view"},
                 "receipt.id == canonical id")

    legacy = load_json(args.legacy_native)
    for index, x in enumerate(legacy["details"]):
        register(x["id"], "legacy_native", index,
                 {"path": x["native_path"], "sha256": x["native_sha256"], "channels": x["native_channels"], "duration_s": x["native_duration_s"], "sample_rate_hz": x["native_sample_rate_hz"], "channel_field": "native_channels"},
                 {"path": master[x["id"]][1]["audio_path"] if x["id"] in selected else "", "sha256": None, "channels": None, "duration_s": 30.0, "role": "registry_exact30_derivative_lineage_unresolved"},
                 "receipt.id == canonical id")

    for line, x in load_jsonl(args.materialization):
        register(x["item_id"], "materialization", line,
                 {"path": x["native_path"], "sha256": x["native_sha256"], "channels": x["native_channels"], "duration_s": x["native_duration_s"], "sample_rate_hz": x["native_sample_rate_hz"], "channel_field": "native_channels"},
                 {"path": x["view_max60s_path"], "sha256": x["view_max60s_sha256"], "channels": x["view_max60s_channels"], "duration_s": x["view_max60s_duration_s"], "padding_s": x["view_max60s_padded_s"], "role": "receipt_hashed_unpadded_max60_view"},
                 "receipt.item_id == canonical id")

    missing = sorted(set(selected) - set(evidence))
    if missing:
        raise SystemExit(f"missing receipt evidence for {len(missing)} IDs: {missing[:5]}")

    freeze = load_json(args.accepted60_freeze)
    freeze_selected = {x["metadata"]["id"] for x in freeze["selected"]}
    freeze_excluded = {x["metadata"]["id"] for x in freeze["excluded"]}
    external = {}
    for line, x in load_jsonl(args.external500):
        if x["id"] in external:
            raise ValueError(f"duplicate external500 id: {x['id']}")
        external[x["id"]] = (line, x)
    candidates = []
    for ident in sorted(selected):
        screen_row = selected[ident]
        registry_line, registry = master[ident]
        external_evidence = None
        if ident in external:
            external_line, external_row = external[ident]
            for field, observed, expected in (
                ("raw_sha256", external_row["raw_sha256"], evidence[ident]["native_evidence"]["sha256"]),
                ("original_channels", external_row["original_channels"], evidence[ident]["native_evidence"]["channels"]),
                ("original_duration_s", external_row["original_duration_s"], evidence[ident]["native_evidence"]["duration_s"]),
            ):
                equal = decimal_equal(observed, expected) if field == "original_duration_s" else observed == expected
                if not equal:
                    raise ValueError(f"external500 corroboration conflict for {ident}: {field}")
            external_evidence = {
                "row": external_line,
                "raw_sha256": external_row["raw_sha256"],
                "original_channels": external_row["original_channels"],
                "original_duration_s": external_row["original_duration_s"],
                "original_sample_rate": external_row["original_sample_rate"],
                "standardized_relpath": external_row["standardized_relpath"],
                "path_role": "corroborating_original_metadata_plus_10s_derivative_only",
            }
        item = {
            "id": ident,
            "label": screen_row["label"],
            "source_group": screen_row["source_group"],
            "component_id": screen_row["component_id"],
            "receipt_binding": evidence[ident]["receipt_binding"],
            "receipt_row": evidence[ident]["receipt_row"],
            "id_join_rule": evidence[ident]["id_join_rule"],
            "join_fields": evidence[ident]["join_fields"],
            "native_evidence": evidence[ident]["native_evidence"],
            "view_evidence": evidence[ident]["view_evidence"],
            "registry": {
                "row": registry_line,
                "source_audio_path": registry["source_audio_path"],
                "audio_path": registry["audio_path"],
                "crop_start_s": float(registry["crop_start_s"] or 0),
                "audio_offset_s": float(registry["audio_offset_s"] or 0),
                "source_audio_is_native_origin": registry["source_audio_path"] == evidence[ident]["native_evidence"]["path"],
            },
            "accepted60_freeze_relation": "selected" if ident in freeze_selected else ("excluded" if ident in freeze_excluded else "absent"),
            "external500_evidence": external_evidence,
            "external500_id_overlap": external_evidence is not None,
            "native_stereo_metadata_eligible": evidence[ident]["native_evidence"]["channels"] == 2,
        }
        candidates.append(item)

    replay = {"performed": False, "scope": "header metadata only; no sample decoding"}
    if args.probe_headers:
        remote_records, local_records = [], []
        for item in candidates:
            for field, role in (("native_evidence", "native_origin"), ("view_evidence", item["view_evidence"]["role"])):
                path = item[field]["path"]
                record = {"id": item["id"], "role": role, "path": path}
                (remote_records if path.startswith("/mnt/") else local_records).append(record)
        remote_results = header_probe_remote(remote_records, args.remote_host, args.remote_python)
        local_results = header_probe_local(local_records)
        indexed = {(x["id"], x["role"]): x for x in remote_results + local_results}
        for item in candidates:
            item["native_header_probe"] = indexed[(item["id"], "native_origin")]
            item["view_header_probe"] = indexed[(item["id"], item["view_evidence"]["role"])]
        replay = {
            "performed": True,
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "remote_host": args.remote_host,
            "remote_python": args.remote_python,
            "remote_records": len(remote_records),
            "local_records": len(local_records),
            "method": "remote soundfile.info and local ffprobe stream headers; no sample decoding",
            "summary": summarize_probes(candidates),
        }

    source_summary = []
    for source in sorted({x["source_group"] for x in candidates}):
        rows = [x for x in candidates if x["source_group"] == source]
        channels = Counter(str(x["native_evidence"]["channels"]) for x in rows)
        source_summary.append({
            "source_group": source,
            "candidates": len(rows),
            "native_channels": dict(sorted(channels.items())),
            "native_duration_s_min": min(x["native_evidence"]["duration_s"] for x in rows),
            "native_duration_s_max": max(x["native_evidence"]["duration_s"] for x in rows),
            "receipt_hashed_view": sum(x["view_evidence"]["sha256"] is not None for x in rows),
            "unresolved_derivative_lineage": sum(x["view_evidence"]["sha256"] is None for x in rows),
        })

    report = {
        "schema_version": "native30-evidence-review-v2",
        "status": "completed_metadata_reconciliation_and_header_replay" if args.probe_headers else "completed_metadata_reconciliation_only",
        "scope": "Source-intake evidence only; no cohort admission, waveform decode, feature extraction, classifier fit, or acoustic-stereo certification.",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_bindings": bindings,
        "implementation_binding": {"path": str(Path(__file__).resolve()), "sha256_at_start": script_sha_at_start},
        "counts": {
            "candidate_rows": len(candidates),
            "receipt_native_stereo": sum(x["native_stereo_metadata_eligible"] for x in candidates),
            "receipt_native_mono": sum(not x["native_stereo_metadata_eligible"] for x in candidates),
            "receipt_hashed_stereo_approved_views": sum(x["native_stereo_metadata_eligible"] and x["view_evidence"]["sha256"] is not None for x in candidates),
            "stereo_registry_derivatives_unresolved_lineage": sum(x["native_stereo_metadata_eligible"] and x["view_evidence"]["sha256"] is None for x in candidates),
            "accepted60_selected_overlap": sum(x["accepted60_freeze_relation"] == "selected" for x in candidates),
            "accepted60_excluded_overlap": sum(x["accepted60_freeze_relation"] == "excluded" for x in candidates),
            "external500_id_overlap": sum(x["external500_id_overlap"] for x in candidates),
            "classifier_fits": 0,
        },
        "source_summary": source_summary,
        "path_role_warning": "source_audio_path is a registry field, not intrinsically a native origin. Trust native_evidence.path and its receipt binding. Standardized/view header channels never prove native channels.",
        "investigation_history": {
            "original_interactive_probe": {
                "performed_before_script_persistence": True,
                "remote_records": 3088,
                "remote_present_and_header_ok": 3088,
                "local_fma_suno_native_records": 404,
                "local_present_and_header_ok": 404,
                "note": "Ephemeral independent check reported in the parent task; retained separately from the reproducible replay below.",
            },
            "script_replay": replay,
        },
        "replay_command": "python artifacts/audio_phenomena_expansion_20260907/code/probe_native30_evidence_v2.py --probe-headers --output <new-json-path>",
        "remaining_boundaries": [
            "Container channel count is not acoustic nonduplicated-stereo certification.",
            "Header probes establish current presence and header metadata, not current full-file SHA256.",
            "FMA/Suno registry exact30 paths are legacy inputs under demucs_bias_corrected_1000_20260901/input. They are regular files, not symlinks, but no creation receipt was found; directory naming alone does not prove Demucs or spectral correction, and no equivalence to native full mixes or simple resampling is assumed.",
            "The preferred future treatment for the 397 native-stereo FMA/Suno rows is one uniform new exact30 native-origin preprocessing rule without source-specific correction, unless the legacy-input transformation is first proven and uniformly admissible.",
            "Three native registered-offset checks are sub-frame rounding failures in the replay (one MedleyDB and two URMP); each native still exceeds 30 seconds and its receipt-hashed max60 view supports the registered window.",
            "This review does not authorize feature extraction or admit a cohort.",
        ],
        "candidates": candidates,
    }
    # Detect input and implementation drift across the slower header phase.
    for name, path in inputs.items():
        final_sha = sha256(path)
        if final_sha != bindings[name]["sha256"]:
            raise SystemExit(f"input changed during probe: {name}")
    script_sha_at_end = sha256(Path(__file__).resolve())
    if script_sha_at_end != script_sha_at_start:
        raise SystemExit("implementation changed during probe")
    report["implementation_binding"]["sha256_at_end"] = script_sha_at_end
    report["implementation_binding"]["unchanged_during_run"] = True
    report["input_recheck_before_publication"] = "all_bound_inputs_unchanged"
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output_file:
            output_file.write(payload)
            output_file.flush()
            os.fsync(output_file.fileno())
        # Hard-link publication is atomic and fails if output appeared meanwhile.
        os.link(temporary, args.output)
        directory_fd = os.open(args.output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "counts": report["counts"], "probe_summary": replay.get("summary")}, indent=2))


if __name__ == "__main__":
    main()
