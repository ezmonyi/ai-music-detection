#!/usr/bin/env python3
"""Build the auditable union of prior tracks and the frozen expansion.

This script reads metadata only. It does not fit, score, or inspect audio features.
Every output row is one track/view, with explicit availability and independent-group
identity. Absolute remote audio paths deliberately remain remote paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path


def jsonl(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Required metadata input is absent: {path}")
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def csvrows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile("w", newline="", dir=path.parent, prefix="." + path.name, delete=False) as handle:
        temporary = handle.name
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def latest(path, key="id"):
    return {row[key]: row for row in jsonl(path) if row.get("status", "ok") == "ok"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materialized", type=Path)
    parser.add_argument("--aime-long", type=Path)
    parser.add_argument("--legacy-native-audit", type=Path)
    args = parser.parse_args()
    base = args.artifacts
    remote = Path("/mnt/nfs-code/users/yi")
    old = {f"{r['class_name']}_{r['source']}_{r['id']}": r for r in
           jsonl(base / "demucs_bias_corrected_1000_20260901/manifest.jsonl")}
    legacy_native = {}
    if args.legacy_native_audit:
        audit = json.loads(args.legacy_native_audit.read_text())
        legacy_native = {r["id"]: r for r in audit["details"]}
        if (audit.get("passed") is not True or audit.get("verified") != 1000
                or set(legacy_native) != set(old)
                or audit.get("manifest_sha256") != hashlib.sha256(
                    (base / "demucs_bias_corrected_1000_20260901/manifest.jsonl").read_bytes()).hexdigest()):
            raise ValueError("Legacy native audit does not match the frozen 1,000 original records")
    prompts = {r["id"]: r for r in jsonl(base / "open_models_spectral_500_20260901/prompts/prompt_manifest.jsonl")}
    generated = {g: latest(base / f"open_models_spectral_500_20260901/state/{g}_standardization.jsonl")
                 for g in ("heartmula", "acestep")}
    legacy = csvrows(base / "dynamics_rhythm_change_detector_extension_20260903/detector_cohort/detector_manifest.csv")
    aime = jsonl(base / "external_generator_500_testset_20260904/testset/manifest.jsonl")
    aime_long = {r["id"]: r for r in jsonl(args.aime_long)} if args.aime_long else {}
    frozen = csvrows(base / "source_diversity_expansion_20260904/manifests/v2/frozen_item_manifest.csv")
    materialized = {}
    if args.materialized:
        for record in jsonl(args.materialized):
            if record.get("status", "ok") in ("ok", "success", "completed"):
                materialized[record["item_id"]] = record

    rows = []
    for r in legacy:
        track, source = r["track"], r["source"]
        label = int(r["label"])
        if source in generated:
            raw = generated[source][r["id"]]
            prompt = prompts[r["id"]]
            condition = "muse:" + prompt["source_song_id"]
            group = condition
            native_sr = raw["source_sample_rate"]
            native_duration = raw["source_duration_s"]
            creator = ""
            rawhash = raw["source_sha256"]
        else:
            raw = old[track]
            condition = ""
            creator = raw.get("artist_name", raw.get("creator", ""))
            key = raw.get("group_id") or r["id"]
            group = source + ":" + key
            native_sr = raw["sample_rate"]
            native_duration = raw["duration"]
            rawhash = legacy_native.get(track, {}).get("native_sha256", "")
        source_group = {"humair_suno": "Suno", "suno_unknown": "Suno",
                        "fma_medium": "FMA", "heartmula": "HeartMuLa", "acestep": "ACE-Step"}[source]
        common = dict(id=track, track=track, label=label, source_id=source,
                      source_group=source_group,
                      role="development" if r["split"] == "development" else "locked",
                      original_role=r["split"], group_id=group, condition_id=condition,
                      artist_or_creator=creator, native_sample_rate_hz=native_sr,
                      native_duration_s=native_duration, raw_sha256=rawhash,
                      provenance_status="legacy", acquisition="prior",
                      available=1, source_audio_path=r["source_audio_path"])
        for view in (10, 30):
            if view == 30 and float(native_duration) < 30 - 1e-6:
                continue
            audio_path = r["source_audio_path"]
            if view == 10 and r["split"] == "development":
                audio_path = str(remote / "external_generator_500_heuristics_20260904/cohort/audio" / f"{track}.flac")
            rows.append({**common, "duration_sec": view, "duration_view": f"{view}s",
                         "audio_path": audio_path, "crop_start_s": 0.0,
                         "audio_offset_s": 0.0,
                         "requires_crop": int(view == 10 and r["split"] != "development")})

    for r in aime:
        # The former AIME external collection was explicitly consumed as development.
        condition = "aime:" + r["condition_id"]
        rows.append(dict(id=r["id"], track=r["id"], label=int(r["label"]), source_id=r["source"],
                         source_group=r["model"], role="development", original_role="consumed_aime_test",
                         group_id=condition, condition_id=condition, artist_or_creator="",
                         native_sample_rate_hz=r["original_sample_rate"], native_duration_s=r["original_duration_s"],
                         raw_sha256=r["raw_sha256"], provenance_status="AIME_pinned", acquisition="prior",
                         available=1, source_audio_path=str(remote / "external_generator_500_testset_20260904/testset" / r["standardized_relpath"]),
                         duration_sec=10, duration_view="10s",
                         audio_path=str(remote / "external_generator_500_testset_20260904/testset" / r["standardized_relpath"]),
                         crop_start_s=0.0, audio_offset_s=0.0, requires_crop=0))
        if r["id"] in aime_long:
            long = aime_long[r["id"]]
            rows.append({**rows[-1], "duration_sec": 30, "duration_view": "30s",
                         "audio_path": long["view_30s_path"], "requires_crop": 0,
                         "crop_start_s": long["view_30s_crop_start_s"]})

    roles = {"development": "development", "locked_catalogue_test": "locked",
             "stress_test_only": "stress", "provisional_development": "provisional",
             "locked_pilot_only": "pilot"}
    for r in frozen:
        m = materialized.get(r["item_id"], {})
        source = r["source_id"]
        # Creator grouping is conservative only when the metadata identifies an
        # actual artist/composer; generic placeholders must not merge whole corpora.
        creator = r["artist_or_creator"]
        identity = r["group_id"]
        if source in ("human_magnatagatune", "human_medleydb", "human_maestro_v3") and creator:
            identity = "creator:" + creator
        group = source + ":" + identity
        native_sr = m.get("native_sample_rate_hz", m.get("native_sr", ""))
        duration = m.get("native_duration_s", "")
        common = dict(id=r["item_id"], track=r["item_id"], label=int(r["label"] == "ai"),
                      source_id=source, source_group=source, role=roles[r["role"]], original_role=r["role"],
                      group_id=group, condition_id="", artist_or_creator=creator,
                      native_sample_rate_hz=native_sr, native_duration_s=duration,
                      raw_sha256=m.get("raw_sha256", m.get("native_sha256", "")),
                      provenance_status=r["provenance_grade"], acquisition="expansion",
                      available=int(bool(m)), source_audio_path=m.get("native_path", ""),
                      source_locator=r["source_locator"], evaluation_allowed=r["evaluation_allowed"])
        path10 = m.get("view_10s_path", m.get("standardized_path", ""))
        pathlong = m.get("view_max60s_path", m.get("long_path", ""))
        rows.append({**common, "duration_sec": 10, "duration_view": "10s", "audio_path": path10,
                     "crop_start_s": m.get("crop_start_s", r["window_start_s"]),
                     "audio_offset_s": 0.0, "requires_crop": 0})
        actual_long_duration = float(m.get("view_max60s_duration_s", 0) or 0)
        if duration and float(duration) >= 30 - 1e-6 and actual_long_duration >= 30 - 1e-6:
            offset = max(0.0, min(float(m.get("crop_start_s", 0)) - float(m.get("long_crop_start_s", 0)) - 10.0,
                                  actual_long_duration - 30.0))
            rows.append({**common, "duration_sec": 30, "duration_view": "30s", "audio_path": pathlong,
                         "crop_start_s": float(m.get("long_crop_start_s", 0)) + offset,
                         "audio_offset_s": offset, "requires_crop": 1})

    seen = set()
    generator_families = {
        "MusicGen Small": "MusicGen", "MusicGen Medium": "MusicGen", "MusicGen Large": "MusicGen",
        "AudioLDM 2 Large": "AudioLDM2", "AudioLDM 2 Music": "AudioLDM2",
        "Stable Audio v1": "Stable Audio", "Stable Audio v2": "Stable Audio",
    }
    for row in rows:
        key = (row["id"], row["duration_view"])
        assert key not in seen, key
        seen.add(key)
        sr = row["native_sample_rate_hz"]
        row["eligible_common8"] = int(bool(sr) and float(sr) >= 16000)
        row["eligible_fullband"] = int(bool(sr) and float(sr) >= 40000)
        row["generator_family"] = (generator_families.get(row["source_group"], row["source_group"])
                                   if row["label"] == 1 else "")
    # Exact native-byte duplicates across catalogues are one leakage group. The
    # group graph also retains all artist/condition links already established.
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    hashes = {}
    duplicate_hashes = set()
    for row in rows:
        digest = row.get("raw_sha256")
        if not digest:
            continue
        current = find(row["group_id"])
        if digest in hashes:
            other, label, item_id = hashes[digest]
            if row["id"] != item_id:
                duplicate_hashes.add(digest)
            if label != row["label"]:
                raise RuntimeError("Conflicting labels on an exact native hash: " + digest)
            a, b = sorted((current, find(other)))
            parent[b] = a
        else:
            hashes[digest] = (current, row["label"], row["id"])
    for row in rows:
        row["group_id"] = find(row["group_id"])
    # A cross-role creator/condition group cannot be trained then called held-out.
    locked_groups = {r["group_id"] for r in rows if r["role"] == "locked"}
    overlap = {r["group_id"] for r in rows if r["role"] == "development"} & locked_groups
    for row in rows:
        if row["role"] == "development" and row["group_id"] in overlap:
            row["role"] = "excluded_group_overlap"
    args.output.mkdir(parents=True, exist_ok=True)
    for view in ("10s", "30s"):
        selected = [r for r in rows if r["duration_view"] == view]
        write_csv(args.output / f"metadata_{view}.csv", selected)
    summary = dict(rows=len(rows), by_view=dict(Counter(r["duration_view"] for r in rows)),
                   counts=dict(Counter(f"{r['duration_view']}|{r['role']}|{r['source_group']}" for r in rows)),
                   materialized_expansion_records=len(materialized), group_overlap_excluded=sorted(overlap),
                   cross_item_duplicate_raw_hashes=sorted(duplicate_hashes))
    input_paths = [
        base / "demucs_bias_corrected_1000_20260901/manifest.jsonl",
        base / "open_models_spectral_500_20260901/prompts/prompt_manifest.jsonl",
        base / "open_models_spectral_500_20260901/state/heartmula_standardization.jsonl",
        base / "open_models_spectral_500_20260901/state/acestep_standardization.jsonl",
        base / "dynamics_rhythm_change_detector_extension_20260903/detector_cohort/detector_manifest.csv",
        base / "external_generator_500_testset_20260904/testset/manifest.jsonl",
        base / "source_diversity_expansion_20260904/manifests/v2/frozen_item_manifest.csv",
    ]
    summary["frozen_input_sha256"] = {
        str(path.relative_to(base)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in input_paths
    }
    if args.legacy_native_audit:
        summary["frozen_input_sha256"]["legacy_native_audit"] = hashlib.sha256(
            args.legacy_native_audit.read_bytes()).hexdigest()
        summary["legacy_originals_physically_verified"] = len(legacy_native)
    for view in ("10s", "30s"):
        path = args.output / f"metadata_{view}.csv"
        summary[f"metadata_{view}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "registry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
