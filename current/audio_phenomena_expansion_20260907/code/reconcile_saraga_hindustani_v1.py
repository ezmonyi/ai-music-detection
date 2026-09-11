#!/usr/bin/env python3
"""Reconcile verified Saraga Hindustani archive members with pinned metadata.

This is an inventory/identity check only.  It does not open audio payloads,
decode media, select a cohort, or admit any recording to a classifier.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import unicodedata


COMMIT = "bd103295689ad11a5ed19038b7469e91b5c3c124"
ARCHIVE_ROOT = "saraga1.5_hindustani"
ARCHIVE_BYTES = 4_109_172_493
ARCHIVE_MD5 = "ea9ed2885ea37a1b10e42f60cf299702"
ARCHIVE_SHA256 = "cd3abd54288efd95e85ae3bf8b13e770e0914bfac12fa07dcba04c6ddc641fb7"
MD5_RE = re.compile(r"[0-9a-f]{32}")

# Repository companions describe availability.  Only pitch changes its physical
# archive suffix; the repository stores a checksum while the ZIP stores text.
ANNOTATION_SUFFIXES = {
    "bpm-manual.txt": "bpm-manual.txt",
    "ctonic.txt": "ctonic.txt",
    "mphrases-manual.txt": "mphrases-manual.txt",
    "pitch.md5": "pitch.txt",
    "sama-manual.txt": "sama-manual.txt",
    "sections-manual-p.txt": "sections-manual-p.txt",
    "tempo-manual.txt": "tempo-manual.txt",
}
NON_ANNOTATION_COMPANIONS = {"json", "mp3.md5"}


class ReconciliationError(ValueError):
    """A hard reconciliation failure carrying machine-readable diagnostics."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_files(named_paths):
    return {
        label: {"path": str(path), "sha256": sha256_file(path)}
        for label, path in named_paths.items()
    }


def require_snapshot_unchanged(snapshot):
    changes = []
    for label, before in snapshot.items():
        path = Path(before["path"])
        after = sha256_file(path) if path.is_file() else None
        if after != before["sha256"]:
            changes.append({"input": label, "path": str(path),
                            "before_sha256": before["sha256"],
                            "after_sha256": after})
    if changes:
        raise ReconciliationError(
            "Input/code changed during reconciliation",
            {"status": "failed_input_mutation_during_reconciliation",
             "changed_files": changes},
        )


def normalized_path(raw):
    """Return a safe NFC POSIX path without applying lossy/fuzzy rewrites."""
    require(isinstance(raw, str) and raw, "Empty/non-string path")
    require("\\" not in raw, f"Backslash forbidden in path: {raw!r}")
    path = PurePosixPath(raw)
    require(not path.is_absolute() and ".." not in path.parts and "." not in path.parts,
            f"Unsafe path: {raw!r}")
    normalized = unicodedata.normalize("NFC", raw)
    require(str(PurePosixPath(normalized)) == normalized,
            f"Non-canonical POSIX spelling: {raw!r}")
    return normalized


def checked_normalized_map(items, label):
    """Map normalized paths to values and fail with both raw collision sources."""
    result = {}
    raw_by_key = defaultdict(list)
    values = defaultdict(list)
    for raw, value in items:
        key = normalized_path(raw)
        raw_by_key[key].append(raw)
        values[key].append(value)
    collisions = [
        {"normalized_path": key, "raw_paths": paths}
        for key, paths in sorted(raw_by_key.items()) if len(paths) != 1
    ]
    if collisions:
        raise ReconciliationError(
            f"Ambiguous {label} normalization collision",
            {"status": "failed_ambiguous_path_normalization", "label": label,
             "collisions": collisions},
        )
    for key, vals in values.items():
        result[key] = vals[0]
    return result


def is_appledouble(path):
    parts = PurePosixPath(path).parts
    return "__MACOSX" in parts or any(part.startswith("._") for part in parts)


def compact_archive_record(record):
    return {key: record[key] for key in
            ("path", "bytes", "md5", "sha256") if key in record}


def read_pinned_metadata(repository_tar, catalog):
    """Read the observed Git-tar schema and cross-check it to the catalog audit."""
    expected_tar_sha = catalog.get("source_inputs_sha256", {}).get(
        repository_tar.name)
    require(expected_tar_sha and sha256_file(repository_tar) == expected_tar_sha,
            "Repository metadata tar does not match catalog-audit input")
    require(catalog.get("status") == "passed_metadata_only"
            and catalog.get("repository_commit") == COMMIT,
            "Catalog audit is not the pinned passed metadata-only audit")
    require(catalog.get("classifier_admission") is False
            and catalog.get("audio_downloaded") is False,
            "Catalog audit crosses the metadata-only boundary")

    prefix = f"saraga-{COMMIT}/"
    blobs = {}
    with tarfile.open(repository_tar, "r:gz") as archive:
        for member in archive:
            raw = member.name
            require(raw == prefix[:-1] or raw.startswith(prefix),
                    f"Unexpected repository-tar root: {raw}")
            if member.isdir():
                continue
            require(member.isfile() and member.size <= 5_000_000,
                    f"Unsafe repository-tar member: {raw}")
            relative = normalized_path(raw[len(prefix):])
            require(relative not in blobs, f"Duplicate repository member: {relative}")
            blobs[relative] = archive.extractfile(member).read()

    csv_name = "dataset/hindustani/file_paths.csv"
    require(csv_name in blobs, "Pinned Hindustani file_paths.csv missing")
    reader = csv.DictReader(io.StringIO(blobs[csv_name].decode("utf-8-sig"), newline=""))
    require(reader.fieldnames == ["", "filepath", "mbid"],
            f"Unexpected file_paths.csv schema: {reader.fieldnames!r}")
    declared = []
    for ordinal, row in enumerate(reader):
        require(set(row) == {"", "filepath", "mbid"} and row[""] == str(ordinal),
                f"Unexpected file_paths.csv row/index at {ordinal}")
        declared.append((row["filepath"], row["mbid"]))
    declared_by_path = checked_normalized_map(declared, "pinned file_paths.csv")
    require(len(set(declared_by_path.values())) == len(declared_by_path),
            "Duplicate MBID in pinned file_paths.csv")

    catalog_rows = [row for row in catalog.get("rows", [])
                    if row.get("tradition") == "hindustani"]
    catalog_by_mbid = {}
    metadata_by_mbid = {}
    for row in catalog_rows:
        mbid = row.get("mbid")
        require(mbid and mbid not in catalog_by_mbid, f"Duplicate catalog MBID: {mbid}")
        catalog_by_mbid[mbid] = row
        metadata_path = normalized_path(row["metadata_path"])
        require(metadata_path in blobs, f"Catalog metadata missing from tar: {metadata_path}")
        data = blobs[metadata_path]
        require(hashlib.sha256(data).hexdigest() == row["metadata_sha256"],
                f"Metadata SHA-256 changed for {mbid}")
        metadata = json.loads(data)
        require(metadata.get("mbid") == mbid, f"Metadata MBID mismatch for {mbid}")
        metadata_by_mbid[mbid] = metadata
        checksum_path = metadata_path[:-5] + ".mp3.md5"
        require(checksum_path in blobs, f"Pinned MP3 checksum missing for {mbid}")
        checksum = blobs[checksum_path].decode("ascii").strip().split()[0].lower()
        require(MD5_RE.fullmatch(checksum) and checksum == row.get("audio_md5"),
                f"Catalog/tar MP3 checksum mismatch for {mbid}")

    require(set(catalog_by_mbid) == set(declared_by_path.values()),
            "Pinned catalog MBIDs differ from file_paths.csv MBIDs")
    return catalog_by_mbid, metadata_by_mbid, declared_by_path


def validate_archive_audit(audit):
    require(audit.get("status") == "passed_archive_integrity_not_audio_admission",
            "Archive audit did not pass integrity verification")
    require(audit.get("source_record") == 4301737,
            "Archive audit is not from pinned Zenodo record 4301737")
    hashes = audit.get("archive_hashes", {})
    require(audit.get("archive_bytes") == ARCHIVE_BYTES
            and hashes.get("md5") == ARCHIVE_MD5
            and hashes.get("sha256") == ARCHIVE_SHA256,
            "Archive audit does not match pinned Hindustani archive bytes/MD5/SHA-256")
    require(audit.get("classifier_admission") is False
            and audit.get("physically_decoded_audio_files") == 0
            and audit.get("extracted_audio_files") == 0,
            "Archive audit crosses the no-extraction/no-decoding boundary")
    records = audit.get("records")
    require(isinstance(records, list)
            and len(records) == audit.get("archive_members_verified"),
            "Archive record inventory count mismatch")
    raw_paths = []
    for record in records:
        require(set(("path", "bytes", "md5", "sha256", "crc_verified")) <= set(record),
                "Archive record schema incomplete")
        require(record["crc_verified"] is True and MD5_RE.fullmatch(record["md5"]),
                f"Unverified/invalid archive record: {record.get('path')}")
        require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]),
                f"Invalid archive SHA-256: {record.get('path')}")
        raw_paths.append((record["path"], record))
    require(sum(record["path"].lower().endswith(".mp3") for record in records)
            == audit.get("mp3_members"), "Archive MP3 member count mismatch")
    return checked_normalized_map(raw_paths, "archive audit")


def parse_credit(credit):
    artist = credit.get("artist") or {}
    instrument = credit.get("instrument") or {}
    require(artist.get("mbid") and artist.get("name"), "Incomplete performer credit")
    return {
        "artist_mbid": artist["mbid"],
        "artist_name": artist["name"],
        "instrument_mbid": instrument.get("mbid"),
        "instrument_name": instrument.get("name"),
        "lead": credit.get("lead") is True,
        "attributes": credit.get("attributes", ""),
    }


def build_connected_groups(tracks):
    """Conservatively connect tracks through any performer, album artist, or event."""
    parent = {track["mbid"]: track["mbid"] for track in tracks}

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    bridges = defaultdict(set)
    bridge_roles = defaultdict(set)
    for track in tracks:
        mbid = track["mbid"]
        for credit in track["performer_credits"]:
            key = ("artist", credit["artist_mbid"])
            bridges[key].add(mbid)
            bridge_roles[key].add("performer_credit")
        for artist in track["album_artists"]:
            key = ("artist", artist["mbid"])
            bridges[key].add(mbid)
            bridge_roles[key].add("album_artist")
        for group in track["release_or_concert_groups"]:
            key = ("release_or_concert", group["mbid"])
            bridges[key].add(mbid)
            bridge_roles[key].add(group["kind"])
    for members in bridges.values():
        ordered = sorted(members)
        for other in ordered[1:]:
            union(ordered[0], other)

    components = defaultdict(list)
    for mbid in sorted(parent):
        components[find(mbid)].append(mbid)
    result = []
    for members in sorted(components.values(), key=lambda value: tuple(value)):
        member_set = set(members)
        component_bridges = [
            {"kind": kind, "identity_mbid": identity,
             "source_roles": sorted(bridge_roles[(kind, identity)]),
             "track_mbids": sorted(track_ids)}
            for (kind, identity), track_ids in sorted(bridges.items())
            if len(track_ids & member_set) >= 2
        ]
        digest = hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]
        result.append({
            "component_id": f"performer-connected-{digest}",
            "track_count": len(members),
            "track_mbids": members,
            "shared_identity_bridges": component_bridges,
        })
    return result


def reconcile(catalog_by_mbid, metadata_by_mbid, declared_by_path, archive_records):
    """Reconcile already-loaded inputs without reading or decoding audio."""
    expected_by_path = {}
    declared_path_by_mbid = {}
    for base, mbid in declared_by_path.items():
        require(mbid in catalog_by_mbid and mbid in metadata_by_mbid,
                f"Declared path has unknown MBID: {mbid}")
        expected = normalized_path(f"{ARCHIVE_ROOT}/{base}.mp3.mp3")
        require(expected not in expected_by_path,
                f"Duplicate expected archive audio path: {expected}")
        expected_by_path[expected] = mbid
        declared_path_by_mbid[mbid] = base

    data_mp3 = {}
    resource_mp3 = []
    other_mp3 = []
    resource_all = []
    auxiliary = []
    unclassified = []
    member_classes = Counter()
    for path, record in archive_records.items():
        lower = path.lower()
        if is_appledouble(path):
            member_classes["appledouble_resource_fork"] += 1
            resource_all.append(compact_archive_record(record))
            if lower.endswith(".mp3"):
                resource_mp3.append(compact_archive_record(record))
            continue
        if lower.endswith(".mp3.mp3") and path.startswith(ARCHIVE_ROOT + "/"):
            member_classes["data_mp3"] += 1
            data_mp3[path] = record
        elif lower.endswith(".mp3"):
            member_classes["unexpected_mp3_naming"] += 1
            other_mp3.append(compact_archive_record(record))
        elif path.endswith("/.DS_Store") or path == ARCHIVE_ROOT + "/.DS_Store":
            member_classes["finder_metadata"] += 1
            auxiliary.append({"kind": "finder_metadata", **compact_archive_record(record)})
        elif path == ARCHIVE_ROOT + "/file_paths.csv":
            member_classes["archive_manifest"] += 1
            auxiliary.append({"kind": "archive_manifest", **compact_archive_record(record)})
        elif path.startswith(ARCHIVE_ROOT + "/") and path.endswith(".json"):
            member_classes["track_metadata"] += 1
        elif (path.startswith(ARCHIVE_ROOT + "/")
              and any(path.endswith("." + suffix)
                      for suffix in set(ANNOTATION_SUFFIXES.values()))):
            member_classes["annotation"] += 1
        else:
            member_classes["unclassified"] += 1
            unclassified.append(compact_archive_record(record))

    missing_paths = sorted(set(expected_by_path) - set(data_mp3))
    extra_paths = sorted(set(data_mp3) - set(expected_by_path))
    mismatched = []
    matched = []
    for path in sorted(set(expected_by_path) & set(data_mp3)):
        mbid = expected_by_path[path]
        expected_md5 = catalog_by_mbid[mbid]["audio_md5"].lower()
        actual_md5 = data_mp3[path]["md5"].lower()
        item = {
            "mbid": mbid,
            "catalog_metadata_path": catalog_by_mbid[mbid]["metadata_path"],
            "declared_archive_base_path": declared_path_by_mbid[mbid],
            "archive_audio_path": path,
            "catalog_md5": expected_md5,
            "archive_md5": actual_md5,
        }
        if expected_md5 == actual_md5:
            matched.append(item)
        else:
            mismatched.append(item)

    annotation_actual_suffixes = set(ANNOTATION_SUFFIXES.values())
    missing_annotations = []
    expected_annotation_paths = set()
    tracks = []
    declared_aliases = []
    performer_names = defaultdict(set)
    album_names = defaultdict(set)
    group_names = defaultdict(set)
    instrument_names = defaultdict(set)
    lead_ids = set()
    for mbid in sorted(catalog_by_mbid):
        row = catalog_by_mbid[mbid]
        metadata = metadata_by_mbid[mbid]
        base = declared_path_by_mbid[mbid]
        catalog_base = row["metadata_path"][len("dataset/hindustani/"):-5]
        if normalized_path(catalog_base) != base:
            declared_aliases.append({"mbid": mbid, "catalog_metadata_base": catalog_base,
                                     "declared_archive_base": base})

        credits = [parse_credit(value) for value in metadata.get("artists", [])]
        for credit in credits:
            performer_names[credit["artist_mbid"]].add(credit["artist_name"])
            if credit["instrument_mbid"]:
                instrument_names[credit["instrument_mbid"]].add(credit["instrument_name"])
            if credit["lead"]:
                lead_ids.add(credit["artist_mbid"])
        album_artists = []
        for artist in metadata.get("album_artists", []):
            require(artist.get("mbid") and artist.get("name"), "Incomplete album artist")
            album_artists.append({"mbid": artist["mbid"], "name": artist["name"]})
            album_names[artist["mbid"]].add(artist["name"])
        groups = []
        for kind in ("concert", "release"):
            for group in metadata.get(kind, []):
                name = group.get("title") or group.get("name")
                require(group.get("mbid") and name, f"Incomplete {kind} identity")
                groups.append({"kind": kind, "mbid": group["mbid"], "name": name})
                group_names[(kind, group["mbid"])].add(name)

        companion = set(row.get("companion_files", []))
        unknown_companions = companion - set(ANNOTATION_SUFFIXES) - NON_ANNOTATION_COMPANIONS
        require(not unknown_companions,
                f"Unknown catalog companions for {mbid}: {sorted(unknown_companions)}")
        annotations = []
        for catalog_suffix, archive_suffix in ANNOTATION_SUFFIXES.items():
            catalog_present = catalog_suffix in companion
            archive_path = normalized_path(f"{ARCHIVE_ROOT}/{base}.{archive_suffix}")
            archive_present = archive_path in archive_records and not is_appledouble(archive_path)
            if catalog_present:
                expected_annotation_paths.add(archive_path)
                if not archive_present:
                    missing_annotations.append({"mbid": mbid, "kind": catalog_suffix,
                                                "archive_path": archive_path})
            annotations.append({
                "kind": catalog_suffix,
                "catalog_presence": catalog_present,
                "archive_member_presence": archive_present,
                "archive_path": archive_path if archive_present else None,
                "content_or_accuracy_verified": False,
            })
        tracks.append({
            "mbid": mbid,
            "title": metadata.get("title"),
            "advertised_length_ms": metadata.get("length"),
            "speech_title_review_flag": row.get("speech_title_review_flag", False),
            "catalog_metadata_path": row["metadata_path"],
            "declared_archive_base_path": base,
            "archive_audio_path": f"{ARCHIVE_ROOT}/{base}.mp3.mp3",
            "performer_credits": credits,
            "album_artists": album_artists,
            "release_or_concert_groups": groups,
            "annotations": annotations,
        })

    actual_annotation_paths = set()
    for path in archive_records:
        if is_appledouble(path) or not path.startswith(ARCHIVE_ROOT + "/"):
            continue
        if any(path.endswith("." + suffix) for suffix in annotation_actual_suffixes):
            actual_annotation_paths.add(path)
    extra_annotations = [
        {"archive_path": path} for path in sorted(actual_annotation_paths - expected_annotation_paths)
    ]
    expected_metadata_paths = {
        normalized_path(f"{ARCHIVE_ROOT}/{base}.json")
        for base in declared_by_path
    }
    actual_metadata_paths = {
        path for path in archive_records
        if not is_appledouble(path) and path.startswith(ARCHIVE_ROOT + "/")
        and path.endswith(".json")
    }
    missing_metadata = [{"archive_path": path}
                        for path in sorted(expected_metadata_paths - actual_metadata_paths)]
    extra_metadata = [{"archive_path": path}
                      for path in sorted(actual_metadata_paths - expected_metadata_paths)]

    identity_conflicts = []
    for kind, registry in (("performer_artist", performer_names),
                           ("album_artist", album_names),
                           ("instrument", instrument_names)):
        for mbid, names in sorted(registry.items()):
            if len(names) != 1:
                identity_conflicts.append({"kind": kind, "mbid": mbid,
                                           "names": sorted(names)})
    for (kind, mbid), names in sorted(group_names.items()):
        if len(names) != 1:
            identity_conflicts.append({"kind": kind, "mbid": mbid,
                                       "names": sorted(names)})
    all_artist_names = defaultdict(set)
    for registry in (performer_names, album_names):
        for mbid, names in registry.items():
            all_artist_names[mbid].update(names)
    for mbid, names in sorted(all_artist_names.items()):
        if len(names) != 1 and not any(
                item["kind"] == "artist_any_role" and item["mbid"] == mbid
                for item in identity_conflicts):
            identity_conflicts.append({"kind": "artist_any_role", "mbid": mbid,
                                       "names": sorted(names)})

    failures = []
    if missing_paths:
        failures.append("missing_archive_audio")
    if extra_paths or other_mp3:
        failures.append("extra_or_unexpected_archive_audio")
    if mismatched:
        failures.append("audio_md5_mismatch")
    if missing_annotations or extra_annotations:
        failures.append("annotation_presence_mismatch")
    if missing_metadata or extra_metadata:
        failures.append("track_metadata_presence_mismatch")
    if identity_conflicts:
        failures.append("metadata_identity_name_conflict")
    if unclassified:
        failures.append("unclassified_archive_members")

    connected_groups = build_connected_groups(tracks)
    tracks_without_performers = sorted(
        track["mbid"] for track in tracks if not track["performer_credits"])

    report = {
        "status": ("passed_archive_catalog_reconciliation_not_audio_admission"
                   if not failures else "failed_archive_catalog_reconciliation"),
        "failure_reasons": failures,
        "scope": "hindustani_inventory_and_metadata_identity_only",
        "inventory": {
            "archive_records_total": len(archive_records),
            "archive_mp3_suffix_records": len(data_mp3) + len(resource_mp3) + len(other_mp3),
            "data_mp3_records": len(data_mp3),
            "appledouble_resource_fork_mp3_records": len(resource_mp3),
            "other_mp3_records": len(other_mp3),
            "matched_audio_records": len(matched),
            "missing_audio_records": len(missing_paths),
            "extra_audio_records": len(extra_paths) + len(other_mp3),
            "mismatched_audio_md5_records": len(mismatched),
            "all_appledouble_resource_fork_records": len(resource_all),
            "expected_annotation_members": len(expected_annotation_paths),
            "missing_annotation_members": len(missing_annotations),
            "extra_annotation_members": len(extra_annotations),
            "missing_track_metadata_members": len(missing_metadata),
            "extra_track_metadata_members": len(extra_metadata),
        },
        "archive_member_class_counts": dict(sorted(member_classes.items())),
        "matched_audio": matched,
        "missing_audio": [
            {"mbid": expected_by_path[path], "expected_archive_path": path,
             "catalog_md5": catalog_by_mbid[expected_by_path[path]]["audio_md5"]}
            for path in missing_paths
        ],
        "extra_archive_audio": [compact_archive_record(data_mp3[path]) for path in extra_paths]
                               + other_mp3,
        "mismatched_audio_md5": mismatched,
        "appledouble_resource_fork_mp3_records": sorted(resource_mp3, key=lambda x: x["path"]),
        "auxiliary_archive_records": sorted(auxiliary, key=lambda x: x["path"]),
        "unclassified_archive_records": sorted(unclassified, key=lambda x: x["path"]),
        "declared_path_aliases": declared_aliases,
        "missing_annotations": missing_annotations,
        "extra_annotations": extra_annotations,
        "missing_track_metadata": missing_metadata,
        "extra_track_metadata": extra_metadata,
        "identity_summary": {
            "track_mbids": len(tracks),
            "performer_artist_mbids": len(performer_names),
            "lead_performer_artist_mbids": len(lead_ids),
            "album_artist_mbids": len(album_names),
            "release_or_concert_mbids": len({mbid for _, mbid in group_names}),
            "instrument_mbids": len(instrument_names),
            "tracks_without_performer_credits": tracks_without_performers,
            "performer_connected_group_count": len(connected_groups),
            "performer_connected_group_size_counts": dict(sorted(Counter(
                group["track_count"] for group in connected_groups).items())),
            "identity_name_conflicts": identity_conflicts,
        },
        "tracks": tracks,
        "performer_connected_groups": connected_groups,
        "performer_grouping_rule": (
            "connected components over all credited performer artist MBIDs, including "
            "accompanists, plus album-artist and release/concert MBIDs; no cohorts selected"
        ),
        "annotation_presence_only": True,
        "annotation_content_or_accuracy_verified": False,
        "physical_audio_decoded": False,
        "classifier_admission": False,
        "cohort_selected": False,
    }
    return report


def load_and_reconcile(catalog_path, repository_tar, archive_audit_path):
    inputs = snapshot_files({
        "catalog_audit": catalog_path,
        "repository_metadata_tar": repository_tar,
        "archive_audit": archive_audit_path,
        "reconciliation_code": Path(__file__),
    })
    catalog = json.loads(catalog_path.read_text())
    archive_audit = json.loads(archive_audit_path.read_text())
    catalog_by_mbid, metadata_by_mbid, declared_by_path = read_pinned_metadata(
        repository_tar, catalog)
    archive_records = validate_archive_audit(archive_audit)
    report = reconcile(catalog_by_mbid, metadata_by_mbid, declared_by_path,
                       archive_records)
    require_snapshot_unchanged(inputs)
    report["inputs"] = {
        "catalog_audit": str(catalog_path),
        "catalog_audit_sha256": inputs["catalog_audit"]["sha256"],
        "repository_metadata_tar": str(repository_tar),
        "repository_metadata_tar_sha256": inputs["repository_metadata_tar"]["sha256"],
        "archive_audit": str(archive_audit_path),
        "archive_audit_sha256": inputs["archive_audit"]["sha256"],
    }
    report["repository_commit"] = COMMIT
    report["archive_root"] = ARCHIVE_ROOT
    report["archive_integrity_evidence"] = {
        "source_record": archive_audit["source_record"],
        "archive_path": archive_audit.get("archive_path"),
        "archive_bytes": archive_audit["archive_bytes"],
        "archive_hashes": archive_audit["archive_hashes"],
        "archive_members_verified_to_eof_for_crc": archive_audit["archive_members_verified"],
    }
    report["code_sha256"] = inputs["reconciliation_code"]["sha256"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-audit", type=Path, required=True)
    parser.add_argument("--repository-metadata-tar", type=Path, required=True)
    parser.add_argument("--archive-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Refusing existing output")
    try:
        report = load_and_reconcile(args.catalog_audit, args.repository_metadata_tar,
                                    args.archive_audit)
    except ReconciliationError as exc:
        report = exc.report
        report["error"] = str(exc)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("tracks", "matched_audio",
                                     "appledouble_resource_fork_mp3_records")},
                     indent=2, ensure_ascii=False), flush=True)
    return 0 if report.get("status", "").startswith("passed_") else 2


if __name__ == "__main__":
    sys.exit(main())
