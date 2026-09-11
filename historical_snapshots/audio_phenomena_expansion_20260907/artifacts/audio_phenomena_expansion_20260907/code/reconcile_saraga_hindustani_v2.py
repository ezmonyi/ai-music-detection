#!/usr/bin/env python3
"""Reconcile Saraga v1.5 paths through unique pinned MP3 checksums.

Version 2 preserves the v1 receipt and code as provenance.  It resolves path
spelling differences only through a one-to-one exact MD5 relation; it never
rewrites names heuristically, opens audio, decodes media, or selects a cohort.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import reconcile_saraga_hindustani_v1 as v1


ADAPTER_VERSION = "saraga_hindustani_checksum_path_reconciliation_v2"
V1_CODE_SHA256 = "36101cd425cf242c1de5ae1784dc3b25f1ed6a7476c438974810ec5a87a9bdb9"
V1_TEST_SHA256 = "f57ddd5febcc8618e66da4d1f39730a217fc930319c4ef9a7288d7011b9c60da"
V1_FAILED_RECEIPT_SHA256 = "e626fa8155cf11919ef0c17cad3b2b5baeb3f041977ace7b45375cfd89921b43"
AUDIO_SUFFIX = ".mp3.mp3"


def require(value, message):
    if not value:
        raise ValueError(message)


def checksum_groups(items):
    groups = defaultdict(list)
    for checksum, value in items:
        require(isinstance(checksum, str) and v1.MD5_RE.fullmatch(checksum.lower()),
                f"Invalid MP3 MD5: {checksum!r}")
        groups[checksum.lower()].append(value)
    return groups


def archive_audio_candidates(archive_records):
    candidates = []
    resource_forks = []
    unexpected_mp3 = []
    base_sources = []
    for path, record in archive_records.items():
        lower = path.lower()
        if not lower.endswith(".mp3"):
            continue
        if v1.is_appledouble(path):
            resource_forks.append(v1.compact_archive_record(record))
            continue
        if lower.endswith(AUDIO_SUFFIX) and path.startswith(v1.ARCHIVE_ROOT + "/"):
            raw_base = path[len(v1.ARCHIVE_ROOT) + 1:-len(AUDIO_SUFFIX)]
            base = v1.normalized_path(raw_base)
            base_sources.append((raw_base, path))
            candidates.append((base, record))
        else:
            unexpected_mp3.append(v1.compact_archive_record(record))
    # A collision here would make an actual archive base ambiguous even if the
    # ZIP member spellings differed only by Unicode normalization.
    v1.checked_normalized_map(base_sources, "checksum-resolved archive base")
    return candidates, resource_forks, unexpected_mp3


def resolve_audio_paths_by_checksum(catalog_by_mbid, declared_by_path, archive_records):
    """Resolve every catalog row to an archive base solely by exact unique MD5."""
    pinned_by_mbid = {}
    for base, mbid in declared_by_path.items():
        require(mbid not in pinned_by_mbid, f"Duplicate declared MBID: {mbid}")
        pinned_by_mbid[mbid] = base
    require(set(pinned_by_mbid) == set(catalog_by_mbid),
            "Catalog and pinned file_paths.csv MBIDs differ")

    candidates, resource_forks, unexpected_mp3 = archive_audio_candidates(archive_records)
    catalog_groups = checksum_groups(
        (row["audio_md5"], {"mbid": mbid,
                            "pinned_file_paths_csv_base_path": pinned_by_mbid[mbid],
                            "catalog_md5": row["audio_md5"].lower()})
        for mbid, row in catalog_by_mbid.items())
    archive_groups = checksum_groups(
        (record["md5"], {"checksum_resolved_archive_base_path": base,
                         **v1.compact_archive_record(record)})
        for base, record in candidates)

    duplicate_catalog = [
        {"md5": checksum, "catalog_records": sorted(values, key=lambda x: x["mbid"])}
        for checksum, values in sorted(catalog_groups.items()) if len(values) != 1
    ]
    duplicate_archive = [
        {"md5": checksum, "archive_records": sorted(values, key=lambda x: x["path"])}
        for checksum, values in sorted(archive_groups.items()) if len(values) != 1
    ]
    missing_catalog = [
        value for checksum, values in sorted(catalog_groups.items())
        if checksum not in archive_groups for value in values
    ]
    unmatched_archive = [
        value for checksum, values in sorted(archive_groups.items())
        if checksum not in catalog_groups for value in values
    ] + unexpected_mp3

    failure_reasons = []
    if duplicate_catalog:
        failure_reasons.append("duplicate_catalog_mp3_checksum")
    if duplicate_archive:
        failure_reasons.append("duplicate_archive_mp3_checksum")
    if missing_catalog:
        failure_reasons.append("catalog_checksum_missing_from_archive")
    if unmatched_archive:
        failure_reasons.append("archive_checksum_unmatched_to_catalog")

    resolution = {
        "status": ("passed_unique_checksum_path_resolution" if not failure_reasons
                   else "failed_unique_checksum_path_resolution"),
        "failure_reasons": failure_reasons,
        "inventory": {
            "catalog_audio_records": len(catalog_by_mbid),
            "archive_data_mp3_records": len(candidates),
            "appledouble_resource_fork_mp3_records": len(resource_forks),
            "unexpected_nonresource_mp3_records": len(unexpected_mp3),
            "unique_catalog_checksums": len(catalog_groups),
            "unique_archive_data_checksums": len(archive_groups),
            "duplicate_catalog_checksums": len(duplicate_catalog),
            "duplicate_archive_checksums": len(duplicate_archive),
            "catalog_checksums_missing_from_archive": len(missing_catalog),
            "archive_checksums_unmatched_to_catalog": len(unmatched_archive),
        },
        "duplicate_catalog_checksums": duplicate_catalog,
        "duplicate_archive_checksums": duplicate_archive,
        "catalog_checksums_missing_from_archive": missing_catalog,
        "archive_checksums_unmatched_to_catalog": unmatched_archive,
        "appledouble_resource_fork_mp3_records": sorted(resource_forks,
                                                          key=lambda x: x["path"]),
        "resolution_rule": "exact unique catalog MP3 MD5 equals exact unique archive member MD5",
        "fuzzy_path_rewriting_used": False,
    }
    if failure_reasons:
        return resolution, None

    resolved_by_path = {}
    matches = []
    for checksum in sorted(catalog_groups):
        catalog_item = catalog_groups[checksum][0]
        archive_item = archive_groups[checksum][0]
        base = archive_item["checksum_resolved_archive_base_path"]
        mbid = catalog_item["mbid"]
        require(base not in resolved_by_path, f"Resolved archive base collision: {base}")
        resolved_by_path[base] = mbid
        matches.append({
            "mbid": mbid,
            "catalog_md5": checksum,
            "pinned_file_paths_csv_base_path": catalog_item[
                "pinned_file_paths_csv_base_path"],
            "checksum_resolved_archive_base_path": base,
            "archive_audio_path": archive_item["path"],
            "archive_bytes": archive_item.get("bytes"),
            "archive_sha256": archive_item.get("sha256"),
            "resolution": "exact_unique_mp3_md5",
        })
    resolution["matches"] = matches
    return resolution, resolved_by_path


def metadata_base(row):
    prefix = "dataset/hindustani/"
    path = row["metadata_path"]
    require(path.startswith(prefix) and path.endswith(".json"),
            f"Unexpected Hindustani metadata path: {path}")
    return v1.normalized_path(path[len(prefix):-5])


def adapt_success_report(base_report, resolution, catalog_by_mbid, declared_by_path):
    pinned_by_mbid = {mbid: base for base, mbid in declared_by_path.items()}
    resolved_by_mbid = {
        item["mbid"]: item["checksum_resolved_archive_base_path"]
        for item in resolution["matches"]
    }
    metadata_vs_csv = []
    csv_vs_archive = []
    for mbid in sorted(catalog_by_mbid):
        metadata_path_base = metadata_base(catalog_by_mbid[mbid])
        pinned_base = pinned_by_mbid[mbid]
        actual_base = resolved_by_mbid[mbid]
        if metadata_path_base != pinned_base:
            metadata_vs_csv.append({
                "mbid": mbid,
                "pinned_metadata_base_path": metadata_path_base,
                "pinned_file_paths_csv_base_path": pinned_base,
                "relation": "explicit_same_mbid_in_pinned_repository",
            })
        if pinned_base != actual_base:
            csv_vs_archive.append({
                "mbid": mbid,
                "catalog_md5": catalog_by_mbid[mbid]["audio_md5"],
                "pinned_file_paths_csv_base_path": pinned_base,
                "checksum_resolved_archive_base_path": actual_base,
                "pinned_expected_audio_path": (
                    f"{v1.ARCHIVE_ROOT}/{pinned_base}{AUDIO_SUFFIX}"),
                "actual_archive_audio_path": (
                    f"{v1.ARCHIVE_ROOT}/{actual_base}{AUDIO_SUFFIX}"),
                "relation": "exact_unique_mp3_md5",
            })

    base_report.pop("declared_path_aliases", None)
    for track in base_report["tracks"]:
        actual = track.pop("declared_archive_base_path")
        track["pinned_file_paths_csv_base_path"] = pinned_by_mbid[track["mbid"]]
        track["checksum_resolved_archive_base_path"] = actual
        track["path_resolution"] = "exact_unique_mp3_md5"
    for match in base_report["matched_audio"]:
        actual = match.pop("declared_archive_base_path")
        match["pinned_file_paths_csv_base_path"] = pinned_by_mbid[match["mbid"]]
        match["checksum_resolved_archive_base_path"] = actual
        match["path_resolution"] = "exact_unique_mp3_md5"

    passed = base_report["status"].startswith("passed_")
    base_report["status"] = (
        "passed_archive_catalog_checksum_path_reconciliation_v2_not_audio_admission"
        if passed else "failed_archive_catalog_checksum_path_reconciliation_v2")
    base_report["adapter_version"] = ADAPTER_VERSION
    base_report["scope"] = "hindustani_checksum_resolved_inventory_and_metadata_identity_only"
    base_report["checksum_path_resolution"] = {
        key: value for key, value in resolution.items()
        if key not in ("matches", "appledouble_resource_fork_mp3_records")
    }
    base_report["inventory"]["checksum_resolved_audio_records"] = len(
        resolution["matches"])
    base_report["inventory"]["exact_pinned_csv_path_audio_records"] = (
        len(resolution["matches"]) - len(csv_vs_archive))
    base_report["inventory"]["checksum_resolved_path_alias_records"] = len(csv_vs_archive)
    base_report["pinned_metadata_vs_file_paths_csv_differences"] = metadata_vs_csv
    base_report["file_paths_csv_vs_checksum_resolved_archive_path_differences"] = csv_vs_archive
    base_report["path_resolution_statement"] = (
        "Archive bases are resolved only by one-to-one exact MP3 MD5; original strings are "
        "retained and no path-character substitution or fuzzy matching is applied"
    )
    return base_report


def reconcile_v2(catalog_by_mbid, metadata_by_mbid, declared_by_path, archive_records):
    resolution, resolved_by_path = resolve_audio_paths_by_checksum(
        catalog_by_mbid, declared_by_path, archive_records)
    if resolved_by_path is None:
        return {
            "status": "failed_archive_catalog_checksum_path_reconciliation_v2",
            "failure_reasons": resolution["failure_reasons"],
            "adapter_version": ADAPTER_VERSION,
            "scope": "hindustani_checksum_resolved_inventory_and_metadata_identity_only",
            "checksum_path_resolution": resolution,
            "physical_audio_decoded": False,
            "classifier_admission": False,
            "cohort_selected": False,
        }
    base_report = v1.reconcile(catalog_by_mbid, metadata_by_mbid,
                               resolved_by_path, archive_records)
    return adapt_success_report(base_report, resolution, catalog_by_mbid,
                                declared_by_path)


def validate_frozen_v1(v1_code, v1_failed_receipt):
    require(v1_code.resolve() == Path(v1.__file__).resolve(),
            "Provided v1 code is not the imported v1 implementation")
    require(v1.sha256_file(v1_code) == V1_CODE_SHA256,
            "Frozen v1 reconciliation code hash changed")
    require(v1.sha256_file(v1_code.with_name("test_reconcile_saraga_hindustani_v1.py"))
            == V1_TEST_SHA256, "Frozen v1 reconciliation test hash changed")
    require(v1.sha256_file(v1_failed_receipt) == V1_FAILED_RECEIPT_SHA256,
            "Frozen v1 failed receipt hash changed")
    receipt = json.loads(v1_failed_receipt.read_text())
    require(receipt.get("status") == "failed_archive_catalog_reconciliation"
            and receipt.get("inventory", {}).get("matched_audio_records") == 92
            and receipt.get("inventory", {}).get("missing_audio_records") == 16
            and receipt.get("inventory", {}).get("extra_audio_records") == 16
            and receipt.get("inventory", {}).get("mismatched_audio_md5_records") == 0,
            "Frozen v1 receipt no longer has the reviewed 92+16 path-only failure")


def load_and_reconcile_v2(catalog_path, repository_tar, archive_audit_path,
                          v1_code, v1_failed_receipt):
    named_inputs = {
        "catalog_audit": catalog_path,
        "repository_metadata_tar": repository_tar,
        "archive_audit": archive_audit_path,
        "v1_reconciliation_code": v1_code,
        "v1_reconciliation_test": v1_code.with_name(
            "test_reconcile_saraga_hindustani_v1.py"),
        "v1_failed_receipt": v1_failed_receipt,
        "v2_reconciliation_code": Path(__file__),
    }
    snapshots = v1.snapshot_files(named_inputs)
    validate_frozen_v1(v1_code, v1_failed_receipt)
    catalog = json.loads(catalog_path.read_text())
    archive_audit = json.loads(archive_audit_path.read_text())
    catalog_by_mbid, metadata_by_mbid, declared_by_path = v1.read_pinned_metadata(
        repository_tar, catalog)
    require(len(catalog_by_mbid) == len(metadata_by_mbid) == len(declared_by_path) == 108,
            "Pinned Hindustani catalog must contain exactly 108 tracks")
    archive_records = v1.validate_archive_audit(archive_audit)
    report = reconcile_v2(catalog_by_mbid, metadata_by_mbid, declared_by_path,
                          archive_records)
    v1.require_snapshot_unchanged(snapshots)
    report["inputs"] = snapshots
    report["repository_commit"] = v1.COMMIT
    report["archive_root"] = v1.ARCHIVE_ROOT
    report["archive_integrity_evidence"] = {
        "source_record": archive_audit["source_record"],
        "archive_path": archive_audit.get("archive_path"),
        "archive_bytes": archive_audit["archive_bytes"],
        "archive_hashes": archive_audit["archive_hashes"],
        "archive_members_verified_to_eof_for_crc": archive_audit["archive_members_verified"],
    }
    report["provenance"] = {
        "frozen_v1_code_sha256": V1_CODE_SHA256,
        "frozen_v1_test_sha256": V1_TEST_SHA256,
        "frozen_v1_failed_receipt_sha256": V1_FAILED_RECEIPT_SHA256,
        "v1_receipt_retained_as_failed_path_mapping_evidence": True,
        "v2_code_sha256": snapshots["v2_reconciliation_code"]["sha256"],
    }
    return report


def stdout_summary(report, output):
    inventory = report.get("inventory", {})
    return {
        "status": report.get("status"),
        "failure_reasons": report.get("failure_reasons", []),
        "matched_audio_records": inventory.get("matched_audio_records"),
        "checksum_resolved_audio_records": inventory.get("checksum_resolved_audio_records"),
        "checksum_resolved_path_alias_records": inventory.get(
            "checksum_resolved_path_alias_records"),
        "missing_audio_records": inventory.get("missing_audio_records"),
        "extra_audio_records": inventory.get("extra_audio_records"),
        "mismatched_audio_md5_records": inventory.get("mismatched_audio_md5_records"),
        "identity_summary": report.get("identity_summary"),
        "physical_audio_decoded": report.get("physical_audio_decoded"),
        "classifier_admission": report.get("classifier_admission"),
        "cohort_selected": report.get("cohort_selected"),
        "output": str(output),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-audit", type=Path, required=True)
    parser.add_argument("--repository-metadata-tar", type=Path, required=True)
    parser.add_argument("--archive-audit", type=Path, required=True)
    parser.add_argument("--v1-code", type=Path, required=True)
    parser.add_argument("--v1-failed-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "Refusing existing output")
    try:
        report = load_and_reconcile_v2(
            args.catalog_audit, args.repository_metadata_tar, args.archive_audit,
            args.v1_code, args.v1_failed_receipt)
    except v1.ReconciliationError as exc:
        report = exc.report
        report["error"] = str(exc)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps(stdout_summary(report, args.output), indent=2, ensure_ascii=False),
          flush=True)
    return 0 if report.get("status", "").startswith("passed_") else 2


if __name__ == "__main__":
    sys.exit(main())
