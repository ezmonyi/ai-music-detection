#!/usr/bin/env python3
"""Build a metadata-only VocalSet straight/vibrato matched-pair protocol.

This program never opens audio payloads and never computes pitch or class
scores.  It consumes the independently audited VocalSet-Breath acquisition,
checks the official archive-path technique metadata, excludes the known
m9/male8 identity conflict, and freezes a singer-disjoint development/evaluation
allocation for later review.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "vocalset-vibrato-metadata-protocol-v1"
EXPECTED_ACQUISITION_ROWS = 244
SPLIT_SEED = "vocalset-vibrato-presence-v1"
EVALUATION_QUOTA_BY_SEX = {"f": 5, "m": 5}
TARGET_TECHNIQUES = {"straight", "vibrato"}
TARGET_CONTEXTS = {"arpeggios", "scales", "excerpts"}
EXCLUDED_IDENTITY_FILES = {
    "m9_caro_vibrato.wav": (
        "filename/annotation singer m9 conflicts with selected official archive "
        "directory FULL/male8; duration resolved bytes but not performer identity"
    )
}
SINGER_RE = re.compile(r"^(?P<sex>[fm])(?P<number>[1-9][0-9]*)$")
EXCERPT_RE = re.compile(
    r"^(?P<singer>[fm][1-9][0-9]*)_(?P<content>caro|dona|row)_"
    r"(?P<technique>straight|vibrato)\.wav$"
)
EXERCISE_RE = re.compile(
    r"^(?P<singer>[fm][1-9][0-9]*)_(?P<context>arpeggios|scales)_"
    r"(?P<technique>straight|vibrato)(?P<suffix>(?:_[A-Za-z0-9]+)+)\.wav$"
)


def load_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite/nonstandard JSON constant: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expected_archive_singer(singer: str) -> str:
    match = SINGER_RE.fullmatch(singer)
    if match is None:
        raise ValueError(f"invalid singer identifier: {singer!r}")
    prefix = "female" if match.group("sex") == "f" else "male"
    return prefix + match.group("number")


def parse_target_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Validate one row and return target-pair metadata, or None if non-target."""

    filename = row.get("filename")
    singer = row.get("singer")
    archive_member = row.get("archive_member")
    if not isinstance(filename, str) or not isinstance(singer, str):
        raise ValueError("manifest filename and singer must be strings")
    if not isinstance(archive_member, str):
        raise ValueError(f"{filename}: archive_member must be a string")
    parts = archive_member.split("/")
    if len(parts) != 5 or parts[0] != "FULL" or parts[-1] != filename:
        raise ValueError(f"{filename}: unexpected archive member layout {archive_member!r}")
    archive_singer, context, technique = parts[1:4]
    if context not in TARGET_CONTEXTS:
        raise ValueError(f"{filename}: unexpected context {context!r}")
    if filename not in EXCLUDED_IDENTITY_FILES and archive_singer != expected_archive_singer(singer):
        raise ValueError(
            f"{filename}: archive singer {archive_singer!r} disagrees with {singer!r}"
        )
    if technique not in TARGET_TECHNIQUES:
        return None

    if context == "excerpts":
        match = EXCERPT_RE.fullmatch(filename)
        if match is None:
            raise ValueError(f"{filename}: target excerpt name is not parseable")
        parsed_singer = match.group("singer")
        parsed_technique = match.group("technique")
        content_id = match.group("content")
    else:
        match = EXERCISE_RE.fullmatch(filename)
        if match is None:
            raise ValueError(f"{filename}: target exercise name is not parseable")
        parsed_singer = match.group("singer")
        parsed_technique = match.group("technique")
        if match.group("context") != context:
            raise ValueError(f"{filename}: filename/archive context disagreement")
        content_id = context + match.group("suffix").lower()
    if parsed_singer != singer or parsed_technique != technique:
        raise ValueError(f"{filename}: filename/archive technique or singer disagreement")

    required_physical = (
        "audio_path",
        "audio_sha256",
        "sample_rate",
        "frames",
        "channels",
        "duration_verified",
        "member_crc32",
        "member_bytes",
    )
    missing = [key for key in required_physical if row.get(key) in (None, "")]
    if missing:
        raise ValueError(f"{filename}: missing audited physical fields {missing}")
    return {
        "filename": filename,
        "singer": singer,
        "sex": singer[0],
        "context": context,
        "content_id": content_id,
        "technique": technique,
        "audio_path": row["audio_path"],
        "audio_sha256": row["audio_sha256"],
        "archive_member": archive_member,
        "member_crc32": row["member_crc32"],
        "member_bytes": int(row["member_bytes"]),
        "sample_rate_hz": int(row["sample_rate"]),
        "frames": int(row["frames"]),
        "channels": int(row["channels"]),
        "duration_sec": float(row["duration_verified"]),
        "official_label_source": "VocalSet archive technique directory plus filename token",
    }


def allocate_singers(singers: list[str]) -> tuple[dict[str, str], dict[str, Any]]:
    """Deterministically select five singers per sex for held-out evaluation."""

    allocation: dict[str, str] = {}
    detail: dict[str, Any] = {"seed": SPLIT_SEED, "algorithm": "SHA256(seed|singer) rank within sex"}
    for sex in ("f", "m"):
        members = sorted(
            (s for s in singers if s.startswith(sex)),
            key=lambda singer: (hashlib.sha256(f"{SPLIT_SEED}|{singer}".encode()).hexdigest(), singer),
        )
        quota = EVALUATION_QUOTA_BY_SEX[sex]
        if len(members) <= quota:
            raise ValueError(f"not enough {sex!r} singers for the frozen split")
        evaluation = members[:quota]
        development = members[quota:]
        for singer in evaluation:
            allocation[singer] = "evaluation"
        for singer in development:
            allocation[singer] = "development"
        detail[sex] = {"evaluation": evaluation, "development": development}
    return allocation, detail


def build_protocol(acquisition_summary: Path, independent_audit: Path) -> dict[str, Any]:
    summary = load_json(acquisition_summary)
    audit = load_json(independent_audit)
    rows = summary.get("manifest")
    if (
        summary.get("status") != "complete_accounting"
        or summary.get("selected") != EXPECTED_ACQUISITION_ROWS
        or summary.get("downloaded_verified") != EXPECTED_ACQUISITION_ROWS
        or summary.get("errors") != []
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_ACQUISITION_ROWS
    ):
        raise ValueError("VocalSet acquisition v2 is not complete 244/244 clean accounting")
    if not audit.get("gate_passed") or audit.get("status") != "passed":
        raise ValueError("independent acquisition audit did not pass")
    if audit.get("acquisition_summary_sha256") != sha256_file(acquisition_summary):
        raise ValueError("independent audit does not bind the supplied acquisition summary")

    filenames = [row.get("filename") for row in rows]
    if len(set(filenames)) != len(filenames):
        raise ValueError("acquisition manifest contains duplicate filenames")
    if set(EXCLUDED_IDENTITY_FILES) - set(filenames):
        raise ValueError("known identity-conflict file is missing instead of explicitly excluded")

    target_rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    non_target_count = 0
    all_singers: set[str] = set()
    for row in rows:
        singer = row.get("singer")
        if not isinstance(singer, str) or SINGER_RE.fullmatch(singer) is None:
            raise ValueError(f"invalid singer in acquisition row: {singer!r}")
        all_singers.add(singer)
        if row.get("filename") in EXCLUDED_IDENTITY_FILES:
            excluded.append(
                {
                    "filename": row["filename"],
                    "archive_member": row["archive_member"],
                    "reason": EXCLUDED_IDENTITY_FILES[row["filename"]],
                }
            )
            continue
        parsed = parse_target_row(row)
        if parsed is None:
            non_target_count += 1
        else:
            target_rows.append(parsed)

    allocation, split_detail = allocate_singers(sorted(all_singers))
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for row in target_rows:
        key = (row["singer"], row["context"], row["content_id"])
        if row["technique"] in groups[key]:
            raise ValueError(f"duplicate technique recording for matched key {key}")
        groups[key][row["technique"]] = row

    pairs: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for key, techniques in sorted(groups.items()):
        singer, context, content_id = key
        if set(techniques) != TARGET_TECHNIQUES:
            unmatched.append(
                {
                    "singer": singer,
                    "context": context,
                    "content_id": content_id,
                    "present_techniques": sorted(techniques),
                    "filenames": sorted(row["filename"] for row in techniques.values()),
                    "reason": "one-sided source subset; no exact matched counterpart",
                }
            )
            continue
        straight = techniques["straight"]
        vibrato = techniques["vibrato"]
        pair = {
            "pair_id": f"{singer}__{context}__{content_id}",
            "split": allocation[singer],
            "singer": singer,
            "sex": singer[0],
            "context": context,
            "content_id": content_id,
            "official_label_scope": "clip-level instructed technique",
            "straight_filename": straight["filename"],
            "straight_audio_path": straight["audio_path"],
            "straight_audio_sha256": straight["audio_sha256"],
            "straight_archive_member": straight["archive_member"],
            "straight_duration_sec": straight["duration_sec"],
            "straight_sample_rate_hz": straight["sample_rate_hz"],
            "straight_frames": straight["frames"],
            "vibrato_filename": vibrato["filename"],
            "vibrato_audio_path": vibrato["audio_path"],
            "vibrato_audio_sha256": vibrato["audio_sha256"],
            "vibrato_archive_member": vibrato["archive_member"],
            "vibrato_duration_sec": vibrato["duration_sec"],
            "vibrato_sample_rate_hz": vibrato["sample_rate_hz"],
            "vibrato_frames": vibrato["frames"],
            "absolute_duration_difference_sec": abs(
                straight["duration_sec"] - vibrato["duration_sec"]
            ),
        }
        pairs.append(pair)

    used_files = [name for pair in pairs for name in (pair["straight_filename"], pair["vibrato_filename"])]
    if len(used_files) != len(set(used_files)):
        raise ValueError("a source recording was reused across matched pairs")
    if {pair["singer"] for pair in pairs if pair["split"] == "development"} & {
        pair["singer"] for pair in pairs if pair["split"] == "evaluation"
    }:
        raise ValueError("singer leakage across development/evaluation splits")

    pair_counts_by_split = collections.Counter(pair["split"] for pair in pairs)
    pair_counts_by_context = collections.Counter(pair["context"] for pair in pairs)
    pair_counts_by_split_context = collections.Counter(
        (pair["split"], pair["context"]) for pair in pairs
    )
    pair_counts_by_singer = collections.Counter(pair["singer"] for pair in pairs)
    return {
        "schema": SCHEMA,
        "status": "metadata_ready_for_root_review_no_scoring",
        "no_audio_opened": True,
        "no_pitch_extraction": True,
        "no_ai_human_labels": True,
        "acquisition_summary_path": str(acquisition_summary.resolve()),
        "acquisition_summary_sha256": sha256_file(acquisition_summary),
        "independent_audit_path": str(independent_audit.resolve()),
        "independent_audit_sha256": sha256_file(independent_audit),
        "split": split_detail,
        "counts": {
            "acquisition_rows": len(rows),
            "all_singers": len(all_singers),
            "non_target_rows": non_target_count,
            "identity_conflict_excluded_files": len(excluded),
            "target_rows_after_identity_exclusion": len(target_rows),
            "eligible_pairs": len(pairs),
            "eligible_clips": len(used_files),
            "unmatched_target_recordings": len(unmatched),
            "pairs_by_split": dict(sorted(pair_counts_by_split.items())),
            "pairs_by_context": dict(sorted(pair_counts_by_context.items())),
            "pairs_by_split_context": {
                f"{split}:{context}": count
                for (split, context), count in sorted(pair_counts_by_split_context.items())
            },
            "pairs_by_singer": dict(sorted(pair_counts_by_singer.items())),
        },
        "exclusions": excluded,
        "unmatched": unmatched,
        "pairs": pairs,
        "interpretation_limit": (
            "VocalSet paths provide clip-level instructed straight/vibrato labels, not "
            "framewise vibrato presence or continuous rate/extent ground truth."
        ),
        "source_subset_limit": (
            "These 244 files are an already-local VocalSet-Breath-selected subset, not "
            "a complete or random sample of VocalSet."
        ),
    }


PAIR_FIELDS = (
    "pair_id",
    "split",
    "singer",
    "sex",
    "context",
    "content_id",
    "official_label_scope",
    "straight_filename",
    "straight_audio_path",
    "straight_audio_sha256",
    "straight_archive_member",
    "straight_duration_sec",
    "straight_sample_rate_hz",
    "straight_frames",
    "vibrato_filename",
    "vibrato_audio_path",
    "vibrato_audio_sha256",
    "vibrato_archive_member",
    "vibrato_duration_sec",
    "vibrato_sample_rate_hz",
    "vibrato_frames",
    "absolute_duration_difference_sec",
)


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acquisition-summary",
        type=Path,
        default=here / "external_validation/vocalset_breath_original/acquisition_summary_v2.json",
    )
    parser.add_argument(
        "--independent-audit",
        type=Path,
        default=here / "external_validation/vocalset_breath_original/independent_audit.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "external_validation/vocalset_vibrato_protocol_v1",
    )
    args = parser.parse_args()
    result = build_protocol(args.acquisition_summary.resolve(), args.independent_audit.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "pair_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(result["pairs"])
    summary = {key: value for key, value in result.items() if key != "pairs"}
    summary["pair_manifest_path"] = str((args.output_dir / "pair_manifest.csv").resolve())
    summary["pair_manifest_sha256"] = sha256_file(args.output_dir / "pair_manifest.csv")
    summary["code_sha256"] = sha256_file(Path(__file__).resolve())
    feature_code = Path(__file__).with_name("vocalset_vibrato_features.py")
    if not feature_code.is_file():
        raise FileNotFoundError(f"missing proposed feature module: {feature_code}")
    summary["proposed_feature_code_path"] = str(feature_code.resolve())
    summary["proposed_feature_code_sha256"] = sha256_file(feature_code)
    summary["proposed_feature_status"] = "draft_unscored_root_review_required"
    write_json(args.output_dir / "metadata_audit.json", summary)
    print(json.dumps(summary["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
