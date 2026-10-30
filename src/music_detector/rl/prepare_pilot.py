"""Prepare a text-only MusicCaps engineering pilot, never a cleared music test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .data import (MUSICCAPS_DATASET, MUSICCAPS_LICENSE, MUSICCAPS_REVISION,
                   build_prompt_dataset, file_sha256, load_local_records,
                   write_prompt_dataset)

# LFS SHA-256 from the pinned dataset tree, checked against the mirror download.
SOURCE_SHA256 = "8cc109ac507b5e5aed8ea73a5c8f897a9d0793f7397427e000adf4c05f6021c7"
RULES = {
    "explicit_mono": r"\bmono\b",
    "degraded_recording": r"\b(?:low[- ]quality|poor[- ]quality|bad[- ]quality|muffled|distorted|distortion)\b",
    "noise_instruction": r"\b(?:noisy|noise|hiss|hissing|crackle|crackling)\b",
    "speech_instruction": r"\b(?:talking|speaking|speech|spoken)\b",
}


def exclusion_reasons(caption: str) -> list[str]:
    """Conservative lexical screening, not audio quality or vocal annotation.

    Negations can be false positives. Captions are never rewritten and no
    quality claim follows for retained prompts. Report this selection bias.
    """
    return [name for name, pattern in RULES.items()
            if re.search(pattern, caption, flags=re.IGNORECASE)]


def prepare(source: Path, output: Path, *, seed: int = 20261003,
            exclusion_manifests: tuple[Path, ...] = ()) -> dict:
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError("Expected the pinned MusicCaps CSV; source SHA-256 differs")
    rows, files = load_local_records(source, source_dataset=MUSICCAPS_DATASET,
                                    source_revision=MUSICCAPS_REVISION,
                                    license_name=MUSICCAPS_LICENSE,
                                    force_empty_lyrics=True)
    kept, rejected, counts = [], [], dict.fromkeys(RULES, 0)
    for row in rows:
        reasons = exclusion_reasons(row["caption"])
        if reasons:
            rejected.append({"source_id": row["source_id"], "reasons": reasons})
            for reason in reasons:
                counts[reason] += 1
        else:
            kept.append(row)
    provenance = {
        "dataset": MUSICCAPS_DATASET, "revision": MUSICCAPS_REVISION,
        "license": MUSICCAPS_LICENSE, "metadata_only": True,
        "metadata_sha256": SOURCE_SHA256,
        "metadata_url": f"https://hf-mirror.com/datasets/{MUSICCAPS_DATASET}/resolve/{MUSICCAPS_REVISION}/musiccaps-public.csv",
        "download_policy": "hf-mirror direct; no Clash HTTP proxy; no audio/weights",
        "screening": {"version": 1, "rules": RULES, "source_rows": len(rows),
                      "retained_rows": len(kept), "rejected_rows": len(rejected),
                      "reason_counts_nonexclusive": counts,
                      "rejected_source_ids_and_reasons": rejected,
                      "caption_rewriting": False, "quality_labels": False,
                      "selection_bias": "Excludes explicit mono/degradation/noise/speech text; not the full MusicCaps distribution."},
        "readiness": {
            "status": "engineering_pilot_only",
            "historical_manifests_supplied": bool(exclusion_manifests),
            "historical_and_detector_holdout_clearance": "not_established",
            "semantic_overlap_audited": False, "vocal_eligibility_verified": False,
            "external_test_claim_permitted": False,
            "lyrics": "empty source field; no lyrics invented; vocals not guaranteed",
        },
    }
    result = build_prompt_dataset(kept, seed=seed, input_files=files,
                                  exclusion_manifests=exclusion_manifests,
                                  source_provenance=provenance)
    write_prompt_dataset(result, output)
    return {"output": str(output.resolve()), "counts": result.manifest["counts"],
            "source_rows": len(rows), "screened_out": len(rejected),
            "manifest_sha256": result.manifest["manifest_sha256"],
            "status": "engineering_pilot_only", "audio_downloaded": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.source_csv, args.output, seed=args.seed,
                             exclusion_manifests=tuple(args.exclude_manifest)),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
