#!/usr/bin/env python3
"""Validate the frozen catalogue-first manifest and its leakage guards."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "v2" / "frozen_item_manifest.csv"

EXPECTED_ITEMS = {
    "human_magnatagatune": 500,
    "human_maestro_v3": 300,
    "human_musicnet": 330,
    "human_medleydb": 178,
    "human_moisesdb": 239,
    "human_urmp": 44,
    "human_deam": 300,
    "human_gtzan": 100,
    "human_hindustani_raag_hf": 100,
    "ai_audiox_third_party": 500,
    "ai_diffrhythm_pilot": 50,
}

EXPECTED_GROUPS = {**EXPECTED_ITEMS, "ai_diffrhythm_pilot": 10}


def main() -> None:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2641
    assert len({row["item_id"] for row in rows}) == len(rows)
    assert Counter(row["source_id"] for row in rows) == EXPECTED_ITEMS
    assert Counter(row["label"] for row in rows) == {"human": 2091, "ai": 550}
    assert Counter(row["role"] for row in rows) == {
        "development": 1261,
        "locked_catalogue_test": 630,
        "stress_test_only": 200,
        "provisional_development": 500,
        "locked_pilot_only": 50,
    }

    sources = sorted(EXPECTED_ITEMS)
    group_counts = {
        source: len({row["group_id"] for row in rows if row["source_id"] == source})
        for source in sources
    }
    assert group_counts == EXPECTED_GROUPS
    assert all(row["source_revision"] and row["source_locator"] for row in rows)

    locked = {row["source_id"] for row in rows if row["role"] == "locked_catalogue_test"}
    assert locked == {"human_musicnet", "human_deam"}
    assert all(
        row["evaluation_allowed"] == "locked_only"
        for row in rows if row["source_id"] in locked
    )

    mtt = [row for row in rows if row["source_id"] == "human_magnatagatune"]
    assert Counter(row["notes"].rsplit("; ", 1)[-1] for row in mtt) == {
        "vocal stratum": 250,
        "non-vocal stratum": 250,
    }

    hindustani = [row for row in rows if row["source_id"] == "human_hindustani_raag_hf"]
    raags = Counter(row["genre_or_tags"].split("raag=", 1)[1] for row in hindustani)
    assert len(raags) == 50 and set(raags.values()) == {2}

    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    metadata = json.loads((MANIFEST.parent / "manifest_metadata.json").read_text())
    assert metadata["manifest_sha256"] == digest

    print(json.dumps({
        "status": "PASS",
        "sha256": digest,
        "items": len(rows),
        "class_counts": Counter(row["label"] for row in rows),
        "role_counts": Counter(row["role"] for row in rows),
        "source_item_counts": EXPECTED_ITEMS,
        "source_group_counts": group_counts,
        "locked_sources": sorted(locked),
        "hindustani_raag_count": len(raags),
    }, indent=2, default=dict))


if __name__ == "__main__":
    main()
