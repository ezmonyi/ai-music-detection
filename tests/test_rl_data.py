"""Contract tests for the deterministic RL prompt-data builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from music_detector.rl.data import (
    CANONICAL_FIELDS,
    DataPreparationError,
    ExistingOutputError,
    build_prompt_dataset,
    caption_hash,
    canonicalize_record,
    load_local_records,
    write_prompt_dataset,
)


def _rows(count: int = 20) -> list[dict[str, str]]:
    return [
        {
            "caption": f"A distinct music prompt {index}",
            "source_dataset": "fixture/prompts",
            "source_revision": "fixture-v1",
            "source_id": f"fixture-{index:04d}",
            "license": "cc-by-4.0",
        }
        for index in range(count)
    ]


def _snapshot(result):
    return {
        split: [json.dumps(row, sort_keys=True) for row in result.splits[split]]
        for split in ("train", "validation", "test")
    }


def test_shuffle_does_not_change_rows_or_seeds():
    rows = _rows()
    first = build_prompt_dataset(
        rows,
        total=10,
        split_counts={"train": 6, "validation": 2, "test": 2},
        seed=123,
    )
    shuffled = build_prompt_dataset(
        list(reversed(rows)),
        total=10,
        split_counts={"train": 6, "validation": 2, "test": 2},
        seed=123,
    )
    assert _snapshot(first) == _snapshot(shuffled)
    assert all(tuple(row) == CANONICAL_FIELDS for row in first.records)
    assert all(row["duration_s"] == 30 for row in first.records)


def test_caption_and_source_dependencies_are_transitively_grouped():
    rows = _rows(4)
    rows[1]["caption"] = rows[0]["caption"]  # same normalized text, new source ID
    rows[2]["source_id"] = rows[0]["source_id"]  # same source, different caption
    rows[2]["caption"] = "A third description"
    result = build_prompt_dataset(
        rows,
        total=2,
        split_counts={"train": 1, "validation": 1, "test": 0},
        seed=1,
    )
    # The dependency component containing rows 0/1/2 contributes one prompt;
    # row 3 contributes the other.  No duplicate source or caption can leak.
    ids = {row["source_id"] for row in result.records}
    assert len(result.records) == 2
    assert len(ids) == 2


def test_exclusion_manifest_removes_entire_dependency_group(tmp_path: Path):
    rows = _rows(8)
    held_out = {
        "source_ids": [rows[0]["source_id"]],
        "caption_hashes": [caption_hash(rows[1]["caption"])],
    }
    manifest = tmp_path / "heldout.json"
    manifest.write_text(json.dumps(held_out), encoding="utf-8")
    result = build_prompt_dataset(
        rows,
        total=4,
        split_counts={"train": 2, "validation": 1, "test": 1},
        seed=8,
        exclusion_manifests=[manifest],
    )
    assert all(row["source_id"] not in {rows[0]["source_id"], rows[1]["source_id"]} for row in result.records)
    assert result.manifest["exclusions"]["manifests"][0]["sha256"]


def test_missing_provenance_and_invalid_counts_are_rejected():
    incomplete = {"caption": "A caption", "source_id": "x"}
    with pytest.raises(DataPreparationError, match="source_dataset"):
        build_prompt_dataset([incomplete], total=1, split_counts={"train": 1, "validation": 0, "test": 0})

    with pytest.raises(DataPreparationError, match="sum"):
        build_prompt_dataset(
            _rows(4),
            total=3,
            split_counts={"train": 2, "validation": 1, "test": 1},
        )

    with pytest.raises(DataPreparationError, match="insufficient"):
        build_prompt_dataset(
            _rows(2),
            total=3,
            split_counts={"train": 1, "validation": 1, "test": 1},
        )


def test_local_json_jsonl_and_csv_inputs_require_explicit_provenance(tmp_path: Path):
    rows = _rows(3)
    json_path = tmp_path / "rows.json"
    json_path.write_text(json.dumps(rows), encoding="utf-8")
    loaded, files = load_local_records(json_path)
    assert len(loaded) == 3
    assert files[0]["sha256"]

    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        "caption,source_dataset,source_revision,source_id,license\n"
        "A CSV prompt,fixture/prompts,fixture-v1,csv-1,cc-by-4.0\n",
        encoding="utf-8",
    )
    csv_rows, _ = load_local_records(csv_path)
    assert csv_rows[0]["source_id"] == "csv-1"

    jsonl_path = tmp_path / "rows.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    jsonl_rows, _ = load_local_records(jsonl_path)
    assert len(jsonl_rows) == 3

    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps([{"caption": "not attributable"}]), encoding="utf-8")
    with pytest.raises(DataPreparationError, match="source_dataset"):
        load_local_records(missing)


def test_musiccaps_never_accepts_invented_lyrics():
    row = {
        "caption": "A descriptive caption",
        "ytid": "video-1",
        "lyrics": "invented lyrics",
    }
    with pytest.raises(DataPreparationError, match="no lyrics"):
        canonicalize_record(
            row,
            source_dataset="google/MusicCaps",
            source_revision="0a51889b340037bb75a9a0858af2e4ece21f7f89",
            license_name="cc-by-sa-4.0",
            force_empty_lyrics=True,
        )


def test_legacy_duration_s30_input_is_canonicalized_to_duration_s():
    row = canonicalize_record(
        {
            "caption": "A caption",
            "source_dataset": "fixture/prompts",
            "source_revision": "fixture-v1",
            "source_id": "legacy-1",
            "license": "CC0-1.0",
            "duration_s30": 30,
        }
    )
    assert row["duration_s"] == 30
    assert "duration_s30" not in row


def test_language_and_lyric_linebreaks_are_preserved_without_english_inference():
    row = canonicalize_record(
        {
            "caption": "中文歌词提示",
            "lyrics": "第一行\n\n第二行\r\n第三行",
            "language": "zh",
            "source_dataset": "synthetic/prompts",
            "source_revision": "fixture-v1",
            "source_id": "cn-1",
            "license": "CC0-1.0",
        }
    )
    assert row["vocal_language"] == "zh"
    assert row["lyrics"] == "第一行\n\n第二行\n第三行"

    missing_language = canonicalize_record(
        {
            "caption": "Lyrics with unknown language",
            "lyrics": "line one\nline two",
            "source_dataset": "synthetic/prompts",
            "source_revision": "fixture-v1",
            "source_id": "unknown-1",
            "license": "CC0-1.0",
        }
    )
    assert missing_language["vocal_language"] == "unknown"


def test_historical_source_song_and_track_ids_are_exclusion_dependencies(tmp_path: Path):
    rows = _rows(5)
    rows[0]["source_id"] = "song-a"
    rows[1]["source_id"] = "track-b"
    exclusions = tmp_path / "historical.json"
    exclusions.write_text(
        json.dumps({"source_song_id": "song-a", "source_track_id": "track-b"}),
        encoding="utf-8",
    )
    with pytest.raises(DataPreparationError, match="insufficient"):
        build_prompt_dataset(
            rows,
            total=4,
            split_counts={"train": 2, "validation": 1, "test": 1},
            exclusion_manifests=[exclusions],
        )


def test_atomic_write_hashes_and_refuses_overwrite(tmp_path: Path):
    result = build_prompt_dataset(
        _rows(8),
        total=4,
        split_counts={"train": 2, "validation": 1, "test": 1},
        seed=4,
    )
    output = tmp_path / "prompt_data"
    write_prompt_dataset(result, output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"train": 2, "validation": 1, "test": 1}
    for split in ("train", "validation", "test"):
        assert manifest["files"][f"{split}.jsonl"]["sha256"]
    assert (output / "manifest.json.sha256").is_file()
    with pytest.raises(ExistingOutputError):
        write_prompt_dataset(result, output)


def test_failed_build_does_not_create_output_directory(tmp_path: Path):
    output = tmp_path / "should-not-exist"
    with pytest.raises(DataPreparationError):
        result = build_prompt_dataset(
            _rows(2),
            total=5,
            split_counts={"train": 3, "validation": 1, "test": 1},
        )
        write_prompt_dataset(result, output)
    assert not output.exists()
