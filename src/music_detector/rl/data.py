"""Build deterministic, provenance-aware prompt data for online RL runs.

This module intentionally uses only the Python standard library.  It prepares
text prompts and metadata; it never downloads audio or model weights and it
does not start a training job.  A local source is the default and the only
network source supported here is an explicit MusicCaps metadata download.

The public helpers are useful from a trainer or a small project script, while
``python -m music_detector.rl.data`` provides the reproducible CLI.  The
output JSONL files use one stable schema for every training mode, so LoRA and
full-model runs can consume the same prompt and seed values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# MusicCaps is a text/metadata dataset on the official Hugging Face account.
# The commit is deliberately pinned instead of using the mutable ``main``
# branch.  It is the commit returned by the official HF dataset API as of the
# date this builder was authored.
MUSICCAPS_DATASET = "google/MusicCaps"
MUSICCAPS_REVISION = "0a51889b340037bb75a9a0858af2e4ece21f7f89"
MUSICCAPS_LICENSE = "cc-by-sa-4.0"
MUSICCAPS_FILE = "musiccaps-public.csv"
MUSICCAPS_API_URL = "https://huggingface.co/api/datasets/google/MusicCaps"
MUSICCAPS_CARD_URL = (
    "https://huggingface.co/datasets/google/MusicCaps/raw/"
    + MUSICCAPS_REVISION
    + "/README.md"
)
MUSICCAPS_CSV_URL = (
    "https://huggingface.co/datasets/google/MusicCaps/resolve/"
    + MUSICCAPS_REVISION
    + "/musiccaps-public.csv?download=true"
)

DEFAULT_TOTAL = 500
DEFAULT_SPLITS: dict[str, int] = {
    "train": 400,
    "validation": 50,
    "test": 50,
}
SPLIT_ORDER = ("train", "validation", "test")
DURATION_S30 = 30
# Public spelling used by the ACE backend; ``DURATION_S30`` remains as a
# descriptive compatibility constant for callers that used the original
# prompt-budget wording.
DURATION_S = float(DURATION_S30)
SCHEMA_VERSION = 1
CANONICAL_FIELDS = (
    "prompt_id",
    "caption",
    "lyrics",
    "vocal_language",
    "duration_s",
    "seed",
    "source_dataset",
    "source_revision",
    "source_id",
    "license",
    "split",
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class DataPreparationError(ValueError):
    """Raised when prompt data cannot satisfy the reproducibility contract."""


class ExistingOutputError(FileExistsError):
    """Raised when a requested output directory already exists."""


@dataclass(frozen=True)
class PromptGroup:
    """A duplicate/dependency-connected set of source records.

    Groups are formed by connected components over normalized caption and
    source-id keys.  Keeping the full component (rather than just a set of
    duplicate keys) makes the split assignment auditable and prevents a
    transitive chain of dependencies from leaking between splits.
    """

    records: tuple[dict[str, Any], ...]
    key: str
    caption_keys: tuple[str, ...]
    source_keys: tuple[str, ...]

    @property
    def representative(self) -> dict[str, Any]:
        return min(self.records, key=_record_sort_key)


@dataclass(frozen=True)
class BuildResult:
    """In-memory result returned by :func:`build_prompt_dataset`."""

    splits: dict[str, list[dict[str, Any]]]
    manifest: dict[str, Any]

    @property
    def records(self) -> list[dict[str, Any]]:
        """Return all split records in stable split order."""

        return [row for split in SPLIT_ORDER for row in self.splits[split]]


def normalize_text(value: str) -> str:
    """Normalize text for duplicate detection and caption hashes.

    Unicode compatibility normalization, case folding, and whitespace
    collapsing catch formatting-only copies while retaining punctuation and
    words.  The normalized value is not used as a replacement for the
    user-facing caption except where two captions are equivalent under this
    rule.
    """

    if not isinstance(value, str):
        raise TypeError("text must be a string")
    value = unicodedata.normalize("NFKC", value)
    value = " ".join(value.split())
    return value.casefold().strip()


def clean_text(value: Any, *, field: str) -> str:
    """Validate and make a canonical display/provenance string."""

    if not isinstance(value, str):
        raise DataPreparationError(f"{field} must be a string")
    value = unicodedata.normalize("NFKC", value)
    value = " ".join(value.split()).strip()
    if not value:
        raise DataPreparationError(f"{field} must not be empty")
    return value


def caption_hash(caption: str) -> str:
    """Return the SHA-256 of a normalized caption."""

    return hashlib.sha256(normalize_text(caption).encode("utf-8")).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_sort_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Order records independently of their input order."""

    return (
        normalize_text(str(record["source_id"])),
        normalize_text(str(record["caption"])),
        normalize_text(str(record.get("vocal_language", "unknown"))),
        str(record["source_dataset"]),
        str(record["source_revision"]),
        str(record["license"]),
        str(record.get("lyrics", "")),
    )


def _first_nonempty(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            return value
    return None


def canonicalize_record(
    record: Mapping[str, Any],
    *,
    row_number: int | None = None,
    source_dataset: str | None = None,
    source_revision: str | None = None,
    license_name: str | None = None,
    force_empty_lyrics: bool = False,
) -> dict[str, Any]:
    """Validate one input row and return a canonical internal record.

    Provenance is intentionally required.  Optional CLI fallback values are
    explicit user inputs and are never inferred from a file name or caption.
    For MusicCaps, ``force_empty_lyrics`` is always enabled; a non-empty
    supplied lyric is rejected rather than fabricated or silently discarded.
    """

    if not isinstance(record, Mapping):
        where = f" at row {row_number}" if row_number is not None else ""
        raise DataPreparationError(f"record{where} must be a JSON object")

    def required_text(
        field: str,
        aliases: Sequence[str] = (),
        fallback: str | None = None,
    ) -> str:
        value = _first_nonempty(record, (field, *aliases))
        if value is None:
            value = fallback
        if value is None:
            where = f" at row {row_number}" if row_number is not None else ""
            raise DataPreparationError(f"missing required provenance field {field!r}{where}")
        try:
            return clean_text(value, field=field)
        except DataPreparationError as exc:
            where = f" at row {row_number}" if row_number is not None else ""
            raise DataPreparationError(f"{exc}{where}") from exc

    caption_value = _first_nonempty(
        record, ("caption", "acestep_caption", "prompt_common", "text", "prompt")
    )
    if caption_value is None:
        where = f" at row {row_number}" if row_number is not None else ""
        raise DataPreparationError(f"missing required field 'caption'{where}")
    caption = clean_text(caption_value, field="caption")

    dataset_value = _first_nonempty(record, ("source_dataset", "dataset"))
    if dataset_value is None:
        dataset_value = source_dataset
    revision_value = _first_nonempty(record, ("source_revision", "revision", "version", "commit"))
    if revision_value is None:
        revision_value = source_revision
    license_value = _first_nonempty(record, ("license", "licence", "source_license"))
    if license_value is None:
        license_value = license_name
    dataset = required_text("source_dataset", fallback=dataset_value)
    revision = required_text("source_revision", fallback=revision_value)
    license_text = required_text("license", aliases=("licence",), fallback=license_value)

    if normalize_text(dataset) in {"unknown", "unspecified", "none", "null", "n/a", "na"}:
        where = f" at row {row_number}" if row_number is not None else ""
        raise DataPreparationError(f"source_dataset must identify a source{where}")
    if normalize_text(revision) in {"main", "master", "latest", "head"}:
        where = f" at row {row_number}" if row_number is not None else ""
        raise DataPreparationError(
            f"source_revision must be a pinned/explicit revision, not {revision!r}{where}"
        )
    if normalize_text(license_text) in {
        "unknown",
        "unspecified",
        "none",
        "null",
        "n/a",
        "na",
        "no license",
    }:
        where = f" at row {row_number}" if row_number is not None else ""
        raise DataPreparationError(f"license must be explicit, not {license_text!r}{where}")

    source_id_value = _first_nonempty(
        record,
        (
            "source_id",
            "source_track_id",
            "source_song_id",
            "ytid",
            "audio_id",
            "recording_id",
            "id",
            "uid",
            "key",
        ),
    )
    if source_id_value is None:
        where = f" at row {row_number}" if row_number is not None else ""
        raise DataPreparationError(f"missing required provenance field 'source_id'{where}")
    # CSV readers provide strings, while JSON datasets frequently use integer
    # IDs.  Numeric IDs are safe to stringify; arbitrary objects are not.
    if isinstance(source_id_value, bool) or not isinstance(source_id_value, (str, int, float)):
        where = f" at row {row_number}" if row_number is not None else ""
        raise DataPreparationError(f"source_id must be a scalar string/number{where}")
    source_id = clean_text(str(source_id_value), field="source_id")

    lyrics_value = _first_nonempty(record, ("lyrics", "lyrics_30s"))
    if lyrics_value is None:
        lyrics_value = ""
    if not isinstance(lyrics_value, str):
        where = f" at row {row_number}" if row_number is not None else ""
        raise DataPreparationError(f"lyrics must be a string when present{where}")
    lyrics = unicodedata.normalize("NFKC", lyrics_value)
    lyrics = lyrics.replace("\r\n", "\n").replace("\r", "\n")
    lyrics = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in lyrics.split("\n")).strip()
    musiccaps_source = normalize_text(dataset) == normalize_text(MUSICCAPS_DATASET)
    if (force_empty_lyrics or musiccaps_source) and lyrics:
        where = f" at row {row_number}" if row_number is not None else ""
        raise DataPreparationError(
            "MusicCaps metadata has no lyrics; non-empty lyrics would be invented" + where
        )
    if force_empty_lyrics or musiccaps_source:
        lyrics = ""

    language_values = [
        value
        for value in (
            _first_nonempty(record, ("vocal_language",)),
            _first_nonempty(record, ("language",)),
        )
        if value is not None
    ]
    if len(language_values) == 2 and normalize_text(str(language_values[0])) != normalize_text(
        str(language_values[1])
    ):
        where = f" at row {row_number}" if row_number is not None else ""
        raise DataPreparationError(f"language and vocal_language disagree{where}")
    if language_values:
        vocal_language = clean_text(language_values[0], field="vocal_language")
    else:
        # Missing language is deliberately explicit.  It must not silently
        # become English when an input contains non-English lyric prompts.
        vocal_language = "unknown"

    duration_value = record.get("duration_s", record.get("duration_s30"))
    if duration_value not in (None, "", 30, 30.0, "30", "30.0"):
        where = f" at row {row_number}" if row_number is not None else ""
        raise DataPreparationError(f"duration_s must be 30 seconds{where}")

    return {
        "caption": caption,
        "lyrics": lyrics,
        "vocal_language": vocal_language,
        "duration_s": DURATION_S30,
        "source_dataset": dataset,
        "source_revision": revision,
        "source_id": source_id,
        "license": license_text,
    }


def _json_rows(payload: Any, *, path: Path) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = None
        for key in ("records", "data", "items", "rows", "examples", "prompts"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            # A single canonical record is a useful and unambiguous JSON form.
            rows = [payload]
    else:
        raise DataPreparationError(f"{path}: top-level JSON must be an object or array")
    if not all(isinstance(row, Mapping) for row in rows):
        raise DataPreparationError(f"{path}: every JSON row must be an object")
    return list(rows)


def load_local_records(
    paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    *,
    source_dataset: str | None = None,
    source_revision: str | None = None,
    license_name: str | None = None,
    force_empty_lyrics: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Read local JSON, JSONL, NDJSON, or CSV files.

    Returns canonical records and input-file provenance (including SHA-256
    hashes).  Files are read in the order supplied, but later grouping and
    selection is independent of this order.
    """

    if isinstance(paths, (str, os.PathLike)):
        path_list = [Path(paths)]
    else:
        path_list = [Path(path) for path in paths]
    if not path_list:
        raise DataPreparationError("at least one local input file is required")

    records: list[dict[str, Any]] = []
    input_files: list[dict[str, str]] = []
    for path in path_list:
        if not path.is_file():
            raise DataPreparationError(f"input file does not exist: {path}")
        suffix = path.suffix.casefold()
        input_files.append({"path": str(path), "sha256": file_sha256(path)})
        try:
            if suffix in {".jsonl", ".ndjson"}:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        try:
                            value = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise DataPreparationError(
                                f"{path}: invalid JSONL at line {line_number}: {exc.msg}"
                            ) from exc
                        if not isinstance(value, Mapping):
                            raise DataPreparationError(
                                f"{path}: JSONL line {line_number} must be an object"
                            )
                        records.append(
                            canonicalize_record(
                                value,
                                row_number=line_number,
                                source_dataset=source_dataset,
                                source_revision=source_revision,
                                license_name=license_name,
                                force_empty_lyrics=force_empty_lyrics,
                            )
                        )
            elif suffix == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise DataPreparationError(f"{path}: invalid JSON: {exc.msg}") from exc
                for row_number, value in enumerate(_json_rows(payload, path=path), start=1):
                    records.append(
                        canonicalize_record(
                            value,
                            row_number=row_number,
                            source_dataset=source_dataset,
                            source_revision=source_revision,
                            license_name=license_name,
                            force_empty_lyrics=force_empty_lyrics,
                        )
                    )
            elif suffix == ".csv":
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames:
                        raise DataPreparationError(f"{path}: CSV is missing a header")
                    for row_number, value in enumerate(reader, start=2):
                        if None in value:
                            raise DataPreparationError(
                                f"{path}: CSV row {row_number} has more fields than its header"
                            )
                        records.append(
                            canonicalize_record(
                                value,
                                row_number=row_number,
                                source_dataset=source_dataset,
                                source_revision=source_revision,
                                license_name=license_name,
                                force_empty_lyrics=force_empty_lyrics,
                            )
                        )
            else:
                raise DataPreparationError(
                    f"unsupported input format {path.suffix!r}; use .json, .jsonl, or .csv"
                )
        except UnicodeDecodeError as exc:
            raise DataPreparationError(f"{path}: input must be UTF-8 text") from exc
    if not records:
        raise DataPreparationError("local input contains no records")
    return records, input_files


# A shorter alias is convenient for callers and preserves a natural API name.
load_records = load_local_records


def _read_json_url(url: str, *, timeout: float = 30.0, opener: Any = None) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "music-detector-rl-data/1"})
    open_fn = opener or urllib.request.urlopen
    try:
        response = open_fn(request, timeout=timeout)
        with response:
            raw = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DataPreparationError(f"failed to read official Hugging Face metadata: {url}") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataPreparationError(f"official Hugging Face API returned invalid JSON: {url}") from exc


def _read_url_bytes(url: str, *, timeout: float = 60.0, opener: Any = None) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "music-detector-rl-data/1"})
    open_fn = opener or urllib.request.urlopen
    try:
        response = open_fn(request, timeout=timeout)
        with response:
            return response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DataPreparationError(f"failed to read Hugging Face metadata: {url}") from exc


def verify_musiccaps_source(
    revision: str = MUSICCAPS_REVISION,
    *,
    timeout: float = 30.0,
    opener: Any = None,
) -> dict[str, Any]:
    """Verify the pinned official MusicCaps card and API metadata.

    The API is checked for the repository identity, exact commit, and declared
    license.  The dataset card at that commit is also checked for the same
    license.  A mutable branch name is never accepted as a revision.
    """

    if not _REVISION_RE.fullmatch(revision):
        raise DataPreparationError(
            "MusicCaps revision must be a full 40-character commit SHA; mutable branches are not allowed"
        )
    api = _read_json_url(MUSICCAPS_API_URL, timeout=timeout, opener=opener)
    if not isinstance(api, Mapping) or api.get("id") != MUSICCAPS_DATASET:
        raise DataPreparationError("official Hugging Face API identity is not google/MusicCaps")
    actual_revision = api.get("sha")
    if actual_revision != revision:
        raise DataPreparationError(
            f"pinned MusicCaps revision {revision} is not the official API revision {actual_revision}"
        )
    licenses: list[str] = []
    card_data = api.get("cardData")
    if isinstance(card_data, Mapping):
        value = card_data.get("license")
        if isinstance(value, str):
            licenses.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            licenses.extend(str(item) for item in value)
    tags = api.get("tags")
    if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
        licenses.extend(
            str(item).split(":", 1)[1]
            for item in tags
            if isinstance(item, str) and item.startswith("license:")
        )
    if MUSICCAPS_LICENSE not in {item.casefold() for item in licenses}:
        raise DataPreparationError("official MusicCaps API does not declare cc-by-sa-4.0")

    # Fetch only the text card, never an audio path.  This is intentionally a
    # second verification source; a metadata API response alone is not the
    # license text/card users should audit.
    card_url = (
        "https://huggingface.co/datasets/google/MusicCaps/raw/"
        + revision
        + "/README.md"
    )
    card = _read_url_bytes(card_url, timeout=timeout, opener=opener).decode("utf-8", "replace")
    if MUSICCAPS_LICENSE not in card.casefold():
        raise DataPreparationError("pinned MusicCaps dataset card does not declare cc-by-sa-4.0")
    return {
        "dataset": MUSICCAPS_DATASET,
        "revision": revision,
        "license": MUSICCAPS_LICENSE,
        "api_url": MUSICCAPS_API_URL,
        "card_url": card_url,
        "metadata_only": True,
    }


def fetch_musiccaps_metadata(
    *,
    revision: str = MUSICCAPS_REVISION,
    timeout: float = 60.0,
    opener: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Explicitly download the MusicCaps CSV metadata, without audio.

    ``verify_musiccaps_source`` is called first.  The returned provenance
    records the pinned source and license; no rights beyond the card's
    declared license are inferred.
    """

    verification = verify_musiccaps_source(revision, timeout=timeout, opener=opener)
    csv_url = (
        "https://huggingface.co/datasets/google/MusicCaps/resolve/"
        + revision
        + "/"
        + MUSICCAPS_FILE
        + "?download=true"
    )
    raw = _read_url_bytes(csv_url, timeout=timeout, opener=opener)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataPreparationError("MusicCaps CSV is not valid UTF-8") from exc
    reader = csv.DictReader(text.splitlines())
    if not reader.fieldnames:
        raise DataPreparationError("MusicCaps CSV is missing a header")
    records: list[dict[str, Any]] = []
    for row_number, row in enumerate(reader, start=2):
        if None in row:
            raise DataPreparationError(f"MusicCaps CSV row {row_number} is malformed")
        records.append(
            canonicalize_record(
                row,
                row_number=row_number,
                source_dataset=MUSICCAPS_DATASET,
                source_revision=revision,
                license_name=MUSICCAPS_LICENSE,
                force_empty_lyrics=True,
            )
        )
    if not records:
        raise DataPreparationError("MusicCaps CSV contains no records")
    provenance = {
        **verification,
        "metadata_url": csv_url,
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "audio_downloaded": False,
        "row_count": len(records),
    }
    return records, provenance


def _component_groups(records: Sequence[dict[str, Any]]) -> list[PromptGroup]:
    """Build caption/source connected components using union-find."""

    n = len(records)
    parent = list(range(n))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    owners: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        keys = (
            ("caption", normalize_text(record["caption"])),
            ("source", normalize_text(record["source_id"])),
        )
        for key in keys:
            previous = owners.get(key)
            if previous is None:
                owners[key] = index
            else:
                union(index, previous)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[find(index)].append(record)

    groups: list[PromptGroup] = []
    for rows in grouped.values():
        ordered = tuple(sorted(rows, key=_record_sort_key))
        caption_keys = tuple(sorted({normalize_text(row["caption"]) for row in ordered}))
        source_keys = tuple(sorted({normalize_text(row["source_id"]) for row in ordered}))
        key_material = "\x00".join(
            [*caption_keys, "\x01", *source_keys, "\x02", str(ordered[0]["source_dataset"])]
        )
        key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
        groups.append(
            PromptGroup(
                records=ordered,
                key=key,
                caption_keys=caption_keys,
                source_keys=source_keys,
            )
        )
    return sorted(groups, key=lambda group: group.key)


def group_records(records: Sequence[dict[str, Any]]) -> list[PromptGroup]:
    """Public wrapper for duplicate/dependency grouping."""

    return _component_groups(records)


def _load_exclusions(paths: Sequence[str | os.PathLike[str]]) -> tuple[set[str], set[str], list[dict[str, str]]]:
    """Read exclusion manifests and return source IDs/caption hashes."""

    source_ids: set[str] = set()
    caption_hashes: set[str] = set()
    provenance: list[dict[str, str]] = []

    def add_source(value: Any) -> None:
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            try:
                normalized = normalize_text(str(value))
                if normalized:
                    source_ids.add(normalized)
            except TypeError:
                pass

    def add_caption_hash(value: Any) -> None:
        if isinstance(value, str) and _SHA256_RE.fullmatch(value.strip()):
            caption_hashes.add(value.strip().casefold())

    def walk(value: Any, *, key_hint: str = "") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                lowered = str(key).casefold()
                if lowered in {
                    "source_id",
                    "source_ids",
                    "sourceid",
                    "sourceids",
                    "source_song_id",
                    "source_song_ids",
                    "source_songid",
                    "source_songids",
                    "source_track_id",
                    "source_track_ids",
                    "source_trackid",
                    "source_trackids",
                    "excluded_source_id",
                    "excluded_source_ids",
                    "selected_source_id",
                    "selected_source_ids",
                    "heldout_source_id",
                    "heldout_source_ids",
                }:
                    if isinstance(item, list):
                        for entry in item:
                            add_source(entry)
                    else:
                        add_source(item)
                elif lowered in {
                    "caption_hash",
                    "caption_hashes",
                    "caption_sha256",
                    "caption_sha256s",
                    "selected_caption_hash",
                    "selected_caption_hashes",
                    "captionhash",
                    "captionhashes",
                }:
                    if isinstance(item, list):
                        for entry in item:
                            add_caption_hash(entry)
                    else:
                        add_caption_hash(item)
                elif lowered in {"caption", "acestep_caption", "prompt_common"}:
                    if isinstance(item, str):
                        add_caption_hash(caption_hash(item))
                else:
                    walk(item, key_hint=lowered)
        elif isinstance(value, list):
            for item in value:
                walk(item, key_hint=key_hint)
        elif key_hint in {
            "source_id",
            "sourceid",
            "source_song_id",
            "source_songid",
            "source_track_id",
            "source_trackid",
        }:
            add_source(value)
        elif key_hint in {"caption_hash", "captionhash"}:
            add_caption_hash(value)

    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise DataPreparationError(f"exclusion manifest does not exist: {path}")
        provenance.append({"path": str(path), "sha256": file_sha256(path)})
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DataPreparationError(f"exclusion manifest must be UTF-8: {path}") from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # A plain line-oriented manifest is useful for held-out IDs and
            # remains explicit: 64-hex lines are caption hashes, others IDs.
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                raise DataPreparationError(f"exclusion manifest is empty: {path}")
            for line in lines:
                if _SHA256_RE.fullmatch(line):
                    add_caption_hash(line)
                else:
                    add_source(line)
            continue
        if isinstance(payload, list):
            # A JSON list is accepted as a concise source-ID/hash manifest;
            # unlike a structured manifest it has no other fields to inspect.
            for item in payload:
                if isinstance(item, str) and _SHA256_RE.fullmatch(item.strip()):
                    add_caption_hash(item)
                else:
                    add_source(item)
            continue
        walk(payload)
    return source_ids, caption_hashes, provenance


def _prompt_id(record: Mapping[str, Any]) -> str:
    material = "\x00".join(
        (
            str(record["source_dataset"]),
            str(record["source_revision"]),
            normalize_text(str(record["source_id"])),
            normalize_text(str(record["caption"])),
        )
    )
    return "p-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _prompt_seed(record: Mapping[str, Any], seed: int) -> int:
    material = f"{seed}\x00{_prompt_id(record)}".encode("utf-8")
    # Keep the value in the commonly supported signed 32-bit range while
    # deriving it from identity rather than ordinal position.  This means a
    # prompt has the same seed when selected for LoRA or full-model training.
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2_147_483_647


def _output_record(record: Mapping[str, Any], *, split: str, seed: int) -> dict[str, Any]:
    return {
        "prompt_id": _prompt_id(record),
        "caption": str(record["caption"]),
        "lyrics": str(record.get("lyrics", "")),
        "vocal_language": str(record.get("vocal_language", "unknown")),
        "duration_s": DURATION_S30,
        "seed": _prompt_seed(record, seed),
        "source_dataset": str(record["source_dataset"]),
        "source_revision": str(record["source_revision"]),
        "source_id": str(record["source_id"]),
        "license": str(record["license"]),
        "split": split,
    }


def _validate_counts(
    total: int,
    split_counts: Mapping[str, int] | None,
) -> dict[str, int]:
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise DataPreparationError("total must be a positive integer")
    counts = dict(DEFAULT_SPLITS if split_counts is None else split_counts)
    if set(counts) != set(SPLIT_ORDER):
        raise DataPreparationError("split_counts must contain exactly train, validation, and test")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values()):
        raise DataPreparationError("split counts must be non-negative integers")
    if sum(counts.values()) != total:
        raise DataPreparationError(
            f"split counts sum to {sum(counts.values())}, but total is {total}"
        )
    return {name: counts[name] for name in SPLIT_ORDER}


def build_prompt_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    total: int = DEFAULT_TOTAL,
    split_counts: Mapping[str, int] | None = None,
    seed: int = 0,
    excluded_source_ids: Iterable[str | int | float] = (),
    excluded_caption_hashes: Iterable[str] = (),
    exclusion_manifests: Sequence[str | os.PathLike[str]] = (),
    input_files: Sequence[Mapping[str, str]] = (),
    source_provenance: Mapping[str, Any] | None = None,
) -> BuildResult:
    """Deduplicate, split, and canonicalize prompt records deterministically.

    The caller may provide raw rows or rows already returned by
    :func:`canonicalize_record`; validation is applied either way.  Input row
    order never affects group selection, split assignment, prompt IDs, or
    per-prompt seeds.
    """

    counts = _validate_counts(total, split_counts)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise DataPreparationError("seed must be an integer")

    canonical: list[dict[str, Any]] = []
    for row_number, row in enumerate(records, start=1):
        canonical.append(canonicalize_record(row, row_number=row_number))
    if not canonical:
        raise DataPreparationError("input contains no records")

    manifest_sources, manifest_hashes, exclusion_files = _load_exclusions(exclusion_manifests)
    source_exclusions = set(manifest_sources)
    for value in excluded_source_ids:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise DataPreparationError(f"invalid source ID exclusion: {value!r}")
        normalized = normalize_text(str(value))
        if not normalized:
            raise DataPreparationError("source ID exclusions must not be empty")
        source_exclusions.add(normalized)
    caption_exclusions = set(manifest_hashes)
    for value in excluded_caption_hashes:
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value.strip()):
            raise DataPreparationError(f"invalid caption hash exclusion: {value!r}")
        caption_exclusions.add(value.strip().casefold())

    groups = _component_groups(canonical)
    eligible: list[PromptGroup] = []
    excluded_group_count = 0
    for group in groups:
        excluded = any(key in source_exclusions for key in group.source_keys)
        excluded = excluded or any(
            (
                caption_hash(row["caption"]).casefold() in caption_exclusions
                or hashlib.sha256(str(row["caption"]).encode("utf-8")).hexdigest().casefold()
                in caption_exclusions
            )
            for row in group.records
        )
        if excluded:
            excluded_group_count += 1
        else:
            eligible.append(group)

    if len(eligible) < total:
        raise DataPreparationError(
            "insufficient unique prompts after duplicate grouping/exclusions: "
            f"need {total}, have {len(eligible)}"
        )

    # Hash ranking is a deterministic randomization independent of source file
    # order and Python's process-randomized hash().
    ranked = sorted(
        eligible,
        key=lambda group: hashlib.sha256(f"{seed}\x00{group.key}".encode("utf-8")).hexdigest(),
    )
    selected = ranked[:total]
    split_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_ORDER}
    cursor = 0
    for split in SPLIT_ORDER:
        for group in selected[cursor : cursor + counts[split]]:
            split_rows[split].append(_output_record(group.representative, split=split, seed=seed))
        cursor += counts[split]

    # Sort rows by stable prompt identity within each file.  The ranked order
    # remains the source-selection order recorded in the manifest, but stable
    # prompt order makes diffs and file hashes easy to reproduce.
    for rows in split_rows.values():
        rows.sort(key=lambda row: row["prompt_id"])

    all_ids = [row["prompt_id"] for rows in split_rows.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise DataPreparationError("prompt ID collision detected")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "builder": "music_detector.rl.data",
        "canonical_fields": list(CANONICAL_FIELDS),
        "duration_s": DURATION_S30,
        "seed": seed,
        "total": total,
        "target_counts": dict(counts),
        "counts": {name: len(split_rows[name]) for name in SPLIT_ORDER},
        "split_order": list(SPLIT_ORDER),
        "selection": {
            "raw_records": len(canonical),
            "unique_groups_before_exclusions": len(groups),
            "excluded_groups": excluded_group_count,
            "eligible_groups": len(eligible),
            "selected_groups": len(selected),
            "duplicate_records_removed": len(canonical) - len(groups),
            "selection_rule": "sha256(seed + dependency_group_key), then target counts",
            "dependency_keys": ["normalized_caption", "normalized_source_id"],
            "selected_source_ids": sorted({
                row["source_id"] for group in selected for row in group.records
            }),
            "selected_caption_hashes": sorted({
                caption_hash(row["caption"]) for group in selected for row in group.records
            }),
            "selected_prompt_ids": sorted({
                _prompt_id(group.representative) for group in selected
            }),
        },
        "exclusions": {
            "source_ids": sorted(source_exclusions),
            "caption_hashes": sorted(caption_exclusions),
            "manifests": exclusion_files,
        },
        "source_provenance": dict(source_provenance or {}),
        "input_files": [dict(item) for item in input_files],
        "license_policy": (
            "licenses are copied from explicit source provenance; no rights or permissions are inferred"
        ),
        "audio_policy": "text/metadata only; no audio or model weights are downloaded",
        "splits": {
            name: {"file": f"{name}.jsonl", "count": len(split_rows[name])}
            for name in SPLIT_ORDER
        },
    }
    return BuildResult(splits=split_rows, manifest=manifest)


# Alias for callers that prefer the shorter name.
build_dataset = build_prompt_dataset
prepare_prompt_data = build_prompt_dataset
build_prompt_data = build_prompt_dataset


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=False).encode("utf-8")
            + b"\n"
        )
        for row in rows
    )


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_prompt_dataset(
    result: BuildResult,
    output_dir: str | os.PathLike[str],
    *,
    refuse_existing: bool = True,
) -> Path:
    """Atomically write JSONL files and a manifest, refusing overwrite.

    The destination directory must not already exist.  All files are first
    written to a private sibling staging directory, fsynced, and then the
    completed directory is installed with one rename.  A failed build or
    serialization therefore leaves no partial destination directory.
    """

    destination = Path(output_dir)
    if destination.exists() and refuse_existing:
        raise ExistingOutputError(f"refusing to overwrite existing output: {destination}")
    if destination.exists():
        # Keep the safety default even if an accidental false flag is passed;
        # this function is a reproducibility artifact writer, not a clobbering
        # utility.
        raise ExistingOutputError(f"refusing to overwrite existing output: {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)

    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(parent)))
    try:
        manifest = json.loads(json.dumps(result.manifest))
        files: dict[str, dict[str, Any]] = {}
        for split in SPLIT_ORDER:
            name = f"{split}.jsonl"
            payload = _jsonl_bytes(result.splits[split])
            path = stage / name
            with path.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            files[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}

        manifest["files"] = files
        for split in SPLIT_ORDER:
            manifest["splits"][split]["sha256"] = files[f"{split}.jsonl"]["sha256"]
        manifest["manifest_format"] = "JSON manifest + SHA-256 sidecar"
        manifest_path = stage / "manifest.json"
        manifest_payload = _manifest_bytes(manifest)
        with manifest_path.open("wb") as handle:
            handle.write(manifest_payload)
            handle.flush()
            os.fsync(handle.fileno())
        manifest_hash = hashlib.sha256(manifest_payload).hexdigest()
        sidecar = stage / "manifest.json.sha256"
        sidecar_payload = (manifest_hash + "  manifest.json\n").encode("ascii")
        with sidecar.open("wb") as handle:
            handle.write(sidecar_payload)
            handle.flush()
            os.fsync(handle.fileno())

        # Add manifest hash to the in-memory object for callers without
        # mutating the on-disk manifest (which would invalidate the hash).
        result.manifest["files"] = files
        result.manifest["manifest_sha256"] = manifest_hash
        result.manifest["manifest_sidecar"] = "manifest.json.sha256"

        if destination.exists():
            raise ExistingOutputError(f"refusing to overwrite existing output: {destination}")
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return destination


def write_bundle(*args: Any, **kwargs: Any) -> Path:
    """Backward-compatible alias for :func:`write_prompt_dataset`."""

    return write_prompt_dataset(*args, **kwargs)


write_dataset = write_prompt_dataset


def _build_from_cli(args: argparse.Namespace) -> tuple[BuildResult, Path]:
    if args.musiccaps_hf:
        if args.input:
            raise DataPreparationError("choose local --input or --musiccaps-hf, not both")
        records, provenance = fetch_musiccaps_metadata(revision=args.hf_revision)
        input_files: list[dict[str, str]] = []
    else:
        if not args.input:
            raise DataPreparationError(
                "no input selected; local files are the default, and --musiccaps-hf is explicit opt-in"
            )
        records, input_files = load_local_records(
            args.input,
            source_dataset=args.source_dataset,
            source_revision=args.source_revision,
            license_name=args.license_name,
            force_empty_lyrics=(args.source_dataset or "").casefold() == MUSICCAPS_DATASET.casefold(),
        )
        provenance = {"kind": "local", "metadata_only": True}

    split_counts = {
        "train": args.train_count,
        "validation": args.validation_count,
        "test": args.test_count,
    }
    result = build_prompt_dataset(
        records,
        total=args.total,
        split_counts=split_counts,
        seed=args.seed,
        excluded_source_ids=args.exclude_source_id,
        excluded_caption_hashes=args.exclude_caption_hash,
        exclusion_manifests=args.exclude_manifest,
        input_files=input_files,
        source_provenance=provenance,
    )
    destination = write_prompt_dataset(result, args.output_dir)
    return result, destination


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m music_detector.rl.data",
        description=(
            "Build deterministic ACE-Step prompt JSONL data from local metadata. "
            "HF MusicCaps metadata download is explicit and text-only."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input",
        action="append",
        type=Path,
        metavar="PATH",
        help="local .json, .jsonl/.ndjson, or .csv input (repeat for multiple files)",
    )
    source.add_argument(
        "--musiccaps-hf",
        "--hf-musiccaps",
        "--from-musiccaps-hf",
        dest="musiccaps_hf",
        action="store_true",
        help="explicitly fetch official MusicCaps CSV text metadata; never downloads audio",
    )
    parser.add_argument(
        "--hf-revision",
        default=MUSICCAPS_REVISION,
        help="full pinned MusicCaps commit SHA (default: %(default)s)",
    )
    parser.add_argument("--source-dataset", help="explicit provenance fallback for local rows")
    parser.add_argument("--source-revision", help="explicit provenance fallback for local rows")
    parser.add_argument("--license", dest="license_name", help="explicit source license fallback for local rows")
    parser.add_argument(
        "--output-dir",
        "--output",
        "--out-dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="new output directory (must not exist)",
    )
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--train-count", "--train", type=int, default=DEFAULT_SPLITS["train"])
    parser.add_argument(
        "--validation-count", "--validation", type=int, default=DEFAULT_SPLITS["validation"]
    )
    parser.add_argument("--test-count", "--test", type=int, default=DEFAULT_SPLITS["test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        type=Path,
        default=[],
        help="historical/held-out JSON or line manifest (repeatable)",
    )
    parser.add_argument("--exclude-source-id", action="append", default=[], help="held-out source ID (repeatable)")
    parser.add_argument(
        "--exclude-caption-hash",
        action="append",
        default=[],
        help="SHA-256 of normalized held-out caption (repeatable)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result, destination = _build_from_cli(args)
    except (DataPreparationError, ExistingOutputError, OSError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "output_dir": str(destination),
                "counts": result.manifest["counts"],
                "manifest": str(destination / "manifest.json"),
                "metadata_only": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests
    raise SystemExit(main())
