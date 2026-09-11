#!/usr/bin/env python3
"""Build a source-aware, deterministic expansion manifest without reading audio."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "source_metadata"
OUT = ROOT / "manifests" / "v2"
SEED = "ai-human-source-diversity-v2-catalogue-first-20260904"

REVISIONS = {
    "magnatagatune": "042e24c2c26a71168dd35edc3bc0cb47b2f3c3cd",
    "maestro": "67e586b1dcf5925c94a6729fcd5eb7f23d504fac",
    "musicnet": "15078b3037afe83bdcacb9cea14232e4debffd2b",
    "deam": "f83e169f346b17c44e5c213e3d87c88772f8b506",
    "gtzan": "d2146561ecc7df707d9e6b8318885fe6a39668a2",
    "medleydb": "99e4ae2e79451a8189c2d8d2bc231c2ee74091de",
    "moisesdb": "353a59cfd90514ea71e35480b5ab8ffb57d14598",
    "urmp": "3f2da70e1f46a8806d15b02e6586cfc7ae5bf782",
    "hindustani_raag": "326caef0bc01da44ad46e4d9c65a5146da6bcc5b",
    "audiox": "703918956d5ed91a71e76a3b280ed9ee1a7baa81",
    "diffrhythm": "2be7bd4fb18d317111c0fece3c077c3587bf6f23",
}

VOCAL_TAGS = {
    "singer", "duet", "female singing", "female opera", "male vocal",
    "vocals", "chorus", "female voice", "male voice", "girl", "voice",
    "male singer", "man singing", "female vocal", "male vocals", "opera",
    "soprano", "vocal", "woman", "woman singing", "singing",
    "female vocals", "voices", "choir", "female singer", "rap",
    "male opera", "operatic", "chanting", "chant", "men", "man", "women",
}

FIELDS = [
    "item_id", "label", "source_id", "source_type", "role",
    "source_revision", "source_path", "source_locator", "container_member",
    "original_duration_s", "window_start_s", "window_duration_s",
    "sampling_target_hz", "channels_target", "codec_target", "group_id",
    "artist_or_creator", "title", "genre_or_tags", "provenance_grade",
    "license", "evaluation_allowed", "notes",
]


def stable_hex(value: str) -> str:
    return hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest()


def stable_fraction(value: str) -> float:
    return int(stable_hex(value)[:16], 16) / float(16**16 - 1)


def window_start(value: str, duration: float) -> str:
    if duration <= 10.0:
        return "0.000"
    available = duration - 10.0
    return f"{available * (0.1 + 0.8 * stable_fraction(value)):.3f}"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle))


def base_row(**values: object) -> dict[str, str]:
    row = {key: "" for key in FIELDS}
    row.update({key: str(value) for key, value in values.items()})
    row.update(
        window_duration_s="10.000",
        sampling_target_hz="44100",
        channels_target="2",
        codec_target="FLAC PCM-16",
    )
    return row


def pick_artist_balanced(
    candidates: list[dict[str, object]], count: int, cap: int, key_prefix: str
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in candidates:
        groups[str(item["artist"])].append(item)
    for artist, items in groups.items():
        items.sort(key=lambda x: stable_hex(f"{key_prefix}|{artist}|{x['group_id']}|{x['clip_id']}"))
    artists = sorted(groups, key=lambda x: stable_hex(f"{key_prefix}|artist|{x}"))
    chosen: list[dict[str, object]] = []
    for rank in range(cap):
        for artist in artists:
            if rank < len(groups[artist]):
                chosen.append(groups[artist][rank])
                if len(chosen) == count:
                    return chosen
    raise RuntimeError(f"Only {len(chosen)} candidates under artist cap {cap}; need {count}")


def build_magnatagatune() -> list[dict[str, str]]:
    folder = META / "magnatagatune"
    info = {row["clip_id"]: row for row in read_tsv(folder / "clip_info_final.csv")}
    annotations = read_tsv(folder / "annotations_final.csv")
    by_track: dict[str, dict[str, object]] = {}
    for ann in annotations:
        meta = info.get(ann["clip_id"])
        if not meta:
            continue
        positive = sorted(
            key for key, value in ann.items()
            if key not in {"clip_id", "mp3_path"} and value and float(value) > 0
        )
        candidate: dict[str, object] = {
            **meta,
            "group_id": meta["original_url"],
            "positive_tags": positive,
            "vocal_active": any(tag in VOCAL_TAGS for tag in positive),
        }
        old = by_track.get(meta["original_url"])
        if old is None or stable_hex(str(candidate["clip_id"])) < stable_hex(str(old["clip_id"])):
            by_track[meta["original_url"]] = candidate
    vocal = pick_artist_balanced(
        [x for x in by_track.values() if x["vocal_active"]], 250, 2, "mtt-vocal"
    )
    nonvocal = pick_artist_balanced(
        [x for x in by_track.values() if not x["vocal_active"]], 250, 2, "mtt-nonvocal"
    )
    rows = []
    for item in vocal + nonvocal:
        clip_id = str(item["clip_id"])
        member = str(item["mp3_path"])
        rows.append(base_row(
            item_id=f"human_mtt_{int(clip_id):06d}", label="human",
            source_id="human_magnatagatune", source_type="independent-label catalogue",
            role="development", source_revision=REVISIONS["magnatagatune"],
            source_path="mp3.zip", source_locator=(
                f"hf://datasets/confit/magnatagatune@{REVISIONS['magnatagatune']}/mp3.zip"
            ), container_member=member, original_duration_s="29.000",
            window_start_s=window_start(f"mtt|{clip_id}", 29.0),
            group_id=item["group_id"], artist_or_creator=item["artist"],
            title=item["title"], genre_or_tags="|".join(item["positive_tags"]),
            provenance_grade="A-", license="CC BY-NC-SA (Magnatune research use)",
            evaluation_allowed="yes", notes=(
                "artist-balanced; one excerpt per original track; "
                + ("vocal stratum" if item["vocal_active"] else "non-vocal stratum")
            ),
        ))
    assert len(rows) == 500 and len({x["group_id"] for x in rows}) == 500
    strata = Counter(x["notes"].rsplit("; ", 1)[-1] for x in rows)
    assert strata == {"vocal stratum": 250, "non-vocal stratum": 250}
    return rows


def build_maestro() -> list[dict[str, str]]:
    rows = read_csv(META / "maestro_metadata" / "maestro-v3.0.0.csv")
    excluded_lines = (META / "maestro_external_exclusions.txt").read_text().splitlines()
    excluded = set(excluded_lines[3:])
    candidates = [x for x in rows if x["audio_filename"] not in excluded]
    candidates.sort(key=lambda x: stable_hex(f"maestro|{x['audio_filename']}"))
    composer_counts: Counter[str] = Counter()
    work_seen: set[tuple[str, str]] = set()
    selected = []
    for item in candidates:
        composer = item["canonical_composer"]
        work = (composer, item["canonical_title"])
        if composer_counts[composer] >= 20 or work in work_seen:
            continue
        composer_counts[composer] += 1
        work_seen.add(work)
        selected.append(item)
        if len(selected) == 300:
            break
    assert len(selected) == 300 and not ({x["audio_filename"] for x in selected} & excluded)
    output = []
    for rank, item in enumerate(selected, 1):
        duration = float(item["duration"])
        path = item["audio_filename"]
        locator = (
            "https://huggingface.co/datasets/ddPn08/maestro-v3.0.0/resolve/"
            f"{REVISIONS['maestro']}/{quote(path, safe='/')}"
        )
        output.append(base_row(
            item_id=f"human_maestro_{rank:04d}", label="human",
            source_id="human_maestro_v3", source_type="acoustic piano competition",
            role="development", source_revision=REVISIONS["maestro"],
            source_path=path, source_locator=locator,
            original_duration_s=f"{duration:.3f}",
            window_start_s=window_start(f"maestro|{path}", duration),
            group_id=f"{item['canonical_composer']}|{item['canonical_title']}",
            artist_or_creator=item["canonical_composer"], title=item["canonical_title"],
            genre_or_tags=f"classical piano|performance_year={item['year']}",
            provenance_grade="A", license="CC BY-NC-SA 4.0",
            evaluation_allowed="yes", notes=(
                "unique composer-title work; composer cap 20; excludes the 50-recording "
                "external dynamics benchmark"
            ),
        ))
    return output


def build_musicnet() -> list[dict[str, str]]:
    info = json.loads((META / "musicnet_info.json").read_text())
    paths = sorted(
        x["rfilename"] for x in info["siblings"] if x["rfilename"].endswith(".wav")
    )
    assert len(paths) == 330
    return [base_row(
        item_id=f"human_musicnet_{Path(path).stem}", label="human",
        source_id="human_musicnet", source_type="freely licensed chamber/classical recordings",
        role="locked_catalogue_test", source_revision=REVISIONS["musicnet"],
        source_path=path, source_locator=(
            "https://huggingface.co/datasets/DreamyWanderer/MusicNet/resolve/"
            f"{REVISIONS['musicnet']}/{quote(path, safe='/')}"
        ), original_duration_s="probe_required", window_start_s="deterministic_after_probe",
        group_id=Path(path).stem, genre_or_tags="classical|chamber|multi-instrument",
        provenance_grade="A-", license="public-domain or Creative Commons per MusicNet",
        evaluation_allowed="locked_only", notes="all 330 recordings; no detector inspection before lock",
    ) for path in paths]


def build_deam() -> list[dict[str, str]]:
    items = json.loads((META / "deam_song_ids.json").read_text())
    items.sort(key=lambda x: stable_hex(f"deam|{x['song_id']}"))
    selected = items[:300]
    assert len(selected) == 300 and len({x["song_id"] for x in selected}) == 300
    return [base_row(
        item_id=f"human_deam_{int(item['song_id']):05d}", label="human",
        source_id="human_deam", source_type="emotion-annotated mixed music catalogue",
        role="locked_catalogue_test", source_revision=REVISIONS["deam"],
        source_path=item["filename"],
        source_locator=f"{item['filename']}#song_id={item['song_id']}",
        original_duration_s="probe_required", window_start_s="deterministic_after_probe",
        group_id=item["song_id"], genre_or_tags="mixed genre|emotion annotated",
        provenance_grade="B+", license="CC BY-NC (dataset terms; verify per-track terms)",
        evaluation_allowed="locked_only", notes="selected by hash before audio or feature inspection",
    ) for item in selected]


def build_gtzan() -> list[dict[str, str]]:
    rows = []
    for genre in ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]:
        paths = [f"genres/{genre}/{genre}.{idx:05d}.wav" for idx in range(100)]
        paths.sort(key=lambda x: stable_hex(f"gtzan|{x}"))
        for path in paths[:10]:
            rows.append(base_row(
                item_id=f"human_gtzan_{genre}_{Path(path).stem.split('.')[-1]}", label="human",
                source_id="human_gtzan", source_type="legacy genre benchmark",
                role="stress_test_only", source_revision=REVISIONS["gtzan"],
                source_path="data/genres.tar.gz", source_locator=(
                    f"hf://datasets/marsyas/gtzan@{REVISIONS['gtzan']}/data/genres.tar.gz"
                ), container_member=path, original_duration_s="30.000",
                window_start_s=window_start(f"gtzan|{path}", 30.0), group_id=Path(path).stem,
                genre_or_tags=genre, provenance_grade="C",
                license="not stated in HF card", evaluation_allowed="stress_only",
                notes="10 per genre; known duplicates/faults and unclear rights; never headline evidence",
            ))
    assert len(rows) == 100
    return rows


def build_medleydb() -> list[dict[str, str]]:
    items = json.loads((META / "medleydb_records.json").read_text())
    # The only MedleyDB item inspected in the earlier separator study is not
    # present in this 178-track mirror.  Keep the assertion so a future mirror
    # update cannot silently re-introduce it.
    prior_separator_tracks = {"LizNelson_Rainfall"}
    assert not ({str(x["medleydb_id"]) for x in items} & prior_separator_tracks)
    assert len(items) == 178
    rows = []
    for item in sorted(items, key=lambda x: stable_hex(f"medleydb|{x['medleydb_id']}")):
        track_id = str(item["medleydb_id"])
        artist = track_id.split("_", 1)[0]
        rows.append(base_row(
            item_id=f"human_medleydb_{item['id']}", label="human",
            source_id="human_medleydb", source_type="royalty-free studio multitracks",
            role="development", source_revision=REVISIONS["medleydb"],
            source_path="medleydb.tar.gz", source_locator=(
                "hf://datasets/seungheondoh/cmd-medleydb-metadata@"
                f"{REVISIONS['medleydb']}/medleydb.tar.gz"
            ), container_member=item["audio_path"],
            original_duration_s="probe_required", window_start_s="deterministic_after_probe",
            group_id=track_id, artist_or_creator=artist, title=track_id,
            genre_or_tags="|".join(item.get("genre") or []), provenance_grade="A-",
            license=str(item["license"]), evaluation_allowed="yes", notes=(
                "full mixture; all 178 independent tracks in this pinned mirror; "
                "development only because MedleyDB informed the earlier separator analysis"
            ),
        ))
    return rows


def build_moisesdb() -> list[dict[str, str]]:
    items = json.loads((META / "moisesdb_records.json").read_text())
    assert len(items) == 239 and len({str(x["id"]) for x in items}) == 239
    rows = []
    for item in sorted(items, key=lambda x: stable_hex(f"moisesdb|{x['id']}")):
        rows.append(base_row(
            item_id=f"human_moisesdb_{item['id']}", label="human",
            source_id="human_moisesdb", source_type="original multitrack studio catalogue",
            role="development", source_revision=REVISIONS["moisesdb"],
            source_path="moisesdb.tar.gz", source_locator=(
                "hf://datasets/seungheondoh/cmd-moisesdb-metadata@"
                f"{REVISIONS['moisesdb']}/moisesdb.tar.gz"
            ), container_member=item["audio_path"],
            original_duration_s="probe_required", window_start_s="deterministic_after_probe",
            group_id=item["id"], artist_or_creator="artist id unavailable in HF metadata",
            title=item["id"], genre_or_tags="|".join(item.get("genre") or []),
            provenance_grade="A-", license=str(item["license"]),
            evaluation_allowed="yes", notes=(
                "239 of the official catalogue's 240 tracks are present in the pinned mirror; "
                "track—not stem—is the statistical unit"
            ),
        ))
    return rows


def build_urmp() -> list[dict[str, str]]:
    paths = json.loads((META / "urmp_mix_paths.json").read_text())
    assert len(paths) == 44
    rows = []
    for path in paths:
        piece = Path(path).parent.name
        rows.append(base_row(
            item_id=f"human_urmp_{piece}", label="human",
            source_id="human_urmp", source_type="controlled chamber-performance recordings",
            role="development", source_revision=REVISIONS["urmp"],
            source_path=path, source_locator=(
                "https://huggingface.co/datasets/DreamyWanderer/URMP-Reduced/resolve/"
                f"{REVISIONS['urmp']}/{quote(path, safe='/')}"
            ), original_duration_s="probe_required", window_start_s="deterministic_after_probe",
            group_id=piece, artist_or_creator="URMP performers", title=piece,
            genre_or_tags="classical chamber|duet-to-quintet", provenance_grade="A",
            license="research dataset; verify distribution terms before redistribution",
            evaluation_allowed="yes", notes=(
                "all 44 pieces; real performers with aligned MIDI scores; each ensemble mix is one unit"
            ),
        ))
    return rows


def build_hindustani_stress() -> list[dict[str, str]]:
    paths = json.loads((META / "hindustani_raag_paths.json").read_text())
    pattern = re.compile(r"^([^/]+)/(?:train|test)_\[([^]]+)\]_chunk(\d+)\.mp3$")
    recordings: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in paths:
        match = pattern.match(path)
        assert match, path
        recordings[(match.group(1), match.group(2))].append(path)
    by_raag: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for raag, video_id in recordings:
        by_raag[raag].append((raag, video_id))
    assert len(by_raag) == 50
    selected: list[tuple[str, str]] = []
    for raag in sorted(by_raag):
        candidates = sorted(by_raag[raag], key=lambda x: stable_hex(f"hindustani|{x[0]}|{x[1]}"))
        assert len(candidates) >= 2
        selected.extend(candidates[:2])
    assert len(selected) == 100
    rows = []
    for raag, video_id in selected:
        path = sorted(recordings[(raag, video_id)], key=lambda x: stable_hex(f"clip|{x}"))[0]
        rows.append(base_row(
            item_id=f"human_hindustani_{video_id}", label="human",
            source_id="human_hindustani_raag_hf", source_type="YouTube-derived Hindustani recordings",
            role="stress_test_only", source_revision=REVISIONS["hindustani_raag"],
            source_path=path, source_locator=(
                "https://huggingface.co/datasets/neerajaabhyankar/hindustani-raag-small/resolve/"
                f"{REVISIONS['hindustani_raag']}/{quote(path, safe='/')}"
            ), original_duration_s="20_to_60", window_start_s="deterministic_after_probe",
            group_id=video_id, artist_or_creator="upstream performer metadata unavailable",
            title=video_id, genre_or_tags=f"Hindustani classical|raag={raag}",
            provenance_grade="B-", license="HF release declares CC BY 4.0; upstream video audit pending",
            evaluation_allowed="stress_only", notes=(
                "two independent source videos per raag and one clip per video; excluded from headline "
                "results until upstream rights and performer metadata are verified"
            ),
        ))
    return rows


def build_audiox() -> list[dict[str, str]]:
    info = json.loads((META / "audiox_info.json").read_text())
    paths = [
        x["rfilename"] for x in info["siblings"]
        if x["rfilename"].startswith("no_help_audio/") and x["rfilename"].endswith(".wav")
    ]
    paths.sort(key=lambda x: stable_hex(f"audiox|{x}"))
    selected = paths[:500]
    assert len(selected) == 500
    return [base_row(
        item_id=f"ai_audiox_{Path(path).stem.removeprefix('output_')}", label="ai",
        source_id="ai_audiox_third_party", source_type="third-party AudioX output collection",
        role="provisional_development", source_revision=REVISIONS["audiox"],
        source_path=path, source_locator=(
            "https://huggingface.co/datasets/sheng22213/audioX_dataset/resolve/"
            f"{REVISIONS['audiox']}/{quote(path, safe='/')}"
        ), original_duration_s="probe_required", window_start_s="deterministic_after_probe",
        group_id=Path(path).stem, artist_or_creator="AudioX (claimed by repository name)",
        genre_or_tags="unknown", provenance_grade="C+", license="not stated",
        evaluation_allowed="no_until_verified", notes=(
            "audio exists, but the repository has no dataset card, prompt, seed, or checkpoint record; "
            "feature extraction is allowed but detector fitting is gated"
        ),
    ) for path in selected]


def build_diffrhythm() -> list[dict[str, str]]:
    info = json.loads((META / "diffrhythm_info.json").read_text())
    paths = sorted(
        x["rfilename"] for x in info["siblings"]
        if x["rfilename"].startswith("generated/") and x["rfilename"].endswith(".wav")
    )
    assert len(paths) == 50
    rows = []
    for rank, path in enumerate(paths, 1):
        base = re.sub(r"_[1-5]$", "", Path(path).stem)
        rows.append(base_row(
            item_id=f"ai_diffrhythm_pilot_{rank:03d}", label="ai",
            source_id="ai_diffrhythm_pilot", source_type="third-party DiffRhythm inference pilot",
            role="locked_pilot_only", source_revision=REVISIONS["diffrhythm"],
            source_path=path, source_locator=(
                "https://huggingface.co/datasets/Akjava/"
                "diffrhythm-instrument-cc0-oepngamearg-10x5-generated/resolve/"
                f"{REVISIONS['diffrhythm']}/{quote(path, safe='/')}"
            ), original_duration_s="probe_required", window_start_s="deterministic_after_probe",
            group_id=base, artist_or_creator="DiffRhythm, exact architecture unspecified",
            genre_or_tags="instrumental", provenance_grade="B-", license="Apache-2.0 stated",
            evaluation_allowed="pilot_only", notes=(
                "50 outputs but only 10 independent input-audio conditions; not a 50-condition test"
            ),
        ))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = (
        build_magnatagatune() + build_maestro() + build_musicnet() +
        build_medleydb() + build_moisesdb() + build_urmp() +
        build_deam() + build_gtzan() + build_hindustani_stress() +
        build_audiox() + build_diffrhythm()
    )
    assert len({x["item_id"] for x in rows}) == len(rows)
    manifest = OUT / "frozen_item_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    source_ids = sorted({x["source_id"] for x in rows})
    source_identity = {
        source_id: next(x for x in rows if x["source_id"] == source_id)
        for source_id in source_ids
    }
    sources_per_label_role = Counter(
        (item["label"], item["role"]) for item in source_identity.values()
    )
    for source_id in source_ids:
        subset = [x for x in rows if x["source_id"] == source_id]
        independent_groups = len({x["group_id"] for x in subset})
        source_weight = 1.0 / sources_per_label_role[(subset[0]["label"], subset[0]["role"])]
        summary_rows.append({
            "source_id": source_id,
            "label": subset[0]["label"],
            "role": subset[0]["role"],
            "items": len(subset),
            "independent_groups": independent_groups,
            "equal_source_weight_within_label_role": f"{source_weight:.9f}",
            "per_group_weight_within_label_role": f"{source_weight / independent_groups:.12f}",
            "provenance_grade": subset[0]["provenance_grade"],
            "evaluation_allowed": subset[0]["evaluation_allowed"],
        })
    summary = OUT / "source_summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    metadata = {
        "manifest_version": "v2-catalogue-first",
        "supersedes": "../frozen_item_manifest.csv (v1, replaced before audio decoding)",
        "selection_seed": SEED,
        "selection_before_audio_or_feature_inspection": True,
        "manifest_sha256": digest,
        "total_items": len(rows),
        "class_counts": Counter(x["label"] for x in rows),
        "role_counts": Counter(x["role"] for x in rows),
        "source_revisions": REVISIONS,
        "weighting_rule": (
            "balance AI/Human classes, then give every eligible source equal total weight "
            "within class; divide a source's weight uniformly over independent groups"
        ),
        "standard_view": {
            "duration_seconds": 10.0,
            "sample_rate_hz": 44100,
            "channels": 2,
            "codec": "FLAC PCM-16",
            "position": "deterministic interior window; computed before feature extraction",
        },
    }
    (OUT / "manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=dict) + "\n", encoding="utf-8"
    )
    (OUT / "frozen_item_manifest.sha256").write_text(
        f"{digest}  frozen_item_manifest.csv\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, default=dict))


if __name__ == "__main__":
    main()
