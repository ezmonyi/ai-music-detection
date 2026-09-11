import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from build_vocalset_vibrato_protocol import (
    EXCLUDED_IDENTITY_FILES,
    allocate_singers,
    build_protocol,
    parse_target_row,
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def row(singer, context, technique, content):
    directory_singer = ("female" if singer.startswith("f") else "male") + singer[1:]
    if context == "excerpts":
        filename = f"{singer}_{content}_{technique}.wav"
    else:
        filename = f"{singer}_{context}_{technique}_{content}.wav"
    return {
        "filename": filename,
        "singer": singer,
        "audio_path": f"/fixture/{filename}",
        "audio_sha256": "a" * 64,
        "archive_member": f"FULL/{directory_singer}/{context}/{technique}/{filename}",
        "member_crc32": "12345678",
        "member_bytes": 100,
        "sample_rate": 44100,
        "frames": 441000,
        "channels": 1,
        "duration_verified": 10.0,
    }


class VocalSetVibratoProtocolTest(unittest.TestCase):
    def test_target_parser_preserves_exact_content_match(self):
        parsed = parse_target_row(row("f1", "scales", "vibrato", "a"))
        self.assertEqual(parsed["content_id"], "scales_a")
        self.assertEqual(parsed["technique"], "vibrato")
        parsed = parse_target_row(row("m2", "excerpts", "straight", "caro"))
        self.assertEqual(parsed["content_id"], "caro")

    def test_parser_rejects_archive_singer_disagreement(self):
        bad = row("f1", "scales", "straight", "a")
        bad["archive_member"] = bad["archive_member"].replace("female1", "female2")
        with self.assertRaisesRegex(ValueError, "archive singer"):
            parse_target_row(bad)

    def test_split_is_deterministic_and_singer_disjoint(self):
        singers = [f"f{i}" for i in range(1, 10)] + [f"m{i}" for i in range(1, 12)]
        first, detail = allocate_singers(singers)
        second, _ = allocate_singers(list(reversed(singers)))
        self.assertEqual(first, second)
        self.assertEqual(sum(value == "evaluation" for value in first.values()), 10)
        self.assertEqual(len(detail["f"]["evaluation"]), 5)
        self.assertEqual(len(detail["m"]["evaluation"]), 5)

    def test_build_excludes_conflict_and_does_not_reuse_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            singers = [f"f{i}" for i in range(1, 10)] + [f"m{i}" for i in range(1, 12)]
            for singer in singers:
                rows.extend(
                    [
                        row(singer, "excerpts", "straight", "caro"),
                        row(singer, "excerpts", "vibrato", "caro"),
                    ]
                )
            conflict = next(item for item in rows if item["filename"] == "m9_caro_vibrato.wav")
            conflict["archive_member"] = conflict["archive_member"].replace("male9", "male8")
            while len(rows) < 244:
                index = len(rows)
                singer = singers[index % len(singers)]
                directory_singer = ("female" if singer[0] == "f" else "male") + singer[1:]
                filename = f"{singer}_belt_fixture_{index}.wav"
                filler = row(singer, "scales", "straight", "a")
                filler.update(
                    filename=filename,
                    archive_member=f"FULL/{directory_singer}/scales/belt/{filename}",
                )
                rows.append(filler)
            summary_path = root / "acquisition_summary_v2.json"
            write_json(
                summary_path,
                {
                    "status": "complete_accounting",
                    "selected": 244,
                    "downloaded_verified": 244,
                    "errors": [],
                    "manifest": rows,
                },
            )
            audit_path = root / "independent_audit.json"
            write_json(
                audit_path,
                {
                    "status": "passed",
                    "gate_passed": True,
                    "acquisition_summary_sha256": sha256_file(summary_path),
                },
            )
            result = build_protocol(summary_path, audit_path)
            self.assertEqual(result["counts"]["eligible_pairs"], 19)
            self.assertEqual(result["counts"]["unmatched_target_recordings"], 1)
            self.assertEqual(result["exclusions"][0]["filename"], next(iter(EXCLUDED_IDENTITY_FILES)))
            used = [
                name
                for pair in result["pairs"]
                for name in (pair["straight_filename"], pair["vibrato_filename"])
            ]
            self.assertEqual(len(used), len(set(used)))


if __name__ == "__main__":
    unittest.main()
