"""Safety and correctness checks for large-archive sample reads."""
import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from music_detector.drive import DriveRangeReader, RangeReadError, download_member, find_tar_members


class MemoryReader:
    def __init__(self, data, budget=100000):
        self.data, self.size = data, len(data)
        self.bytes_read, self.max_bytes, self.file_id = 0, budget, "fixture"

    def read_at(self, offset, length):
        if self.bytes_read + length > self.max_bytes:
            raise RangeReadError("budget")
        self.bytes_read += length
        return self.data[offset:offset + length]


class FakeResponse(io.BytesIO):
    def __init__(self, data, status, content_range):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range}


class FakeOpener:
    def __init__(self, response):
        self.response = response

    def open(self, request, timeout):
        return self.response


class DriveTests(unittest.TestCase):
    def archive(self):
        output = io.BytesIO()
        long_name = "audio/" + "a" * 110 + "/sample.wav"
        with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name, data in [("irrelevant.wav", b"X" * 2000000), (long_name, b"small payload")]:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return output.getvalue(), long_name

    def test_skips_large_payload_and_resolves_pax_path(self):
        data, name = self.archive()
        reader = MemoryReader(data)
        member = find_tar_members(reader, [name])[name]
        self.assertLess(reader.bytes_read, 4096)
        self.assertEqual(data[member.offset:member.offset + member.size], b"small payload")
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "sample.wav"
            download_member(reader, member, target, hashlib.sha256(b"small payload").hexdigest())
            self.assertEqual(target.read_bytes(), b"small payload")
            with self.assertRaises(FileExistsError):
                download_member(reader, member, target, hashlib.sha256(b"small payload").hexdigest())

    def test_hash_failure_removes_only_new_partial_file(self):
        data, name = self.archive()
        reader = MemoryReader(data)
        member = find_tar_members(reader, [name])[name]
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "sample.wav"
            with self.assertRaises(RangeReadError):
                download_member(reader, member, target, "0" * 64)
            self.assertFalse(target.exists())

    def test_traversal_and_budget_are_rejected(self):
        with self.assertRaises(RangeReadError):
            find_tar_members(MemoryReader(b""), ["../escape.wav"])
        data, name = self.archive()
        with self.assertRaises(RangeReadError):
            find_tar_members(MemoryReader(data, budget=511), [name])

    def test_ignored_and_incorrect_ranges_fail_closed(self):
        for status, content_range, payload in [(200, "bytes 0-3/10", b"abcd"),
                                                (206, "bytes 1-4/10", b"abcd"),
                                                (206, "bytes 0-3/10", b"abc")]:
            reader = DriveRangeReader("fixture", 10, token=lambda: "test",
                                      opener=FakeOpener(FakeResponse(payload, status, content_range)))
            with self.assertRaises(RangeReadError):
                reader.read_at(0, 4)

    def test_valid_range_and_pre_request_limit(self):
        reader = DriveRangeReader("fixture", 10, token=lambda: "test", max_bytes=4,
                                  opener=FakeOpener(FakeResponse(b"abcd", 206, "bytes 0-3/10")))
        self.assertEqual(reader.read_at(0, 4), b"abcd")
        with self.assertRaises(RangeReadError):
            reader.read_at(4, 1)


if __name__ == "__main__":
    unittest.main()
