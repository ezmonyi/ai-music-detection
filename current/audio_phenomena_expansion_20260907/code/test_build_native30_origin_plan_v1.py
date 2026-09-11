#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("origin_plan", HERE / "build_native30_origin_plan_v1.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OriginPlanHelpersTest(unittest.TestCase):
    def test_strict_json_rejects_duplicate_and_nonfinite(self):
        with self.assertRaises(ValueError):
            MODULE.strict_json_bytes(b'{"a":1,"a":2}')
        with self.assertRaises(ValueError):
            MODULE.strict_json_bytes(b'{"a":NaN}')

    def test_csv_rejects_duplicate_headers_and_ids(self):
        with self.assertRaises(ValueError):
            MODULE.strict_csv_bytes(b"id,id\na,b\n", "id")
        with self.assertRaises(ValueError):
            MODULE.strict_csv_bytes(b"id,x\na,1\na,2\n", "id")

    def test_full_native_center_branch(self):
        result = MODULE.full_native_center30(30.08, 48000, 1443840)
        self.assertEqual(result["start_frame"], 1920)
        self.assertEqual(result["frames"], 1440000)
        self.assertFalse(result["physical_frame_validation_required"])
        unresolved = MODULE.full_native_center30(30.0, 44100)
        self.assertIsNone(unresolved["start_frame"])
        self.assertTrue(unresolved["physical_frame_validation_required"])
        with self.assertRaises(ValueError):
            MODULE.full_native_center30(60.01, 44100)

    def test_approved_region_requires_exact60_and_hash(self):
        sha = "a" * 64
        result = MODULE.approved_region_center30(100, 60 * 48000, 48000, sha, True)
        self.assertEqual(result["start_frame"], 100 + 15 * 48000)
        self.assertEqual(result["frames"], 30 * 48000)
        with self.assertRaises(ValueError):
            MODULE.approved_region_center30(0, 59 * 48000, 48000, sha, True)
        with self.assertRaises(ValueError):
            MODULE.approved_region_center30(0, 60 * 48000, 48000, None, True)

    def test_atomic_exclusive_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "plan.json"
            first = MODULE.atomic_exclusive_json(target, {"status": "draft"})
            self.assertEqual(first, MODULE.digest(target.read_bytes()))
            with self.assertRaises(FileExistsError):
                MODULE.atomic_exclusive_json(target, {"status": "changed"})
            self.assertEqual(json.loads(target.read_text()), {"status": "draft"})


if __name__ == "__main__":
    unittest.main()
