import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from run_vocalset_vibrato_assay import (
    PREPROCESS_CONFIG,
    consolidate,
    select_role_rows,
)


class VocalSetVibratoRunnerTest(unittest.TestCase):
    def test_role_selection_is_exact_and_singer_disjoint(self):
        rows = []
        for index in range(43):
            rows.append({"pair_id": f"d{index}", "split": "development", "singer": f"d{index}"})
        for index in range(47):
            rows.append({"pair_id": f"e{index}", "split": "evaluation", "singer": f"e{index}"})
        selected = select_role_rows(rows, "development")
        self.assertEqual(len(selected), 43)
        self.assertFalse(any(row["split"] == "evaluation" for row in selected))

    def test_preprocessor_freeze_is_explicit(self):
        self.assertEqual(PREPROCESS_CONFIG["resample_up"], 160)
        self.assertEqual(PREPROCESS_CONFIG["resample_down"], 441)
        self.assertEqual(PREPROCESS_CONFIG["resample_window"], ["kaiser", 5.0])
        self.assertFalse(PREPROCESS_CONFIG["dc_removal"])
        self.assertFalse(PREPROCESS_CONFIG["amplitude_normalization"])

    def test_development_consolidation_counts_ties_as_nonpositive(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "receipts").mkdir()
            specs = []
            # 43 pairs across the three required contexts.  All clips are valid;
            # 28 positive differences is >=65%, 15 ties remain non-positive.
            contexts = ["excerpts"] * 26 + ["scales"] * 9 + ["arpeggios"] * 8
            for index, context in enumerate(contexts):
                pair_id = f"pair-{index}"
                singer = f"s{index % 10}"
                for technique in ("straight", "vibrato"):
                    clip_id = f"{pair_id}::{technique}"
                    specs.append(
                        {
                            "clip_id": clip_id,
                            "pair_id": pair_id,
                            "role": "development",
                            "singer": singer,
                            "context": context,
                            "content_id": str(index),
                            "technique": technique,
                            "filename": clip_id + ".wav",
                            "source_path": "/fixture/" + clip_id,
                            "expected_source_sha256": "a" * 64,
                            "expected_sample_rate_hz": 44100,
                            "expected_frames": 441000,
                        }
                    )
                    straight = 0.0
                    value = straight if technique == "straight" else (0.5 if index < 28 else 0.0)
                    record = {
                        **specs[-1],
                        "status": "ok",
                        "contract_hash": "contract",
                        "source_sha256": "a" * 64,
                        "V_status": "ok",
                        "V_periodic_modulation_window_fraction": value,
                    }
                    key = __import__("hashlib").sha256(clip_id.encode()).hexdigest()
                    (output / "receipts" / f"{key}.json").write_text(json.dumps(record))
            summary = consolidate(output, specs, "contract", "development")
            self.assertEqual(summary["pair_direction_counts_including_missing"], {"positive": 28, "tie": 15})
            self.assertAlmostEqual(summary["strictly_positive_pair_fraction"], 28 / 43)
            self.assertTrue(summary["development_gate_checks"]["median_paired_difference_gt_0"])
            self.assertTrue(summary["development_gate_passed"])
            with (output / "per_pair_differences.csv").open(newline="", encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual(len(pair_rows), 43)


if __name__ == "__main__":
    unittest.main()
