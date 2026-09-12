# Code delivery checkpoint

The 66 Python/shell scripts currently at the top level of the local YuE2
extension directory all exist byte-for-byte in
`current/yue2_500_extension_20260911/`. No missing or differing script was found
in this bounded comparison. This does not establish coverage of every historical
directory or remote-only script.

`PUBLICATION_MANIFEST_20260912_v14.json` binds 1,126 files under `current/` and
`historical_snapshots/`. Its literal credential-pattern scan passed. The check
does not cover other folders and is not a complete security audit. Dataset
packages have their own product hashes; the v7 thesis source snapshot was
separately compared with all 54 local source files.

Targeted validation repeated on 12 September 2026:

    python3 -m unittest test_native60_score_kernel_v1.py test_verify_evaluation_archive_v1.py

All ten tests passed. These tests cover the scalar score kernel and archive
verifier fixtures, not a fresh reproduction of every experiment. The completed
experiment receipts and full results provide separate execution evidence.

Latest thesis entry: `thesis_releases/v7/thesis_v7.pdf`; the matching source entry
is `thesis_releases/v7/sources/interpretable_audio_thesis_working_v7_20260912_en.tex`.
Older working masters are retained for provenance, not preferred build targets.
Raw audio, model weights, environments and credentials are not part of this code
repository. Audio publication is still in progress in the separate HF dataset.
