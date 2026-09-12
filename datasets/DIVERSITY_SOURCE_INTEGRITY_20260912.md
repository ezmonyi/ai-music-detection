# Historical diversity source integrity

A fresh server read verified all 2,641 selected source files against the frozen
historical raw SHA-256 values: 74,350,492,316 bytes, zero missing files, zero
hash mismatches and zero read errors. This is source integrity, not publication
or permission acceptance. The audit did not regenerate or alter audio.

| Source | Verified files |
|---|---:|
| MagnaTagATune | 500 |
| MAESTRO v3 | 300 |
| MusicNet | 330 |
| MedleyDB | 178 |
| MoisesDB | 239 |
| URMP | 44 |
| DEAM | 300 |
| GTZAN | 100 |
| Hindustani Raag HF | 100 |
| AudioX third-party | 500 |
| DiffRhythm pilot | 50 |

Scope is exactly the historical 10s metadata rows whose source_audio_path is
under the preserved September 5 diversity native directory. Files are the
historically recorded source assets, not necessarily untouched upstream originals.
Other sources, derived views and separated stems are not covered by this audit.
Provisional, stress, pilot and locked roles remain unchanged.

Input metadata SHA-256:
`a03aec930130fcb942c40b35ba5fb49109ceee588894124010448f041d9d1ef4`.
Audit records SHA-256:
`ec874be2d40b987facd745262a5664554495634fa72ffa2109691ff1679fba9a`.
Terminal COMMIT SHA-256:
`811bb52731cd61286ce8fc408df50f92dae1251e58aa8cd0bc877ab0509cf3cc`.

Complete records, terminal receipt and runtime log are retained locally under
`Documents/report/music_ai_detection_20260912/current_results/diversity_source_audio_audit_v1/`.
The downloaded records were independently checked for all 2,641 distinct IDs,
all expected/actual hash pairs and total bytes. Code:
`current/yue2_500_extension_20260911/audit_diversity_source_audio_v1.py`.
