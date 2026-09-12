# Interpretable AI and Human Music Detection

Research code and preserved source snapshots for an English graduation thesis
on interpretable audio phenomena in AI-generated and human music.

This repository is a work-in-progress reconstruction from preserved experiment
archives. It is not a reconstruction of the original Git history and does not
claim that all experiments or final deliverables are complete.

## Layout

- `current/`: currently recovered source, tests, working LaTeX, and YuE2 extension drivers.
- `historical_snapshots/`: source and LaTeX recovered separately from dated experiment archives.
- `SOURCE_MANIFEST.json`: file hashes and archive provenance for the recovered files.

The native30 analysis code is under
`current/audio_phenomena_expansion_20260907/code/`.
New YuE2 drivers are under `current/yue2_500_extension_20260911/`.
The working English thesis is under
`current/audio_phenomena_expansion_20260907/latex/`.

## Data and reports

### Completed experiment snapshot (12 September 2026)

- Latest editorial revision: [English thesis v7 PDF](thesis_releases/v7/thesis_v7.pdf),
  [self-contained sources](thesis_releases/v7/sources/), and
  [differential layout acceptance](thesis_releases/v7/ACCEPTANCE_EN.md).
  V7 corrects stale experiment-status wording without changing numerical results.
  The v6 links below remain immutable historical releases.

- [English thesis v6 PDF](https://huggingface.co/datasets/EZMONYI/music-ai-human-test-audio/resolve/eb0e35208bb9229e1a2ee1a389e181ce7ddee8d6/reports/completed_experiments_thesis_v6_20260912/thesis_v6.pdf)
- [Self-contained thesis source bundle](https://huggingface.co/datasets/EZMONYI/music-ai-human-test-audio/resolve/eb0e35208bb9229e1a2ee1a389e181ce7ddee8d6/reports/completed_experiments_thesis_v6_20260912/thesis_v6_sources.tar.gz)
- [Complete BC/YuE2 aggregate reports, Native60 predictions and features](https://huggingface.co/datasets/EZMONYI/music-ai-human-test-audio/tree/eb0e35208bb9229e1a2ee1a389e181ce7ddee8d6/reports/completed_experiments_thesis_v6_20260912)
- Compact tables: `results/native30_aggregate_tables_20260912/` and
  `results/native60_transfer_report_v1/`.
- Dataset indexes: `datasets/native30_expanded_v1/`,
  `datasets/historical_input_inventory_v1/`,
  `datasets/historical_measurement_coverage_v1/`, `datasets/EXTERNAL_CONTROLS.md`
  and `datasets/FMA_PUBLICATION.md`. Historical inputs are not all confirmed
  classifier test members; preserve the recorded roles and limitations.

The historical measurement index joins 10,141 unique input IDs across 26 sources
to 10s/30s feature status and verified original-audio publication receipts.
The 10s view has 10,085 complete rows and 56 recorded serialization failures;
the 30s view has 4,493 complete rows and four recorded serialization failures.
Per-family unavailability is retained even for complete rows. Original-audio
links are established for 692 IDs (392 FMA excerpts and 300 MAESTRO recordings),
not for every measured view or stem. Later sources and external controls are
indexed separately; these counts are not the complete project's song count.

The seven files in the completed-result HF snapshot were verified against
remote hashes and sizes. Its bundle COMMIT is
`d392cff45fc86beac373faca19fe55aa2349e47fdbf93a5b8111976b37bfd152`.
Thesis v6 includes completed BC, expanded YuE2 and Native60 transfer results;
its self-contained source rebuild passed. Full-corpus audio publication is
still incomplete, so this is not a claim of whole-project delivery completion.

Public data/report archive:
https://huggingface.co/datasets/EZMONYI/music-ai-human-test-audio

The HF archive is incomplete. Its first publication contains verified
intermediate reports and tables, not the full tested audio collection.
Audio redistribution requires source-specific permission and attribution.
Do not infer a blanket audio license from the visibility of this repository.

## Reproduction constraints

Many historical scripts contain absolute paths to the original Linux servers.
They are preserved for provenance; a portable entry point is not yet complete.
Frozen native30 runners additionally bind code, checkpoints, input hashes,
runtime versions, and separate authorization contracts. Do not edit their
scientific gates or silently substitute missing models/data to make a run pass.

The existing neural analysis runtime used Python 3.11.15, NumPy 1.26.4,
Torch 2.8.0+cu128, All-In-One Infer 3.1.0, and Beat This 1.1.0.
YuE2 generation used a separate runtime. Refer to experiment-specific
contracts and source rather than mixing their dependencies.

All source snapshots should be treated as research code. Preserved scripts
include intermediate and failed approaches; presence is not endorsement as a
final method. No global success claim follows from a subset of passing tests.

## Security and completeness

Raw audio, model weights, environments, private keys, tokens, and recovery keys
are excluded. A literal private-key/HF/GitHub token-pattern scan was performed;
this is not a complete security audit. The source manifest covers recovered
files, not these staging metadata files. Additional ongoing scripts and final
reports will be added after verification.
