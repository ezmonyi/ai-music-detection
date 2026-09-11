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
