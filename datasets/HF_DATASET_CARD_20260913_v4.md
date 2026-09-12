---
pretty_name: Interpretable AI and Human Music Evaluation Archive
language:
- en
tags:
- music
- ai-music-detection
- audio
- research
---

# Interpretable AI and Human Music Evaluation Archive

Research audio and versioned experiment outputs for an English graduation thesis.
**The audio archive is incomplete.** Completed experiments and verified partial
audio publications must not be confused with whole-project delivery completion.
No blanket license is assigned to this mixed-source archive.

## Completed experiments and thesis

The BC extension, expanded YuE Native30 evaluation, locked YuE Native30 scoring,
legacy spectral comparison and Native60 frozen transfer have completed. Older
reports stating these were pending are retained as historical checkpoints.

- [Latest English thesis v7 and source code](https://github.com/ezmonyi/ai-music-detection).
- [Completed BC, expanded YuE and Native60 report bundles, including thesis v6](https://huggingface.co/datasets/EZMONYI/music-ai-human-test-audio/tree/eb0e35208bb9229e1a2ee1a389e181ce7ddee8d6/reports/completed_experiments_thesis_v6_20260912).
- [Portable Native60 report reproduction from pinned public inputs](https://github.com/ezmonyi/ai-music-detection/blob/main/REPRODUCE_NATIVE60_REPORT.md).
- [Dataset identities, group memberships, source retrieval and measurement coverage](https://github.com/ezmonyi/ai-music-detection/tree/main/datasets).

The report replay verifies 876,300 persisted predictions and reconstructs 1,905
aggregate cells. It does not rerun audio inference or refit models.

## Audio publication coverage

Verified publications include:

- 500 YuE original FLAC recordings under `audio/yue2/originals_v1/`.
- 300 MAESTRO original recordings and their 300 Native30 views under
  `audio/human_maestro_v3/` (the same 300 recordings, not 600 songs).
- 392 unchanged, explicitly licensed FMA excerpts under
  `fma_originals_explicit_license_v1/`; 108 other selected FMA originals are not
  included in that publication.
- 244 selected VocalSet external-control recordings under
  `external_controls/vocalset_selected_originals_v1/`.
- Official NSynth test and GuitarSet source archives under `external_controls/`.
  Source archive membership does not mean every member was tested.

- AIME: 5,000 generated originals; ACE-Step/HeartMuLa: 2,000 original/view
  files representing 1,000 recordings; historical Suno: 500 originals;
  MagnaTagATune: 500 excerpts; early FMA: 162 originals; early Suno: 100
  originals. All six publications passed independent acceptance; receipts:
  https://github.com/ezmonyi/ai-music-detection/tree/main/datasets/completed_audio_acceptance_v1
- Mureka: 500 originals, independently accepted.
- Early humair: 100 originals with recovered attribution variants, independently
  accepted. https://github.com/ezmonyi/ai-music-detection/tree/main/datasets/early_humair_publication_v2
- YuE legacy Demucs: 1,000 vocal/accompaniment stems from 500 recordings,
  independently accepted. https://github.com/ezmonyi/ai-music-detection/tree/main/datasets/yue2_demucs_publication_v1

- Saraga: all 108 unchanged originals (103 classifier identities) independently
  accepted; two records retain incomplete detailed performer credits.
- YuE derived audio: all 5,900 unique objects independently accepted at revision
  `caed9220f68b258e9c2dcb89792ccb1fe4d1dc61`, including 4,900 additional objects
  and the 1,000 legacy stems. These represent 6,922 source-path memberships
  across seven inventoried directories, not a new population of songs.
- MusicNet Ishizaka: 39 recordings independently accepted after source-specific
  CC0 review; the other 291 MusicNet selections are not covered by this review.
- Early external MAESTRO: 50 WAV and 50 paired MIDI files independently accepted,
  separate from the later 300-recording classifier source.
- DiffRhythm pilot: 50 outputs independently accepted, with ten conditioning
  groups; no promotion from pilot to primary evaluation is implied.

The fixed snapshot at `caed9220f68b258e9c2dcb89792ccb1fe4d1dc61` contains
16,795 direct audio/MIDI paths, 130,247,953,279 logical bytes, excluding archive
members. These are file versions, not unique songs. Against this snapshot,
7,781 of 10,141 historical recorded raw hashes match public objects, with
2,360 unmatched. This is not whole-project or exact processed-view coverage.
See the versioned acceptance and coverage records in the GitHub datasets index.
Not every derived crop, separated stem or earlier-only control is published.

The catalogues overlap. Original recordings, normalized views, stems and
features are different objects, not independent songs. The 11,242-ID
cross-experiment index is an exact-ID union within its stated scope, not a
complete content-deduplicated count of every project recording.

## Source rights and attribution

Consult each source-specific README and manifest before reuse. Noncommercial,
share-alike, no-derivatives and attribution restrictions remain applicable where
stated. A model license or public dataset tag is not a blanket recording license.
Restricted or unresolved audio is represented by retrieval/provenance records,
not by a false claim that the files have been republished. No credentials,
private keys or recovery keys belong in this dataset.

## Scientific limitations

Balanced accuracy is distinct from ordinary accuracy. Source-held-out and
grouped descriptive evaluations answer different questions. Separable features
do not by themselves establish an intrinsic or universal AI fingerprint.
Production, codec, instrument and dataset-source effects remain relevant.

The Native60 YuE transfer population is duration-selected and AI-only; it
supports AI sensitivity, not human specificity, balanced accuracy or ROC AUC.
Shared prompts and recording groups must remain grouped under the published
protocol. Missing feature families, excluded short generations, pilot-only
features and failed earlier approaches are retained in the provenance.

## Version history

Earlier thesis versions, intermediate tables and failed-attempt evidence are
preserved. This card corrects stale pending-status prose; it does not change
experimental memberships, thresholds or numerical results. The latest thesis
is a working research draft, not a claim of institutional thesis acceptance.
