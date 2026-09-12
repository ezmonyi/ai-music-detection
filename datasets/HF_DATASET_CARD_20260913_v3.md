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

Two further publications were launched but are not yet independently accepted
at this card checkpoint: 108 Saraga unchanged originals (103 classifier
members), and 4,900 additional unique YuE derived audio objects. Their planned
manifests do not establish completion. The YuE mapping retains 6,922 paths for
5,900 unique byte objects across seven inventoried directories, including the
1,000 already accepted legacy stems. It is not a new population of songs.
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
