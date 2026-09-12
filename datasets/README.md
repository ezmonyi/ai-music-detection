# Dataset navigation and release boundaries

This is the metadata/code-facing dataset release. Audio and large numerical
reports live in [the HF archive](https://huggingface.co/datasets/EZMONYI/music-ai-human-test-audio).
The release is still incomplete; repository visibility is not an audio license.

| Catalogue | Purpose | Scope |
|---|---|---|
| [Cross-experiment identities](cross_experiment_catalogue_v1/README.md) | Exact-ID union with separate cohort memberships | 11,242 IDs, 29 sources; not content-deduplicated |
| [Historical inputs](historical_input_inventory_v1/README.md) | Input roles and provenance | 10,141 IDs; 10s and 30s views overlap |
| [Audited diversity retrieval index](diversity_retrieval_catalogue_v1/README.md) | Fixed source locators, source hashes, recorded terms and historical groups | 2,641 existing IDs; all source bytes freshly verified |
| [Historical measurement status](historical_measurement_coverage_v1/README.md) | Successful/failed and per-family availability | 10,141 10s rows and 4,497 30s rows |
| [Expanded Native30](native30_expanded_v1/README.md) | Completed feature cohort index | 4,228 development plus 100 locked YuE2 IDs |
| [All YuE2 originals and duration eligibility](yue2_duration_catalogue_v1/README.md) | Preserve all generations, including short outputs | 500 IDs; 498 Native30 and 276 Native60 eligible |
| [Selected AIME generated originals](aime_generated_catalogue_v1/README.md) | Raw-source hashes linked to historical views | 5,000 existing historical IDs; ten generator labels, 500 each |
| [ACE-Step/HeartMuLa original-view pairs](open_model_audio_catalogue_v1/README.md) | Fresh byte audit linked to historical IDs | 1,000 existing identities, 2,000 original/view files; not yet uploaded |
| [External controls](EXTERNAL_CONTROLS.md) | Acoustic-phenomenon validation sources | Separate from song-classifier populations |

Do not add these row counts. The cross-experiment union includes 498 YuE2 IDs;
the generation catalogue adds two short IDs, not another 500 songs. Even that
union is not the full historical corpus: external controls and earlier-only
samples still require reconciliation. Different IDs can share content or prompts.

## How to reproduce membership

Use `memberships.csv` in the cross-experiment catalogue to identify the original
cohort and role. Follow its source_index to the bound catalogue and preserve
group_id/component_id. A role labelled development is not a universal training
partition: each source-holdout protocol has its own frozen model schedule.
Provisional, pilot and stress-only roles are not silently admitted to primary
classification. Feature completeness does not imply every family was observable.

Original-file hashes, normalized-view hashes and stem hashes identify different
objects. Retain native sample rate, duration and exact crop definitions. The
Native60 YuE2 population is duration-selected and AI-only; it cannot estimate
human specificity, balanced accuracy or two-class ROC AUC.

## Audio publication status

- MAESTRO: 300 verified originals and 300 Native30 views, representing the same
  300 recordings, not 600 independent works. Individual receipts bind revisions.
- [FMA](FMA_PUBLICATION.md): 392 unchanged FMA medium excerpts published; 108
  selected originals remain outside this release pending license reconciliation.
- NSynth and GuitarSet: official source archives published as described in the
  external-controls index; not all archive members were tested.
- VocalSet: 244 selected original external-control recordings and two metadata
  files independently verified at revision `65087d237b50f598000d2cf3a5353317bdf5f866`;
  see the external-controls index for acquisition versus analysis scope.
- YuE2: all 500 original FLAC files (6,233,703,879 bytes) were uploaded and
  independently checked at revision `4389bbd7906137b9abbc340fcd743bca451ab092`.
  [Verified originals](https://huggingface.co/datasets/EZMONYI/music-ai-human-test-audio/tree/4389bbd7906137b9abbc340fcd743bca451ab092/audio/yue2/originals_v1/).
  All 50 batch receipts, terminal COMMIT and exact public manifest were checked.
  This covers originals, not every derived view or stem.
- AIME: all 5,000 selected generated source blobs passed byte-level audit
  (8,281,783,640 bytes). All originals were extracted; upload remains incomplete
  (1,650 files in verified batches at the hourly-rate-limit checkpoint).
  These are existing historical IDs, not new test data.
  The separately licensed MTG-labelled human subset is excluded.
- ACE-Step/HeartMuLa: 1,000 originals and their 1,000 standardized views are
  byte-audited and linked to historical IDs. Upload remains incomplete (620
  files in verified batches at the hourly-rate-limit checkpoint). These are
  1,000 recordings, not 2,000 independent songs. The detached resumptions for
  both active publications honor the HF hourly commit limit.
- Other song sources and derived views/stems: publication is not established by
  these catalogues. Missing links do not prove the source was never backed up.

No blanket redistribution right is inferred from an upstream dataset/model tag.
Keep source-specific attribution and conditions. Restricted or unresolved audio
must have explicit retrieval/provenance records rather than a false upload claim.

## Validation and versioning

Each catalogue's COMMIT binds its products by size and SHA-256. These commitments
are scoped evidence, not a declaration that all project requirements are complete.
The construction scripts are under `../current/yue2_500_extension_20260911/`.
Old catalogue versions and failed experiment evidence are retained, not silently
rewritten to fit the latest population or a more favorable outcome.
