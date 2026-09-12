# Completed independent audio-publication acceptance

Six formerly queued publications have now passed independent anonymous public
verification. This package preserves all 208 batch receipts, six upload terminal
receipts and six independent acceptances. SUMMARY.json binds the copied receipt
files by SHA-256 and size. The README itself is explanatory, not a bound receipt.

| Publication | Audio file entries | Verified public prefix |
|---|---:|---|
| AIME generated originals | 5,000 | `audio/aime_generated_originals_v1/` |
| ACE-Step/HeartMuLa originals and standardized views | 2,000 | `audio/project_open_models_v1/` |
| Historical Suno originals | 500 | `audio/historical_suno_originals_v1/` |
| MagnaTagATune excerpts | 500 | `audio/magnatagatune_selected_v1/` |
| Early FMA unchanged excerpts | 162 | `audio/early_fma_originals_v1/` |
| Early Suno originals | 100 | `audio/early_suno_originals_v1/` |

Except for early FMA, the verification revision is
`8cc2c8d6bf54735b5ada8126461c0b3031060fce` in
https://huggingface.co/datasets/EZMONYI/music-ai-human-test-audio .
Early FMA was checked at `1298fbbebdde70fff07e2ef8f2737d8b78351e0b`;
all 38 held-file paths were confirmed absent from that publication prefix.
Use each receipt's exact revision to construct immutable download links.

All 8,262 audio file entries matched expected SHA-256 and size, and manifests
were checked against their source evidence. Verification of source README bytes
is recorded only where the corresponding verifier performs that check; do not
infer identical metadata-check coverage across all six programs.

These are file/version counts, not 8,262 independent songs. The two open-model
views represent 1,000 recordings. Historical and early input memberships are
preserved separately. No new model fitting, threshold selection or experimental
relabeling occurred. Per-source attribution and reuse restrictions still apply.

Mureka and previously accepted YuE, FMA, MAESTRO and external-control releases
have separate receipts; they are not included in this package's totals.
This package does not claim whole-project coverage: unresolved early humair
metadata, source-specific permissions and remaining views/stems still require
reconciliation. Earlier queued-status prose is historical and superseded for
these six publications only.
