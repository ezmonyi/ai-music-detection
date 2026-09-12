# Source delivery coverage, checkpoint v15

The 491 Python files in the current expanded-phenomena `code/` tree and the
top-level YuE extension directory were compared byte-for-byte with their
`current/` copies in the GitHub staging repository. No missing or different
files were found. This explicitly scoped comparison is not a claim that every
historical file on every server has been discovered.

`PUBLICATION_MANIFEST_20260912_v15.json` records 1,160 files under `current/` and
`historical_snapshots/`, each with byte size and SHA-256. A common literal
private-key/token-pattern scan found no matches. The scan does not constitute
a complete semantic security audit. This manifest does not cover dataset,
result, thesis-release or staging-metadata directories; those retain their
own commitments and provenance.

Targeted tests rerun at this checkpoint:

- Native60 scalar score kernel: six passed.
- Upload HTTP 429 backoff behavior: six passed.
- Five-hundred-file publication receipt validation: five passed.

These 17 tests validate their named components, not every historical script or
all remote uploads. The public-input Native60 report replay was separately
verified against all original output bytes; see
`current_results/NATIVE60_PORTABLE_REPLAY_ACCEPTANCE_20260912_EN.md`.

Code publication and audio publication are separate requirements. The latter
is still incomplete. No new experiment, classifier fit, audio generation or
threshold adjustment was performed by this delivery audit.
