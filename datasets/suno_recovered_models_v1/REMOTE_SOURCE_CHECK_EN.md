# Independent source-byte correspondence check

The 250 historical suno_unknown originals were checked anonymously against
Kukedlc/suno-ai-music-dataset at revision
`bdff424e70c10ef62dca13ba43659ad9e7e1fcdb`, using five batches of 50 exact
`audio/<source UUID>.mp3` paths. All 250 remote file sizes and LFS SHA-256 values
matched the freshly audited preserved originals. The check completed normally.

The corresponding pinned metadata contains these model_name counts:

| Source model label | Selected recordings |
|---|---:|
| chirp-fenix | 126 |
| chirp-crow | 114 |
| chirp-v3 | 3 |
| chirp-v4 | 3 |
| chirp-auk | 2 |
| chirp-v2 | 2 |

These are publisher-supplied model labels, not independently attested model
execution records. The source card's general V5.5 description must not replace
these row-level distinctions. The original experiment's unknown labels and
Suno grouping are retained. No fresh per-version performance is claimed.

This verifies the upstream source, not arrival in EZMONYI's archive. The local
catalogue COMMIT binds the exported CSV/README; this later supplementary check
is a separate report and is not included in that earlier product manifest.
