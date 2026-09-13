# Current nonvocal inventory review

The completed audit records 27,300 upstream nonvocal memberships. Its downloaded
JSONL matches terminal SHA-256
`34cfb5894eb3ce71828eea3b6d9c63ed88772f8606ca9fbecfb216298c50d9f2`.

- 27,174 current paths were readable and hashed, totaling 68,659,603,656 bytes
  before cross-inventory byte deduplication.
- 126 unanchored memberships correspond to 42 HeartMuLa IDs, three stems each.
  Exact set comparison confirms these are the upstream `new_features.csv` IDs
  excluded from the merged `metadata_30s.csv`. They are not missing-file findings.
- No attempted path was unreadable.

Paths were reconstructed as siblings of historically hash-verified vocals.
The old dynamics tables did not record per-stem hashes, and their directory
arguments have not been independently established. This is current-file
inventory evidence, not historical-byte integrity acceptance. Preserve the 42
upstream IDs as a separate historical scope; do not silently discard them.
Publication and cross-inventory reconciliation remain separate work.
