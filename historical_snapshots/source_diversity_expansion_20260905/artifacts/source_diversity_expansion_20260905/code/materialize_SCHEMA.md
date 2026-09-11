# Frozen v2 audio materialization contract

Heavy output root: `/mnt/nfs-data/users/yi/source_diversity_expansion_20260905`

- `materialization_manifest.jsonl`: atomic, de-duplicated latest record per item.
- `materialization_events.jsonl`: append-only attempt and error audit trail.
- `materialization_summary.json`: current totals, bytes, source counts, and schema.
- `native/<source_id>/`: exact downloaded files or exact embedded/archive members.
- `views_10s/<source_id>/`: frozen 10 s, 44.1 kHz, stereo, PCM16 FLAC views.
- `views_max60s/<source_id>/`: unpadded, at-most-60 s analysis views containing
  the frozen 10 s selection.
- `cache/archives/` and `cache/parquet/`: retained pinned source containers.

Consumers must select `status == "success"`. The frozen `role`,
`provenance_grade`, and `evaluation_allowed` fields are copied verbatim; in
particular, AudioX remains `provisional_development/no_until_verified` and
DiffRhythm remains `locked_pilot_only/pilot_only`.

Path/hash columns are `native_*`, `view_10s_*`, and `view_max60s_*`.
`crop_start_s` is the frozen 10 s anchor. `long_crop_start_s` is chosen so the
unpadded long view contains that anchor. `view_10s_padded_s` discloses any short
source padding; `view_max60s_padded_s` is always zero.
