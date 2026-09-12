# Early humair attribution recovery v2

All 100 early humair Suno original-file UUIDs were found in the pinned upstream
nyuuzyou/suno Parquet. The source revision and complete Parquet SHA-256 are bound
in COMMIT.json. The source has 659,788 rows; only selected attribution fields
were read into the recovered output. Lyrics, prompts and artwork are excluded.

The 100 UUIDs matched 132 source rows. Identical attribution records were
deduplicated within each UUID. One UUID has multiple distinct title/creator
variants; all variants remain in metadata_variants and none is silently selected
as the sole historical value. Every UUID has at least one creator display name
or handle. Original file hashes and historical memberships remain attached.

The initial recovery attempt rejected duplicate UUIDs before writing output.
After inspecting which fields differed, the implementation was corrected to
preserve variants. Previous viewer failures remain documented separately.
The local records were checked against COMMIT.json, with 100 unique IDs, one
multi-variant identity and no missing creator attribution.

This is an exact UUID provenance join, not a claim that the recovered metadata
is byte-identical to the now-missing historical Arrow metadata. Audio publication
has its own completion receipt; this recovery COMMIT alone is not upload proof.
