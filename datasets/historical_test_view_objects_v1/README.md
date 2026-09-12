# Deduplicated historical test-input publication gap

The actual-input audit read every one of 14,246 distinct filesystem paths.
Across 14,638 10s/30s memberships, exact SHA-256 deduplication yields 14,244
unique byte objects. At fixed HF revision
`caed9220f68b258e9c2dcb89792ccb1fe4d1dc61`, 966 objects match public hashes
and 13,278 objects do not, totaling 19,083,227,708 unpublished bytes.

These are processed file objects, not independent recordings. The earlier
13,480 unmatched figure counts memberships, not unique bytes. No unreadable
paths were found. This audit excludes separated stems unless directly named
by audio_path, and does not materialize crops applied at runtime.

`objects.json` retains all source paths, exact ID/view memberships, crop
parameters and existing public matches. `COMMIT.json` binds its SHA-256 and
the original audit, and gives per-source object counts and sizes. Per-source
counts need not be additive if byte-identical objects cross source labels.

Publication authorization remains false in this inventory: source-specific
derivative licensing and attribution must be attached before publication.
This flag is not a conclusion that all objects are legally restricted.
Priority candidates include 600 MAESTRO objects, 100 DiffRhythm pilot objects,
the Ishizaka-linked subset, and 800 additional ACE-Step/HeartMuLa objects;
each still needs exact source membership and publication-notice binding.
The existing original-release acceptance is not an acceptance of these bytes.
