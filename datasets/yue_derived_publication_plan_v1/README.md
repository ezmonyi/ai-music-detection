# YuE derived-audio publication plan

The pinned seven-directory inventory contains 6,922 audio path memberships
bound to the 500 published YuE original IDs. Exact SHA-256/size deduplication
produces 5,900 byte objects (56,849,461,990 bytes), not 5,900 recordings.

Of these, 1,000 objects belong to the separately running legacy Demucs upload.
The remaining 4,900 objects total 46,265,373,990 bytes and are not yet uploaded
by this plan. `memberships.json` retains every original relative path, ID, role
and group; `objects.json` supplies proposed public object paths and all aliases.
The final uploader must rehash source files and independently verify arrival.
No object is marked publication-verified here, including the running batch.

The 1,000 non-YuE audio memberships are retained in a separate exclusion
manifest for source-specific review, not discarded or authorized for upload.
This fixed-scope plan does not cover every historical project directory.

Validation: all 6,922 member paths occur exactly once in the object aliases,
all 500 original IDs remain represented, and all 5,900 proposed paths are unique.
The source inventory and original manifest are SHA-256 pinned in COMMIT.json.
