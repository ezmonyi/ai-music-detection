# Four-stem historical integrity acceptance

All 22,320 recorded memberships in the three selected runs passed fresh SHA-256
comparison. There were 22,320 distinct paths and byte objects, with no missing,
unreadable, unrecorded or mismatching entries.

Audit records SHA-256:
`7b311d0a2dfd3c8aab3079b04ee377612bf715619f2c5c08ed00b8091f5cedfc`.
The downloaded JSONL was checked against the terminal COMMIT, and the downstream
inventory required all row hashes and status counts to agree.

Inventory `../../datasets/historical_four_stem_objects_v1/` contains 22,320 objects,
75,203,830,080 bytes. None matched the immutable public snapshot
`caed9220f68b258e9c2dcb89792ccb1fe4d1dc61`.
Object inventory SHA-256:
`fd17c91421ae9693a8a15ba4eae5c3959dfcca3524a30861366ae864dc51a292`.

This is integrity acceptance, not publication acceptance. Snapshot absence is
not a current-HF audit or publication permission. Cross-inventory deduplication
with spectral-only vocals remains necessary. Earlier nonvocal inputs without
historical byte hashes and other experiment scopes are separate.
