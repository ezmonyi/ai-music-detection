# Artifact recovery acceptance

## Preserved experiment history

Fresh SSH inspection confirmed that the server's CURRENT_STATE.md, RUN_LOG.md,
and COMPREHENSIVE_PROJECT_SUMMARY_EN_20260908.md exactly match their local
SHA-256 hashes in `../server_history/`. These are historical checkpoints, not
current process status. No raw conversation transcript was found in the
top-level project locations checked; this is not an exhaustive account-history
search and does not establish that deleted chat messages can be restored.

## Derived artifact inventory

`derived_artifact_inventory_v1/` contains the server inventory, terminal receipt
and local acceptance. Seven fixed YuE extension directories contain 11,900
file entries totaling 74,802,816,276 logical bytes. The copied inventory hash is
`f1ce7251de8a3b08f6fe1c0cb629c494cf3cbd354495dafcb135438e709eed8c`.
The local verifier checked manifest hash, unique relative paths, total and
per-directory counts, byte sums and extension counts. Three corrupted temporary
fixtures (hash, count and directory byte total) were rejected.

There are 10,875 distinct (SHA-256, byte-size) objects totaling 67,855,263,326
bytes, with 1,025 duplicate file entries. This is byte-object accounting, not
song deduplication. The local verifier did not reread remote audio; the producer
hashed it on the server. No audio was deleted, copied locally or published by
this inventory operation. It is not a whole-project inventory.

## Saraga originals

The server audit freshly read all 108 preserved MP3 originals and matched their
sizes and SHA-256 hashes to the physical-source contract: 3,956,076,186 bytes.
Exactly 103 IDs bind to the Native30 classifier cohort; five other preserved
recordings are not classifier entries. The local copy of `records.json` was
checked against SHA-256
`5982e37c320db8287cb78f25bad22b49c33a7441f4c7a57fa2f09cd449ea8925`,
with unique-ID, selected-count and aggregate-byte checks.

Records and completion receipt are in `saraga_original_audit_v1/`.
The source archive was not rehashed. Audio redistribution rights are not
established by this integrity audit, and no Saraga audio was uploaded.

## Delivery boundary

These checks preserve evidence and recover context. They do not complete the
remaining HF publications or establish complete local audio mirroring. Existing
upload processes were inspected, not restarted. Earlier experiment versions,
failed attempts and historical reports remain preserved.
