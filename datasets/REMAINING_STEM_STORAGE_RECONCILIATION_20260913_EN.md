# Remaining stem preservation scope — 2026-09-13

Exact SHA-256 set reconciliation gives 14,956 historical stem objects and
23,776 current-only stem objects unmatched to the inspected public/private HF
direct-file inventories. The sets are disjoint: 38,732 unique byte objects.
Their recorded sizes total 101,495,208,208 bytes (approximately 101.50 GB).

Neither set intersects the 4,484-object local rights-held archive manifest.
That accepted 9.22 GB archive therefore does not reduce this remaining scope.
The 19 inspected historical result archives had no recognized audio/MIDI file
members; nested container payloads were not recursively inspected.

Evidence: `private_stem_reconciliation_v1/COMMIT.json` and
`unmatched_current_hashes.json`; repository dataset records under
`stem_public_snapshot_match_v3/`; local
`../private_rights_hold_4484_v1/manifest.json`.

This is an unmatched-backup scope, not a missing-source-file assertion or
permission to publish. Current-only files also lack historical input-hash
provenance; preservation does not establish that experiments used them.
Other uninspected backups may cover some objects. No new audio was uploaded,
deleted, or downloaded by this reconciliation.

The local report volume had approximately 26 GiB available at the preceding
filesystem check. It cannot hold a full additional copy of this scope. A new
private storage destination is required if the whole scope must be copied
locally. Public redistribution remains subject to source-specific permissions.
