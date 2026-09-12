# Historical recorded-source hash coverage

At public revision `b7e95fe5e708628cce259a04e29a767b2cae3dad`, 7,692 of the
10,141 historical IDs have their recorded raw SHA-256 present as a direct
public audio object. The other 2,449 do not match this snapshot. No historical
row lacks a recorded hash. See `counts_by_source.csv` and exact ID/path joins
in `records.json`; all input hashes and revision are recorded in COMMIT.json.

Missing recorded hashes by source: MTG-Jamendo 500; AudioX third-party 500;
MusicNet 330; DEAM 300; MoisesDB 239; MedleyDB 178; FMA 108; GTZAN 100;
Hindustani HF 100; DiffRhythm pilot 50; URMP 44. Source conditions and rights
must be resolved separately. An absent match is not permission to publish.

This check does not inspect compressed archive members, prove the exact
standardized inputs/stems are present, or include later-only Mureka/YuE/Saraga
identities and early-only inputs. Some historical raw hashes identify a saved
view rather than the original full recording. These are identity rows, not
content-deduplicated song counts. Concurrent later uploads do not change this
immutable checkpoint; final coverage requires a fresh fixed revision.

The public inventory was fetched anonymously and every listed audio path had
LFS SHA-256 metadata. This is a metadata equality join, not full audio download.
