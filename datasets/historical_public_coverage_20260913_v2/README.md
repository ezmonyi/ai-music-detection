# Historical public-audio coverage after YuE upload completion

At HF revision `caed9220f68b258e9c2dcb89792ccb1fe4d1dc61`, 7,781 of
10,141 historical recorded source hashes match public direct audio objects.
The remaining 2,360 have no matching hash at that revision. None lack a
recorded source hash. This improves the preceding snapshot by 89 matches:
39 MusicNet Ishizaka recordings and 50 DiffRhythm pilot recordings.

| Source | Unmatched historical records |
|---|---:|
| MTG-Jamendo | 500 |
| AudioX third-party | 500 |
| DEAM | 300 |
| MusicNet | 291 |
| MoisesDB | 239 |
| MedleyDB | 178 |
| FMA | 108 |
| GTZAN | 100 |
| Hindustani Raag HF | 100 |
| URMP | 44 |

Unmatched does not mean lost: source preservation and public redistribution
are different requirements. These results do not grant redistribution rights.
The source-level rights/attribution reviews remain separate evidence.

`records.json` provides per-ID matches and public paths; `counts_by_source.csv`
and `COMMIT.json` bind counts, historical input hash and public snapshot hash.
The public snapshot contains 16,795 direct audio/MIDI paths and
130,247,953,279 logical bytes, including repeated versions. It excludes archive
members. Forty-three paths have no LFS hash (ordinary Git objects); absence of
an LFS hash is not proof those objects are corrupt or absent.

This audit compares recorded raw hashes only. It does not prove exact coverage
of every cropped input, stem, corrected waveform, external control, later-only
identity or early experiment. Do not present 7,781/10,141 as whole-project
delivery completeness. YuE and Saraga have separate accepted inventories.
