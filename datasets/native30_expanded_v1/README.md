# Native30 expanded dataset catalogue v1

This is a verified index of the completed expanded Native30 feature cohort,
not the entire historical corpus and not a claim that all audio is uploaded.
It contains 4,228 development rows and 100 locked YuE2 rows (4,328 unique IDs).
Label 0 means source-labelled human music; label 1 means source-labelled AI.
Labels describe source provenance, not a claim of forensic ground truth for
every recording. Shared source-song identities must not cross training/test
boundaries. Preserve both group_id and component_id when reproducing splits.
The development role denotes eligibility for development protocols, not one
universal training partition. Use the frozen per-protocol schedules.

The input hash identifies the analysis view, not necessarily the original
recording or any HF object. Absolute server paths and source receipts have
been omitted. This catalogue does not grant redistribution rights or imply
that every dataset member is public-domain material.

| Source | Label | Role | Rows |
|---|---:|---|---:|
| ACE-Step | 1 | development | 400 |
| FMA | 0 | development | 354 |
| HeartMuLa | 1 | development | 366 |
| MTG-Jamendo | 0 | development | 500 |
| Mureka_v9 | 1 | development | 500 |
| Suno | 1 | development | 400 |
| Udio | 1 | development | 500 |
| YuE2 | 1 | development | 398 |
| YuE2 | 1 | locked_test | 100 |
| human_maestro_v3 | 0 | development | 300 |
| human_medleydb | 0 | development | 168 |
| human_moisesdb | 0 | development | 239 |
| human_saraga_hindustani_v1 | 0 | development | 103 |

## Audio and scope

The audio delivery destination is
https://huggingface.co/datasets/EZMONYI/music-ai-human-test-audio .
Only per-file publication receipts establish uploaded content. At this
checkpoint the verified audio publication covers 300 MAESTRO originals and
300 derived views, representing the same 300 works, not 600 unique songs.
The remaining source audio needs source-specific rights review and upload.

YuE2 generation produced 500 recordings. Two were too short for Native30;
398 eligible rows are development and 100 are locked. This catalogue contains
498 YuE2 rows, not all generated audio. Historical spectral and native60
cohorts have different eligibility and are not interchangeable with this one.
