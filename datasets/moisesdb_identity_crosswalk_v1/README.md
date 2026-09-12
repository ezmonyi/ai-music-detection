# MoisesDB mirror identity crosswalk

The historical mirror uses new UUIDs. None of its 239 selected IDs directly
matches the official htdemucs4 benchmark IDs. Its confusingly named medleydb_id
field provides 239 unique source IDs; 235 match that official benchmark's 235
IDs. Four are absent from that benchmark; absence is not evidence of invalid
audio. This is not a cross-dataset link to MedleyDB inferred from the field name.

Official benchmark is pinned to wearemusicai/moisesdb revision
3162378eb0653d9f9831a6051110822ab067b8b6, benchmark/htdemucs4.csv.
The mirror is seungheondoh/cmd-moisesdb-metadata at revision
353a59cfd90514ea71e35480b5ab8ffb57d14598.

This link improves source retrieval but does not recover artist names, song
titles, original audio equivalence or permission to republish. Keep experiment
IDs and historical grouping unchanged. Do not silently substitute source IDs
into already-run evaluations. Artist attribution remains unresolved.
