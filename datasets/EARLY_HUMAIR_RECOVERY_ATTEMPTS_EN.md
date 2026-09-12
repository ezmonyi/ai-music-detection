# Early humair metadata recovery attempts

The preserved local source metadata tree contains README, dataset_info.json
and state.json files but no Arrow data files. All 100 selected original MP3s
remain preserved and were already freshly hashed in the early-original audit.

The public HF repository API remains available at revision
344c67dd2992063779b8f40504ff112f6083e7f2. The splits endpoint reports default/train.
A one-row viewer request succeeded, reported 1,206 viewer rows and exposed id,
title, display_name, handle, model_name and other fields. This is not evidence
that the viewer covers the card's stated 49,698 recordings.

A filtered UUID query timed out. The first metadata recovery script attempt
failed with SSL UNEXPECTED_EOF_WHILE_READING. A second attempt failed at the
first 100-row request with HTTP 500. No completed recovery output was produced.
No audio, prompts, lyrics or signed media URLs were saved by these requests.

The selected 100 local files span batches 0, 1, 10, 11, 12, 13, 14, 16 and 17.
Attribution recovery remains unresolved. Do not infer that a nonempty or partial
viewer response establishes coverage of those 100 IDs, or that this attempt
published their audio to the project dataset.
