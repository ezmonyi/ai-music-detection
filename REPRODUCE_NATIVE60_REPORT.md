# Reproduce the completed Native60 report

This is a portable, standard-library-only replay of the published prediction
report. It does **not** regenerate audio, extract features, refit classifiers,
or establish end-to-end reproducibility of every historical experiment.

Requirements: Python 3.11 or newer, network access to public Hugging Face files,
and approximately 60 MB of available disk space. Do not use Python `-O`: the
historical report runner uses assertions for its scientific integrity gates.
Run from this repository's root. Both destination directories must be new.

```sh
python3 current/yue2_500_extension_20260911/fetch_native60_replay_inputs_v1.py --destination ./replay-inputs
python3 current/yue2_500_extension_20260911/report_native60_transfer_v1.py --input-root ./replay-inputs --output ./replayed-native60-report
```

The downloader pins HF revision
`eb0e35208bb9229e1a2ee1a389e181ce7ddee8d6`, verifies the SHA-256 and byte size
of both public archives, rejects unsafe archive entries, and leaves existing
destinations untouched. It downloads result records, not music or model weights.

The reporter pins the original scoring COMMIT and cohort metadata, checks every
scoring product hash and size, then reconciles 876,300 persisted predictions
against 9,525 per-model/population summaries. It exports all 1,905 aggregate
subset/cap/population cells. No fitting or operating-point selection occurs.

Expected output SHA-256 hashes:

| File | SHA-256 |
|---|---|
| `all_subsets_caps_populations.csv` | `fbf495559da43be2fa878576236e530f666e7647aabb406546dc935deec6ea23` |
| `native60_transfer_tables.tex` | `31e152b47931e6970525e011d714d09a56735ba6145bc8ae479e076041526009` |
| `COMMIT.json` | `9487f6b5ac4bc21f113f1c426d8c38e9da51700467feaa0f5c9c7c9878c959ec` |

These files match `results/native60_transfer_report_v1/`.
The 276 recordings are all AI-generated and duration-selected: 223 development
and 53 previously locked recordings. The endpoint is AI sensitivity, **not**
balanced accuracy, human specificity or ROC AUC. M-containing combinations
are secondary historical diagnostics. This report must not be conflated with
the Native30 eight-family evaluation.
