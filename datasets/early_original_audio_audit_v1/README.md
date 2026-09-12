# Early original audio recovery

All 400 distinct relative source paths referenced by the early frequency study
exist in the preserved server workspace. The selected 20-file stem pilot uses
a subset of those paths, giving 420 experiment memberships rather than 420
distinct source files. The 400 files total 1,114,016,270 bytes and have 400 unique
SHA-256 values: 200 FMA, 100 humair Suno and 100 other historical Suno files.

Each file was freshly hashed, with size and modification time checked before
and after reading. The copied originals.json was verified against its terminal
manifest hash, file count, aggregate bytes and membership count. No historical
expected raw hash was available in the three early manifests: this proves
present-day preserved bytes at those locators, not an exact historical-byte
comparison or a new measurement replay.

None of these fresh hashes matched the raw_sha256 field of the 10,141-row
historical-input index. This comparison does not establish musical-content
independence, because alternate encodings can have different hashes. No dataset
labels, splits or model metrics were changed. Publication and source-license
reconciliation for this additional early scope remain pending.

The source input memberships are in datasets/early_experiment_inputs_v1 within
the project report/repository. This README is explanatory and was added after
the original audit COMMIT, which binds originals.json rather than this README.
