# Native60 portable report replay acceptance

The report runner now accepts `--input-root` and `--output`, while retaining the
historical default paths and original scoring COMMIT. Cohort metadata is also
bound to its published SHA-256. Existing output directories are refused before
any report writing. Numerical aggregation, subset definitions and thresholds
are unchanged.

Two actual complete replays succeeded:

1. Existing locally preserved scores and metadata, output in
   `native60_portable_replay_20260912/`.
2. Inputs freshly downloaded anonymously from the pinned public HF revision,
   output in `native60_public_replay_report_20260912/`. Downloaded and hash-verified
   inputs and receipts remain in `native60_public_replay_inputs_20260912/`.

Each run reconciled 876,300 predictions and 9,525 per-model/population summaries
and regenerated all 1,905 aggregate cells. Both outputs' CSV, LaTeX and COMMIT
were byte-identical to the preserved original report. Negative checks rejected
an existing output directory and a corrupted temporary metadata fixture; no
output directory was created for the corrupt fixture.

| Product | SHA-256 |
|---|---|
| CSV | `fbf495559da43be2fa878576236e530f666e7647aabb406546dc935deec6ea23` |
| LaTeX | `31e152b47931e6970525e011d714d09a56735ba6145bc8ae479e076041526009` |
| COMMIT | `9487f6b5ac4bc21f113f1c426d8c38e9da51700467feaa0f5c9c7c9878c959ec` |

Repository instructions: `REPRODUCE_NATIVE60_REPORT.md`. Implementations are
`current/yue2_500_extension_20260911/fetch_native60_replay_inputs_v1.py` and
`report_native60_transfer_v1.py`.

Scope: report reproduction from persisted predictions, not new audio generation,
feature extraction or numerical model replay. This does not certify whole-project
portability. The single-class sensitivity endpoint remains distinct from
balanced accuracy, specificity and ROC AUC. No scientific results changed.
