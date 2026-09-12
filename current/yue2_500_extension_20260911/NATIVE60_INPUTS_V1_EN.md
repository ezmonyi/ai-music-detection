# YuE2 native60 input preparation

This step prepares inputs for the remaining exact-duration extension. It does
not run a neural model, extract features, fit a classifier or score a test set.
The full frozen 500-prompt manifest is checked; the earlier native-duration
inventory predicts 276 eligible recordings, but completion requires validating
every original against its generation receipt and reaching a terminal COMMIT.

At native 48 kHz, the crop contains 2,880,000 frames and begins at
`floor((native_frames - 2,880,000)/2)`. Shorter files are recorded as excluded;
there is no padding, resampling, downmix, normalization or gain limiting.
Stereo float32 crops are stored as FLOAT WAV and read back for exact numerical
equality. The source is hashed before and after processing. Input metadata
retains the generation role and `muse:` plus the original `source_song_id` as
the group, not an independently generated group inferred from filename text.

The crop-boundary helper was checked on one sample below threshold, exact
threshold, and an odd surplus of three samples. These three checks are not a
full independent audit of the real output. The final acceptance must check
all output hashes, crop identity, exclusions and role counts.

Native60 inputs use a separate output directory and do not alter Native30,
legacy first30, generation outputs, or the completed classifier catalogues.
The current runner is `prepare_yue2_native60_v1.py`; PID 3085416 was observed
live after launch on 12 September 2026. Server output:
`/mnt/nfs-data/users/yi/yue2_500_extension_20260911/native60_inputs_v1`.
Log: `/mnt/nfs-code/users/yi/yue2_500_extension_20260911/native60_inputs_v1.log`.

Remaining steps include input acceptance, duration-compatible neural output
validation, feature extraction, group-aware comparative evaluation and report
integration. Native30 validators cannot simply be used as native60 validators.
