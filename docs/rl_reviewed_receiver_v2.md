# User-reviewed receiver protocol v2

The v1 numerical study failed its precommitted budget. Its report and all raw
measurements remain immutable under `reports/rl/numerical_calibration_20261031`.
After reading that result, the user approved a separate fixed `3e-4` audit on
24 unused train prompt groups, all four candidates and all 31 stochastic times,
and authorized formal RL only after this independent audit passes. This is a
new receiver-specific numerical protocol, not a relabeling of v1 as successful.

`music_detector.rl.reviewed_receiver issue` recomputes the independent audit
statistics and coverage from all 2,976 observations, checks strict native
controls, the actual user-reviewed plan SHA, corrupted-likelihood controls and
actual receiver numerical flags. It refuses a failed/incomplete audit. The
resulting profile binds every evidence-file SHA and the current host/flags.

The `train` entry point validates that profile on each start/resume and keeps
an immutable `.NUMERICAL_PROTOCOL.json` beside each run directory. Its
configuration checker pins the original offline objective and all drift/quality
settings. Only the reviewed replay tolerance (`1e-6` to `3e-4`) and the stronger
initial probe coverage (4 to 31 stochastic steps) differ. The original
`configs/rl/offline_1000.json`, core replay default and sampler stay unchanged.
Original cached behavior probabilities remain the denominators; they must not
be replaced with receiver recomputations. The existing core manifest's
`guards_changed: false` refers to unchanged objective/drift/quality guards;
the sidecar explicitly records the separately reviewed numerical gate change.

```bash
python -m music_detector.rl.reviewed_receiver issue \
  --audit-root /persistent/v2-audit \
  --previous-root /persistent/v1-study \
  --authorization /persistent/PROTOCOL_V2_AUTHORIZATION.json \
  --output /persistent/REVIEWED_RECEIVER_PROFILE.json

python -m music_detector.rl.reviewed_receiver train \
  --profile /persistent/REVIEWED_RECEIVER_PROFILE.json \
  --original-options configs/rl/offline_1000.json \
  --options configs/rl/offline_1000_rtx3090_v2.json \
  --index /verified/INDEX.json \
  --checkpoint /original/group_000100.pt --checkpoint-sha256 VERIFIED_SHA \
  --validation-data /frozen/validation.jsonl --test-data /frozen/test.jsonl \
  --runtime /frozen/runtime.json --output /fresh/two-update-smoke --max-updates 2
```

Resume the same offline checkpoint and unchanged 1,000-update options into a
fresh output directory. Resume requires the identical numerical-profile
sidecar in addition to the core optimizer/RNG/cursor/data/code boundary. There
must be 1,000 additional **real optimizer updates**, not 1,000 visited groups;
no additional epochs or recycled data may be introduced to reach that number.

The final fixed 50-prompt comparison uses identical seeds/sampler for the new
offline-RL checkpoint and official SFT checkpoint. The existing evaluation also
retains the online initialization as a third branch. Deliver the 50 requested
base/candidate pairs and metrics under `~/Downloads`, with member checksums.
This previously inspected prompt set is a longitudinal comparison, not a new
blind generalization test. Keep source data, intermediate reports and all
failed/invalid candidates, and verify persistent artifacts before node release.
