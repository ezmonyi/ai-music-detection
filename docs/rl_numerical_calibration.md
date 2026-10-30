# Independent cross-GPU numerical calibration

This is a **zero-optimizer-update study**, approved after the RTX3090 admission
failed. It does not activate a new training tolerance. The original `1e-6`
verdict, production options, cached states/actions and old likelihoods remain
unchanged. A proposed budget must be reviewed separately before formal GRPO.

The prior discovery used 64 transitions and cannot independently calibrate its
own acceptance rule. The new plan excludes portable archive block 0 and legacy
24-group shard groups 0 and 1. A fixed hash seed selects two other portable
25-group archives. Six calibration and six audit groups are selected per archive
and per legacy shard: **12 calibration + 12 audit groups per behavior**, 48
groups altogether. All four candidates and eight evenly spaced stochastic
timesteps are measured, including the stochastic-interval endpoints: 768
calibration and 768 independent-audit transitions. Each forward is repeated.
No validation/test audio or prompt enters numerical calibration; prompt,
caption and source identities are checked against the fixed heldout files and
against the opposite numerical-study split.

Before cross-GPU fitting, the engine generates **one native diagnostic
trajectory for each behavior** on the receiver, using a selected training
condition and a distinct fixed seed. These are two diagnostic trajectories,
not two complete GRPO groups or new heldout tests. It saves the raw native
audio and trajectory. Every stochastic step must pass the original `1e-6`
rule both on the staged collector and on the regular replay backend on the
same GPU. This distinguishes a collect/replay implementation mismatch from
device differences. Failure ends this study without proposing a new epsilon.

The study records numerical flags without changing them. Wrong behavior
fingerprints must be rejected, a deliberately altered in-memory variance must
fail the existing sampler validator, and shifted old log probabilities must
fail any proposed budget. Original files are never changed.

The precommitted ratio-uncertainty ceiling is 0.01%, corresponding to
`abs(delta_logprob) <= log1p(1e-4)` under the existing **dimension-mean**
surrogate convention. Candidate budgets are a fixed grid up to that ceiling.
The smallest candidate at least twice the calibration maximum can be proposed,
provided native and repeated-forward controls pass. `CANDIDATE_FROZEN.json` is
written **before the first audit forward**. Every audit observation must fit
that frozen budget; an audit failure cannot select a larger budget.

This error bound is not a theorem that optimization or audio quality is
unchanged. In particular, numerical uncertainty close to PPO clipping
boundaries can change which branch is active. The study supports a documented
proposal, not an exact joint-trajectory importance-sampling claim or universal
portability guarantee. Sparse group-level audit coverage is stated explicitly.

```bash
export PYTHONPATH=/path/to/repository/src
python -m music_detector.rl.numerical_calibration_cli plan \
  --options configs/rl/numerical_calibration_v1.json \
  --output /fresh/PRE_REGISTERED_SELECTION.json

# Read-only reanalysis of the already observed discovery data; no torch needed.
python -m music_detector.rl.numerical_calibration_cli analyse-discovery \
  --diagnosis /preserved/reports/NUMERICAL_DIAGNOSIS.json \
  --output /fresh/DISCOVERY_REANALYSIS.json

# Run only on the expressly provisioned RTX3090 with the locked cloud Python.
python -m music_detector.rl.numerical_calibration_cli run \
  --options /frozen/numerical_calibration_v1.json \
  --plan /frozen/PRE_REGISTERED_SELECTION.json \
  --runtime /frozen/runtime.json --base /mnt/ai-music-rl-20261003 \
  --export /mnt/upload-export-20261009 \
  --scratch /fresh/system-disk-scratch --output /fresh/persistent-study \
  --execute
```

The real runner requires Torch 2.10.0+cu128, native BF16 and the selected
RTX3090. The 3000-second timebox includes selected-archive verification and
measurements. A timeout saves failure/partial evidence and performs no update.
All new small outputs and the two native diagnostic trajectories/audio are
persistent. The approximately 7 GB restored archive subset is derived scratch;
the original 132 GB archive collection stays untouched. This is a selected
subset audit, not full-cache verification.

Existing snapshot 229669 already protects the unchanged runtime. No package,
model-weight or dependency change is part of this study; all new study code
and artifacts must be separately persisted before the new node is released.
The released previous node must never be contacted or restarted implicitly.

Reference: [PyTorch 2.10 numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html).
