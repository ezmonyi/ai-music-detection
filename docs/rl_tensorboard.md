# ACE-Step RL TensorBoard monitoring

Status: implemented and CPU/toy validated. CUDA memory collection is wired and
unit-tested with mocked CUDA readings; no A6000 training/VRAM profile is claimed.
The dashboard is diagnostic, not evidence of improved perceived music quality.

## Setup and launch

Install the RL extra in the dedicated training environment (it pins
`tensorboard==2.20.0` and `setuptools>=70,<81` to retain the dashboard's
`pkg_resources` runtime dependency):

```bash
python -m pip install -e '.[rl,test]'
tensorboard --logdir .rl-runs --host 127.0.0.1 --port 6006 --reload_interval 2
```

Use http://127.0.0.1:6006 locally. For a remote GPU host, keep TensorBoard bound to
loopback and use an SSH tunnel from the viewing computer:

```bash
ssh -N -L 6006:127.0.0.1:6006 -p PORT USER@HOST
```

Do not expose an unauthenticated TensorBoard port publicly. The server has no
need for model weights, HF credentials or proxy/VPN changes.

All scalars use **completed prompt groups** as the x-axis, not solver timesteps or
candidate count. `train/optimizer_updates` distinguishes executed optimizer
steps from zero-advantage groups. Resume writes into a new output directory and
continues the original absolute group index; select the partial and resumed runs
together in TensorBoard. An event stream is under `OUTPUT/tensorboard/run-*`;
existing machine-readable JSON outputs remain authoritative and are preserved.

## Required metrics and their definitions

| Requested measurement | TensorBoard tags | Definition |
| --- | --- | --- |
| Train loss | `train/loss`, `train/policy_loss`, `train/weighted_kl`, `train/reference_kl` | Replay policy surrogate + KL coefficient × raw reference KL, normalized by scored transitions. Not supervised audio reconstruction loss. |
| Eval loss | `eval/loss`, `eval/reward_mean`, `eval/valid_count`, `eval/valid_fraction` | **Negative mean admitted held-out reward** from same-seed candidate/frozen-reference ODE pairs. Not a held-out GRPO loss and not scale-comparable to train loss. All-invalid evaluation has JSON null and no loss event, with coverage zero. |
| Gradient norm | `train/grad_norm` | Global trainable-parameter L2 norm **before** clipping; `max_grad_norm` is the configured clipping threshold. Zero indicates an explicitly skipped zero-advantage group. |
| Learning rate | `train/learning_rate` | Actual AdamW param-group LR, not a copied schedule estimate. The current trainer has a constant LR. |
| Throughput | `throughput/candidate_clips_per_second`, `throughput/generated_clips_per_second`, `throughput/audio_seconds_per_second`, `throughput/groups_per_second` | Group size / measured training-group wall seconds; generated clips include both candidates and paired references (2 × group size). Audio throughput uses their configured duration. Group time includes conditioning, reward computation, paired audio writes and checkpoint writes, but excludes periodic validation and event flush. |
| GPU memory | `system/cuda_allocated_gib`, `system/cuda_reserved_gib`, `system/cuda_peak_allocated_gib`, `system/cuda_peak_reserved_gib`, `system/cuda_device_used_gib`, `system/cuda_device_free_gib`, `system/cuda_device_total_gib` | Current/peak PyTorch allocations and cache, plus device-wide used/free/total, in GiB. CPU runs emit no fabricated CUDA zeros. Peaks cover the current process/run invocation, not the whole historical experiment. |

Additional tags include reward mean, admission rate, absolute old/new log-density
difference, cumulative optimizer updates, and per-phase timings. The
`throughput/end_to_end_groups_per_second` denominator also includes periodic
validation (but excludes the current group's event flush). The JSON
`whole_group_seconds` uses the same end-to-end denominator.

Training deliberately includes invalid-candidate penalties when calculating
group advantages. In contrast, validation loss averages **admitted** candidates
only. Always inspect coverage and saved rejected pairs alongside eval loss;
improving the valid-only average while losing admission coverage is not success.
Group-centered on-policy advantages can make train loss nearly zero despite a
nonzero gradient. Reward/coverage/KL and blinded listening matter more than
monotonic train-loss descent for this protocol.

## Periodic held-out validation

`configs/rl/acestep_a6000_lora_pilot.json` enables TensorBoard, validation every
10 groups, and a fixed subset of 10 prompts. Pass the separately prepared
validation JSONL:

```bash
PYTHONPATH=src python -m music_detector.rl.cli train \
  --config /path/to/resolved_a6000_config.json \
  --data .rl-data/musiccaps_a6000_curated_500_20261003/train.jsonl \
  --validation-data .rl-data/musiccaps_a6000_curated_500_20261003/validation.jsonl \
  --output .rl-runs/a6000-pilot
```

The subset is selected by hashed prompt ID, independent of file order, and its
IDs/full validation file SHA are pinned in the manifest and resume boundary.
Prompt-ID/caption/source overlap, wrong split, and duration mismatch fail before a GPU
backend loads. Evaluation reuses the existing policy/reference, changes no
optimizer state, and restores decoder train flags and Torch CPU/all-CUDA,
NumPy, and Python RNG states. It runs at the configured cadence and full group
budget, not at every short budget-slice boundary. No test prompts enter this
periodic path. Standalone `evaluate --split test` uses separate `test/*` tags.

Generic `acestep_lora.json` and `acestep_full.json` enable events but leave
periodic validation disabled (`evaluation_every=0`) for compatibility with old
commands. Set it to a positive integer and provide `--validation-data` to add
live eval. Older configs without a monitoring block remain logging-disabled.

Explicit final evaluation can use all 50 validation rows and produces its own
event stream at the checkpoint's absolute group step:

```bash
PYTHONPATH=src python -m music_detector.rl.cli evaluate \
  --config /path/to/resolved_a6000_config.json \
  --data .rl-data/musiccaps_a6000_curated_500_20261003/validation.jsonl \
  --checkpoint .rl-runs/a6000-pilot/checkpoints/group_000100.pt \
  --output .rl-runs/a6000-validation-final
```

## Safe local CPU preview

```bash
PYTHONPATH=src python -m music_detector.rl.cli train \
  --config configs/rl/toy_monitoring.json \
  --data configs/rl/toy_train.jsonl \
  --validation-data configs/rl/toy_validation.jsonl \
  --output .rl-runs/toy-tensorboard-preview
```

This executes three real toy optimizer updates and three held-out toy
evaluations. It downloads no audio or model weights and has no music-quality
meaning. Use another output name if it exists. GPU tags appear only after an
actual CUDA run.

Implementation follows the [PyTorch SummaryWriter API](https://docs.pytorch.org/docs/2.8/tensorboard.html).
