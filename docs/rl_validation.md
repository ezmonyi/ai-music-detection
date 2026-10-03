# RL preparation validation — 2026-10-01

This report records **software validation**, not an ACE-Step training outcome.
No pretrained generation weights, neural analysis weights, or music recordings
were downloaded for these tests. No GPU was rented and no real ACE training ran.

## Verified locally

Environment: macOS, Python 3.12.14, PyTorch 2.8.0, CPU; existing project virtual
environment with the new worktree's `src` first on `PYTHONPATH`.

```bash
PYTHONPATH=src python -m pytest -q
```

Result: **141 passed**, one pre-existing Starlette/AnyIO deprecation warning,
8.11 seconds in the final full-suite run. Tests include existing detector/API
regressions as well as the new RL tests.

| Check | Observed result | Boundary |
|---|---|---|
| Gaussian transitions, density, KL, zero-variance endpoints | Tests pass | Numerical helpers, not audio quality |
| LoRA/full toy training | Both update trainable parameters; frozen weights preserved | Tiny CPU model |
| Resume | Uninterrupted vs 1-group + resume yields identical state tensors | Same CPU environment and source/config/data hashes |
| Holdout control | Training caption/source overlap rejected | Exact/dependency keys, not semantic similarity |
| Invalid/constant rewards | All-invalid groups stop; constant groups skip updates | Does not validate gate thresholds for real ACE audio |
| ACE condition/velocity/decode boundary | Mocked shape, backward, precision/reference and loading tests pass | Real pretrained ACE and CUDA kernels not exercised |
| Artifact bridge | Actual F + SC extractor admitted a synthetic 30-second native-stereo signal | No Demucs/BeatThis/AllInOne weights used |
| Prompt builder | Determinism, split counts, exclusions, language/lyric preservation, overwrite refusal pass | No new 500-prompt scientific set finalized |
| Ablation planner | 45 configurations produced, none launched | Hyperparameters are starting grids |

The actual F+SC reward integration yielded raw score `0.2662668935`, reward
`-0.1323524085`, and paired delta `0` for identical candidate/base input.
All 15 F and 6 SC features were finite. This verifies the bridge to the frozen
detector; the synthetic signal is not music and the score has no authorship
interpretation here.

## End-to-end CLI fixture

Using `configs/rl/toy_smoke.json` and the authored toy JSONL fixtures:

- Three prompt groups completed, three optimizer updates, no skipped groups.
- Gradient norms: approximately 0.02127, 0.01631, 0.01363.
- Replayed old/current log densities matched exactly before each update.
- A separate held-out ODE evaluation completed with one valid synthetic pair.
- Config, source/data hashes, parameter dtypes, logs, checkpoints and paired
  synthetic audio were written under ignored `.rl-runs/toy-validated/` and
  `.rl-runs/toy-validated-eval/` in the development worktree.

Toy reward movement is intentionally **not reported as evidence of improved
music**. The run is only an executable check of the policy-gradient pipeline.

## Not yet validated

1. Loading the complete pinned ACE SFT/VAE/text weights in a real CUDA environment.
2. Native ACE waveform admission rates, all-family reward dependencies and exact
   GPU rollout/replay equality under mixed precision/checkpointing.
3. Peak VRAM, throughput, cost, or fit on any specific GPU including RTX 5060.
4. Training on a cleared 500-prompt split after historical/held-out exclusions.
5. LoRA versus full learning curves, generalization, quality/adherence, diversity,
   VAE reconstruction controls, or blinded listening improvements.

The next execution step is a **single-group real-GPU pilot**, not the 45-run grid.
Preserve its failures and admission/timing/VRAM diagnostics, then decide whether
to proceed or revise the protocol. The existing thesis conclusions are unchanged.

## TensorBoard and A6000 preparation follow-up — 2026-10-03

The historical 141-test report above is preserved. The current full suite passes
**178 tests** (9.90 seconds), with the same Starlette/AnyIO deprecation warning.
Runtime: macOS/Python 3.12.14, CPU PyTorch 2.8.0, TensorBoard 2.20.0 and
setuptools 80.10.2 in an isolated `.tb-venv`. Existing local scientific packages
are reused without changing the running detector application's environment.

The follow-up adds text-only MusicCaps engineering-pilot preparation and
TensorBoard monitoring; neither is a cleared external music test or a GPU result.
Detailed metric definitions/remote startup are in [rl_tensorboard.md](rl_tensorboard.md).

Verified monitoring contracts include real event files/scalars, resumed absolute
group steps, policy/weighted-KL loss decomposition, held-out admitted-reward loss,
no fabricated loss for all-invalid evaluation, explicit test namespace, validation
provenance/ID/caption/source overlap guards, and pinned validation SHA/IDs. Writer
initialization and close failures are recorded without hiding model failures. Periodic validation
preserves policy/optimizer/Torch RNG versus a no-validation run. Decoder train
flags, NumPy/Python/Torch RNG are restored even when evaluation raises. GPU
current/peak/capacity tag mapping is tested with mocked CUDA calls; CPU runs do
not call CUDA memory APIs or create CUDA scalar events.

The final CLI fixture used `configs/rl/toy_monitoring.json`: three optimizer
updates and three periodic held-out evaluations completed. Standalone paired
ODE validation also completed at checkpoint group 3. Final ignored output paths:

- `.rl-runs/toy-tensorboard-final-20261003/`
- `.rl-runs/toy-tensorboard-final-eval-20261003/`

TensorBoard was started privately on `127.0.0.1:6006`; its HTTP scalar API returned
the expected loss/gradient/LR/throughput event series. Older preview directories
are retained as intermediate software-validation outputs. No pretrained model
weights/audio downloads, GPU rental, or real ACE music training occurred.

The A6000 template now enables a fixed 10-prompt ODE validation subset every 10
groups and at the full configured budget, with separate validation input required.
Formal GPU readiness and the historical/held-out prompt-clearance work remain
unchanged and outstanding.
