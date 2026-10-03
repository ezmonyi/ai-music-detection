# Frozen reward selection for the A6000 pilot

The user explicitly requested the existing metric combination that best
separates AI and human recordings on the existing detector test results.
On 2026-10-03 the 127 deployable frozen combinations were ranked by the
experiment's primary class/source/group-weighted balanced accuracy (BA), not
ordinary recording accuracy or the result of a single-class YuE probe.

| Frozen combination | Internal weighted BA | AUC |
|---|---:|---:|
| S+R+F (selected) | 0.8034082812131541 | 0.8509064563860074 |
| S+R+P+F+H | 0.8019964770548711 | 0.8590008456030992 |
| S+R+P+F | 0.8004947076867599 | 0.8516940024537301 |
| S+F+SC (previous proposal) | 0.7539844149562520 | 0.8248410921424222 |
| F+SC (engineering-only template) | 0.7336879261681200 | 0.8011288735381129 |

This is selection on an already observed **internal** test partition of 840
recordings (327 human, 513 AI), drawn from the 4,228-recording retrospective
development experiment. It is not independent unknown-generator validation;
the maximum after searching combinations is an optimistically selected
reference statistic. The original reports remain unchanged and retain their
original no-winner-selection declaration; this new RL protocol is the later
selection experiment. No new labels, calibration fits, thresholds, or features
are changed. The new RL prompt test split remains untouched until the final
checkpoint is frozen. It is an engineering pilot, not cleared external thesis
evidence, because historical prompt/exclusion manifests are still unavailable.

Sources retained in this repository:

- `src/music_detector/assets/deployment_models.json`: bundle SHA-256
  `05dbe17a012bfec5391dd7c0b203549929e26d1f4612874332000403f71aaf53`;
  S+R+F envelope SHA-256
  `4b12f7319db0a846215445521161398d177168aadbe1012d9dbea8e7e8b4736c`.
- `validation/calibration/internal_test_report.json`: file SHA-256
  `e66dd7288e9b54fb47a4e24501e51b0e37bd916821f6848fc75c88e2d178442f`.
- `validation/calibration/drive_refit_report/internal_test_report.json`: the
  independently reproduced feature-level refit has the same S+R+F BA.

## Measurement and objective

S is the frozen Demucs-vocal/MUSDB-bias-corrected high-frequency spectrum;
R is frozen Beat This! beat-interval/tempo variation; F is fixed STFT
phase/group-delay evolution. S remains measured on every completion under
the detector's original definition. Low separated-vocal activity is recorded
and flagged; it is not evidence of audible singing. Instrumental prompts are
not silently relabelled or excluded, and residual vocal stems are not human
ground truth. Actual feature values, vocal-activity metadata and analyzer
receipts accompany rewards so coverage can be audited after training.

The objective is `-tanh(frozen_raw_ridge_score / 2)`, after the existing
silence/clipping/bandwidth/stereo and paired-baseline guards. The monotonic
Platt transform preserves score ordering but detector evasion is not proof
of improved music quality or human authorship. Baseline score differences
are diagnostics, never the group-relative training reward.

Training uses torch 2.10.0+cu128; analysis runs in an isolated subprocess with
the unchanged frozen torch/torchaudio 2.8.0 contract. A process receipt verifies
both sides' algorithm, worker and asset identities. CUDA allocations are
released when each worker exits. No analysis downloads or dependency setup
occur inside the reward call. The explicit analysis seed is 0 for paired
candidate/reference calls: this improves repeatability but is a newly declared
RL measurement setting, not bitwise historical unseeded Demucs replay.

## Completion and rented-node release gates

The user subsequently authorized autonomous completion while away:
finish the RL pilot, evaluate the held-out test split, save the MatPool `.snap`,
download test audio pairs and checkpoints locally, verify them, and only then
stop the exact rented node (`qKEVx1`, `hz-4.matpool.com:26617`).

Do not equate a process launch with successful training. Require actual finite
optimizer updates and the configured group budget, a completed test report,
a confirmed environment snapshot, and checked local SHA-256 copies. Keep
failed/intermediate results. Do not recharge, rent another node, alter frozen
guards retrospectively, touch unrelated instances, or stop the node before
the recovery gates pass. A normal Linux shutdown is not proof billing stopped;
the provider control-plane state must be verified after its stop operation.
