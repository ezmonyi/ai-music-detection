# Independent cross-GPU numerical calibration, 31 October 2026

**Outcome: v1 completed, no audited replacement budget recommended. Zero new optimizer updates; no production tolerance activation and no new post-RL test audio.** This is a completed numerical study, not completion of the pending 1,000-update experiment.

## Controlled measurements

RTX3090 node 2128889 (ZW703Y), Region 1, CNY1.85/hour, restored snapshot 229669. The locked Torch2.10.0+cu128 runtime, native BF16 backbone, FP32 LoRA/states, original sampler, variance convention, dimension-mean log density, model/VAE weights and original cached old probabilities were retained. No dependency was installed or changed.

A hash-seeded plan selected 48 training prompt groups, excluding the previous discovery groups: portable archives 19 and34, and legacy24/26. Each behavior contributed 12 calibration and 12 audit groups. Four candidates and eight evenly spaced stochastic times per group yielded 768 calibration plus 768 independent-audit observations. Each velocity/logprob forward was repeated. Heldout validation/test prompt, caption and source identities were excluded. The selected TARs and all selected group members were verified; the entire132GB cache was not rehashed.

Two native diagnostic trajectories were generated on this receiver. Each had31 stochastic steps; all 31 replayed exactly on both the staged collector and regular backend for both behavior policies: **maximum absolute logprob error0** (62 unique transitions, checked through both paths). These were diagnostic training-condition trajectories, not post-RL heldout pairs.

Every repeated forward in the1,536 cross-environment observations also had velocity error0 and logprob error0. Variance corruption was rejected; wrong behavior fingerprints differed and were rejected by the identity requirement; all 24 shifted-logprob controls were rejected. This establishes deterministic controls for the observations, not universal platform determinism or isolation of a particular CUDA kernel.

| Split / source | Steps | Median abs error | P95 | Maximum | Pass `1e-6` |
|---|---:|---:|---:|---:|---:|
| calibration/g100 | 384 | 8.9556e-06 | 4.45068e-05 | 8.59499e-05 | 21/384 |
| calibration/g10 | 384 | 9.14931e-06 | 4.21703e-05 | 9.60827e-05 | 25/384 |
| audit/g100 | 384 | 8.35955e-06 | 4.94182e-05 | 0.000103474 | 21/384 |
| audit/g10 | 384 | 9.41753e-06 | 4.46796e-05 | 0.000116467 | 25/384 |

Only **92/1,536** observations passed the original1e-6 threshold. The audit maximum was **0.000116467475891**, giving a measured dimension-mean surrogate ratio uncertainty bound of **0.01164743%**. 3 audit observations also exceeded the precommitted ceiling log1p(1e-4). Observations within each prompt group are correlated;1,536 is not an independent-sample count for a universal portability claim.

## Why v1 did not promote a new tolerance

The fixed candidate grid ended at log1p(1e-4), a0.01% surrogate-ratio error budget. The rule required a candidate at least twice the calibration maximum. Twice9.608268737792969e-5 exceeds that ceiling, so CANDIDATE_FROZEN.json correctly stored null **before** any audit forward. Audit data were retained and reported without enlarging the grid or modifying the original failure. The ratio ceiling is a conservative study design choice, not a theorem that a larger deviation makes RL impossible.

The broader timestep coverage revealed a substantially larger tail than the previous64-step discovery. ERROR_BY_STOCHASTIC_TIME.csv records the measured dependence. Native controls support a cross-environment numerical difference in this setup rather than a demonstrated staged/regular replay implementation mismatch. They do not identify a sole hardware or library cause.

## Concrete proposal for review, not activated

A possible **separate v2** would freeze an absolute logprob budget3e-4 (ratio uncertainty ceiling0.0300045%), rounded above twice the maximum now observed in v1. This is explicitly **post hoc**; the v1 audit cannot be reused as an independent v2 audit. PROTOCOL_V2_PROPOSAL.json preselects24 unused training groups (12 per behavior; new portable blocks[18, 8]), all four candidates and all 31 stochastic times, totaling2,976 observations with repeated forwards. Native/repeat controls stay at1e-6, all identities/variance/precision/sampler/reward/quality guards remain unchanged, and no audit refit is allowed. Nothing in this proposal has run or altered training.

Even a successful new audit would not prove identical gradients or audio quality; near a PPO clipping boundary a small ratio perturbation may change the active branch. A reviewed, versioned receiver-specific protocol would be needed before the formal1,000 real optimizer updates. All invalid candidates and original behavior denominators must remain preserved. Test results must not choose this budget.

The user's requested final output remains pending: fixed50 test prompts, matched seeds and sampler, final **new** RL checkpoint versus official open SFT checkpoint,50 audio pairs/100WAVs with metrics, delivered under /Users/yi/Downloads and SHA-verified before the training instance is released. The previous online group100 checkpoint must not be mislabeled as a newly trained offline checkpoint. Reusing this previously inspected test set is longitudinal comparison, not a fresh blind test.

## Code and artifact references

- Code: https://github.com/ezmonyi/ai-music-detection/tree/7317005/src/music_detector/rl ; study commit7317005 on codex/acestep-online-rl, immutable scientific base1480889d7b7098e27c3259193a9b16be031a37f0.
- Executable study: numerical_calibration_cli.py, numerical_calibration_gpu.py; pre-registration: configs/rl/numerical_calibration_v1.json; safeguards: tests/test_rl_numerical_calibration.py. All12 safeguards passed locally and in the locked cloud Python.
- Local packet: /Users/yi/Documents/projects/music/rl_results/numerical_calibration_20261031
- Original observations: remote_reports/study/calibration_measurements.jsonl and audit_measurements.jsonl.
- Ordering/provenance: PREREGISTRATION.json, PRE_REGISTERED_SELECTION.json, SOURCE_DELTA_SHA256.json, remote_reports/DEPLOYED_SOURCE_SHA256.json, INPUT_VERIFICATION.json, CANDIDATE_FROZEN.json and study.log.
- Results: remote_reports/study/SUMMARY.json, NATIVE_CONTROLS.json, DESCRIPTIVE_ANALYSIS.json, BUDGET_SENSITIVITY_DESCRIPTIVE_ONLY.csv, ERROR_BY_STOCHASTIC_TIME.csv.
- Persistent raw/code/log bundle: /mnt/ai-music-numerical-calibration-20261031-2128889 ;50 original payload members/44,856,728bytes fully SHA-256 readback-verified before adding this report. Any later report additions require their own receipt.
- Original source data: /mnt/upload-export-20261009 (RTX5060/g100), /mnt/ai-music-rl-20261003/data/acestep_grpo_offline_g10_24x4_20261003 and .../acestep_grpo_offline_g10_extension_26x4_20261003 (A6000/g10).
- SHA verification receipts: remote_reports/FILES_SHA256.json, remote_reports/PERSISTENT_VERIFICATION.json and LOCAL_REPORTS_VERIFICATION.json. Native WAV/PT payloads are on persistent storage; only small reports/code were fetched to this Mac.
- The current node closeout/billing receipt is recorded separately; this methods/results report does not imply the node has already been released.

Reference: [PyTorch2.10 numerical accuracy](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html). Cross-platform bitwise equality is not generally promised; this study measured the actual locked configuration rather than treating that warning as permission to bypass a gate.
