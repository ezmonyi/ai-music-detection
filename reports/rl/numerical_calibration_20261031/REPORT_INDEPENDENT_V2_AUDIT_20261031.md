# Independent v2 receiver audit for offline ACE-Step Flow-GRPO

The separately approved v2 audit passed on the current RTX 3090 receiver. All
2,976 independently selected transitions satisfied the fixed `3e-4` absolute
log-density tolerance. This result admits the reviewed numerical receiver
profile; it does not establish RL improvement or completion of the requested
1,000 optimizer updates.

## Why a new audit was necessary

The preceding zero-update study found a maximum cross-GPU discrepancy of
`1.1646747589111328e-4`, and failed its original `1e-6` tolerance and predeclared
budget-selection rule. That failed study remains unchanged in
`REPORT_NUMERICAL_CALIBRATION_20261031.md`. After reviewing the observations,
the user approved a new, explicitly versioned `3e-4` receiver budget, conditional
on an independent audit. The earlier observations therefore serve as discovery
evidence for v2, rather than being reused as its independent validation.

The v2 plan fixed the budget and 24 new training prompt groups before the new
forwards. Twelve groups came from the RTX 5060 / group-100 behavior cache and
twelve from the older A6000 / group-10 behavior caches. All four candidates and
all 31 stochastic transitions were measured: `24 × 4 × 31 = 2,976` observations.
No validation/test prompts or labels selected the numerical threshold. The
exact identities and their disjointness checks are in `v2/FROZEN_V2_PLAN.json`
and `v2/INPUT_VERIFICATION.json`.

## Measured results

| Measurement | Independent v2 audit |
| --- | ---: |
| Prompt groups | 24 |
| Transitions | 2,976 |
| Absolute log-density error, median | 9.059906005859375e-6 |
| Absolute log-density error, 95th percentile | 3.9637088775634766e-5 |
| Absolute log-density error, 99th percentile | 6.0677528381347656e-5 |
| Absolute log-density error, maximum | 1.02996826171875e-4 |
| Passing original 1e-6 budget | 183 / 2,976 |
| Passing user-reviewed 3e-4 budget | 2,976 / 2,976 |
| Maximum same-receiver repeat difference | 0 |
| Native group-100 control | 31 / 31, maximum difference 0 |
| Native group-10 control | 31 / 31, maximum difference 0 |
| Corrupted-probability negative controls | Rejected |
| Audit optimizer updates | 0 |
| Audit elapsed time | 933.856 seconds |

The stored quantities are dimension-mean Gaussian transition log densities.
For an absolute log-density error bound `b`, the corresponding multiplicative
surrogate-ratio uncertainty is bounded by `exp(b)-1`. The fixed v2 budget is
therefore approximately **0.0300045%**, while the largest observed audit error
corresponds to **0.0103002%**. These percentages describe this mean-density
surrogate, not an exact joint-trajectory importance-sampling guarantee.

Neither the audit budget nor the numerical flags were refitted using these
audit observations. Cached denominators, full FP32 trajectories, BF16 backbone,
FP32 adapters, sampler, official/VAE weights, reward and quality guards remained
unchanged. Native/repeat controls retained `1e-6`. The conclusion is specific to
this receiver, model identities, two behavior policies and locked runtimes.

## Explicit implementation boundary

The original scientific core remains byte-identical to commit `1480889`.
`src/music_detector/rl/numerical_audit_v2.py` implements the new independent
zero-update audit. `src/music_detector/rl/reviewed_receiver.py` recomputes the
raw audit and validates user authorization, input identities, native/negative
controls, receiver flags and evidence hashes before issuing a receiver profile.

The reviewed training options are separate in
`configs/rl/offline_1000_rtx3090_v2.json`. The original
`configs/rl/offline_1000.json` retains its strict default. The wrapper allows the
approved `3e-4` replay tolerance and checks all 31 stochastic steps, while
pinning the remaining options to the original configuration digest. Each fresh
run has an external numerical-protocol sidecar; continuation must match that
same sidecar and resume from the same offline optimizer/RNG/cursor checkpoint.

Source commits are `3fd46e8` for the independent audit and `8857c3e` for the
reviewed receiver and authenticated bounded cache restore. The issued profile
SHA-256 is `b1d7fe89f2b6437fa96abf60d1245871db1bbf42ec7a48bee02cba31f80335f7`.

## Subsequent training and fixed-test plan

The cache restoration verified all 42 original TARs, 132,352,450,560 bytes, and
all restored members: 1,000 new groups, 4,000 full FP32 trajectories, and 8,000
candidate/base WAVs. Fifty older behavior groups are available separately.
Frozen S+R+F scoring and exact eligibility preflight precede training. The goal
is 1,000 **additional real optimizer updates**, after 96 earlier online updates,
using one epoch without recycling invalid or exhausted data.

The admitted run must first complete two real updates and then resume the same
offline checkpoint for the remaining 998. Invalid candidates, skipped groups,
drift failures and intermediate checkpoints remain observable. A group count
cannot substitute for an optimizer-update count. If the number of eligible
groups makes the fixed target impossible, the correct result is an explicit
data-feasibility failure, not a relaxed gate or an extra epoch.

After successful training, the fixed 50 held-out prompts compare the new offline
checkpoint with official SFT under identical prompts, seeds, sampler, duration
and VAE. The prior online group-100 policy is retained as a separate third arm.
The requested delivery contains 50 official/new-RL pairs, or 100 WAVs, with
per-member SHA-256 verification. No new RL benefit or held-out comparison is
claimed by this numerical-audit report.

## Artifact references

- Raw per-transition evidence: `v2/audit_measurements.jsonl`.
- Frozen selection and audit controls: `v2/FROZEN_V2_PLAN.json`,
  `v2/INPUT_VERIFICATION.json`, `v2/NATIVE_CONTROLS.json`, `v2/SUMMARY.json`.
- LaTeX text: `thesis_independent_v2_audit_appendix.tex`.
- Verified local packet:
  `/Users/yi/Documents/projects/music/rl_results/numerical_calibration_20261031/v2`.
- Persistent audit:
  `/mnt/ai-music-numerical-audit-v2-20261031-2128889`.
- Training configuration, scores, protocol and controller receipts:
  `/mnt/ai-music-offline-grpo1000-v2-20261031-2128889`.
- Ongoing training outputs are system scratch until verified protection:
  `/root/music-offline-grpo1000-v2-20261031-2128889/runs`.

The persistent audit packet's 24 members (873,010 bytes) and the downloaded
small-report packet were verified member by member. Earlier v1 results and
intermediate versions are preserved independently. Training/test terminal
results belong to their subsequent report and must not be inferred from this
passing numerical audit.
