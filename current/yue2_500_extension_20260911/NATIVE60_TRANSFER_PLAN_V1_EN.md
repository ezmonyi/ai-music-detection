# YuE2 Native60 frozen-transfer plan (not a completion receipt)

## Purpose and scope

Finish the duration-matched historical transfer arm without fitting a new model
to the YuE2 measurements. This is separate from the expanded Native30 experiment
and its S/D/R/P/F/H/SC/BC family definition. Historical equal60 models instead
use S/D/R/P/F/H/M. Do not relabel M as SC or BC, or silently apply Native30
normalization/thresholds to these models.

## Fixed population

The accepted input package is `native60_inputs_v1`, COMMIT SHA256
`cde960d1f4ddb29d9751ac8d3bbecf28b6806a5fc9dc05cf299618ec2d8b02ea`.
It contains 276 original YuE2 recordings long enough for an exact native
center-60-second interval: 223 development and 53 previously locked records.
The other 224 of the original 500 are excluded by duration, never padded.
Roles and shared Muse source-song groups must remain unchanged.

Report the 53-record locked subset separately from the 223 development subset
and the descriptive all-276 population. Locked status does not mean pristine
study-wide non-exposure: Native30 and legacy analyses preceded this transfer.
Length eligibility also induces selection; no extrapolation to all 500 songs.

## Admission before scoring

1. Require completed neural and SDRP producer COMMITs, the independent native
   crop audit, and the completed F/H/M package; verify every committed product.
2. Reconcile exact IDs, roles, groups and feature columns, preserving nulls.
   Verify that numerical feature extraction and preprocessing match the
   historical equal60 definitions. The native48 crops, neural44.1k views and
   mono16k F/H/M loader are distinct documented views, not interchangeable files.
3. Inspect the existing independent equal60 model audit and pin its exact
   model publication and all selected model hashes in a new transfer contract.
   Check training ID/group disjointness from every scored YuE2 record.
4. Test a new YuE2 adapter on synthetic fixtures. Do not bypass Mureka's
   exactly-500/source/role guards by changing metadata to look like Mureka.

## Scoring and interpretation

Apply all historical five folds and five training-group caps (25, 50, 100,
200, all). Preserve saved medians, missing indicators, scales, coefficients
and threshold 0.5; no refitting, calibration or threshold search.

The 63 nonempty S/D/R/P/F/H subsets form the non-M transfer table (1,575
models). For comparability with the historical seven-family report, retain
the 64 M-containing subsets as explicitly secondary historical diagnostics
(1,600 models), not new evidence admitting M as a validated detector family.
Together these reproduce 127 subsets and 3,175 frozen model applications.

Each scored population is AI-only. Report TP, FN, sensitivity and false-negative
rate per model, then descriptive fold means/ranges per subset and cap. Do not
report accuracy, balanced accuracy, specificity or ROC AUC for these single-class
populations. Repeated model applications are not independent recordings.
No best-model selection or significance claim is planned.

Independently reconstruct scores and threshold decisions from saved parameters
before accepting the transfer outputs. Preserve per-record predictions, model
hashes, population denominators, missingness, audit outputs and all failures.
Only then add results to the English thesis and final dataset/code index.
