# Completed Native30 aggregate tables

Both packages are reporting-only projections of completed evaluations. Each
contains all 3,825 primary cells: 255 subsets × 5 training-group caps × 3
evaluation protocols, with eligibility counts and omitted-model counts.
No model is selected, refitted, calibrated or tuned by these exporters.

- `bc_thesis_tables_v1`: original 3,830-recording development cohort,
  103,530 completed primary/diagnostic model evaluations.
- `yue2_expanded_thesis_tables_v1`: expanded 4,228-recording development cohort,
  112,455 completed primary/diagnostic model evaluations. The 100 locked YuE2
  recordings are separate, not additional training rows.

Each package COMMIT binds its products and the full source-report COMMIT.
Full report files and historical versions are retained locally under
`Documents/report/music_ai_detection_20260912/current_results/`.
Primary accuracy is balanced by group, source and class, not ordinary
recording-weighted accuracy. Fold AUC means are descriptive; scores from
different fitted models are never pooled for AUC. Different cohorts change
both training and held-generator composition: their difference is not a causal
estimate of the benefit of adding YuE2 training recordings.

English thesis v5 includes these results. Native60 transfer scripts in the
code tree remain work in progress: the model audit and synthetic score checks
passed, but the real feature acceptance and scoring had not completed when
this package was published. This is not whole-project completion.
