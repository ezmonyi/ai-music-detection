# Formal expanded-evaluation commands

Do not mix a macOS Python runtime with remote `/mnt/nfs-data` paths. First mirror
the final combined S/D/R/P CSVs and serialization manifests into the local
artifact tree, then regenerate the positive union-verification JSONs locally so
their strict paths and byte hashes name those local copies. Retain the original
remote verification JSONs separately as provenance. Do not use the old
7,500-row spectral-only CSV. The preflight command is safe while materialization
is incomplete: it exits nonzero before starting CV and archives any prior audit
before writing a new one.

The authoritative frozen registries are `manifests/final/metadata_10s.csv`
(10,141 rows; SHA-256
`a03aec930130fcb942c40b35ba5fb49109ceee588894124010448f041d9d1ef4`)
and `manifests/final/metadata_30s.csv` (4,497 rows; SHA-256
`eb313f97e2d69d743423efc170eecb24f1f00ee77ad789461772daecd058c242`).
The older files one directory above are partial snapshots and are not valid
formal inputs.

```bash
EXP_PYTHON=/Users/yi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
EXP_ARTIFACT=/Users/yi/Documents/code/music/artifacts/source_diversity_expansion_20260905
EXP_RESULTS="$EXP_ARTIFACT/results"

"$EXP_PYTHON" "$EXP_ARTIFACT/code/evaluate_expanded_orchestrate.py" \
  --mode preflight \
  --metadata-10s "$EXP_ARTIFACT/manifests/final/metadata_10s.csv" \
  --metadata-30s "$EXP_ARTIFACT/manifests/final/metadata_30s.csv" \
  --features-10s "$EXP_RESULTS/four_family_10s/expanded_features_10s.csv" \
  --features-30s "$EXP_RESULTS/four_family_30s/expanded_features_30s.csv" \
  --verification-10s "$EXP_RESULTS/four_family_10s_verification.json" \
  --verification-30s "$EXP_RESULTS/four_family_30s_verification.json" \
  --materialization-audit "$EXP_ARTIFACT/audit/materialization_validation.json" \
  --legacy-audit "$EXP_ARTIFACT/audit/legacy_native_1000.json" \
  --prior-heldout "/Users/yi/Documents/code/music/artifacts/four_family_diversity_ablation_20260904/results/evaluation/evaluation_summary.json" \
  --prior-heldout "/Users/yi/Documents/code/music/artifacts/external_generator_500_heuristics_20260904/results/evaluation/evaluation_summary.json" \
  --output-root "$EXP_RESULTS/formal_evaluation"
```

After the preflight audit says `preflight_passed`, run the same frozen inputs
with the explicit formal confirmation:

```bash
"$EXP_PYTHON" "$EXP_ARTIFACT/code/evaluate_expanded_orchestrate.py" \
  --mode run --confirm-formal-ready \
  --metadata-10s "$EXP_ARTIFACT/manifests/final/metadata_10s.csv" \
  --metadata-30s "$EXP_ARTIFACT/manifests/final/metadata_30s.csv" \
  --features-10s "$EXP_RESULTS/four_family_10s/expanded_features_10s.csv" \
  --features-30s "$EXP_RESULTS/four_family_30s/expanded_features_30s.csv" \
  --verification-10s "$EXP_RESULTS/four_family_10s_verification.json" \
  --verification-30s "$EXP_RESULTS/four_family_30s_verification.json" \
  --materialization-audit "$EXP_ARTIFACT/audit/materialization_validation.json" \
  --legacy-audit "$EXP_ARTIFACT/audit/legacy_native_1000.json" \
  --prior-heldout "/Users/yi/Documents/code/music/artifacts/four_family_diversity_ablation_20260904/results/evaluation/evaluation_summary.json" \
  --prior-heldout "/Users/yi/Documents/code/music/artifacts/external_generator_500_heuristics_20260904/results/evaluation/evaluation_summary.json" \
  --output-root "$EXP_RESULTS/formal_evaluation"
```

The orchestrator always passes `--source-column source_group` and
`--group-column group_id`. It freezes both 10 s and 30 s CV leaders and their
input/code hashes before opening either duration's locked scores. The two
independent CV subprocesses run concurrently, but the scheduler joins both and
verifies both frozen outputs before constructing any locked command. It then runs
locked scoring, 2,000-replicate group-block bootstrap, Markdown reporting, and
the official 15-by-5 AUC/BA overview in PNG/PDF plus readable standalone AUC and BA PNGs for each duration. After formal scoring is
complete it also writes English thesis fragments to
`$EXP_ARTIFACT/latex/results_{10s,30s}_*.tex`; this reporting-only step cannot
change either frozen leader.

Finally run the independent, fitter-free result audit. This must pass before
the generated CSV, LaTeX, or figure artifacts are cited:

```bash
"$EXP_PYTHON" "$EXP_ARTIFACT/code/audit_formal_results.py" \
  --results-root "$EXP_RESULTS/formal_evaluation" \
  --latex-root "$EXP_ARTIFACT/latex" \
  --output "$EXP_ARTIFACT/audit/formal_results_consistency.json"
```

Each verification JSON must positively satisfy the frozen schema-v1 contract:
`status="passed"`, `passed=true`, current metadata/feature byte hashes, matching
sorted-ID digests and exact counts, no duplicates or ID differences, zero
unexplained failures, and complete mapping of any serialization failures back
to the corresponding feature-row reason. Natural D/R/P measurement missingness
is explicitly counted and allowed; it is not treated as a failed record.

If the final validator produces a duplicate-exclusion manifest, pass it to both
commands as `--exclusion-manifest PATH`. Only unavailable IDs with a non-empty,
duplicate-specific justification are accepted; all other missing records keep
the preflight closed.
