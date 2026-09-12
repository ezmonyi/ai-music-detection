# Additive locked YuE2 scoring protocol

This is a separate scoring operation, not a change to the original evaluator's
no-locked-scoring contract. Original evaluation artifacts remain immutable.

Use every primary values-plus-missing model at quantity `all` from the original
BC ordinary-group descriptive protocol: 255 feature subsets times five folds,
1,275 models. Do not select a winner, refit, ensemble scores, tune threshold,
or use expanded YuE2-trained models. Keep ridge 10 and threshold 0.5 with the
original per-model training medians, means and scales. No new audio inference.

Score exactly the 100 eligible YuE2 locked rows from the committed feature
package. Before scoring, verify source COMMIT hashes, every model receipt hash,
and disjoint item, conditioning-group and component identities against each
training set. Check all old development metadata, not only test fold metadata.
Technical failures abort; do not silently omit locked rows or models.

Report per-model AI sensitivity and per-item predictions. These 100 rows are
all AI, so specificity, balanced accuracy and ROC AUC are undefined and must not
be reported. Five models score the same samples: their prediction occurrences
are not 500 independent test songs. Any mean across folds is descriptive model
variation, not an independent-sample confidence interval.

The earlier generation protocol requested old-detector held-out testing before
expanded development. Expanded fitting was already completed before this new
scoring operation. Disclose that ordering deviation; do not retroactively call
this a preregistered pre-expansion experiment. The model catalogue is exhaustive
and fixed without consulting locked outcome scores. Never regenerate audio
based on its detector score.

Acceptance: full metadata preflight before any score, separate exact freeze,
1,275 verified model results and 127,500 predictions over exactly 100 locked
items; preserve individual model identities, final hashes, logs and exceptions.
This test does not replace the remaining two-class source-held-out reports or
other applicable spectral/native-duration analysis arms.
