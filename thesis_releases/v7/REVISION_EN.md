# Thesis v7 editorial revision

Preserves v6 unchanged. Corrects four stale temporal statements in sections 3.3,
7, 9 and 10.6; updates the title revision and appendix build command. No numerical
tables, feature definitions, thresholds, cohorts or experiment results changed.

Build from sources with:

    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=.. interpretable_audio_thesis_working_v7_20260912_en.tex

The build completed successfully: 35 pages. Final log search found no Warning,
Overfull or undefined-reference matches. PDF SHA-256:
`aa3487255739f1e9d7d693f205be2d8e0dd38886688f4c1b2d7d1fcefc84fe04`.

The full visual pass applies to v6; affected v7 pages still require a visual
check. Older master filenames inside the copied source snapshot are historical;
the v7 master above is the entry point for this revision. Existing source filenames
were retained within this self-contained version directory, not overwritten in
the original v6 directory. Audio publication remains incomplete.
