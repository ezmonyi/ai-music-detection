# v7 differential layout acceptance

PDF SHA-256: `aa3487255739f1e9d7d693f205be2d8e0dd38886688f4c1b2d7d1fcefc84fe04`.

All 35 v7 pages were rendered with pdftoppm at maximum dimension 1100 pixels.
The v6 pages were compared at the identical resolution. Exact PNG byte comparison
found changes only on pages 1, 2, 8, 15, 20, 21, 22, 23, 24 and 34. Extracted
page-text comparison independently identified precisely the same ten pages.

All ten changed pages were visually inspected. No clipped tables, overlapping
text/equations, missing glyphs or unresolved reference placeholders were seen.
The four stale status statements identified in the v6 errata are corrected.
Pagination remains 35 pages; text reflow across pages 20–24 is intact. The other
25 pages are raster-identical to the already visually reviewed v6 pages.
This establishes full-document layout coverage by differential inspection, not
a new scientific validation of all reported results.

All 54 files in the local v7 source snapshot match the GitHub staging snapshot
byte-for-byte. The build completed with no Warning, Overfull or undefined-reference
matches in the final log. Earlier versions and their QA records are preserved.
Dataset/audio publication and whole-project delivery remain separate and incomplete.
