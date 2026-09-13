# Thesis v8 verification

Entry point: `sources/interpretable_audio_thesis_working_v8_20260913_en.tex`.
Build: `latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=..`
followed by the entry-point filename, run from `sources/`.

The 37-page PDF compiled successfully. The final log contains no Overfull,
Warning or undefined-reference matches. Its SHA-256 is
`ccdb9bc9420e3044f82cff6d9f28729e6659e25267eddf20b324ea5ab2ae2350`.

All 54 prior v7 source files remain byte-identical in the preserved copy.
The new entry point changes the revision date and includes one new delivery
appendix; experimental numerical results are unchanged.

Pages 1-3 (cover and contents) and 36-37 (new appendix) were rendered and
visually inspected. No clipping, overlap or illegible paths were observed.
Pages 4-35 were rendered for both v7 and v8 with identical Poppler settings;
all 32 PNGs are byte-identical. This is layout regression evidence, not a new
independent scientific review of the inherited chapters.

Earlier failed-layout artifacts and visual checks remain local. V7 is unchanged.
This is a verified working thesis revision, not institutional acceptance or
proof of complete audio delivery. Historical storage and rights gaps remain
explicit in Appendix B.
