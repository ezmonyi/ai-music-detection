# Early server launcher preservation

Two scripts retrieved on 2026-09-13 from the 2026-08-14 Demucs artifact
experiment server directory were absent by content from the previous staged
Python/shell inventory. Both downloads match the server SHA-256 inventory,
were read for inspection, and passed Python syntax parsing. Neither was run.
Existing analysis versions are not replaced. These historical launchers do
not imply authorization to run inference on an arbitrary device today.

The bounded early audit also found two analysis files already preserved and
one third-party get-pip.py installer. The installer is not project-authored
experimental code and has not been republished here. This audit covers
top-level Python/shell files plus immediate code/scripts Python files in the
two identified August music experiment directories, not unrelated projects
or vendor/environment trees.
