# Historical server source variants

Nine project Python/shell byte variants were recovered on 2026-09-13 from
the twelve-root historical server inventory. Each download matched its
recorded SHA-256. Existing repository versions are preserved, not overwritten.
These are historical variants, not replacement evaluators or new results.
Seven Python files parsed successfully; shell files are archived launchers,
not executed jobs. A common literal credential-pattern scan found no matches;
this is not a comprehensive security or correctness audit.

The inventory contains 728 Python/shell files: 100 already matched repository
bytes, nine project variants are preserved here, and 619 unmatched files
belong to dependency checkouts. Dependency Git HEAD/status checks returned:

- ACE-Step-1.5: `ca1e85fe9430179831e6bc6be790c332190a3866`, clean status.
- heartlib: `3783bdb8441f2c298b1e64c8651173aac200361c`, clean status.

Clean status is not a backup of ignored files or proof that the commits are
reachable upstream. Those dependencies still require reproducibility reference
verification. The inventory excludes virtual environments and is bounded to
top-level scripts plus code/scripts directories, not every server file.
