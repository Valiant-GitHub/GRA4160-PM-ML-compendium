# Phase 1 — Workspace Setup Summary

Run on 2026-05-21. Setup Prompt v2, Phase 1.

## Directories created (9)

All created at the working-directory root via `mkdir -p` (none pre-existed):

```
phase0/        # Source extracts (one .md per source file)
theme/         # custom.scss + toggle.js
big_picture/
methods/
cross_method/
drill/
appendix/
data/          # JS arrays exported from CSVs
_site/         # Quarto render output (gitignored)
```

## Files created

| File | Note |
|---|---|
| `workspace_inventory.md` | Single source of truth for the workspace (Phase 1.1) |
| `.gitignore` | Build artifacts + backups (Phase 1.3) |
| `phase1_setup_summary.md` | This file |

**No `.bak` renames were needed** — none of the files written in this phase
pre-existed in the working directory (G2 not triggered).

## Source files

**Not touched.** All lecture PDFs, notebooks, datasets, exam PDFs, and the two
reference texts (ESL, ISL) remain read-only per G1. They live in subfolders
(`Data/`, `Exercises and solutions(VHL)/`, `Lecture notebooks/`,
`Lecture slides/`, `Lecture slides/Past exams/`), not at the root — see
`workspace_inventory.md` for full paths.

## Version control (Phase 1.4)

- No `.git/` existed, so initialized a fresh repository.
- Staged everything (`git add -A`) and committed: **`eeb53b3 Setup: workspace structure`**.
- 71 files tracked. Working tree clean.
- `.gitignore` excludes `_site/`, `_freeze/`, `.quarto/`, `*.bak.*`, caches.
- Empty output folders (`phase0/`, `theme/`, etc.) are not tracked (git does not
  track empty directories); they will be populated by the master prompt.
- LF→CRLF line-ending warnings appeared (Windows default `core.autocrlf`); cosmetic only.

## Skipped / deviations

- **Source layout differs from the setup prompt's flat-root assumption** — files
  are organized into subfolders. Recorded in `workspace_inventory.md`; not a blocker.
- macOS zip noise (`__MACOSX/`, `._*` AppleDouble files) present inside the
  extracted 2025 exam bundle — flagged for the master prompt to ignore.
- Nothing else skipped.
