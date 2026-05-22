# Phase 2 — Toolchain & Skills Summary

Run on 2026-05-21. Setup Prompt v2, Phase 2.

## Hard requirements (all satisfied — no halt under G3)

| Tool | Min | Found | Status |
|---|---|---|---|
| Quarto | 1.4 | **1.8.27** | OK |
| Python | 3.10 | **3.11.9** | OK |
| Pandoc | recent | **3.9.0.2** (system; Quarto also bundles its own) | OK |

## Optional tools (Phase 2.1)

| Tool | Found | Status |
|---|---|---|
| Git | **2.48.1.windows.1** | Present (used for the Phase 1 commit) |
| Node.js | **v24.15.0** | Present |

## Python libraries (Phase 2.2)

`requirements-build.txt` created with the 9 specified packages.

**Install approach: dedicated virtual environment at `.venv/`** (not system, not
`--break-system-packages`). Reason below.

### Why a venv (important for the master prompt)

The host Python is the **Windows Store / app-execution-alias** build
(`...\WindowsApps\PythonSoftwareFoundation.Python.3.11_...\python.exe`). Quarto's
Deno runtime cannot `statSync` that app-alias reparse point, so `quarto check jupyter`
failed with *"Unable to locate an installed version of Python 3"* even though
`python --version` works and `QUARTO_PYTHON` pointed at a path that `Test-Path`
confirmed exists. Creating a normal venv produces a plain `python.exe` Quarto can
locate and execute. After switching, `quarto check jupyter` → **OK**.

### Installed (in `.venv`)

All 9 build libs from `requirements-build.txt`, plus **`ipykernel`** (required by
Quarto's jupyter engine to execute `{python}` cells), plus transitive deps:

| Package | Version | Role |
|---|---|---|
| nbformat | 5.10.4 | notebook parsing |
| nbclient | 0.10.4 | notebook execution |
| pymupdf (fitz) | 1.27.2.3 | **primary** PDF extractor |
| pdfplumber | 0.11.9 | **fallback** PDF extractor (layout-sensitive pages) |
| pandas | 3.0.3 | CSV/data handling |
| numpy | 2.4.6 | numerics |
| scikit-learn | 1.8.0 | ML idioms / verification |
| matplotlib | 3.10.9 | build-time plotting (verification only) |
| pyyaml | 6.0.3 | YAML parsing |
| ipykernel | 7.2.0 | Quarto jupyter engine kernel |

These are **build-time only** (source extraction + verification). The rendered
website is JS-side (Observable Plot + Plotly), not Python.

> A duplicate set of these libraries was also installed into the Store-Python
> user site-packages earlier (before the venv decision). Harmless, but the venv
> is the canonical environment the master prompt should use.

## REQUIRED for the master prompt: how to render

Quarto will **not** find Python on its own here. Before any `quarto render` /
`quarto preview`, set `QUARTO_PYTHON` to the venv interpreter:

- **PowerShell:** `$env:QUARTO_PYTHON = (Resolve-Path ".\.venv\Scripts\python.exe").Path`
- **git-bash:** avoid — backslash path translation mangles the var; use PowerShell for Quarto.

`.venv/` is git-ignored. Verified working: `quarto check jupyter` → OK, and the
full smoke-test render succeeded with this setting.

## Skills discovered (Phase 2.3)

The three standard skill dirs (`~/.claude/skills/`, `/mnt/skills/*`,
`./.claude/skills/`) do **not** exist. File-based skills live in the plugin
**marketplace catalog** (`~/.claude/plugins/marketplaces/claude-plugins-official/`,
28 `SKILL.md` files). Full detail in `phase2_skills_inventory.md`.

| Need | Skill | Status |
|---|---|---|
| Frontend / theming | **frontend-design** | Found — apply manually for `theme/` CSS, typography, color system |
| PDF extraction | — | **None found** — master prompt relies on pymupdf + pdfplumber |
| Document handling | — | None found — use built-in `Read` + Python libs |

Harness-active skills (via the `Skill` tool) — `update-config`, `simplify`,
`claude-api`, `init`, `review`, `security-review`, etc. — none relevant to the build.

## Smoke test (Phase 2.4) — ALL PASS

`phase2_smoke_test.qmd` rendered with `html-math-method: katex` (Quarto defaults to
MathJax; KaTeX was requested explicitly so the math method matches the spec).

| Check | Evidence in rendered HTML | Result |
|---|---|---|
| 1. Prose | paragraph text present | PASS |
| 2. KaTeX math | `katex.min.js`, `katex.min.css`, `katex.render(...)`, `katex-math` class, `<span class="math inline">\hat{\beta}=…</span>` (KaTeX renders client-side, so static `class="katex"` spans appear at runtime, not in the file — "or similar" evidence is conclusive) | PASS |
| 3. Observable JS chart | `<script type="module" src=".../quarto-ojs-runtime.js">` + container divs `ojs-cell-1-1`, `ojs-cell-1-2` | PASS |
| 4. Python nbformat parse | `nbformat OK -- cell count: 15` from `Lecture notebooks\02_OLS (1).ipynb` (the spec's default `02_OLS__1_.ipynb` doesn't exist; globbed the real file) | PASS |

Per §2.4, the smoke-test `.qmd`, its `.html`, the `_files/` support dir, and the
`.quarto/` cache were deleted after verification.

## Skipped / notes under G4

- No PDF or document-handling skill available (logged; not a blocker).
- Switched from system/`--break-system-packages` install to a venv due to the
  Windows Store Python app-alias breaking Quarto's Python detection.
- `.venv/` added to `.gitignore` (venv created after the Phase 1 commit).
