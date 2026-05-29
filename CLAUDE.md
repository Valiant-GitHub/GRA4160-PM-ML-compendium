# GRA 4160 — Interactive Study Companion (project context)

A study companion for **GRA 4160 (Predictive Modelling with Machine Learning), BI Norwegian Business School**. It is a Quarto website rendered to `_site/`, produced across 7 phases by a prior Claude Code session. The `.qmd` files are the source; `_site/` is the rendered output and the shipped deliverable.

## Ground truth vs. generated content

**Ground truth (authoritative, NOT in this repo).** The GRA 4160 course material — lecture slides, lecture notebooks, exercise notebooks, past exams, course datasets — plus the two textbooks **ESL** (*The Elements of Statistical Learning*) and **ISL** (*An Introduction to Statistical Learning*). These are intentionally excluded from the repo (instructor's IP + copyrighted texts). To verify any claim, the user must supply the relevant source file separately; it will not be present after a fresh clone.

**AI-generated (study material, not authoritative, IS in this repo).** Everything in `_site/` (rendered HTML) and all `.qmd` source files — at the project root and in `methods/`, `cross_method/`, `drill/`, `appendix/`, `big_picture/`. These are derived from the ground-truth material above; they are not themselves sources.

**Build provenance (NOT in this repo).** The original build produced structured source extracts in `phase0/` and phase summary logs (`phase*_summary.md`, `workspace_inventory.md`). These were removed from git tracking during repo cleanup. They may still exist on the build author's local disk, but won't be present after a fresh clone.

## Rules for future sessions

1. **Never cite a `.qmd` file or an `_site/` HTML page as a source.** Every claim in those files derives from external course material or the textbooks. Cite the underlying ground-truth file instead.
2. **To verify a claim, open the cited ground-truth source directly.** Don't trust that the citation in the `.qmd` matches the source — verify it. If the cited source isn't locally available, ask the user to supply it rather than guess.
3. **To extend or fix the site, apply the original source-fidelity rules.** Every claim, formula, code snippet, hyperparameter, and dataset reference must trace to a real ground-truth source at a citable location. Use `[VERIFY: ...]` markers rather than invent.
4. **Read `appendix/build_log.qmd` before trusting any specific page.** It documents deviations, known limitations, and unresolved `[VERIFY]` items from the original build.
5. **Known specific limitations carried over from the build:**
   - `methods/nn_pytorch.qmd` §4 has qualitative numbers (PyTorch wasn't installed in `.venv` at build time).
   - Two course datasets (`SMSSpamCollection.csv`, `bank-additional-full.csv`) weren't present at build time; those drill pages document idioms without computed outputs.
   - The site uses Observable JS and requires an HTTP origin to function (must be served, not opened via `file://`).

## How to render

Windows requires an explicit Quarto Python path. The build's `.venv` is not tracked, so recreate it from `requirements-build.txt`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-build.txt
$env:QUARTO_PYTHON = ".venv\Scripts\python.exe"
quarto render
```

## How to view the rendered site

The site uses interactive widgets (Observable JS) that need a local web server — opening the HTML files directly in a browser will not work.

From the project root:

```powershell
cd _site
python -m http.server 8000
```

Then open **http://localhost:8000/** in any browser. To stop the server, press `Ctrl+C` in the terminal.

## What this project is NOT

Not source material. Not graded material. Not a substitute for the lectures, notebooks, or textbooks. Not authoritative when in conflict with ground truth.
