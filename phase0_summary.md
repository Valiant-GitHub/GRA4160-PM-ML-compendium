# Phase 0 Summary — Inventory & Planning (STOP gate)

Run 2026-05-22. Master Prompt v2, Phase 0. **All Phase-0 work produced on Opus 4.7**
(see `phase0/sonnet_produced_files.md` — four early extraction batches were run on
Sonnet, then re-done on Opus per the mid-build model-override; Sonnet originals archived
to `phase0/_sonnet_provisional/`).

## Produced in Phase 0

| Item | File(s) | Status |
|---|---|---|
| 0.0 File manifest | `phase0/file_manifest.md` | ✓ |
| 0.1 Lecture extracts (10) | `phase0/lecture_01..10_extract.md` | ✓ all 10, formulas transcribed |
| 0.2 Method-notebook extracts (20) | `phase0/notebook_*.md` | ✓ all 20 |
| 0.2 Exercise extracts (6 topics, 11 nbs) | `phase0/exercise_01..06_*.md` | ✓ |
| 0.3 Exam extracts | `phase0/exam_spring2025_notebook.md`, `phase0/exam_question_papers.md` | ✓ |
| 0.4 Guidelines | `phase0/guidelines_extract.md` | ✓ |
| 0.5 References | `phase0/references.md` (ISL + ESL both present) | ✓ |
| 0.6 Notation analysis | `phase0/notation_table.md` → `appendix/notation_table.qmd` in Phase 1.5 | ✓ (data) |
| 0.7 Recurrence map | `phase0/recurrence_map.md` | ✓ |
| 0.8 Concept inventory | `phase0/concept_inventory.md` (per-method build plan) | ✓ |
| 0.9 Disagreements | `phase0/disagreements.md` (R9/R10/NB/DATA/GR4) | ✓ |
| Raw PDF text | `phase0/raw_text/*.txt` + `phase0/_extract_pdfs.py` | ✓ (deterministic) |

## Key planning conclusions
- **Lecture→topic map (confirmed from source, differs from prompt's illustrative order):**
  L1 intro/data · L2 ML basics/OLS/kNN/NaiveBayes · L3 LDA+regularization · L4 classification (logistic+trees) · L5 model selection/eval (metrics/AIC-BIC/CV/bias-var) · L6 unsupervised (PCA/K-means) · L7 ensembles (bagging/boosting/RF/extra-trees) · L8 backprop/GD (+ADALINE, logistic-from-ADALINE) · L9 neural nets · L10 traditional-vs-DL survey.
- **16 method pages** all have ≥1 lecture + ≥1 notebook source (GR3 coverage satisfiable). Naive Bayes is drill-only (no method page) — consistent with the prompt's 16-method list.
- **Tier split validated** (`recurrence_map.md`): every Tier-1 method recurs across lecture+notebook+exam; the **entire NN block has zero exam appearances** — corroborates Tier 2.
- **Exam structure → drill + Past-Exam Lens:** 30-hr take-home, 4 assignments ×25%; Asgmt1=regularized regression (Hitters), Asgmt2=PCA+K-means (WholesaleCustomers), Asgmt3=tree/RF/gradient-boosting (WDBC), Asgmt4=binary-classification challenge (SimulatedData, hidden hold-out). Justification graded as heavily as the choice; concise prose; AI allowed only if explainable.
- **From-scratch implementations exist** for OLS, kNN, LDA, ADALINE, logistic, RF, autodiff, NN (numpy/custom/PyTorch) — rich material for "how it works" + Mode B.

## Gaps & flags (none are halts)
- **[DATA-1] spam `SMSSpamCollection.csv` missing**, **[DATA-2] bank `bank-additional-full.csv` missing** — affect only 2 drill pages (C7 skip-and-log); no method page depends on them. Method-page clustering/PCA use present datasets (seeds, WholesaleCustomers).
- **[DATA-4] notebook data paths differ from repo layout** — the `data/` JS export (C8) reads real paths from `file_manifest.md`.
- **R9 supplements needed in Phase 3** (marked supplemental, not from slides): PCA eigen/SVD math, AdaBoost/gradient-boosting formulas, RF regression `p/3`, bias-variance `σ²` term.
- **Notebook code/prose mismatches** ([NB-1..10]) catalogued — quote *actual* code values per R3.
- **Notation collisions** (k/K, p, η/α, δ, λ, sklearn-`alpha`-vs-EN-`α`) resolved in `notation_table.md`.

## C6 halt check — NOT triggered
- No source-verification failure (canary spot-check on `notebook_13` confirmed deep reading).
- No required Tier-1 source missing or unreadable (all 10 lectures + 20 notebooks read).
- No conflict requiring human input.

**Decision: PROCEED to Phase 1 (site skeleton & shared infrastructure).**
