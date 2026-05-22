# Phase 0.7 — Recurrence Map

Appearances of each method/concept across **lectures (L)**, **notebooks (nb)**, and the
**spring-2025 take-home exam (E)**. Used to verify the Tier-1 / Tier-2 split is consistent
with what the sources actually emphasize. (Exam = lens only; counted to confirm emphasis,
not as a content source.)

| Method / concept | Lectures | Notebooks | Exam 2025 | Σ weight | Tier (planned) | Consistent? |
|---|---|---|---|---|---|---|
| OLS / linear regression | L1, L2, L3(recap) | nb02 | Asgmt 1 (Hitters) | high | **1** | ✓ |
| Regularization (Ridge/Lasso/EN) | L3, L4(recap) | nb05, nb08(Lasso) | Asgmt 1 (core) | high | **1** | ✓ |
| kNN | L2, L3(recap), L10 | nb03 | Asgmt 4 (candidate) | med-high | **1** | ✓ |
| Logistic regression | L4, L8 | nb06, nb15 | Asgmt 4 (candidate) | high | **1** | ✓ |
| LDA | L3, L4(recap) | nb04 | — | med | **1** | ✓ |
| Adaline | L8 | nb14 | — | low-med | **1** | ✓ (pedagogical bridge to logistic/NN) |
| Bias-variance | L1, L2, L3, L5 | nb09 | Asgmt 1 (interpret) | high | **1** | ✓ (most-recurring concept) |
| CV / info criteria | L3, L5 | nb08 | all assignments (tuning) | high | **1** | ✓ |
| Decision trees | L4, L7(recap) | nb07_trees | Asgmt 3 | high | **1** | ✓ |
| Random forests | L7 | nb13 | Asgmt 3 (core) | high | **1** | ✓ |
| Ensembles (bag/boost/stack/BMA) | L7 | nb12, nb07_income(GB) | Asgmt 3 (gradient boosting) | high | **1** | ✓ |
| PCA | L6 | nb10 | Asgmt 2 (core) | high | **1** | ✓ |
| K-means | L6 | nb11 | Asgmt 2 (core) | high | **1** | ✓ |
| NN basics | L8, L9 | nb17 | — | med | **2** | ✓ (taught, not examined) |
| Autodiff | L8 | nb16 | — | low-med | **2** | ✓ |
| NN with PyTorch | L9 | nb18 | — | low-med | **2** | ✓ |
| Build a NN | (L9) | nb19 (MNIST) | — | low-med | **2** | ✓ |
| Naive Bayes | L2, L3(recap) | — | — | low | *exercise-only* | nb in exercise 02; **no method page** (drill only) |
| Backprop / gradient descent | L8, L9 | nb16/17/18/19 | — | med | cross-cutting (NN math) | ✓ |
| Metrics (confusion/precision/recall/AUC) | L5 | nb09, exercises 04/05 | all (evaluation) | high | cross-cutting (cv_info / big_picture) | ✓ |

## Findings that confirm the Tier split
- **Every Tier-1 method recurs across all three source types** (lecture + notebook + exam), except LDA and Adaline which lack an exam appearance but are lecture-core with dedicated notebooks → kept Tier 1 (LDA has a full comparison-table lecture treatment; Adaline is the pedagogical bridge L8→logistic→NN).
- **The entire NN block (Tier 2) has ZERO exam appearances.** The 2025 take-home has no neural-network assignment. This directly corroborates the prompt's rationale: "the course de-emphasized the NN block in the final lectures" — L10 is a *survey* (no formulas), and the exam omits NNs entirely. NN block → Tier 2 confirmed.
- **Bias-variance and CV are the most cross-cutting concepts** (4 lectures each, every exam assignment tunes via CV) → justifies their dedicated cross-cutting pages and heavy cross-linking.
- **Naive Bayes** is taught in lecture (L2/L3) and has a worked exercise (02_VHL) but no dedicated notebook beyond the exercise and no exam appearance → **drill-only**, no method page (consistent with the prompt's 16-method list, which excludes it).

## Exam-assignment → method mapping (drives Past-Exam Lens, R8)
- **Asgmt 1** (Hitters): OLS, Ridge, Lasso, CV α-tuning, bias-variance interpretation, coefficient reading.
- **Asgmt 2** (WholesaleCustomers): scaling, PCA (90% var), K-means (elbow + silhouette), cluster interpretation.
- **Asgmt 3** (WDBC): decision tree, random forest, gradient boosting, CV tuning, feature importance, RF-vs-GB discussion.
- **Asgmt 4** (SimulatedData, hidden hold-out): pick ≥2 tuned classifiers (logistic / kNN / tree / RF / etc.), build a generalizable predict pipeline.
