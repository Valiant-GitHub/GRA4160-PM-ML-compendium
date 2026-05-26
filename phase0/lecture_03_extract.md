# Lecture 3 — LDA and Regularized Regression

**Source:** `course_materials/Lecture slides/lecture3_260521_184932.pdf` (24 PDF pages = 20 logical slides; cite `n/20`). Date: Jan 21 2026.

## Topic
Linear Discriminant Analysis (classification + dim reduction); Ridge, Lasso, Elastic Net regularization.

## Key concepts taught — LDA
- **Definition (slide 6/20):** both a classification and dimensionality-reduction technique. Core idea: project data onto lower-dim space (often 1D) maximizing class separability.
- **When to use (slide 7/20):** 2+ classes (multi-class via multiple discriminant components); works when classes well-separated and **covariance similar across classes**; for dimensionality reduction. Comparison: vs **Logistic Regression** (also linear, no dim reduction); vs **QDA** (different covariance per class, more complex); vs **kNN** (non-parametric, less interpretable in high-dim); vs **PCA** (unsupervised, variance not class separation).
- **Math foundations (slide 8/20):** maximize between-class variance / minimize within-class variance. Scatter matrices: `S_B` (between-class, how class means differ), `S_W` (within-class, spread around class means). Optimization:
  - `w = argmax_w (wᵀ S_B w) / (wᵀ S_W w)`
- **How w computed (slide 9/20):** class means `m_i = (1/N_i) Σ_{x∈class i} x`; overall mean `m = (1/N) Σ_i Σ_{x∈class i} x`. Generalized eigenvalue problem: `S_W⁻¹ S_B w = λ w`; eigenvectors with largest λ = most discriminative directions. **Two-class analytic solution:** `w ∝ S_W⁻¹ (m_1 − m_2)`.
- **Assumptions (slide 10/20):** 1. Normality (each class Gaussian); 2. Identical covariance Σ across classes; 3. (usually) feature independence, mild correlations tolerated. If covariances differ → QDA. Normality violations often not fatal but reduce optimality.
- **sklearn (slide 11/20):** `LinearDiscriminantAnalysis` from `sklearn.discriminant_analysis`. Params: `solver` (svd/lsqr/eigen), `shrinkage`, `n_components` (≤ #classes − 1). Attributes: `coef_`, `explained_variance_ratio_`, `means_`. Tip: scale/standardize features.

## Key concepts taught — Regularized regression
- **Overview (slide 13/20):** add penalty to cost `Σ(y−ŷ)²` to control coefficient magnitude, improve generalization, some do feature selection.
- **Ridge (L2) (slide 14/20):** `J_ridge(β) = (1/2n) Σ_i (y_i − β_0 − Σ_j β_j x_ij)² + λ_r Σ_j β_j²`. Shrinks coefficients but rarely exactly zero. Good when many small nonzero effects; helps with multicollinearity. Larger λ_r → stronger shrinkage.
- **Lasso (L1) (slide 15/20):** `J_lasso(β) = (1/2n) Σ_i (y_i − β_0 − Σ_j β_j x_ij)² + λ_ℓ Σ_j |β_j|`. Coefficients can become **exactly zero** (feature selection). **LARS** computes full regularization path. For high-dim with irrelevant features.
- **Elastic Net (slide 17/20):** `J(β) = (1/2n) Σ_i (y_i − β_0 − Σ_j β_j x_ij)² + λ_e [ (1−α) Σ_j β_j² + α Σ_j |β_j| ]`. α∈[0,1] mixes L2/L1. Balances ridge stability + lasso sparsity; good with correlated features.
- **Geometry (slide 16/20):** Ridge penalty region circular (L2), uniform shrink; Lasso diamond-shaped (L1), corners force coefficients to zero. *Source attributed on slide: Hastie, Tibshirani & Friedman (2009)* = ESL.
- **Choosing λ (slide 19/20):** λ=0 → OLS; λ→∞ → all coefficients → 0. Use cross-validation over a grid; pick λ minimizing validation error or **1-SE rule**. Standardize features first.
- **sklearn (slide 20/20):** `Ridge(alpha=1.0)`, `Lasso()`, `ElasticNet()` from `sklearn.linear_model`. Tune with `GridSearchCV`/`RandomizedSearchCV`. Pitfalls: forgetting to scale; over-penalizing → underfit.

## Notation (LDA + reg) — for notation_table
- `S_B`, `S_W` scatter matrices; `m_i` class mean; `w` projection/discriminant direction; `λ` eigenvalue (LDA) AND regularization strength (reg) — **symbol collision**: LDA uses λ for eigenvalues, regularization uses λ_r/λ_ℓ/λ_e. Note in notation_table.
- Reg penalty uses `(1/2n)` normalization on the SSE term (matches sklearn's mean-based objective). β_0 intercept un-penalized.

## R9 cross-check flags (vs ESL)
- Ridge/Lasso objectives: slide uses `1/(2n)` prefactor and explicit intercept β_0; ESL §3.4 writes RSS without the `1/(2n)` and centers data so intercept drops. **Equivalent up to scaling of λ** — note in math section, not a disagreement.
- LDA Fisher criterion and `w ∝ S_W⁻¹(m_1−m_2)`: matches ESL §4.3 / standard Fisher LDA. Agree.

## Professor emphasis cues
- LDA taught with explicit **comparison table** vs logistic/QDA/kNN/PCA — strong "when to use which" framing → feeds decision dashboard & showdowns.
- Standardization-before-regularization stressed twice (pitfall).

## Companion materials
Lecture notebooks `04_Linear_discriminant_analysis`, `05_Regularised_regressions`; exercise `03_Predicting_house_prices`.

## Cross-refs
→ `methods/lda.qmd`, `methods/regularization.qmd`, `methods/logistic.qmd` (comparison), `methods/pca.qmd` (comparison), `methods/cv_info_criteria.qmd` (λ selection).
