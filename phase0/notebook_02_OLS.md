# Notebook extract: 02_OLS (1).ipynb

**Source path:** course_materials/Lecture notebooks/02_OLS (1).ipynb
**Cell count:** 15 cells (indices cell-0 through cell-14 as reported by Read; cell-14 empty)

## Dataset(s) loaded
- NO external dataset. Data is synthetically generated in [cell 3]:
  - `X = np.random.rand(n_samples, n_features)` — features from Uniform(0,1), shape (100, 1).
  - `coeffs = np.random.rand(n_features).round(2)` — true coefficients from Uniform(0,1), rounded to 2 dp.
  - `intercept = np.random.rand(1).round(2)[0]` — true intercept from Uniform(0,1).
  - Data-generating process (with intercept): `y = intercept + np.dot(X, coeffs) + 0.2*np.random.randn(n_samples)`; noiseless line `y_true = intercept + np.dot(X, coeffs)`.
- No target column name (synthetic `y` array). No `np.random.seed` set, so values differ per run.

## Preprocessing steps
- [cell 3] config flags: `n_features = 1`, `n_samples = 100`, `has_intercept = True`.
- [cell 8] manual design-matrix augmentation: `X_ = np.column_stack((np.ones(n_samples), X))` (prepend column of ones for intercept).
- [cell 10] inside class: `X = np.hstack([np.ones((X.shape[0], 1)), X])` (same intercept-augmentation idiom).
- No train/test split; whole synthetic sample used for fitting and prediction.

## Method(s) demonstrated
Ordinary Least Squares (OLS) regression, demonstrated THREE ways:
1. FROM SCRATCH (numpy, normal equations) — [cell 8]: `betas = np.linalg.inv(X_.T@X_)@(X_.T@y)` (closed form `(XᵀX)⁻¹Xᵀy`).
2. FROM SCRATCH (numpy, OOP) — [cell 10]: custom `class OLS` with `fit_intercept`, `fit()`, `predict()`. Uses `np.linalg.solve(XTX, XTY)` (solves the normal equations rather than explicit inverse).
3. SKLEARN cross-check — [cell 13]: `import sklearn.linear_model as lm`; `model_sk = lm.LinearRegression(fit_intercept=has_intercept)`.

Import paths:
- `numpy as np`, `pandas as pd`, `matplotlib.pyplot as plt` [cell 2].
- `import sklearn.linear_model as lm` [cell 13]; class `lm.LinearRegression`.

## Hyperparameters set
- [cell 11] `OLS(fit_intercept=has_intercept)` → `fit_intercept=True` (from cell 3 flag).
- [cell 13] `lm.LinearRegression(fit_intercept=has_intercept)` → `fit_intercept=True`. All other LinearRegression params: `default`.
- DGP noise scale: `0.2 * np.random.randn(...)` [cell 3].

## Plots produced
- [cell 5] Scatter + regression-line plot (only when `n_features == 1`): `plt.scatter(X, y)` (data points), `plt.plot(X, y_true, color='red')` (true DGP line labeled `$y={intercept} + {coeffs[0]}x$`), plus green dashed residual segments from 20 randomly selected points to the true line (`plt.plot([X[i],X[i]], [y[i],y_true[i]], color='green', linestyle='--')`). x-axis `$x$-value`, y-axis `$y$-value`. figsize=(6,3). Saved to file via `plt.savefig('regression_example.png')`.

## What is left as an exercise to the student
- No explicit exercise prompts. cell-14 is empty. The notebook is fully worked. Implicit extension: the plotting branch prints "Can only plot a two-dimensional figure." when `n_features != 1`, inviting experimentation with more features.

## Key cell indices for code idiom extraction
- "[cell 3]: synthetic-DGP idiom — `X = np.random.rand(n_samples, n_features)`, true coeffs, and `y = intercept + X@coeffs + 0.2*np.random.randn(n_samples)`."
- "[cell 8]: closed-form normal-equation OLS — `betas = np.linalg.inv(X_.T@X_)@(X_.T@y)` with `X_ = np.column_stack((np.ones(n_samples), X))`."
- "[cell 10]: the canonical from-scratch `class OLS` (fit_intercept / fit / predict) using `np.linalg.solve(XTX, XTY)` — lift verbatim."
- "[cell 13]: sklearn cross-check — `lm.LinearRegression(fit_intercept=...).fit(X,y)` then `.coef_` and `.intercept_`."

## Notes / [VERIFY] flags
- Markdown math (cell 7) transcribed exactly:
  - Optimization problem: `argmin_β Σ_{i=1}^n (y_i − βᵀx_i)²`
  - Closed-form solution: `β = (XᵀX)⁻¹Xᵀy`, where X is (n_samples, n_features), y is (n_samples).
- Two solver styles to note: cell 8 uses explicit `np.linalg.inv(...)@...`; cell 10 (OOP) uses the numerically preferable `np.linalg.solve(XTX, XTY)`. Worth flagging the difference when teaching.
- Spelling artifacts in source comments: "regressors or futures" / "random futures" (means features) in cell 3 — verbatim transcription should preserve or silently correct depending on study-site policy. [VERIFY: keep "futures" typo or normalize to "features"?]
- No random seed → coefficient/intercept outputs are non-deterministic. The notebook as read shows no execution outputs for cells 4, 8, 11, 12, 13 (no stored outputs in the file), so exact estimated values are not citable.
- cell 5 redundantly re-imports matplotlib/numpy.
