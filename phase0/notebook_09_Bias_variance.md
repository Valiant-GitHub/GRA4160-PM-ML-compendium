# Notebook extract: 09_Bias_variance_tradeoff.ipynb

**Source path:** Lecture notebooks/09_Bias_variance_tradeoff.ipynb
**Cell count:** 14 (cells 0-13)

## Dataset(s) loaded
- **Synthetic noisy sinusoidal data** (generated, no file) [cell 3]:
  - `np.random.seed(10)`
  - `X = np.sort(5 * np.random.rand(80, 1), axis=0)` — 80 points in [0, 5)
  - `y = np.sin(X).ravel() + 0.2 * np.random.randn(80)` — sine + Gaussian noise (std 0.2)
  - No external dataset, no train/test split (uses cross-validation instead).

## Preprocessing steps
- [cell 3] Data generated as above; X sorted ascending.
- No scaling/encoding. `PolynomialFeatures` expansion done inside the pipeline (per call).

## Method(s) demonstrated
- **Polynomial regression** = `PolynomialFeatures` + `LinearRegression` in a `Pipeline`, evaluated with k-fold cross-validation — sklearn. [cell 5]
- Imports [cell 3]: `from sklearn.pipeline import Pipeline`, `from sklearn.preprocessing import PolynomialFeatures`, `from sklearn.linear_model import LinearRegression`, `from sklearn.model_selection import cross_val_score`
- No from-scratch implementation. Demonstrates bias-variance via increasing polynomial degree.

## Hyperparameters set
- [cell 5] `polynomial_regression(degree, cv=10)` function:
  - `Pipeline([('poly', PolynomialFeatures(degree=degree)), ('linear', LinearRegression(fit_intercept=False))])`
  - `LinearRegression(fit_intercept=False)` — intercept absorbed by the degree-0 polynomial term
  - `cross_val_score(model, X, y, cv=cv)` with `cv=10` default
- Degrees actually called: **1** [cell 6], **2** [cell 8], **3** [cell 9], **4** [cell 10], **8** [cell 11], **20** [cell 12]

## Plots produced
- [cell 4] Scatter of raw data: `plt.plot(X, y, 'o', ...)`, figsize (15, 5).
- [cells 6, 8, 9, 10, 11, 12] One plot per degree (figsize (12, 5)): scatter of data (`'o'`) + dashed fitted curve labeled `PR with degree={degree} (error={mean abs CV score})`. x = X, y = y / prediction.
- Saved figures: [cell 6] `../tex/figures/bias_variance_tradeoff_1.png`; [cell 8] `..._2.png`; [cell 10] `..._4.png`; [cell 12] `..._20.png`. (degrees 3 and 8 NOT saved.)

## What is left as an exercise to the student
- None explicitly stated (no exercise cells). Conceptual prose only.

## Key cell indices for code idiom extraction
- "[cell 3]: synthetic sine data — `X = np.sort(5 * np.random.rand(80, 1), axis=0); y = np.sin(X).ravel() + 0.2 * np.random.randn(80)`"
- "[cell 5]: `Pipeline([('poly', PolynomialFeatures(degree=degree)), ('linear', LinearRegression(fit_intercept=False))])`" — poly-regression pipeline idiom
- "[cell 5]: `cross_val_score(model, X, y, cv=cv)` then `np.mean(np.abs(scores))`" — CV scoring inside a plotting helper

## Notes / [VERIFY] flags
- No markdown formulas (conceptual discussion of bias vs variance only).
- **[VERIFY] scoring sign/metric:** `cross_val_score` is called with no `scoring=` argument. For a regression pipeline this defaults to R^2, NOT MSE. The plot label and markdown call this an "error" (`error={np.mean(np.abs(scores)):.3f}`), and markdown [cell 2] says "10-fold cross-validation to estimate the mean squared error" — but the code actually averages |R^2|. This is a **code/prose mismatch**: prose says MSE, code computes mean absolute R^2.
- `LinearRegression(fit_intercept=False)` is intentional because `PolynomialFeatures` includes a bias (degree-0) column by default.
- Figure save paths assume a `../tex/figures/` directory exists relative to the notebook; will error if absent.
