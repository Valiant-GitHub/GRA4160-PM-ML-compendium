# Notebook extract: 02_OLS (1).ipynb

**Source path:** `course_materials/Lecture notebooks/02_OLS (1).ipynb`
**Cell count:** 15 (cell-0 through cell-14)

## Dataset(s) loaded

No external file loaded. Data is **synthetically generated** in cell-3:
- `X = np.random.rand(n_samples, n_features)` — uniform random features, shape `(100, 1)`
- `y = intercept + np.dot(X, coeffs) + 0.2 * np.random.randn(n_samples)` — linear DGP with Gaussian noise (std=0.2)
- True coefficients drawn from `np.random.rand(n_features).round(2)`; true intercept from `np.random.rand(1).round(2)[0]`

Parameters set in cell-3:
- `n_features = 1`
- `n_samples = 100`
- `has_intercept = True`

## Preprocessing steps

- [cell-8] When `has_intercept=True`: `X_ = np.column_stack((np.ones(n_samples), X))` — prepend column of ones for intercept term (inline, not a separate step)
- [cell-10] OLS class `fit` method: `X = np.hstack([np.ones((X.shape[0], 1)), X])` — same intercept augmentation inside OOP wrapper

## Method(s) demonstrated

Two implementations side by side:

1. **FROM SCRATCH (NumPy)** — direct closed-form solution:
   - [cell-8] `betas = np.linalg.inv(X_.T@X_)@(X_.T@y)`
   - [cell-10] OOP class `OLS` using `np.linalg.solve(XTX, XTY)` (numerically preferred over explicit inverse)

2. **sklearn** — [cell-13] `import sklearn.linear_model as lm; model_sk = lm.LinearRegression(fit_intercept=has_intercept)`

The primary pedagogical focus is the from-scratch implementation. sklearn is shown only for verification.

## Hyperparameters set

- `OLS(fit_intercept=True)` — class parameter
- `lm.LinearRegression(fit_intercept=has_intercept)` — `has_intercept=True` as set in cell-3; all other params `default`

## Plots produced

- [cell-5] Scatter plot: `X` vs `y` (observed data points); line: `X` vs `y_true` (true DGP line, red); residual lines for 20 randomly selected points (green dashed). Axes: `$x$-value` / `$y$-value`. Figure saved as `'regression_example.png'`.

## What is left as an exercise to the student

No explicit TODO / "your turn" cells found in this notebook.

## Key cell indices for code idiom extraction

- [cell-3]: DGP construction — `y = intercept + np.dot(X, coeffs) + 0.2*np.random.randn(n_samples)`
- [cell-8]: One-liner closed-form OLS — `betas = np.linalg.inv(X_.T@X_)@(X_.T@y)`
- [cell-10]: Full OOP `OLS` class with `fit` / `predict` methods using `np.linalg.solve`
- [cell-13]: sklearn cross-check — `lm.LinearRegression(fit_intercept=has_intercept).fit(X, y)`

## Notes / [VERIFY] flags

- The key OLS formula is stated in cell-7 markdown:
  $$\arg\min_{\beta} \sum_{i=1}^n (y_i - \beta^T x_i)^2$$
  Closed-form solution:
  $$\beta = (X^T X)^{-1} X^T y$$
- The from-scratch class in cell-10 uses `np.linalg.solve` (not `np.linalg.inv`) for numerical stability — distinct from the one-liner in cell-8.
- The `predict` method in the OOP class stacks intercept and coefficients as `np.hstack([self.intercept, self.coeffs])` for the dot product.
- cell-14 is blank (empty output cell).
- No train/test split is performed — the entire synthetic dataset is used for fitting and demonstration.
- The notebook does not compute MSE or any evaluation metric beyond printing estimated vs true coefficients.
