# Notebook extract: 09_Bias_variance_tradeoff.ipynb

**Source path:** course_materials\Lecture notebooks\09_Bias_variance_tradeoff.ipynb
**Cell count:** 14 (cell-0 through cell-13)

## Dataset(s) loaded
- Synthetic dataset generated in-notebook (cell-3):
  - `np.random.seed(10)`
  - `X = np.sort(5 * np.random.rand(80, 1), axis=0)` — 80 samples, 1 feature, uniform in [0, 5]
  - `y = np.sin(X).ravel() + 0.2 * np.random.randn(80)` — noisy sine wave (noise std = 0.2)

## Preprocessing steps
- No explicit preprocessing. Data generated directly as numpy arrays.
- Feature transformation happens inside the pipeline via `PolynomialFeatures`.

## Method(s) demonstrated
- `from sklearn.pipeline import Pipeline` (cell-3)
- `from sklearn.preprocessing import PolynomialFeatures` (cell-3)
- `from sklearn.linear_model import LinearRegression` (cell-3)
- `from sklearn.model_selection import cross_val_score` (cell-3)
- Polynomial regression of varying degree as sklearn Pipeline — NOT from scratch
- Function `polynomial_regression(degree, cv=10)` defined in cell-5:
  - Builds `Pipeline([('poly', PolynomialFeatures(degree=degree)), ('linear', LinearRegression(fit_intercept=False))])`
  - Computes `cross_val_score(model, X, y, cv=cv)` (10-fold by default)
  - Fits model on full data; plots predictions

## Hyperparameters set
- `LinearRegression(fit_intercept=False)` — intercept suppressed (cell-5)
- `PolynomialFeatures(degree=degree)` — degree varied: 1, 2, 3, 4, 8, 20 (cells 6, 8, 9, 10, 11, 12)
- `cross_val_score(..., cv=10)` — 10-fold CV default in `polynomial_regression()` function (cell-5)

## Plots produced
- [cell-4]: Scatter plot — X vs y (raw data); marker 'o', markersize=10; label='data'; figure size (15, 5)
- [cell-6]: Line + scatter — data points + PR degree=1 predictions; label includes cross-val error; figure size (12, 5); saved to `../tex/figures/bias_variance_tradeoff_1.png`
- [cell-8]: Same format for degree=2; saved to `../tex/figures/bias_variance_tradeoff_2.png`
- [cell-9]: Same format for degree=3 (no save)
- [cell-10]: Same format for degree=4; saved to `../tex/figures/bias_variance_tradeoff_4.png`
- [cell-11]: Same format for degree=8 (no save)
- [cell-12]: Same format for degree=20; saved to `../tex/figures/bias_variance_tradeoff_20.png`

## What is left as an exercise to the student
- No explicit TODO/exercise cells. All degrees are pre-run. Markdown cells (cell-7, cell-13) invite reflection on underfitting (degree=1) and overfitting (degree=20).

## Key cell indices for code idiom extraction
- "[cell-3]: `X = np.sort(5 * np.random.rand(80, 1), axis=0); y = np.sin(X).ravel() + 0.2 * np.random.randn(80)` — noisy sine wave generation"
- "[cell-5]: `model = Pipeline([('poly', PolynomialFeatures(degree=degree)), ('linear', LinearRegression(fit_intercept=False))]); scores = cross_val_score(model, X, y, cv=cv)` — polynomial regression + CV pipeline"
- "[cell-5]: `label=f'PR with degree={degree} (error={np.mean(np.abs(scores)):.3f})'` — mean absolute CV score in plot label"

## Notes / [VERIFY] flags
- `cross_val_score` default scoring for LinearRegression is R^2, not MSE — but the label says "error" and uses `np.mean(np.abs(scores))`. Since R^2 can be negative for very bad models, `np.abs` prevents display of negative error. [VERIFY: instructor may intend this as |1 - R^2| or simply |R^2| as a rough fit measure.]
- `fit_intercept=False` in LinearRegression combined with PolynomialFeatures — the bias term is included in the polynomial expansion (PolynomialFeatures includes the constant column by default), so intercept is effectively present via the degree-0 term.
- Figures are saved to `../tex/figures/` — indicates integration with a LaTeX document.
- The notebook is labelled "Lecture 5" (cell-0).
- Concept definitions (bias, variance) are in pure markdown (cells 1, 2, 7, 13) with no formulas transcribed — conceptual prose only.
