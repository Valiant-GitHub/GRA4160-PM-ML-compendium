# Notebook extract: 05_Regularised_regressions (1).ipynb

**Source path:** `Lecture notebooks/05_Regularised_regressions (1).ipynb`
**Cell count:** 22 (cell-0 through cell-21)

## Dataset(s) loaded

**Synthetic regression data** — manually generated in cell-3 (no external file):
```python
np.random.seed(15)
X = np.random.rand(n_samples, n_features)   # shape (100, 10)
y = 5.5*X[:,0] + 1.75*X[:,1] + 1*X[:,2] - 2*X[:,4] + 0.6*X[:,6] - 9.6*X[:,8] + 5.0*np.random.randn(n_samples)
```
- `n_samples = 100`, `n_features = 10`
- True coefficient vector: `[5.5, 1.75, 1, 0, -2, 0, 0.6, 0, -9.6, 0]` (columns 3, 5, 7, 9 are zero)
- Noise std: `5.0`
- No named columns; numpy array only.

Note: `make_regression` call is present but commented out in cell-3.

## Preprocessing steps

- [cell-3] `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)`
- No feature scaling applied (not demonstrated in this notebook — regularised regression is fit on raw features)

## Method(s) demonstrated

All methods are **sklearn-based** (no from-scratch implementation in this notebook):

1. `from sklearn.linear_model import LinearRegression` — [cell-6] baseline OLS
2. `from sklearn.linear_model import Ridge` — [cell-10] L2 regularisation
3. `from sklearn.linear_model import Lasso` — [cell-12] L1 regularisation
4. `from sklearn.linear_model import ElasticNet` — [cell-14] combined L1+L2
5. `from sklearn.linear_model import lars_path` — [cell-19] LARS algorithm for Lasso path
6. `from sklearn.metrics import mean_squared_error` — [cell-16] evaluation

## Hyperparameters set

- `LinearRegression()` — all `default`
- `Ridge(alpha=0.8)` — `alpha=0.8`; all others `default`
- `Lasso(alpha=0.1)` — `alpha=0.1`; all others `default`
- `ElasticNet(alpha=0.1, l1_ratio=0.5)` — `alpha=0.1`, `l1_ratio=0.5`; all others `default`
- `lars_path(X_train, y_train, method="lasso", verbose=True)` — `method="lasso"`, `verbose=True`; all others `default`

Note: In sklearn `alpha` corresponds to $\lambda$ in the mathematical notation used in the notebook. What the notebook calls $\alpha$ (mixing parameter) is `l1_ratio` in sklearn — this mapping is explicitly called out in cell-13.

## Plots produced

- [cell-4] Line plot: all 10 feature columns of `X` plotted over sample index. `figsize=(12,5)`. No axis labels. Used to visualise raw feature data.
- [cell-19] LASSO path plot:
  - x-axis: `|coef| / max|coef|` (L1-norm fraction)
  - y-axis: `Coefficients`
  - Lines: one per feature (10 lines), coloured by matplotlib default cycle
  - Vertical dashed lines at each knot
  - Title: `LASSO Path`; `figsize=(12,5)`

## What is left as an exercise to the student

No explicit TODO / "your turn" cells found in this notebook. Cell-17 is a markdown header ("Selecting the hyperparameters (the λ's and α)") but no accompanying code or exercise is provided — [VERIFY: was cross-validation code intended here but omitted?]

## Key cell indices for code idiom extraction

- [cell-3]: True sparse DGP construction with named coefficient magnitudes
- [cell-6]: `lm = LinearRegression(); lm.fit(X_train, y_train); print(lm.coef_)` — OLS baseline
- [cell-10]: `ridge = Ridge(alpha=0.8); ridge.fit(X_train, y_train)` — Ridge fit
- [cell-12]: `lasso = Lasso(alpha=0.1); lasso.fit(X_train, y_train)` — Lasso fit (note zero coefficients in output)
- [cell-14]: `en = ElasticNet(alpha=0.1, l1_ratio=0.5); en.fit(X_train, y_train)` — ElasticNet fit
- [cell-16]: MSE comparison across all 4 models — canonical evaluation block
- [cell-19]: `lars_path(X_train, y_train, method="lasso", verbose=True)` + normalised path plot

## Notes / [VERIFY] flags

- Cell-5 markdown — Ridge cost function:
  $$J^{\text{ridge}}(\mathbf{\beta}) = \frac{1}{2n}\sum_i\left(y_i - \beta_0 - \sum_j\beta_jx_{ij}\right)^2 + \lambda_r \sum_j \beta_j^2$$
- Cell-11 markdown — Lasso cost function:
  $$J^{\text{lasso}}(\mathbf{\beta}) = \frac{1}{2n}\sum_i\left(y_i - \beta_0 - \sum_j\beta_jx_{ij}\right)^2 + \lambda_l \sum_j|\beta_j|$$
- Cell-13 markdown — Elastic Net cost function:
  $$J^{\text{elastic net}}(\mathbf{\beta}) = \frac{1}{2n}\sum_i\left(y_i - \beta_0 - \sum_j\beta_jx_{ij}\right)^2 + \lambda_{e} \left( (1-\alpha)\sum_j \beta_j^2 + \alpha \sum_j|\beta_j|\right)$$
- Cell-16 output (MSE on test set):
  - Linear model: `20.7316`
  - Ridge: `19.9094`
  - Lasso: `19.7805`
  - Elastic Net: `19.7715`
- Cell-12 output shows Lasso zeros out coefficients for columns 6 and 7 (0-indexed) — features `X[:,6]` (true coeff `0.6`) and `X[:,7]` (true coeff `0`). Note it also zeros column 9 (true coeff `0`).
- Cell-14 output shows ElasticNet zeros out columns 6 and 9 but retains column 7 with a small coefficient.
- Cell-21 is empty.
- The LARS path DataFrame (cell-20) has 11 rows (knots) and 10 columns (features). Feature 8 (true coeff `-9.6`) enters the model second (row index 1) with a large negative coefficient, confirming it is the strongest predictor.
- [VERIFY: cell-17 header "Selecting the hyperparameters" has no accompanying cross-validation code — the section appears incomplete.]
