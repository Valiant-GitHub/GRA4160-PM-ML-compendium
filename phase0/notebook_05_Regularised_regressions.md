# Notebook extract: 05_Regularised_regressions (1).ipynb

**Source path:** course_materials/Lecture notebooks/05_Regularised_regressions (1).ipynb
**Cell count:** 22 cells (indices cell-0 through cell-21 as reported by Read; cell-21 empty)

## Dataset(s) loaded
- NO external dataset. Synthetic data generated MANUALLY in [cell 3]:
  - `np.random.seed(15)`
  - `X = np.random.rand(n_samples, n_features)` — shape (100, 10), Uniform(0,1).
  - `y = 5.5*X[:,0] + 1.75*X[:,1] + 1*X[:,2] - 2*X[:,4] + 0.6*X[:,6] - 9.6*X[:,8] + 5.0*np.random.randn(n_samples)`
  - TRUE coefficient vector (stated in comment): `[5.5, 1.75, 1, 0, -2, 0, 0.6, 0, -9.6, 0]` — features 3, 5, 7, 9 (0-indexed) have zero true effect.
  - `n_samples = 100`, `n_features = 10`.
  - A commented-out alternative is shown: `# X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=5, noise=10)` (NOT executed).

## Preprocessing steps
- [cell 3] `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)`
- No scaling applied (note: real-world regularization usually requires standardized features; omitted here).

## Method(s) demonstrated
All SKLEARN (no from-scratch implementations in this notebook). Four estimators + a path algorithm:
1. `from sklearn.linear_model import LinearRegression` — [cell 6] `lm = LinearRegression()`. Output `lm.coef_` = `[4.66641132, 1.54531601, 2.09706399, -2.24700274, -2.64803006, -4.09745672, 0.76401839, 0.89822429, -8.67122458, -0.19510997]`.
2. `from sklearn.linear_model import Ridge` — [cell 10] `ridge = Ridge(alpha=0.8)`. Output `ridge.coef_` = `[4.27719127, 1.40510861, 2.19887963, -2.05421924, -2.23247916, -3.70431413, 0.61717232, 0.80001056, -7.60033337, -0.15918099]`.
3. `from sklearn.linear_model import Lasso` — [cell 12] `lasso = Lasso(alpha=0.1)`. Output `lasso.coef_` = `[3.78905786, 0.52088693, 2.07723694, -1.20311347, -1.32725437, -3.1988485, 0., 0., -7.17367499, 0.]` (3 coefficients driven exactly to zero — feature selection).
4. `from sklearn.linear_model import ElasticNet` — [cell 14] `en = ElasticNet(alpha=0.1, l1_ratio=0.5)`. Output `en.coef_` = `[2.92177487, 0.75918837, 1.92784433, -1.22486922, -1.03115498, -2.39525919, 0., 0.21852892, -4.70707567, -0.]`.
5. `from sklearn.linear_model import lars_path` (LARS / Least Angle Regression) — [cell 19] `_, _, coefs = lars_path(X_train, y_train, method="lasso", verbose=True)` to compute the full LASSO path.
- Evaluation: `from sklearn.metrics import mean_squared_error` [cell 16].

## Hyperparameters set
- [cell 3] `train_test_split(test_size=0.2, random_state=15)`; `np.random.seed(15)`.
- [cell 6] `LinearRegression()` — all `default`.
- [cell 10] `Ridge(alpha=0.8)` — other params `default`.
- [cell 12] `Lasso(alpha=0.1)` — other params `default`.
- [cell 14] `ElasticNet(alpha=0.1, l1_ratio=0.5)` — other params `default`.
- [cell 19] `lars_path(X_train, y_train, method="lasso", verbose=True)` — no `alpha`/max-iter set.
- NOTE (markdown cell 13): in scikit-learn `alpha` == the textbook λ; the textbook mixing parameter α == sklearn's `l1_ratio`.

## Plots produced
- [cell 4] `plt.plot(X)` — line plot of all 10 feature columns (figsize=(12,5)). (Just visualizing the raw design matrix.)
- [cell 19] LASSO path plot (figsize=(12,5)): `plt.plot(xx, coefs.T, lw=2)` where `xx = Σ|coef| / max(Σ|coef|)` (normalized L1 norm on x-axis), coefficient trajectories on y-axis, with `plt.vlines(xx, ymin, ymax, linestyle="dashed")` marking each LARS step. x="|coef| / max|coef|", y="Coefficients", title "LASSO Path".

## What is left as an exercise to the student
- No explicit coded exercise; cell-21 empty. Markdown section "Selecting the hyperparameters (the λ's and α)" [cell 17] is a header with NO accompanying code — the actual selection of α via CV is implicitly left to the student (the notebook only demonstrates the LARS path as a tool for inspecting how features enter). [VERIFY: confirm there is no hidden CV cell — cell 18 onward jumps straight to LARS description.]

## Key cell indices for code idiom extraction
- "[cell 3]: manual sparse-DGP idiom — known true coeffs `[5.5,1.75,1,0,-2,0,0.6,0,-9.6,0]`, `np.random.seed(15)`, additive `5.0*np.random.randn`."
- "[cell 6/10/12/14]: the four-estimator comparison — `LinearRegression()`, `Ridge(alpha=0.8)`, `Lasso(alpha=0.1)`, `ElasticNet(alpha=0.1, l1_ratio=0.5)`, each `.fit(X_train,y_train)` then `.coef_`."
- "[cell 16]: MSE comparison loop — `mean_squared_error(y_test, model.predict(X_test))` across all four models."
- "[cell 19]: LASSO-path idiom — `lars_path(X_train, y_train, method='lasso')` then normalized-L1 path plot."
- "[cell 20]: `pd.DataFrame(coefs.T)` to read off which features survive at each LARS step."

## Notes / [VERIFY] flags
- Markdown cost functions transcribed exactly:
  - Ridge: `J^ridge(β) = (1/2n) Σ_i (y_i − β₀ − Σ_j β_j x_ij)² + λ_r Σ_j β_j²` (L2).
  - Lasso: `J^lasso(β) = (1/2n) Σ_i (...)² + λ_l Σ_j |β_j|` (L1).
  - Elastic Net: `J^en(β) = (1/2n) Σ_i (...)² + λ_e ( (1−α) Σ_j β_j² + α Σ_j |β_j| )`; α=0 → Ridge, α=1 → Lasso.
  - NOTE the elastic-net markdown formula uses `(1-α)` on the L2 term and `α` on the L1 term; sklearn's `ElasticNet` uses `l1_ratio` (= α here) where l1_ratio multiplies the L1 part. With `l1_ratio=0.5` the split is even. [VERIFY: the markdown's 1/2 factor and parameterization differ slightly from sklearn's exact objective (sklearn: `1/(2n)||y-Xβ||² + alpha*l1_ratio*||β||₁ + 0.5*alpha*(1-l1_ratio)*||β||₂²`) — flag for precision when teaching.]
  - RSS (cell 5) has a typo: the squared exponent is dropped on the middle expansion (`Σ_i(y_i − ŷ_i)² = Σ_i(y_i − β₀ − ... − β_n x_in) = ...²`) — the middle term is missing its square. Transcribe corrected.
- MSE outputs [cell 16]: Linear 20.7316, Ridge 19.9094, Lasso 19.7805, Elastic Net 19.7715 — all three regularized models beat plain OLS on this synthetic test set; Elastic Net lowest.
- Lasso ([cell 12]) zeros features at indices 6, 7, 9; ElasticNet ([cell 14]) zeros indices 6, 9 (keeps a small index-7 coeff). Compare against the TRUE zero indices {3,5,7,9} — neither perfectly recovers the sparsity pattern (features 0–9 unscaled, correlated noise).
- [cell 20] `pd.DataFrame(coefs.T)` output (11 LARS steps × 10 features) is fully stored in the read; feature 8 (true coef −9.6) enters first, feature 0 (true 5.5) second — consistent with strongest effects entering earliest.
- Header says "Lecture 3" (same as notebook 04); likely a copy-paste label. [VERIFY: actual lecture number.]
