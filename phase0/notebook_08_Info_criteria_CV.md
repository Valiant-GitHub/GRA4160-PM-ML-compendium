# Notebook extract: 08_Information_criteria_and_cross_validation (1).ipynb

**Source path:** course_materials/Lecture notebooks/08_Information_criteria_and_cross_validation (1).ipynb
**Cell count:** 13 (cells 0-12)

## Dataset(s) loaded
- **Diabetes** dataset from sklearn [cell 2]:
  - `from sklearn.datasets import load_diabetes`
  - `X, y = load_diabetes(return_X_y=True, as_frame=True)`
  - 442 samples, 10 continuous features (age, sex, BMI, BP, S1-S6); target = disease progression (real-valued ~25-346)
  - [cell 3] **8 extra random noise features** appended: column names `random_00 ... random_07`, generated with `rng = np.random.RandomState(10)` then `rng.randn(X.shape[0], 8)`, concatenated via `pd.concat([X, X_random], axis=1)`

## Preprocessing steps
- [cell 2] `X = X/X.std()` — scale features by their standard deviation
- [cell 3] Append 8 random features (see above) — to test whether selection methods correctly drop noise
- [cell 6] Within pipelines: `StandardScaler()` applied before each Lasso estimator via `make_pipeline`
- [cell 6] `simplefilter(action='ignore', category=FutureWarning)`

## Method(s) demonstrated
- **LassoLarsIC** (Lasso with AIC/BIC information criterion) — sklearn. `from sklearn.linear_model import LassoLarsIC` [cell 6]
- **LassoCV** (coordinate descent, cross-validated) — sklearn. `from sklearn.linear_model import LassoCV` [cell 11]
- **LassoLarsCV** (least angle regression, cross-validated) — sklearn. `from sklearn.linear_model import LassoLarsCV` [cell 12]
- Helpers: `from sklearn.preprocessing import StandardScaler`, `from sklearn.pipeline import make_pipeline` [cell 6]
- No from-scratch implementation. AIC/BIC formulas are markdown theory [cell 5].

## Hyperparameters set
- [cell 6] `LassoLarsIC(criterion="aic")` inside `make_pipeline(StandardScaler(), LassoLarsIC(criterion="aic"))`
- [cell 6] Switched via `lasso_lars_ic.set_params(lassolarsic__criterion="bic")` then refit
- [cell 11] `LassoCV(cv=10)` inside `make_pipeline(StandardScaler(), LassoCV(cv=10))`
- [cell 12] `LassoLarsCV(cv=20)` inside `make_pipeline(StandardScaler(), LassoLarsCV(cv=20))`

## Plots produced
- [cell 8] AIC vs BIC criterion as a function of alpha (log x-axis). `results.plot()` with vertical lines at `alpha_aic` (blue dashed) and `alpha_bic` (orange dashed). x = $\alpha$ (log scale), y = "criterion". Title "Information-criterion for model selection".
- [cell 11] MSE path per fold (`lasso.mse_path_`) vs alpha (semilogx), dotted lines per fold + black mean line; vertical dashed line at `lasso.alpha_` (CV estimate). x = $\alpha$, y = "Mean square error". ylim (2300, 3800). Title "Mean square error on each fold: coordinate descent".
- [cell 12] Same style for LassoLarsCV using `lasso.cv_alphas_` and `lasso.mse_path_`. Title "Mean square error on each fold: Lars".

## What is left as an exercise to the student
- None explicitly stated (no exercise/extension cells). The notebook is fully worked.

## Key cell indices for code idiom extraction
- "[cell 3]: random-noise feature injection — `rng = np.random.RandomState(10); pd.DataFrame(rng.randn(X.shape[0], n), columns=[f'random_{i:02d}' for i in range(n)])`"
- "[cell 6]: `make_pipeline(StandardScaler(), LassoLarsIC(criterion='aic')).fit(X, y)`" and accessing last step with `lasso_lars_ic[-1]` (`.alphas_`, `.criterion_`, `.alpha_`, `.coef_`)
- "[cell 6]: switch criterion via `set_params(lassolarsic__criterion='bic')`" — pipeline step-param naming idiom
- "[cell 6]: `zero_coefs = np.where(coefs == 0)[0]`" — find features zeroed by Lasso
- "[cell 11]: `plt.semilogx(lasso.alphas_, lasso.mse_path_, ...)` + `lasso.mse_path_.mean(axis=-1)`" — CV MSE-path plot idiom
- "[cell 12]: `LassoLarsCV` uses `lasso.cv_alphas_` (note different attribute name vs LassoCV's `alphas_`)"

## Notes / [VERIFY] flags
- Markdown formulas [cell 5]: $AIC = 2k - 2\log(L)$; $BIC = k\log(n) - 2\log(L)$.
- Attribute-name difference is real and important: LassoCV exposes `.alphas_` [cell 11], LassoLarsCV exposes `.cv_alphas_` [cell 12]. Not a mismatch — a genuine API distinction worth preserving.
- **Inconsistent cv values across the two CV examples:** LassoCV uses `cv=10` [cell 11] while LassoLarsCV uses `cv=20` [cell 12]; markdown [cell 11/intro] describes "10-fold" generally. Not strictly an error but a code/prose inconsistency for the Lars example. `[VERIFY: intended cv=20 for Lars?]`
- `X = X/X.std()` [cell 2] scales by std but does NOT center; pipelines then also apply `StandardScaler` (double scaling). Not flagged in notebook.
