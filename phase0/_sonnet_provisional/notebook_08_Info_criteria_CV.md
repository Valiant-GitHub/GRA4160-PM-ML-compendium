# Notebook extract: 08_Information_criteria_and_cross_validation (1).ipynb

**Source path:** course_materials\Lecture notebooks\08_Information_criteria_and_cross_validation (1).ipynb
**Cell count:** 13 (cell-0 through cell-12)

## Dataset(s) loaded
- `sklearn.datasets.load_diabetes(return_X_y=True, as_frame=True)` — 442 samples, 10 continuous features (age, sex, bmi, bp, s1–s6); target: quantitative disease progression (cell-2)
- 8 random noise features appended: `rng.randn(X.shape[0], 8)` named `random_00` through `random_07` (cell-3); total features after concat = 18

## Preprocessing steps
- `X = X / X.std()` — scale each feature to unit variance (mean-centering already done in dataset) (cell-2)
- `np.random.RandomState(10)` for reproducibility of random features (cell-3)
- `StandardScaler()` applied inside pipeline before each Lasso estimator (cells 6, 11, 12)
- No explicit train-test split; all fitting is done on full X, y using cross-validation or IC

## Method(s) demonstrated
- `from sklearn.linear_model import LassoLarsIC` — sklearn, IC-based Lasso model selection (cell-6)
- `from sklearn.linear_model import LassoCV` — sklearn, coordinate-descent Lasso with CV (cell-11)
- `from sklearn.linear_model import LassoLarsCV` — sklearn, LARS-based Lasso with CV (cell-12)
- `from sklearn.pipeline import make_pipeline` — used to chain StandardScaler + Lasso (cells 6, 11, 12)
- `from sklearn.preprocessing import StandardScaler` (cell-6)

## Hyperparameters set
- `LassoLarsIC(criterion="aic")` — initial fit; then updated to `criterion="bic"` via `set_params(lassolarsic__criterion="bic")` (cell-6)
- `LassoCV(cv=10)` — 10-fold cross-validation (cell-11)
- `LassoLarsCV(cv=20)` — 20-fold cross-validation (cell-12)
- All other Lasso hyperparameters: `default`

## Plots produced
- [cell-8]: Line plot — AIC and BIC criterion values vs. alpha (log-scale x-axis); vertical dashed lines at `alpha_aic` and `alpha_bic`; xlabel: `$\alpha$`, ylabel: `criterion`; title: "Information-criterion for model selection"
- [cell-11]: `plt.semilogx` — MSE path per fold + average across folds vs. alpha; vertical dashed line at `lasso.alpha_`; xlabel: `$\alpha$`, ylabel: "Mean square error"; title: "Mean square error on each fold: coordinate descent"; y-axis limited to [2300, 3800]
- [cell-12]: `plt.semilogx` — same structure for LARS-based LassoLarsCV; title: "Mean square error on each fold: Lars"; same y-limits [2300, 3800]

## What is left as an exercise to the student
- No explicit TODO or exercise cells. Notebook is fully worked.

## Key cell indices for code idiom extraction
- "[cell-6]: `lasso_lars_ic = make_pipeline(StandardScaler(), LassoLarsIC(criterion='aic')).fit(X, y)` — pipeline with IC-based Lasso"
- "[cell-6]: `lasso_lars_ic.set_params(lassolarsic__criterion='bic').fit(X, y)` — in-place criterion switch via set_params"
- "[cell-6]: `zero_coefs = np.where(coefs == 0)[0]` — identify features zeroed by Lasso"
- "[cell-11]: `lasso.mse_path_.mean(axis=-1)` — average CV MSE path across folds"
- "[cell-11]: `plt.axvline(lasso.alpha_, linestyle='--', color='black', label='alpha: CV estimate')` — annotate optimal alpha"

## Notes / [VERIFY] flags
- AIC formula: $AIC = 2k - 2\log(L)$ (cell-5)
- BIC formula: $BIC = k\log(n) - 2\log(L)$ (cell-5)
- `results` DataFrame has index = alphas; columns = "AIC criterion", "BIC criterion" (cell-6/7)
- `alpha_aic` and `alpha_bic` retrieved from `lasso_lars_ic[-1].alpha_` after each respective fit (cell-6)
- The AIC fit pipeline is reused and updated in-place for BIC — `results["BIC criterion"]` is appended after (cell-6)
- `simplefilter(action='ignore', category=FutureWarning)` suppresses warnings (cell-6)
- `LassoLarsCV` uses `cv=20` vs `LassoCV`'s `cv=10` — asymmetry worth noting for exam
- The notebook is labelled "Lecture 5" (cell-0)
