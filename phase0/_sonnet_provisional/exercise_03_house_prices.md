# Exercise extract: Predicting house prices (stub + VHL solution)

**Stub path:** `Exercises and solutions(VHL)/03_Predicting_house_prices.ipynb`
**Solution path:** `Exercises and solutions(VHL)/03_Predicting_house_prices_VHL.ipynb`
**Cell counts:** stub=1 (single markdown cell); solution=20 cells (cells 0-1 markdown, cells 2-18 code, cells 19-20 empty)

## What the exercise teaches (1-2 sentences)

Compares OLS linear regression with Ridge and Lasso regularisation on a real-estate dataset with many features and missing values, using Lasso for feature selection and building a prediction-inspection utility that maps raw (un-normalised) inputs to dollar price predictions.

## Setup

- **Dataset:** Ames, Iowa house prices (Kaggle competition)
  - **Exact loader call** (solution cell-2):
    ```python
    data = pd.read_csv('../../data/house-prices/train.csv')
    ```
  - **Columns:** 81 total (80 features + target); 1460 rows
  - **Target column:** `SalePrice` (continuous USD, e.g., 208500)
  - **Selected numeric columns** (solution cell-4): `select_dtypes(include=['int64', 'float64'])`; after dropping rows with NaN yields ~37 numeric columns (see runtime output for exact count)
  - **Key numeric features identified by Lasso** (cell-12): `MSSubClass`, `LotArea`, `OverallQual`, `OverallCond`, `YearBuilt`, `MasVnrArea`, `BsmtFinSF1`, `GrLivArea`, `BsmtFullBath`, `GarageCars`
- **Task:** Regression — predict `SalePrice`; compare in-sample and out-of-sample MSE and R²
- **Expected outputs:** MSE and R² for OLS, Ridge, Lasso (in-sample and out-of-sample); list of 10 most important features; predicted vs. actual prices for 5 test houses; counterfactual predictions after editing feature values

## What the student must implement (from the stub)

All 8 tasks are in **stub cell-0** (single markdown):

1. Load `train.csv`; inspect variables; identify best predictors
2. `train_test_split` → `X_train`, `X_test`, `y_train`, `y_test`
3. (a) Keep numeric columns, drop rows with NaN; (b) normalise to mean=0, std=1
4. Train OLS; report in-sample and out-of-sample MSE and R²
5. Repeat with Ridge and Lasso
6. Use Lasso + `SelectFromModel` to identify the 10 most important features
7. Re-train OLS on only those 10 features; report MSE and R²
8. Write a function accepting un-normalised feature vectors → returns predicted price in dollars

## Solution walkthrough (from _VHL)

**Cell-3 — Train/test split** (task 2):
```python
from sklearn.model_selection import train_test_split
X = data.drop("SalePrice", axis=1)
y = data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=10)
```

**Cell-4 — Numeric subsetting + NaN drop** (task 3a):
```python
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
X_train_num = X_train[numeric_cols].dropna()
X_test_num  = X_test[numeric_cols].dropna()
y_train_num = y_train.loc[X_train_num.index]
y_test_num  = y_test.loc[X_test_num.index]
```
- Categorical columns are fully discarded at this stage

**Cell-5 — Dual StandardScaler** (task 3b):
```python
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_x.fit(X_train_num)
scaler_y = StandardScaler()
scaler_y.fit(y_train_num.values.reshape(-1,1))
X_train_norm = pd.DataFrame(scaler_x.transform(X_train_num), ...)
X_test_norm  = pd.DataFrame(scaler_x.transform(X_test_num),  ...)
y_train_norm = pd.DataFrame(scaler_y.transform(...))['SalePrice']
y_test_norm  = pd.DataFrame(scaler_y.transform(...))['SalePrice']
```
- **Both X and y are standardised** — two separate `StandardScaler` objects
- Scalers are fit on training data only; applied (`.transform`) to test

**Cells-6 to 8 — OLS** (task 4):
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lm = LinearRegression()
lm.fit(X_train_norm, y_train_norm)
```
- In-sample results (cell-7): MSE=0.2024, R²=0.7976
- Out-of-sample results (cell-8): MSE=0.1760, R²=0.8223

**Cell-9 — Ridge and Lasso** (task 5):
```python
from sklearn.linear_model import Ridge, Lasso
ridge = Ridge(alpha=0.5)
lasso = Lasso(alpha=0.01)
```
- Ridge `alpha=0.5`; Lasso `alpha=0.01` (both non-default)
- Out-of-sample results (cell-11): Ridge MSE=0.1759, R²=0.8223; Lasso MSE=0.1729, R²=0.8254

**Cell-12 — Feature selection with Lasso + SelectFromModel** (task 6):
```python
from sklearn.feature_selection import SelectFromModel
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_norm, y_train_norm)
sfm = SelectFromModel(lasso, max_features=10)
sfm.fit(X_train_norm, y_train_norm)
important_features = X_train_norm.columns[sfm.get_support()]
```
- `SelectFromModel` import: `sklearn.feature_selection.SelectFromModel`
- `max_features=10` — selects exactly 10 features
- **Resulting features:** `['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'MasVnrArea', 'BsmtFinSF1', 'GrLivArea', 'BsmtFullBath', 'GarageCars']`

**Cell-13 — OLS on 10 features** (task 7):
```python
lm2 = LinearRegression()
lm2.fit(X_train_norm[important_features], y_train_norm)
```
- Results (cell-13): MSE=0.1784, R²=0.8198 — slightly below full-feature Lasso

**Cell-14 — Prediction inspection function** (task 8):
```python
def inspectPrediction(model, observed_units):
    price_prediction = model.predict(
        pd.DataFrame(scaler_x.transform(observed_units),
                     index=observed_units.index,
                     columns=observed_units.columns))
    return scaler_y.inverse_transform(price_prediction.reshape(-1, 1))
```
- Function normalises input with training `scaler_x`, calls model, then inverse-transforms with `scaler_y`
- Used in cells-15 to 18: edits `OverallQual` from 5→8 for house 854 (predicted $173K → $235K) and `YearBuilt` from 2006→1970 for house 381 (predicted $216K → $205K)

## Common pitfalls (inferred from the solution / data)

- Fitting `scaler_x` and `scaler_y` before the train/test split (data leakage)
- Forgetting to also standardise `y`; if only X is normalised, inverse-transform step in task 8 cannot be applied correctly
- Using `.dropna()` only on `X_train` but not re-aligning `y_train` indices (`y_train.loc[X_train_num.index]` is the fix)
- `SelectFromModel` with `max_features=10` will choose features by coefficient magnitude; different `alpha` values will change which features are selected — students may get different top-10 sets
- `Ridge(alpha=0.5)` — the chosen alpha is arbitrary and not cross-validated; students should note that in practice a grid search would be needed
- Not separating `scaler_x` and `scaler_y`; some students apply a single scaler to the whole DataFrame, which conflates X and y scales

## What this exercise teaches that the others don't

The only exercise that applies regularisation (Ridge and Lasso) side-by-side on the same dataset and uses Lasso coefficients for automatic feature selection. It also introduces the dual-scaler pattern (standardise both X and y separately) and the counterfactual prediction function, which connects model outputs to interpretable business questions (what happens to price if I improve quality by 3 points?).

## Method page(s) it links to

- **OLS** (primary)
- **regularization** (Ridge, Lasso — primary)
- touches: feature selection via Lasso coefficients, `StandardScaler`

## Notes / [VERIFY] flags

- [VERIFY] File path `../../data/house-prices/train.csv` — confirm it resolves from the notebook's location in the repo
- [VERIFY] Runtime shape of `X_train_num` after `dropna()` — solution shows 37 numeric columns in `observed_units` (cell-15 output), but the exact number after NaN-dropping may differ run-to-run if test split differs
- Solution cells-19 and -20 are empty — intentional
- The stub has `test_size=0.33, random_state=10`; these are not stated in the markdown prompt and students must check the solution or choose their own
