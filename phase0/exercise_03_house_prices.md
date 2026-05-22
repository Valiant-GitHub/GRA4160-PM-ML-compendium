# Exercise extract: Predicting house prices (stub + VHL solution)

**Stub path:** `Exercises and solutions(VHL)\03_Predicting_house_prices.ipynb`   **Solution path:** `Exercises and solutions(VHL)\03_Predicting_house_prices_VHL.ipynb`
**Cell counts:** stub=1 (1 markdown, 0 code; combined intro+task), solution=21 (markdown cells-0,1; code cells-2..18; trailing empty cells-19,20)

## What the exercise teaches (1-2 sentences)
End-to-end regression on the Ames house-prices dataset: numeric-only cleaning, standardization of both X and y, OLS baseline, then Ridge and Lasso with in-sample vs out-of-sample MSE/R², Lasso-based feature selection, and a re-fit on the selected features. Closes with a "what-if" prediction function that inverse-transforms scaled predictions back to dollar prices.

## Setup
- **Dataset:** Kaggle "House Prices - Advanced Regression Techniques" (Ames, Iowa). 1460 observations, 81 variables. Source cited: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- **Exact loader call (solution cell-2):** `pd.read_csv('../../data/house-prices/train.csv')`
- **Target (exact):** `SalePrice`.
- **Key columns shown in head():** `Id, MSSubClass, MSZoning, LotFrontage, LotArea, Street, Alley, LotShape, LandContour, Utilities, ... PoolArea, PoolQC, Fence, MiscFeature, MiscVal, MoSold, YrSold, SaleType, SaleCondition, SalePrice` (81 columns total).
- **Numeric feature set after cleaning:** 37 numeric columns (per cell-15 `[5 rows x 37 columns]`).
- **Task:** continuous-target regression; report in-sample and out-of-sample MSE and R².
- **Expected output:** MSE/R² for OLS, Ridge, Lasso (in- and out-of-sample); list of 10 most important Lasso features; MSE/R² for OLS on those 10; a function returning un-normalized predicted prices and a comparison of actual/predicted/edited prices.

## What the student must implement (from the stub)
Stub is markdown-only; the 8-part task list is stub cell-0 (repeated in solution cell-1):
1. Load the data; inspect variables; judge best predictors.
2. Train/test split into `X_train, X_test, y_train, y_test`.
3. Cleaning/preprocessing: (a) keep numeric columns and drop missing values; (b) normalize to mean 0, std 1.
4. Train a model on numeric columns; report in-sample and out-of-sample MSE and R².
5. Repeat with Ridge and Lasso.
6. Use Lasso to find the 10 most important features; hint: `from sklearn.feature_selection import SelectFromModel`.
7. Re-fit a linear regression on only those 10 features; report MSE and R².
8. Write a function taking un-normalized feature vectors, normalizing internally, predicting, and returning un-normalized prices; experiment with changing features one by one.

## Solution walkthrough (from _VHL)
- **cell-2 (Ex1):** `import pandas as pd`; `data = pd.read_csv('../../data/house-prices/train.csv')`.
- **cell-3 (Ex2):** `X = data.drop("SalePrice", axis=1)`; `y = data["SalePrice"]`; `train_test_split(X, y, test_size=0.33, random_state=10)`.
- **cell-4 (Ex3a):** `numeric_cols = X_train.select_dtypes(include=['int64','float64']).columns`; `categorical_cols = ... include=['object']`. Drop NaN rows: `X_train_num = X_train[numeric_cols].dropna()` (and test); align y via `y_train.loc[X_train_num.index]`. (Drops rows with any NaN in numeric cols.)
- **cell-5 (Ex3b):** `from sklearn.preprocessing import StandardScaler`. Separate scalers for X and y: `scaler_x.fit(X_train_num)`; `scaler_y.fit(y_train_num.values.reshape(-1,1))`. Transform train and test into DataFrames preserving index/columns; y kept as a Series named `SalePrice`.
- **cell-6..8 (Ex4):** `from sklearn.linear_model import LinearRegression`; `from sklearn.metrics import mean_squared_error, r2_score`. `lm = LinearRegression()` (defaults); fit on `(X_train_norm, y_train_norm)`. **In-sample:** MSE 0.2024, R² 0.7976. **Out-of-sample:** MSE 0.1760, R² 0.8223 (values are on standardized y).
- **cell-9..11 (Ex5):** `from sklearn.linear_model import Ridge, Lasso`; `ridge = Ridge(alpha=0.5)`; `lasso = Lasso(alpha=0.01)`; fit both. In-sample: Ridge MSE 0.2024/R² 0.7976, Lasso MSE 0.2055/R² 0.7945. Out-of-sample: Ridge MSE 0.1759/R² 0.8223, Lasso MSE 0.1729/R² 0.8254 (Lasso best out-of-sample).
- **cell-12 (Ex6):** `from sklearn.feature_selection import SelectFromModel`; re-init `lasso = Lasso(alpha=0.01)`, fit; `sfm = SelectFromModel(lasso, max_features=10)`; `sfm.fit(...)`; `important_features = X_train_norm.columns[sfm.get_support()]`. **Selected 10:** `MSSubClass, LotArea, OverallQual, OverallCond, YearBuilt, MasVnrArea, BsmtFinSF1, GrLivArea, BsmtFullBath, GarageCars`.
- **cell-13 (Ex7):** `lm2 = LinearRegression()`; fit on `X_train_norm[important_features]`; predict on test subset. MSE 0.1784, R² 0.8198.
- **cell-14..18 (Ex8):** `inspectPrediction(model, observed_units)` scales inputs with `scaler_x.transform`, predicts, then `scaler_y.inverse_transform(price_prediction.reshape(-1,1))` to return dollar prices. Demonstrates editing features (`observed_units.at[854,'OverallQual']=8`, `observed_units.at[381,'YearBuilt']=1970`) and compares Actual / Predicted / Edited prices in a DataFrame.
- **Methods/classes:** `sklearn.model_selection.train_test_split`; `sklearn.preprocessing.StandardScaler`; `sklearn.linear_model.LinearRegression`; `sklearn.linear_model.Ridge`; `sklearn.linear_model.Lasso`; `sklearn.feature_selection.SelectFromModel`; `sklearn.metrics.mean_squared_error`, `sklearn.metrics.r2_score`.
- **Hyperparameters:** `train_test_split(test_size=0.33, random_state=10)`; `LinearRegression()` defaults; `Ridge(alpha=0.5)`; `Lasso(alpha=0.01)` (used both for fitting and for selection); `SelectFromModel(max_features=10)` (threshold default); `StandardScaler()` defaults.

## Common pitfalls (inferred from the solution / data)
- `dropna()` is applied independently to numeric train and test; because some numeric columns (e.g., `LotFrontage`, `MasVnrArea`, `GarageYrBlt`) have NaNs, this drops rows and you MUST re-align y with `.loc[...index]` (done here) or X and y desync.
- Both X and y are standardized, so reported MSE/R² are on the standardized scale; the prediction function must `inverse_transform` y to get dollars (done in cell-14).
- `random_state` for split is 10 here (not 42); selected features and metrics are split-dependent.
- Categorical columns are entirely dropped (numeric-only model) — loses signal but simplifies; ordinal categoricals like `OverallQual` survive because they are stored as integers.
- `Lasso(alpha=0.01)` is small; too large an alpha zeroes most coefficients and `SelectFromModel(max_features=10)` could return fewer than 10.
- `SelectFromModel` with `max_features=10` still applies a default importance threshold; results can vary with alpha.

## What this exercise teaches that the others don't
- The only **continuous-target regression** exercise and the only one comparing **OLS vs Ridge vs Lasso** with explicit in-sample vs out-of-sample MSE/R². Uniquely covers **Lasso-based feature selection** (`SelectFromModel`) and an **inverse-transform "what-if" prediction** workflow on standardized data. Real-world messy data with missing values and 81 mixed-type columns.

## Method page(s) it links to
- OLS (LinearRegression baseline) and regularization (Ridge, Lasso). Secondary: feature selection via Lasso, standardization/preprocessing.

## Notes / [VERIFY] flags
- [VERIFY: dataset path] Solution loads `../../data/house-prices/train.csv`; the local repo stores it at `data\house-prices data\train.csv` (folder name has a space and differs from the notebook path). Path needs reconciling for the build.
- Reported metrics above are the exact saved cell outputs in the VHL notebook.
- Solution cells 19 and 20 are empty.
