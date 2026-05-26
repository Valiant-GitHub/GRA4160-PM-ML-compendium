# Notebook extract: 01_Working_with_data_in_jupyter_notebooks.ipynb

**Source path:** `course_materials/Lecture notebooks/01_Working_with_data_in_jupyter_notebooks.ipynb`
**Cell count:** 24 (cell-0 through cell-23)

## Dataset(s) loaded

- `'../data/house-prices/test.csv'` — Kaggle House Prices test split. 80 columns including `Id`, `MSSubClass`, `MSZoning`, `LotFrontage`, `LotArea`, `OverallQual`, `OverallCond`, `YearBuilt`, `GrLivArea`, `SaleType`, `SaleCondition`, and many more (1459 rows). No target column in this file (it is the competition test set, so `SalePrice` is absent). Used only for inspection and writing exercises, not for modelling.
- `sns.load_dataset('tips')` — seaborn built-in tips dataset. Columns: `total_bill`, `tip`, `sex`, `smoker`, `day`, `time`, `size`. Used for visualisation and encoding demos.
- `sklearn.datasets.load_diabetes(as_frame=True)` — sklearn built-in. Columns: `age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`, `target`. Used for missing-value handling demo.
- `sklearn.datasets.load_iris(as_frame=True)` — sklearn built-in. Columns: `sepal length (cm)`, `sepal width (cm)`, `petal length (cm)`, `petal width (cm)`. Target: species class (0/1/2). Used for scaling and pipeline demos.

## Preprocessing steps

- [cell-6] `df = pd.read_csv('../data/house-prices/test.csv')` then `df.head()`
- [cell-9] `df.to_csv('../data/tmp/processed_housing.csv', index=False)`
- [cell-13] `df_missing.iloc[:10, 2] = np.nan` — artificial missing injection into column index 2 (`bmi` after loading diabetes)
- [cell-13] `df_missing.fillna(df_missing.mean(), inplace=True)` — mean imputation
- [cell-13] IQR outlier detection on `bmi`:
  ```python
  Q1 = df_missing['bmi'].quantile(0.25)
  Q3 = df_missing['bmi'].quantile(0.75)
  IQR = Q3 - Q1
  outliers = df_missing[(df_missing['bmi'] < Q1 - 1.5 * IQR) | (df_missing['bmi'] > Q3 + 1.5 * IQR)]
  ```
- [cell-15] `encoded_tips = pd.get_dummies(tips, columns=['day','sex','smoker','time'], drop_first=True)` — one-hot encoding
- [cell-17] `scaler = StandardScaler(); df_standard_scaled = scaler.fit_transform(df_iris)`
- [cell-17] `minmax = MinMaxScaler(); df_minmax_scaled = minmax.fit_transform(df_iris)`
- [cell-20] `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`

## Method(s) demonstrated

- `from sklearn.preprocessing import StandardScaler, MinMaxScaler` — feature scaling (sklearn, not from scratch)
- `from sklearn.pipeline import Pipeline` — sklearn pipeline
- `from sklearn.linear_model import LogisticRegression` — used only as pipeline component for demo, not studied in depth
- `from sklearn.model_selection import train_test_split` — data splitting
- `import torch` — PyTorch tensors introduced (not used for modelling; conversion and GPU check only)

No methods implemented from scratch in this notebook. All ML/preprocessing done via sklearn or pandas built-ins.

## Hyperparameters set

- `train_test_split`: `test_size=0.2`, `random_state=42`
- `Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression())])` — `LogisticRegression()` all default
- `StandardScaler()` — all default
- `MinMaxScaler()` — all default

## Plots produced

- [cell-11] Seaborn `pairplot` — all numeric columns of `tips`, coloured by `hue='time'`
- [cell-11] Seaborn `boxplot` — `x='day'`, `y='total_bill'`, `data=tips`

## What is left as an exercise to the student

No explicit TODO / "your turn" cells found in this notebook.

## Key cell indices for code idiom extraction

- [cell-13]: IQR outlier detection pattern (Q1/Q3/IQR/filter)
- [cell-15]: `pd.get_dummies(tips, columns=['day','sex','smoker','time'], drop_first=True)` — one-hot encoding with drop_first
- [cell-17]: `scaler.fit_transform(df_iris)` vs `minmax.fit_transform(df_iris)` — StandardScaler vs MinMaxScaler side-by-side
- [cell-20]: sklearn Pipeline construction and `.fit()` / `.score()` pattern
- [cell-22]: `torch.tensor(np_array)` — NumPy-to-tensor conversion idiom

## Notes / [VERIFY] flags

- cell-4 output shows `Scikit-learn version: 2.3.3` but this is a copy-paste bug in the notebook source — it prints `pd.__version__` again instead of `sklearn.__version__`. The actual sklearn version printed on the next line is `1.6.1`.
- cell-18 explains the distinction between `.fit()`, `.transform()`, and `.fit_transform()` in a markdown cell — important conceptual content, no formula.
- The diabetes dataset is loaded `as_frame=True`, so `data.frame` gives the full DataFrame including the `target` column; the missing values are injected into `df_missing.iloc[:10, 2]`, which corresponds to the `bmi` column (column index 2 in the diabetes feature set).
- Output of cell-13: 5 outliers detected in `bmi` after mean imputation.
- Pipeline test accuracy on Iris with LogisticRegression + StandardScaler: `1.00` (cell-20 output).
