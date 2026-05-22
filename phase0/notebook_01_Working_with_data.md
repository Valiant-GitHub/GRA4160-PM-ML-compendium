# Notebook extract: 01_Working_with_data_in_jupyter_notebooks.ipynb

**Source path:** Lecture notebooks/01_Working_with_data_in_jupyter_notebooks.ipynb
**Cell count:** 24 cells (indices cell-0 through cell-23 as reported by Read)

## Dataset(s) loaded
- `'../data/house-prices/test.csv'` — loaded via `pd.read_csv()` in [cell 6]. 80 columns, 1459 rows (RangeIndex 0–1458). Columns referenced explicitly later: none individually subset; whole frame inspected. Notable column names confirmed from `.info()` output: `Id`, `MSSubClass`, `MSZoning`, `LotFrontage`, `LotArea`, `Street`, `Alley`, `OverallQual`, `YearBuilt`, `GarageArea`, `PoolQC`, `SaleType`, `SaleCondition`, etc. No target column used (this is the Kaggle house-prices `test.csv`, which has no `SalePrice`).
- `sns.load_dataset('tips')` — seaborn built-in, loaded in [cell 11] and again in [cell 15]. Columns: `total_bill` (float64), `tip` (float64), `sex` (category), `smoker` (category), `day` (category), `time` (category), `size` (int64). In [cell 11] `hue='time'` for pairplot and `x='day', y='total_bill'` for boxplot.
- `load_diabetes(as_frame=True)` from `sklearn.datasets` — [cell 13]. Accessed as `data.frame`. Columns: `age, sex, bmi, bp, s1, s2, s3, s4, s5, s6, target`. Target column name: `target`. Outlier detection done on `bmi` column.
- `load_iris(as_frame=True)` from `sklearn.datasets` — [cell 17] (uses `iris.data`) and [cell 20] (uses `iris.data` as X, `iris.target` as y). Feature columns: `sepal length (cm)`, `sepal width (cm)`, `petal length (cm)`, `petal width (cm)`. Target: `iris.target`.

## Preprocessing steps
- [cell 6] `df = pd.read_csv('../data/house-prices/test.csv')`
- [cell 9] `df.to_csv('../data/tmp/processed_housing.csv', index=False)` (writes data back to disk)
- [cell 13] `df_missing.iloc[:10, 2] = np.nan` (artificially injects missing values into 3rd column = `bmi`)
- [cell 13] `df_missing.fillna(df_missing.mean(), inplace=True)` (mean imputation)
- [cell 13] outlier detection via IQR:
  - `Q1 = df_missing['bmi'].quantile(0.25)`
  - `Q3 = df_missing['bmi'].quantile(0.75)`
  - `IQR = Q3 - Q1`
  - `outliers = df_missing[(df_missing['bmi'] < Q1 - 1.5 * IQR) | (df_missing['bmi'] > Q3 + 1.5 * IQR)]`
- [cell 15] `encoded_tips = pd.get_dummies(tips, columns=['day','sex','smoker','time'], drop_first=True)` (one-hot encoding)
- [cell 17] `scaler = StandardScaler(); df_standard_scaled = scaler.fit_transform(df_iris)`
- [cell 17] `minmax = MinMaxScaler(); df_minmax_scaled = minmax.fit_transform(df_iris)`
- [cell 20] `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`

## Method(s) demonstrated
- This is a data-handling / tooling tour, NOT a modeling notebook. No method implemented from scratch.
- Imports: `pandas as pd`, `numpy as np`, `seaborn as sns`, `matplotlib.pyplot as plt`, `torch`, `sklearn` [cell 3].
- `sklearn.datasets`: `load_diabetes` [cell 13], `load_iris` [cell 17, cell 20].
- `sklearn.preprocessing`: `StandardScaler`, `MinMaxScaler` [cell 17].
- `sklearn.pipeline.Pipeline` [cell 20].
- `sklearn.linear_model.LogisticRegression` [cell 20] (used inside pipeline; sklearn, not from scratch).
- `sklearn.model_selection.train_test_split` [cell 20].
- `pd.get_dummies` (one-hot encoding) [cell 15].
- `torch.tensor` (NumPy ↔ PyTorch tensor conversion) [cell 22].
- Markdown [cell 14] mentions `sklearn.preprocessing.OneHotEncoder` and `sklearn.impute.SimpleImputer` as alternatives, but the CODE only uses `pd.get_dummies` and `df.fillna` — SimpleImputer and OneHotEncoder are NOT actually called in code.

## Hyperparameters set
- [cell 13] `df_missing.iloc[:10, 2]` — first 10 rows, column index 2.
- [cell 13] IQR multiplier `1.5` (standard); quantiles `0.25` / `0.75`.
- [cell 15] `pd.get_dummies(..., columns=['day','sex','smoker','time'], drop_first=True)`
- [cell 17] `StandardScaler()` — all defaults. `MinMaxScaler()` — all defaults.
- [cell 20] `train_test_split(test_size=0.2, random_state=42)`
- [cell 20] `LogisticRegression()` — all defaults (no params set).
- [cell 20] `Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression())])`

## Plots produced
- [cell 11] `sns.pairplot(tips, hue='time')` — pairwise scatter matrix of all numeric tips features colored by `time`. (Output too large to inline; flagged below.)
- [cell 11] `sns.boxplot(x='day', y='total_bill', data=tips)` — boxplot of total_bill grouped by day.
- No plots in cells 13, 17, 20, 22.

## What is left as an exercise to the student
- No explicit "exercise" prompts. The markdown frames the notebook as a guided tour. Implicit takeaway: explore SimpleImputer / OneHotEncoder (mentioned in markdown but not coded) and explore PyTorch further (cell 21 markdown says "you may want to explore it further").

## Key cell indices for code idiom extraction
- "[cell 3]: standard import block — `import pandas as pd / numpy as np / seaborn as sns / matplotlib.pyplot as plt / torch / sklearn`"
- "[cell 6]: `df = pd.read_csv('../data/house-prices/test.csv')` then `df.head()` — the canonical relative-path CSV load idiom for this course."
- "[cell 13]: IQR outlier filter idiom — `outliers = df[(df['col'] < Q1 - 1.5*IQR) | (df['col'] > Q3 + 1.5*IQR)]`"
- "[cell 15]: `pd.get_dummies(df, columns=[...], drop_first=True)` one-hot idiom."
- "[cell 17]: `scaler = StandardScaler(); scaler.fit_transform(df)` and the `MinMaxScaler` twin."
- "[cell 20]: full mini-pipeline — `Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression())])` then `.fit`/`.score`."
- "[cell 22]: `torch.tensor(np_array)` NumPy→tensor conversion and `.to('cuda')` guard with `torch.cuda.is_available()`."

## Notes / [VERIFY] flags
- The course data path convention is `'../data/...'` (relative, parent-of-notebook). House-prices subfolder: `data/house-prices/test.csv`; temp output folder `data/tmp/`.
- [VERIFY] [cell 4] prints "Scikit-learn version: {pd.__version__}" — this is a BUG in the source: it prints the pandas version (2.3.3) labeled as scikit-learn. The actual sklearn version is printed separately on the next line as "Sklearn version: 1.6.1". Library versions in environment: Pandas 2.3.3, Numpy 1.26.4, Matplotlib 3.10.7, Seaborn 0.13.2, PyTorch 2.5.1, Sklearn 1.6.1, GPU not available.
- [cell 11] source contains stray escaped-string artifacts: `# Let's also do a boxplot for 'total_bill' grouped by 'day'\n",` and `print("Note: pairplot can be slow...\n")` — likely leftover from JSON cell editing; harmless but note when lifting verbatim.
- [cell 13] comment `df_missing.iloc[:10, 2] = np.nan  # Suppose the 3rd column has missing` — column index 2 is `bmi` (0=age,1=sex,2=bmi). `.isnull().sum()` confirms `bmi 10`. Output: "Number of outliers in 'bmi': 5".
- Markdown math transcribed: cell 18 explains `.fit()` (computes params, e.g. mean & std), `.transform()` (applies them), `.fit_transform()` (both in one step). No formal equations in this notebook.
- Header numbering in markdown is inconsistent: section "2. Setting up the environment" (cell 2) and "2. Reading and Writing Data" (cell 5) both labeled "2".
