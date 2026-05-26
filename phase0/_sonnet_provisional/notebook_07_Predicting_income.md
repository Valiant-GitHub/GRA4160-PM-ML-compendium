# Notebook extract: 07_Predicting_income.ipynb

**Source path:** course_materials\Lecture notebooks\07_Predicting_income.ipynb
**Cell count:** 7 (cell-0 through cell-6)

## Dataset(s) loaded
- Adult / Census Income dataset loaded from URL: `'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'`
  - Columns assigned (header=None): `['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']`
  - Target: `income` (binary: 1 if `'>50K'` else 0)
  - Features used: all columns except `income` (i.e., `X = df.drop(columns=['income'])`)

## Preprocessing steps
- `df = df.replace('?', pd.NaT)` — replace missing-value sentinel '?' with NaT (cell-4)
- `df = df.dropna()` — drop rows with missing values (cell-4)
- Label-encode 8 categorical features using `LabelEncoder` in a loop (cell-4):
  - `categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']`
  - `encoder = LabelEncoder(); for feature in categorical_features: df[feature] = encoder.fit_transform(df[feature])`
- `df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)` — binary target encoding (cell-4)
- `train_test_split(X, y, test_size=0.2, random_state=42)` (cell-4)

## Method(s) demonstrated
- This notebook is a **setup/exercise notebook only** — no model is trained in the provided cells.
- `from sklearn.model_selection import train_test_split` (cell-4)
- `from sklearn.preprocessing import LabelEncoder` (cell-4)
- The markdown cells describe `GradientBoostingClassifier` from sklearn (cell-2) — students are expected to implement it in exercises.

## Hyperparameters set
- No model is instantiated in notebook code. The markdown (cell-2) lists hyperparameters of `GradientBoostingClassifier` that students should tune:
  - `n_estimators`: number of trees
  - `learning_rate`: step size of gradient descent
  - `max_depth`: max depth of each tree
  - `min_samples_split`: min samples to split a node
  - `min_samples_leaf`: min samples at a leaf node

## Plots produced
- None.

## What is left as an exercise to the student
- Cell-5 (Exercises — the entire modelling portion):
  1. Train a `GradientBoostingClassifier` on the Adult dataset; evaluate with AUC.
  2. Experiment with hyperparameters (`n_estimators`, `learning_rate`, `max_depth`) and observe effect on performance.
  3. Train an `AdaBoostClassifier`; compare to GradientBoosting; optionally try XGBoost.
  4. Perform feature selection using GradientBoostingClassifier; evaluate with only top-5 most important features.

## Key cell indices for code idiom extraction
- "[cell-4]: `df = df.replace('?', pd.NaT); df = df.dropna()` — sentinel replacement then dropna"
- "[cell-4]: `encoder = LabelEncoder(); for feature in categorical_features: df[feature] = encoder.fit_transform(df[feature])` — loop-encode multiple categorical columns"
- "[cell-4]: `df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)` — strip whitespace before binary encode"

## Notes / [VERIFY] flags
- The notebook is labelled "Lecture 7" (cell-0), covering boosting. It is primarily a data-loading and exercise scaffold.
- `replace('?', pd.NaT)` is unusual — normally `np.nan` is used for numeric NaN; `pd.NaT` is for datetime. [VERIFY: this works here because `dropna()` treats NaT as missing, but it is an unconventional choice.]
- Cell-6 is empty.
- No imports of `GradientBoostingClassifier` or `AdaBoostClassifier` appear in the notebook — students must add these themselves.
- The `fnlwgt` column is kept in `X` (not dropped) — students may wish to consider whether it is appropriate.
