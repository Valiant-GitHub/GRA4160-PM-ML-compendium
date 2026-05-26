# Notebook extract: 07_Predicting_income.ipynb

**Source path:** course_materials/Lecture notebooks/07_Predicting_income.ipynb
**Cell count:** 7 (cells 0-6; cell 6 is empty)

## Dataset(s) loaded
- **Adult / "Census Income"** dataset from UCI [cell 4]:
  - `df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None, names=[...])`
  - Column names (exact, assigned manually): `['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']`
  - Features (X): `df.drop(columns=['income'])` (all columns except income)
  - Target (y): `df['income']` — binarized to 1 if `>50K` else 0

## Preprocessing steps
- [cell 4] `df = df.replace('?', pd.NaT)` — replace '?' with missing
- [cell 4] `df = df.dropna()` — drop rows with missing values
- [cell 4] Label-encode categorical features: `categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']`; loop `df[feature] = encoder.fit_transform(df[feature])` with a single shared `encoder = LabelEncoder()`
- [cell 4] Target encoding: `df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)`
- [cell 4] `X = df.drop(columns=['income'])`; `y = df['income']`
- [cell 4] `train_test_split(X, y, test_size=0.2, random_state=42)` — 80/20 split

## Method(s) demonstrated
- **Gradient Boosting** is the *topic* (described in markdown as `GradientBoostingClassifier`) but is NOT instantiated/trained in code — the notebook only loads and preprocesses the data. sklearn-based when the student completes it.
- Imports actually present [cell 4]: `import pandas as pd`, `from sklearn.model_selection import train_test_split`, `from sklearn.preprocessing import LabelEncoder`
- No from-scratch implementation. No model is fit in any code cell.

## Hyperparameters set
- [cell 4] `train_test_split(test_size=0.2, random_state=42)`
- No model hyperparameters are set in code (no model is instantiated). Markdown [cell 2] lists tunable params for GradientBoostingClassifier: `n_estimators`, `learning_rate`, `max_depth`, `min_samples_split`, `min_samples_leaf` — descriptive only, no values.

## Plots produced
- None. (No matplotlib import.)

## What is left as an exercise to the student
- [cell 5] Train a Gradient Boosting Classifier on Adult; evaluate with ROC AUC.
- [cell 5] Experiment with hyperparameters (number of trees, learning rate, max depth); observe effect.
- [cell 5] Train an AdaBoost Classifier and compare to Gradient Boosting; optionally try XGBoost (needs install).
- [cell 5] Feature selection via Gradient Boosting; evaluate using only top-5 most important features.

## Key cell indices for code idiom extraction
- "[cell 4]: `df = df.replace('?', pd.NaT)`" then `df.dropna()` — missing-value sentinel cleanup idiom
- "[cell 4]: shared-LabelEncoder loop — `encoder = LabelEncoder(); for feature in categorical_features: df[feature] = encoder.fit_transform(df[feature])`"
- "[cell 4]: target binarization — `df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)`"

## Notes / [VERIFY] flags
- This is essentially a *setup-only* notebook: the entire boosting workflow (fit/evaluate) is deferred to the student as exercises. No model object exists in code.
- **Potential pitfall (not flagged in notebook):** a single shared `LabelEncoder` instance is re-`fit_transform`'d across all categorical columns. This works (each call re-fits), but reuses one object — worth noting as a non-canonical idiom vs one encoder per column.
- `pd.NaT` is used as the missing-value marker (normally for datetimes); `dropna()` still treats it as missing. `[VERIFY: behavior relies on pd.NaT being recognized as NA across object columns]`.
- Despite filename/header "Lecture 7", markdown header [cell 0] says "Lecture 7"; the boosting topic overlaps lecture 7 ensembles.
