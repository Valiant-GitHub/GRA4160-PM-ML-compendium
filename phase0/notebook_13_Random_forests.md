# Notebook extract: 13_Random_forests.ipynb

**Source path:** Lecture notebooks/13_Random_forests.ipynb
**Cell count:** 19 (cells 0-18; cell 18 is empty)

## Dataset(s) loaded
- **Titanic** training data (loaded twice — cells 3 and 15):
  - `df = pd.read_csv("../data/titanic/train.csv")` [cell 3] and again [cell 15]
  - Features (X): `df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]`
  - Target (y): `df['Survived']`

## Preprocessing steps
- [cell 3 & cell 15] `df = df.dropna()` — drop rows with missing values
- [cell 3 & cell 15] `df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)` — encode Sex (male=1, female=0)
- [cell 3 & cell 15] `train_test_split(X, y, random_state=1)` — **default test_size (0.25)**, random_state=1
- No scaling (tree-based).

## Method(s) demonstrated
- **From-scratch random forest** (numpy) built on top of sklearn `DecisionTreeClassifier` [cells 4-9]:
  - `random_subset(n_features)` [cell 4]: picks `k = int(np.sqrt(n_features))` features via `np.random.choice(n_features, size=k, replace=False)`
  - `train_tree(X_train, y_train, n_features)` [cell 5]: bootstrap sample with `np.random.choice(n_samples, size=n_samples, replace=True)`, subset features, fit a `DecisionTreeClassifier`
  - `predict(X, trees)` [cell 6]: aggregate per-tree predictions by majority vote via `np.bincount(x).argmax()`
- **sklearn RandomForestClassifier** [cell 12]: `from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier` [cell 1]
- **sklearn ExtraTreesClassifier** [cell 16]
- Mixed: section 1 is FROM-SCRATCH (numpy + single DecisionTrees); sections 2-3 are sklearn ensembles.
- Imports [cell 1] also include `from sklearn.base import BaseEstimator, ClassifierMixin`, `check_X_y/check_array/check_is_fitted`, `from scipy.stats import mode` — imported but the sklearn-API custom-estimator scaffolding is NOT actually used in code (mode/BaseEstimator unused).

## Hyperparameters set
- From-scratch [cell 7]: `n_trees = 100`, `max_depth = 3` (variable defined but **not passed** to the tree), `n_features = X_train.shape[1]`
- From-scratch tree [cell 5]: `DecisionTreeClassifier(max_features=None, random_state=1)` — note max_depth NOT set here despite `max_depth=3` defined in cell 7
- [cell 12] `RandomForestClassifier(n_estimators=100, max_depth=3, random_state=30)`
- [cell 16] `ExtraTreesClassifier(n_estimators=100, max_depth=3, random_state=1)`

## Plots produced
- None. (matplotlib imported [cell 1] but never used; all outputs are printed metrics.)

## What is left as an exercise to the student
- None explicitly stated (no exercise cells). The notebook is a worked comparison of hand-built RF vs sklearn RF vs ExtraTrees.

## Key cell indices for code idiom extraction
- "[cell 4]: `k = int(np.sqrt(n_features)); np.random.choice(n_features, size=k, replace=False)`" — random feature subset (sqrt rule)
- "[cell 5]: bootstrap sample — `np.random.choice(n_samples, size=n_samples, replace=True)` then `X_train.iloc[sample_indices]`"
- "[cell 6]: majority-vote aggregation — `np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred)`"
- "[cell 9]: metric quartet — `accuracy_score`, `precision_score`, `recall_score`, `f1_score`"
- "[cell 12]: `RandomForestClassifier(n_estimators=100, max_depth=3, random_state=30).fit(X_train, y_train)`"
- "[cell 16]: `ExtraTreesClassifier(n_estimators=100, max_depth=3, random_state=1)`"

## Notes / [VERIFY] flags
- **CODE BUG / DEAD VARIABLE [cell 7]:** `max_depth = 3` is defined but never used — the from-scratch trees in `train_tree` [cell 5] are built with `DecisionTreeClassifier(max_features=None, random_state=1)` (no max_depth), so the hand-rolled forest grows full-depth trees, unlike the sklearn RF/ExtraTrees which DO use `max_depth=3`. Comparison is therefore not apples-to-apples.
- **From-scratch RF and sklearn RF print identical metrics** (Accuracy 0.7608, Precision 0.7631, Recall 0.9355, F1 0.8406) per the saved outputs [cells 9, 13] — likely coincidental on this small post-dropna test set; worth noting as a teaching artifact.
- `feature subset` in `train_tree` is applied per-tree (once), NOT per-split, so this hand-built version differs from a true random forest where feature subsampling happens at every node. The sklearn `max_features=None` inside each tree confirms no per-split subsetting in the scratch version.
- Saved outputs: ExtraTrees [cell 16] prints Accuracy 0.7391, Precision 0.7879, Recall 0.8387 (recall not labeled in print but third value); survival rate in test set 0.67 [cells 10, 17].
- `from scipy.stats import mode` and `BaseEstimator/ClassifierMixin` imported [cell 1] but unused. Dead imports.
- Cell 18 is empty.
