# Notebook extract: 13_Random_forests.ipynb

**Source path:** course_materials\Lecture notebooks\13_Random_forests.ipynb
**Cell count:** 19 (cell-0 through cell-18)

## Dataset(s) loaded
- `"../data/titanic/train.csv"` — loaded twice (cells 3 and 15); identical preprocessing each time
  - Columns used: `['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']`; target: `'Survived'`

## Preprocessing steps
- `df = df.dropna()` (cells 3, 15)
- `df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)` (cells 3, 15)
- `train_test_split(X, y, random_state=1)` — `test_size` default (0.25) (cells 3, 15)

## Method(s) demonstrated
### FROM SCRATCH (cells 4–9):
- `random_subset(n_features)`: selects `k = int(np.sqrt(n_features))` random features without replacement using `np.random.choice`
- `train_tree(X_train, y_train, n_features)`: bootstrap sample (`np.random.choice(n_samples, size=n_samples, replace=True)`), subset features, fits `DecisionTreeClassifier(max_features=None, random_state=1)` — sklearn tree but manual bagging/feature selection
- `predict(X, trees)`: aggregates predictions with `np.bincount(x).argmax()` (majority vote) via `np.apply_along_axis`
- Manual random forest loop: `n_trees=100`, `max_depth=3` (set but NOT passed to DecisionTreeClassifier — [VERIFY: `max_depth=3` is defined in cell-7 but not used in `train_tree`]) 

### sklearn (cells 12–16):
- `from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier` (cell-1)
- `RandomForestClassifier(n_estimators=100, max_depth=3, random_state=30)` (cell-12)
- `ExtraTreesClassifier(n_estimators=100, max_depth=3, random_state=1)` (cell-16)

### Additional imports used:
- `from sklearn.base import BaseEstimator, ClassifierMixin` (cell-1) — imported but not used in visible code
- `from sklearn.utils.multiclass import check_classification_targets` (cell-1) — imported but not used
- `from sklearn.utils.validation import check_X_y, check_array, check_is_fitted` (cell-1) — imported but not used
- `from scipy.stats import mode` (cell-1) — imported but not used in visible code

## Hyperparameters set
### From-scratch forest:
- `n_trees = 100` (cell-7)
- `max_depth = 3` — defined but NOT passed into `DecisionTreeClassifier` in `train_tree` (cell-7); [VERIFY: appears to be dead code / oversight]
- `DecisionTreeClassifier(max_features=None, random_state=1)` inside `train_tree` — `max_depth` effectively `default` (cell-5)
- Feature subset size: `k = int(np.sqrt(n_features))` — hardcoded in `random_subset` (cell-4)

### sklearn RandomForestClassifier:
- `n_estimators=100`, `max_depth=3`, `random_state=30` (cell-12); all others `default`

### sklearn ExtraTreesClassifier:
- `n_estimators=100`, `max_depth=3`, `random_state=1` (cell-16); all others `default`

## Plots produced
- No plots produced in this notebook.

## What is left as an exercise to the student
- No explicit TODO/exercise cells. Notebook is fully worked.

## Key cell indices for code idiom extraction
- "[cell-4]: `def random_subset(n_features): k = int(np.sqrt(n_features)); features = np.random.choice(n_features, size=k, replace=False); return features` — sqrt feature subset rule"
- "[cell-5]: `sample_indices = np.random.choice(n_samples, size=n_samples, replace=True); X_boot = X_train.iloc[sample_indices]` — bootstrap resample with replacement"
- "[cell-6]: `y_pred_agg = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=y_pred)` — majority vote aggregation"
- "[cell-12]: `rfc = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=30); rfc.fit(X_train, y_train); y_pred = rfc.predict(X_test)` — sklearn RF idiom"
- "[cell-16]: `etc = ExtraTreesClassifier(n_estimators=100, max_depth=3, random_state=1); etc.fit(X_train, y_train)` — ExtraTrees idiom"
- "[cell-9]: `accuracy = accuracy_score(y_test, y_pred); precision = precision_score(y_test, y_pred); recall = recall_score(y_test, y_pred); f1 = f1_score(y_test, y_pred)` — four-metric evaluation pattern"

## Notes / [VERIFY] flags
- The from-scratch forest (cells 4–9) and sklearn RF (cells 12–13) produce identical accuracy output (0.7608695652173914), precision (0.7631578947368421), recall (0.9354838709677419), F1 (0.8405797101449275) — demonstrating equivalence.
- ExtraTreesClassifier produces lower accuracy (0.7391304347826086) at same depth setting.
- `max_depth=3` is declared in cell-7 as a local variable but is never passed to `DecisionTreeClassifier` inside `train_tree`. The DecisionTree in the from-scratch forest therefore grows to `default` (unlimited) depth. [VERIFY: this is likely an oversight; the sklearn RF correctly passes `max_depth=3`.]
- `random_state=30` for sklearn RF vs `random_state=1` for ExtraTreesClassifier — different seeds, inconsistency worth noting.
- Cells 18 is empty.
- The notebook is labelled "Lecture 7" (cell-0).
- Output rows are included in the read (cells 9, 10, 13, 16, 17 show executed output), confirming the notebook has been run.
- Survival rate in test set: 0.67 (cells 10, 17).
