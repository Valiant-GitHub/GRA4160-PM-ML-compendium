# Notebook extract: 07_Decision_trees.ipynb

**Source path:** course_materials\Lecture notebooks\07_Decision_trees.ipynb
**Cell count:** 22 (cell-0 through cell-21)

## Dataset(s) loaded
- Iris dataset loaded from URL: `"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"` (cell-4)
  - Columns assigned: `['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']`
  - Features: all four numeric columns (`df.iloc[:, :-1].values`); target: `'class'` (`df.iloc[:, -1].values`)

## Preprocessing steps
- Load CSV with `header=None`, assign column names manually (cell-4)
- `X = df.iloc[:, :-1].values` and `y = df.iloc[:, -1].values` (cell-7)
- `train_test_split(X, y, test_size=0.3, random_state=0)` — 70/30 split (cell-7)
- No scaling applied (trees are scale-invariant)

## Method(s) demonstrated
- `from sklearn.tree import DecisionTreeClassifier, plot_tree` — sklearn, NOT from scratch (cell-4)
- `DecisionTreeClassifier()` — default (gini, no depth limit) (cell-7)
- `DecisionTreeClassifier(random_state=0, criterion='gini')` — explicit gini, fit on full dataset for visualisation (cell-12)
- `DecisionTreeClassifier(random_state=0, criterion='entropy')` — entropy criterion (cell-13)
- `DecisionTreeClassifier(min_samples_leaf=5, random_state=0)` — pruning via min_samples_leaf (cell-16)
- `DecisionTreeClassifier(min_samples_leaf=10, random_state=0)` — pruning, fit on full dataset for visualisation (cell-17); note: title in plot says "min_samples_leaf=5" but code uses 10 — [VERIFY: title/code mismatch in cell-17]

## Hyperparameters set
- Default tree: all `default` (cell-7)
- Gini tree (for visualisation): `criterion='gini'`, `random_state=0`; all others `default` (cell-12)
- Entropy tree (for visualisation): `criterion='entropy'`, `random_state=0`; all others `default` (cell-13)
- Pruned tree (test evaluation): `min_samples_leaf=5`, `random_state=0`; all others `default` (cell-16)
- Pruned tree (full-data visualisation): `min_samples_leaf=10`, `random_state=0`; all others `default` (cell-17)

## Plots produced
- [cell-12]: Full decision tree with Gini criterion — `plot_tree(clf_gini, filled=True, feature_names=df.columns[:-1], class_names=np.unique(y))`; figure size (14, 7); title "Decision Tree using Gini"
- [cell-13]: Full decision tree with Entropy criterion — same arguments; title "Decision Tree using Entropy"
- [cell-17]: Pruned decision tree (min_samples_leaf=10) on full dataset — same arguments; title "Pruned Decision Tree with min_samples_leaf=5" [VERIFY: title says 5, code uses 10]

## What is left as an exercise to the student
- Cell-19 (Exercises/Extensions):
  1. Adjust `min_samples_leaf` to different values (1, 2, 10) or set `max_depth`; observe accuracy and tree structure changes.
  2. Use `GridSearchCV` or `RandomizedSearchCV` to find the best `max_depth`, `min_samples_leaf`, etc.
  3. Use `clf.feature_importances_` to see which features are most used.
  4. Compare Gini vs. Entropy performance on the same train-test split.
  5. Try the model on another dataset (e.g., Wine dataset).

## Key cell indices for code idiom extraction
- "[cell-7]: `clf = DecisionTreeClassifier(); clf.fit(X_train, y_train); score = clf.score(X_test, y_test); print(clf.get_depth()); print(clf.get_n_leaves())` — default tree with depth/leaf diagnostics"
- "[cell-12]: `plot_tree(clf_gini, filled=True, feature_names=df.columns[:-1], class_names=np.unique(y))` — canonical tree visualisation"
- "[cell-16]: `clf_pruned = DecisionTreeClassifier(min_samples_leaf=5, random_state=0); clf_pruned.fit(X_train, y_train)` — pre-pruning idiom"

## Notes / [VERIFY] flags
- Gini index formula: $\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2$ (cell-2)
- Entropy formula: $\text{Entropy}(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)$ (cell-2)
- Gini/Entropy trees in cells-12 and 13 are fit on the FULL dataset (X, y), not the train split — for visualisation only.
- Cell-6, cell-8, cell-9, cell-10, cell-21 are empty cells.
- The notebook is labelled "Lecture 4" in the header (cell-0) — same lecture number as notebook 06.
- Pruning section header (cell-15) says "min_samples_leaf to 5" but cell-17 code uses `min_samples_leaf=10` — [VERIFY: intentional?]
