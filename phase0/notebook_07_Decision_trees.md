# Notebook extract: 07_Decision_trees.ipynb

**Source path:** course_materials/Lecture notebooks/07_Decision_trees.ipynb
**Cell count:** 22 (cells 0-21; cells 6, 8, 9, 10, 21 are empty)

## Dataset(s) loaded
- **Iris** dataset loaded from a URL via pandas (NOT `sklearn.datasets.load_iris`) [cell 4]:
  - `url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"`
  - `df = pd.read_csv(url, header=None)`
  - `df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']`
  - Features (X): all columns except last — `df.iloc[:, :-1].values` (sepal_length, sepal_width, petal_length, petal_width) [cell 7]
  - Target (y): `df.iloc[:, -1].values` (the `class` column) [cell 7]

## Preprocessing steps
- [cell 4] `df = pd.read_csv(url, header=None)`; then assign `df.columns = [...]`
- [cell 7] `X = df.iloc[:, :-1].values`; `y = df.iloc[:, -1].values`
- [cell 7] `train_test_split(X, y, test_size=0.3, random_state=0)` — 70/30 split
- No scaling, no encoding (tree-based; string class labels kept).

## Method(s) demonstrated
- **Decision Tree Classifier** — sklearn. Import: `from sklearn.tree import DecisionTreeClassifier, plot_tree` [cell 4]
- `from sklearn.model_selection import train_test_split` [cell 4]; `import matplotlib.pyplot as plt` [cell 4]
- No from-scratch implementation. Gini and Entropy formulas appear only in markdown theory [cell 2].

## Hyperparameters set
- [cell 7] `DecisionTreeClassifier()` — all `default` (criterion=gini, max_depth=None, random_state=None)
- [cell 12] `DecisionTreeClassifier(random_state=0, criterion='gini')` — fit on full `X, y`
- [cell 13] `DecisionTreeClassifier(random_state=0, criterion='entropy')` — fit on full `X, y`
- [cell 16] `DecisionTreeClassifier(min_samples_leaf=5, random_state=0)` — fit on `X_train, y_train`
- [cell 17] `DecisionTreeClassifier(min_samples_leaf=10, random_state=0)` — fit on full `X, y`

## Plots produced
- [cell 12] `plot_tree(clf_gini, filled=True, feature_names=df.columns[:-1], class_names=np.unique(y))`; title "Decision Tree using Gini". figsize=(14, 7)
- [cell 13] `plot_tree(clf_entropy, ...)`; title "Decision Tree using Entropy". figsize=(14, 7)
- [cell 17] `plot_tree(clf_pruned_full, ...)`; title "Pruned Decision Tree with min_samples_leaf=5". figsize=(14, 7)

## What is left as an exercise to the student
- [cell 19] Try more pruning: vary `min_samples_leaf` (1, 2, 10) or set `max_depth`.
- [cell 19] Use `GridSearchCV` / `RandomizedSearchCV` to tune `max_depth`, `min_samples_leaf`, etc.
- [cell 19] Inspect `clf.feature_importances_`.
- [cell 19] Compare Gini vs Entropy on the same split.
- [cell 19] Apply to another dataset (e.g. Wine).

## Key cell indices for code idiom extraction
- "[cell 7]: `clf = DecisionTreeClassifier(); clf.fit(X_train, y_train); clf.score(X_test, y_test)`" — default tree fit/score
- "[cell 7]: `clf.get_depth()` and `clf.get_n_leaves()`" — tree-size inspection idiom
- "[cell 12/13]: `plot_tree(clf, filled=True, feature_names=df.columns[:-1], class_names=np.unique(y))`" — tree visualization idiom
- "[cell 16]: `DecisionTreeClassifier(min_samples_leaf=5, random_state=0)`" — pre-pruning idiom

## Notes / [VERIFY] flags
- Markdown formulas [cell 2]: Gini $\text{Gini}(S) = 1 - \sum_{i=1}^{C} p_i^2$; Entropy $\text{Entropy}(S) = -\sum_{i=1}^{C} p_i \log_2(p_i)$.
- **CODE/PROSE MISMATCH** [cell 17]: the plot title string is `"Pruned Decision Tree with min_samples_leaf=5"` but the estimator is actually built with `min_samples_leaf=10`. The comment at the top of cell 17 also says "min_samples_leaf to 5" while building with 10. Markdown [cell 15] separately says setting `min_samples_leaf` to 5 reduces complexity (refers to cell 16, which does use 5).
- **MISMATCH (fit-set inconsistency)** [cell 17]: comment claims trees should be fit on training set "but we'll do it on X, y for illustration" — cells 12, 13, 17 fit on the full dataset, not `X_train`. Only cells 7 and 16 use the train/test split.
- Empty cells: 6, 8, 9, 10, 21 contain no code.
