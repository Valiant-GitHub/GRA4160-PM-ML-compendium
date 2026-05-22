# Notebook extract: 03_Supervised_learning_with_kNN (1).ipynb

**Source path:** Lecture notebooks/03_Supervised_learning_with_kNN (1).ipynb
**Cell count:** 14 cells (indices cell-0 through cell-13 as reported by Read; cell-13 empty)

## Dataset(s) loaded
- `datasets.load_iris()` from `sklearn` (`from sklearn import datasets`) — [cell 4]. `X = iris.data` (150×4: sepal length, sepal width, petal length, petal width), `y = iris.target` (3 classes: 0/1/2). This is the Bunch-style load (NOT `as_frame`), so X is a plain numpy array. Target: `iris.target`.
- `make_blobs(n_samples=150, centers=3, n_features=2)` from `sklearn.datasets` — [cell 10], used ONLY for 2-feature visualization (`X_train_sim`, `y_train_sim`). A single random test point `X_test_sim` is drawn via `np.random.uniform(X_train_sim.min(), X_train_sim.max(), 2)`.

## Preprocessing steps
- [cell 4] `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)`
- No scaling/encoding applied (raw iris features fed directly to the distance metric — worth noting since k-NN is distance-based; scaling is not done here).
- [cell 10] `np.random.seed(0)` set before `make_blobs` (for the viz only).

## Method(s) demonstrated
k-Nearest Neighbors (k-NN) classification, demonstrated TWO ways:
1. FROM SCRATCH (numpy) — [cell 7]:
   - `euclidean_distance(x, y)` → `np.sqrt(np.sum((x - y)**2))`
   - `predict(X_train, y_train, X_test, k)` → computes distances, `k_nearest = np.argsort(distances)[:k]`, returns `y_train[k_nearest]`.
   - `predict_label(...)` → `np.bincount(predict(...)).argmax()` (majority vote).
   - [cell 8] applies it across the whole test set via list comprehension and computes `accuracy = np.mean(y_pred == y_test)`.
2. SKLEARN cross-check — [cell 12]: `from sklearn.neighbors import KNeighborsClassifier`; `knn = KNeighborsClassifier(n_neighbors=3)`; `.fit`, `.predict`, `.score`.

Import paths:
- `from sklearn import datasets`; `from sklearn.model_selection import train_test_split` [cell 4].
- `import numpy as np` [cell 7].
- `from sklearn.datasets import make_blobs`; `import matplotlib.pyplot as plt` [cell 10].
- `from sklearn.neighbors import KNeighborsClassifier` [cell 12].

## Hyperparameters set
- [cell 4] `train_test_split(test_size=0.2, random_state=10)`.
- [cell 7] from-scratch `k = 3` (passed in the test call `predict_label(X_train, y_train, X_test[0], 3)`).
- [cell 8] from-scratch `k = 3` (in the list comprehension over the test set).
- [cell 10] `make_blobs(n_samples=150, centers=3, n_features=2)` (no random_state given; `np.random.seed(0)` set just before). `k_nearest = np.argsort(distances)[:3]` (k=3 for viz).
- [cell 12] `KNeighborsClassifier(n_neighbors=3)` — all other params `default` (e.g. `weights='uniform'`, `metric='minkowski'`, `p=2` are sklearn defaults, not explicitly set).

## Plots produced
- [cell 10] Scatter plot (figsize=(10,6)): training blobs colored by class (`cmap='viridis'`), the single test point as a red star (`marker='*', s=200`), and the 3 nearest neighbors highlighted as black circles (`marker='o', s=100`). x-axis "Feature 1", y-axis "Feature 2", title "k-Nearest Neighbors Classifier Visualization".

## What is left as an exercise to the student
- No explicit exercise prompts; cell-13 is empty. Markdown (cell 3) notes k-NN "can be sensitive to the choice of k and the presence of noisy or irrelevant features" — an implicit invitation to experiment with k and feature scaling, but no coded exercise.

## Key cell indices for code idiom extraction
- "[cell 7]: the from-scratch k-NN trio — `euclidean_distance` (`np.sqrt(np.sum((x-y)**2))`), `predict` (`np.argsort(distances)[:k]`), `predict_label` (`np.bincount(...).argmax()`)."
- "[cell 8]: vectorized eval idiom — `y_pred = [predict_label(X_train, y_train, x, 3) for x in X_test]` then `np.mean(y_pred == y_test)`."
- "[cell 10]: k-NN visualization idiom (highlighting nearest neighbors with a starred test point)."
- "[cell 12]: sklearn cross-check — `KNeighborsClassifier(n_neighbors=3).fit(...).score(...)`."

## Notes / [VERIFY] flags
- Markdown math (cell 3) transcribed: Euclidean distance `d(x,y) = sqrt((x₁−y₁)² + (x₂−y₂)² + ... + (xₙ−yₙ)²)`. Algorithm steps listed: compute distances, sort, take k closest, majority-vote the label.
- Note the from-scratch `predict` relies on `np.bincount` which requires non-negative integer labels — works for iris targets {0,1,2}; would break on non-integer/string labels. Worth flagging as a limitation when teaching.
- No execution outputs are stored in the file for cells 5, 7, 8, 12 (accuracy values not citable from the read). The sklearn and from-scratch versions both use k=3 on the same split (random_state=10), so they should agree.
- k-NN here uses unscaled iris features; markdown elsewhere in the course (notebook 01) stresses scaling for distance-based models, but it is intentionally omitted here.
