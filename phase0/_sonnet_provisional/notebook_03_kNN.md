# Notebook extract: 03_Supervised_learning_with_kNN (1).ipynb

**Source path:** `course_materials/Lecture notebooks/03_Supervised_learning_with_kNN (1).ipynb`
**Cell count:** 14 (cell-0 through cell-13)

## Dataset(s) loaded

1. **Iris dataset** (sklearn built-in) — loaded via `sklearn.datasets.load_iris()`:
   - Features `X = iris.data` — 4 columns: `sepal length`, `sepal width`, `petal length`, `petal width` (column names not accessed explicitly, array form used)
   - Target `y = iris.target` — 3 classes (0, 1, 2)
   - 150 total samples

2. **Simulated blobs** (sklearn) — `make_blobs(n_samples=150, centers=3, n_features=2)` in cell-10 — used only for the visualisation of k-NN neighbour selection, not for accuracy evaluation.

## Preprocessing steps

- [cell-4] `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)`
- No feature scaling applied (k-NN is distance-based but scaling is not demonstrated in this notebook — [VERIFY: intentional omission or oversight?])

## Method(s) demonstrated

Two implementations:

1. **FROM SCRATCH (NumPy)** — cell-7:
   ```python
   def euclidean_distance(x, y):
       return np.sqrt(np.sum((x - y)**2))

   def predict(X_train, y_train, X_test, k):
       distances = [euclidean_distance(X_test, x) for x in X_train]
       k_nearest = np.argsort(distances)[:k]
       return y_train[k_nearest]

   def predict_label(X_train, y_train, X_test, k):
       return np.bincount(predict(X_train, y_train, X_test, k)).argmax()
   ```

2. **sklearn** — cell-12:
   `from sklearn.neighbors import KNeighborsClassifier`
   `knn = KNeighborsClassifier(n_neighbors=3)`

## Hyperparameters set

- From-scratch: `k=3` (hard-coded in cell-7 test call and cell-8 list comprehension)
- sklearn: `KNeighborsClassifier(n_neighbors=3)` — all other parameters `default`

## Plots produced

- [cell-10] Scatter plot (simulated blobs visualisation):
  - Training data coloured by class (`c=y_train_sim`, `cmap='viridis'`)
  - Single test point marked with red star (`c='red'`, `marker='*'`, `s=200`)
  - 3 nearest neighbours marked in black circles (`c='black'`, `marker='o'`, `s=100`)
  - Axes: `Feature 1` / `Feature 2`; title: `k-Nearest Neighbors Classifier Visualization`

## What is left as an exercise to the student

No explicit TODO / "your turn" cells found in this notebook.

## Key cell indices for code idiom extraction

- [cell-7]: From-scratch euclidean distance + k-NN predict functions (the canonical from-scratch idiom)
- [cell-8]: Accuracy calculation — `accuracy = np.mean(y_pred == y_test)` using list comprehension over test set
- [cell-10]: Neighbour visualisation with `make_blobs` and matplotlib scatter
- [cell-12]: sklearn k-NN — `KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)` + `.score(X_test, y_test)`

## Notes / [VERIFY] flags

- Euclidean distance formula stated in cell-3 markdown:
  $$d(x,y) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + ... + (x_n - y_n)^2}$$
- cell-8 output is not shown in the Read output (outputs omitted for that cell). The notebook text confirms accuracy is computed and printed as `"Accuracy: X%"`.
- cell-13 is empty.
- The from-scratch implementation uses a Python list comprehension over the training set for distances — O(n) scan, no KD-tree.
- `np.bincount(...).argmax()` is the majority-vote mechanism (cell-7) — works because labels are integers 0/1/2.
- The simulated blob data uses `np.random.seed(0)` (cell-10); the iris train/test split uses `random_state=10` (cell-4).
- sklearn and from-scratch should give identical results for k=3 on this data (both confirmed to print accuracy, though exact value not captured in the cell output shown).
