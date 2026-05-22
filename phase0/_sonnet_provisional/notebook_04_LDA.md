# Notebook extract: 04_Linear_discriminant_analysis.ipynb

**Source path:** `Lecture notebooks/04_Linear_discriminant_analysis.ipynb`
**Cell count:** 21 (cell-0 through cell-20)

## Dataset(s) loaded

1. **Synthetic binary classification data** (sklearn) — cell-6:
   ```python
   X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, class_sep=0.75, random_state=42)
   ```
   2 features, 2 classes (0/1), 200 samples. No named columns (numpy array).

2. **Iris dataset** (sklearn built-in) — cell-15:
   `from sklearn.datasets import load_iris; iris = load_iris()`
   `X2, y2 = iris.data, iris.target`
   4 features (`sepal length`, `sepal width`, `petal length`, `petal width`), 3 classes (`setosa`, `versicolor`, `virginica`), 150 samples.

## Preprocessing steps

- [cell-6] `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)` — for synthetic data
- [cell-15] `X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=10)` — for iris

No feature scaling applied before LDA (sklearn's LDA handles this internally).

## Method(s) demonstrated

Two implementations:

1. **sklearn** — `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis`
   - Binary data (cell-8): `lda = LinearDiscriminantAnalysis(n_components=1)` — fit, transform, predict, score
   - Iris data (cell-15): `lda = LinearDiscriminantAnalysis(n_components=2)` — fit_transform on train, transform on test

2. **FROM SCRATCH (NumPy)** — cell-18: `class LDAFromScratch` implementing:
   - Within-class scatter matrix `S_W`
   - Pooled covariance `self.cov_ = S_W / (n_samples - len(self.classes_))`
   - Between-class scatter matrix `S_B`
   - Eigendecomposition of `np.linalg.inv(self.cov_) @ S_B`
   - Linear discriminant classification rule: `delta_c(x) = x^T * cov_inv_ * mean_c - 0.5 * mean_c^T * cov_inv_ * mean_c + log(prior_c)`
   - `transform` method: `X_centered = X - self.overall_mean_.ravel(); return X_centered @ self.W_`

## Hyperparameters set

- sklearn binary: `LinearDiscriminantAnalysis(n_components=1)` — all other params `default`
- sklearn iris: `LinearDiscriminantAnalysis(n_components=2)` — all other params `default`
- `make_classification`: `n_samples=200`, `n_features=2`, `n_redundant=0`, `n_informative=2`, `n_clusters_per_class=1`, `class_sep=0.75`, `random_state=42`
- `LDAFromScratch(n_components=1)` — cell-19 (used on binary synthetic data)

## Plots produced

- [cell-7] Scatter plot: training data for binary synthetic set, coloured by class (navy/orange), axes `Feature 1` / `Feature 2`, title `Toy binary classification data`
- [cell-9] Scatter plot: 1D LDA projection — `range(len(X_transformed))` vs `X_transformed`, coloured by `y_train`, `cmap=plt.cm.coolwarm`
- [cell-10] Decision boundary plot (training set): `contourf` on meshgrid over `Feature 1` / `Feature 2`, scatter overlay of training points, title `LDA Decision Boundary`
- [cell-12] Decision boundary plot (test set): same structure as cell-10 but on test points
- [cell-15] Scatter plot: iris training data in LDA space — `LD1` (x) vs `LD2` (y), 3 colours (red/blue/green) for setosa/versicolor/virginica
- [cell-16] Scatter plot: iris test data in LDA space — same axes as cell-15
- [cell-19] Multiple plots: 1D projection scatter (from-scratch) + decision boundary (train) + decision boundary (test) — all for binary synthetic data

## What is left as an exercise to the student

No explicit TODO / "your turn" cells found in this notebook.

## Key cell indices for code idiom extraction

- [cell-6]: `make_classification(...)` data generation pattern
- [cell-8]: sklearn LDA fit + transform — `lda.fit(X_train, y_train)` then `lda.transform(X_train)`
- [cell-10]: Decision boundary meshgrid + `contourf` pattern (reusable boilerplate)
- [cell-15]: `lda.fit_transform(X_train2, y_train2)` then `lda.transform(X_test2)` — supervised dimensionality reduction idiom
- [cell-18]: Full `LDAFromScratch` class — the canonical from-scratch LDA implementation
- [cell-13]: `lda.score(X_test, y_test)` returns `0.7424242424242424`

## Notes / [VERIFY] flags

- Key math in cell-2 markdown — Fisher criterion:
  $$J(w) = \frac{w^T S_B w}{w^T S_W w}$$
  Between-class scatter:
  $$S_B = \sum_{i=1}^{k} N_i (\mu_i - \mu)(\mu_i - \mu)^T$$
  Within-class scatter:
  $$S_W = \sum_{i=1}^{k} \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T$$
- Generalised eigenvalue problem (cell-3 markdown):
  $$S_W^{-1} S_B w = \lambda w$$
- Bayes decision rule (cell-4 markdown):
  $$\text{Classify } x \text{ into class } i \text{ if: } P(C_i | x) > P(C_j | x), \, \forall j \neq i$$
- cell-20 is empty.
- The from-scratch `LDAFromScratch` divides `S_W` by `(n_samples - len(self.classes_))` — this is the unbiased pooled covariance, consistent with sklearn's implementation (noted explicitly in the code comment).
- The `predict` method in `LDAFromScratch` implements the linear discriminant function (log-posterior score), NOT a projection-then-threshold — it computes full class scores.
- sklearn test accuracy on binary synthetic data: `0.7424242424242424` (cell-13 output).
