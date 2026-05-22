# Notebook extract: 04_Linear_discriminant_analysis.ipynb

**Source path:** Lecture notebooks/04_Linear_discriminant_analysis.ipynb
**Cell count:** 21 cells (indices cell-0 through cell-20 as reported by Read; cell-20 empty)

## Dataset(s) loaded
- `make_classification(...)` from `sklearn.datasets` — [cell 6]. Toy 2-feature binary set: `n_samples=200, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, class_sep=0.75, random_state=42`. Produces `X` (200×2), `y` (2 classes 0/1). First 5 X rows / y shown in output (e.g. X[0]=[-0.47131493, -0.71420127], y[0:5]=[0,1,0,0,0]).
- `load_iris()` from `sklearn.datasets` — [cell 15]. `X2, y2 = iris.data, iris.target` (150×4, 3 classes). `target_names` used for legend labels (setosa/versicolor/virginica). Bunch-style (not as_frame).

## Preprocessing steps
- [cell 6] `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)` (the toy classification set).
- [cell 15] `X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=10)` (iris).
- No scaling/encoding applied. LDA is fit on raw features.
- Decision-boundary mesh idiom [cells 10, 12, 19]: `np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))`; predict on `np.c_[xx.ravel(), yy.ravel()]`; reshape to `xx.shape`.

## Method(s) demonstrated
Linear Discriminant Analysis (LDA), demonstrated TWO ways:
1. SKLEARN — `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis`:
   - [cell 8] `lda = LinearDiscriminantAnalysis(n_components=1)` on toy data; `.fit`, `.transform`, `.predict`, `.score`.
   - [cell 15] `lda = LinearDiscriminantAnalysis(n_components=2)` on iris; `.fit_transform` (train) + `.transform` (test).
2. FROM SCRATCH (numpy) — [cell 18] `class LDAFromScratch`:
   - Computes class means, priors, class counts.
   - Within-class scatter `S_W` accumulated per class via outer products; pooled covariance `cov_ = S_W / (n_samples - len(classes))` (unbiased, "consistent with sklearn"); `cov_inv_ = np.linalg.inv(cov_)`.
   - Between-class scatter `S_B = Σ N_c (μ_c − μ)(μ_c − μ)ᵀ`.
   - Eigendecomposition of `A = inv(cov_) @ S_B` via `np.linalg.eig`; take real parts; sort eigvals descending; keep top `n_components` eigenvectors as `W_`.
   - `predict()` uses the linear discriminant rule `δ_c(x) = xᵀ Σ⁻¹ μ_c − 0.5 μ_cᵀ Σ⁻¹ μ_c + log(prior_c)`, argmax over classes.
   - `transform()` centers by overall mean then projects: `(X − overall_mean) @ W_`.
   - [cell 19] instantiated `LDAFromScratch(n_components=1)`, fit on the toy `X_train`/`y_train`, transformed, decision boundaries plotted for train and test, accuracy reported via `np.mean(y_pred_test_scratch == y_test)`.

Import paths:
- `matplotlib.pyplot as plt`, `numpy as np`, `from sklearn.datasets import make_classification`, `from sklearn.model_selection import train_test_split` [cell 6].
- `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis` [cell 8].
- `from sklearn.datasets import load_iris` [cell 15].

## Hyperparameters set
- [cell 6] `make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, class_sep=0.75, random_state=42)`; `train_test_split(test_size=0.33, random_state=42)`.
- [cell 8] `LinearDiscriminantAnalysis(n_components=1)` — solver/shrinkage `default`.
- [cell 15] `LinearDiscriminantAnalysis(n_components=2)`; `train_test_split(test_size=0.2, random_state=10)`.
- [cell 18/19] `LDAFromScratch(n_components=1)`; if `n_components is None` it defaults to `len(classes) - 1`.

## Plots produced
- [cell 7] Scatter of toy training data, two classes colored navy/orange. x="Feature 1", y="Feature 2", title "Toy binary classification data".
- [cell 9] Scatter of the 1D LDA projection: `plt.scatter(range(len(X_transformed)), X_transformed, c=y_train, cmap=plt.cm.coolwarm)`.
- [cell 10] LDA decision boundary on TRAIN: `plt.contourf(xx, yy, Z, cmap=coolwarm, alpha=0.8)` + scatter of train points. title "LDA Decision Boundary".
- [cell 12] LDA decision boundary on TEST (same idiom, test points). title "LDA Decision Boundary".
- [cell 15] Iris LDA train projection: scatter of `X_train_lda` (LD1 vs LD2), colored red/blue/green per class. x="LD1", y="LD2".
- [cell 16] Iris LDA test projection (same axes, test points).
- [cell 19] Four plots: 1D from-scratch projection; decision boundary (train); decision boundary (test); each via contourf+scatter with `edgecolor='k'`.

## What is left as an exercise to the student
- No explicit exercise prompts; cell-20 empty. Markdown (cell 4) states "We will not dive into all the mathematical details of implementing LDA" — the from-scratch class (cell 18) effectively fills that gap. Implicit: compare from-scratch vs sklearn accuracy/boundaries (the notebook does this for the toy set but not numerically side-by-side with sklearn on iris).

## Key cell indices for code idiom extraction
- "[cell 6]: `make_classification(...)` toy-data idiom with all the documented kwargs."
- "[cell 8]: sklearn LDA core — `LinearDiscriminantAnalysis(n_components=1).fit(X,y).transform(X)`."
- "[cell 10]: the reusable decision-boundary mesh+contourf idiom (meshgrid → `np.c_[xx.ravel(), yy.ravel()]` → predict → reshape → contourf)."
- "[cell 15]: `lda.fit_transform(X_train, y_train)` + `lda.transform(X_test)` for 2D LDA dimensionality reduction + LD1/LD2 scatter."
- "[cell 18]: the full `class LDAFromScratch` — S_W/S_B construction, eig of inv(cov)@S_B, discriminant-rule predict. Canonical from-scratch reference."

## Notes / [VERIFY] flags
- Markdown math transcribed (cell 2): Fisher criterion `J(w) = (wᵀ S_B w)/(wᵀ S_W w)`; `S_B = Σ_{i=1}^k N_i (μ_i−μ)(μ_i−μ)ᵀ`; `S_W = Σ_{i=1}^k Σ_{x∈C_i} (x−μ_i)(x−μ_i)ᵀ`. Generalized eigenproblem `S_W⁻¹ S_B w = λ w`. (cell 3) Steps 1–5 listed.
- (cell 4) classifier decision rule: classify x into class i if `P(C_i|x) > P(C_j|x) ∀j≠i`; assumes Gaussian per class with SHARED covariance ⇒ linear boundary.
- Stored output values: [cell 13] `lda.score(X_test, y_test)` = `0.7424242424242424` (toy set, sklearn). [cell 6] sample arrays shown. Other plot cells: outputs too large to inline. [cell 19] from-scratch accuracy NOT stored in the read (output too large) — [VERIFY: run to capture `accuracy_scratch` and confirm it matches sklearn's ~0.742 on the toy set].
- Note from-scratch `cov_` uses pooled covariance `S_W/(n_samples - k)`, matching sklearn's unbiased convention (commented in code). This is a deliberate teaching point.
- [cell 13] begins with a stray `plt.show()` before the score call.
- Minor: in [cell 7] `target_names = list(set(y))` is computed but the loop indexes by `i` (0,1), effectively just the two class ints.
