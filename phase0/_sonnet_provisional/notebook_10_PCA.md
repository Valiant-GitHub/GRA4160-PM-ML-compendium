# Notebook extract: 10_PCA.ipynb

**Source path:** Lecture notebooks\10_PCA.ipynb
**Cell count:** 16 (cell-0 through cell-15)

## Dataset(s) loaded
- `'../data/seeds.csv'` — 140 observations, 7 features + type label (cell-4)
  - Feature columns used: `['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry_coefficient', 'grove_length']`
  - Label column: `'type'`
- `sklearn.datasets.load_wine(return_X_y=True)` — 178 samples, 13 features; target: cultivar class (0, 1, 2) (cell-10)

## Preprocessing steps
- Seeds dataset: no explicit scaling before first PCA in cell-6 (unscaled PCA)
- Wine dataset:
  - `train_test_split(features, target, test_size=0.30)` — no random_state specified (cell-12)
  - Pipeline 1 (unscaled): `make_pipeline(PCA(n_components=2), GaussianNB())` (cell-12)
  - Pipeline 2 (scaled): `make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())` (cell-12)
  - For visualisation: `scaler.transform(X_train)` then `pca_std.transform(...)` (cell-12)

## Method(s) demonstrated
- `from sklearn.decomposition import PCA` — sklearn, NOT from scratch (cell-6)
- `from sklearn.naive_bayes import GaussianNB` — used as downstream classifier after PCA (cell-12)
- `from sklearn.pipeline import make_pipeline` (cell-12)
- `from sklearn.preprocessing import StandardScaler` (cell-12)
- `from sklearn.model_selection import train_test_split` (cell-12)
- `from sklearn import metrics` — for `accuracy_score` (cell-14)
- Mathematical derivation of PCA (covariance-eigendecomposition AND SVD) provided in prose/LaTeX only (cell-1) — NOT implemented in code

## Hyperparameters set
- Seeds PCA: `PCA(n_components=2)` (cell-6)
- Wine unscaled pipeline: `PCA(n_components=2)` (cell-12)
- Wine scaled pipeline: `PCA(n_components=2)` (cell-12)
- `GaussianNB()`: all `default` (cell-12)
- `train_test_split`: `test_size=0.30`; `random_state` = `default` (not set) (cell-12)

## Plots produced
- [cell-4]: Scatter — `area` vs `perimeter`, hue=`type`; `sns.scatterplot(..., sizes=(20, 200))`; title "Scatter plot of the first two features"
- [cell-4]: Scatter — `asymmetry_coefficient` vs `grove_length`, hue=`type`; same format; title "Scatter plot of the last two features"
- [cell-6]: Scatter — first vs second principal component (seeds), hue=label; `sns.scatterplot(data=data_projected, x='first', y='second', hue='label', sizes=(20, 200))`; title "Scatter plot of the first two principal components"
- [cell-12]: Side-by-side subplots (figsize=(10,7)):
  - ax1: "Training dataset after PCA" — raw feature[0] vs feature[1] by class (3 colours/markers)
  - ax2: "Standardized training dataset after PCA" — PC1 vs PC2 by class

## What is left as an exercise to the student
- No explicit TODO/exercise cells. Notebook is fully worked.

## Key cell indices for code idiom extraction
- "[cell-6]: `pca = PCA(n_components=2); pca.fit(X); principal_components = pca.components_; data_projected = pd.DataFrame(pca.transform(X), columns=['first', 'second'])` — fit/transform/inspect pattern"
- "[cell-7]: `print(f'The explained variance ratio is: {sum(pca.explained_variance_ratio_.round(2))}')` — sum of explained variance ratio"
- "[cell-12]: `std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB()); std_clf.fit(X_train, y_train)` — scaled PCA pipeline"
- "[cell-12]: `pca_std = std_clf.named_steps['pca']; scaler = std_clf.named_steps['standardscaler']` — extract named steps from pipeline"

## Notes / [VERIFY] flags
- PCA mathematical steps from cell-1:
  - $\bar{X} = X - mean(X)$
  - $\Sigma = (1/n)\bar{X}'\bar{X}$
  - $\Lambda, V = eig(\Sigma)$
  - Projected data: $\hat{X} = \bar{X}V$
- SVD formulation from cell-1: $X = U\Sigma V'$; columns of V are right singular vectors = principal components; $\hat{X} = XV$
- Seeds dataset has 140 observations per markdown (cell-3), but description says "two different varieties of wheat" — [VERIFY: the UCI seeds dataset actually has 3 varieties and 210 observations; 140 may indicate a subset in the local CSV.]
- `train_test_split` in cell-12 has no `random_state` — results will vary across runs. [VERIFY: intentional for pedagogical comparison or oversight?]
- Cell-15 is empty.
- The notebook is labelled "Lecture 6" (cell-0).
