# Notebook extract: 10_PCA.ipynb

**Source path:** course_materials/Lecture notebooks/10_PCA.ipynb
**Cell count:** 16 (cells 0-15; cell 15 is empty)

## Dataset(s) loaded
- **Seeds** dataset from CSV [cell 4]:
  - `seeds = pd.read_csv('../data/seeds.csv')`
  - Features used (X): `['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry_coefficient', 'grove_length']` (7 features)
  - Label column for coloring: `seeds['type']` (markdown says "two different varieties of wheat" / 140 obs)
- **Wine** dataset from sklearn [cell 10]:
  - `from sklearn.datasets import load_wine`
  - `features, target = load_wine(return_X_y=True)` — 178 samples, 13 features, 3 classes (class_0/1/2)

## Preprocessing steps
- Seeds: no scaling before the first PCA [cell 6] (PCA fit on raw `X`).
- Wine: train/test split [cell 12] `train_test_split(features, target, test_size=0.30)` — **no random_state set** (non-reproducible).
- Wine scaled vs unscaled comparison [cell 12]: one pipeline without scaler, one with `StandardScaler()`.
- Markdown [cell 8] discusses why standardization matters for PCA.

## Method(s) demonstrated
- **PCA** — sklearn. `from sklearn.decomposition import PCA` [cell 6]
- **GaussianNB** classifier used downstream of PCA in pipelines [cell 12]: `from sklearn.naive_bayes import GaussianNB`
- Helpers: `from sklearn.model_selection import train_test_split`, `from sklearn.pipeline import make_pipeline`, `from sklearn.preprocessing import StandardScaler` [cell 12]; `from sklearn import metrics` [cell 14]
- Plotting: `import seaborn as sns` [cell 2]
- No from-scratch implementation. PCA-via-covariance-eig and PCA-via-SVD math are markdown theory [cell 1].

## Hyperparameters set
- [cell 6] `PCA(n_components=2)` — fit on raw seeds `X`
- [cell 12] `make_pipeline(PCA(n_components=2), GaussianNB())` — unscaled
- [cell 12] `make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())` — scaled
- [cell 12] `train_test_split(test_size=0.30)` — random_state=`default` (None)

## Plots produced
- [cell 4] Two seaborn scatter plots: (1) `area` vs `perimeter` colored by `type`; (2) `asymmetry_coefficient` vs `grove_length` colored by `type`.
- [cell 6] Scatter of seeds projected onto first vs second principal component, colored by `label` (= seeds type). Axes "First/Second Principal Component".
- [cell 12] Side-by-side (ncols=2, figsize (10,7)) matplotlib scatter: left = "Training dataset after PCA" (unscaled), right = "Standardized training dataset after PCA"; classes 0/1/2 with markers ^, s, o and colors blue/red/green. Axes "1st/2nd principal component".

## What is left as an exercise to the student
- None explicitly stated (no exercise cells). Notebook is fully worked.

## Key cell indices for code idiom extraction
- "[cell 6]: `pca = PCA(n_components=2); pca.fit(X); pca.components_`" and `pca.transform(X)` into a DataFrame
- "[cell 7]: explained variance — `sum(pca.explained_variance_ratio_.round(2))`"
- "[cell 12]: `make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())`" — scaling+PCA+classifier pipeline
- "[cell 12]: access pipeline steps via `clf.named_steps['pca']`, `clf.named_steps['standardscaler']`"
- "[cell 12]: scaled projection for plotting — `pca_std.transform(scaler.transform(X_train))`"
- "[cell 14]: `metrics.accuracy_score(y_test, pred_test)`" comparing scaled vs unscaled

## Notes / [VERIFY] flags
- Markdown formulas [cell 1]: $\bar{X} = X - mean(X)$; $\Sigma = (1/n)\bar{X}'\bar{X}$; $\Lambda, V = eig(\Sigma)$; projection $\hat{X} = \bar{X}V$. SVD: $X = U\Sigma V'$, $\hat{X} = XV$.
- **[VERIFY] indexing bug in plot [cell 12]:** the unscaled plot uses `X_train[y_train == l, 0]` and `X_train[y_train == l, 1]` — i.e. it plots the first two RAW features, not the first two PCA components, even though `ax1` is titled "Training dataset after PCA". The scaled plot (`X_train_std`) correctly uses PCA-transformed data. So the left panel is mislabeled / does not actually show PCA output. This is a genuine code/prose (title) mismatch.
- **Non-reproducible:** `train_test_split` [cell 12] has no `random_state`, so accuracy printout [cell 14] varies run to run.
- Seeds confirmed against the real file `course_materials/Data/seeds.csv`: columns are exactly `area,perimeter,compactness,length,width,asymmetry_coefficient,grove_length,type`; `type` has exactly 2 classes (values 1 and 2, 70 rows each, 140 total) — matches markdown [cell 3]. NOTE: the notebook path `../data/seeds.csv` (lowercase) differs from the repo's actual `course_materials/Data/seeds.csv` (capital D); on case-insensitive Windows this works, but is a path-case discrepancy.
- Cell 15 is empty.
