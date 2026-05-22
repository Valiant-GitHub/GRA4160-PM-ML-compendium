# Notebook extract: 11_K_means.ipynb

**Source path:** Lecture notebooks/11_K_means.ipynb
**Cell count:** 12 (cells 0-11)

## Dataset(s) loaded
- **Iris** dataset from a URL via pandas (NOT `sklearn.datasets.load_iris`) [cell 2]:
  - `df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])`
  - Features used (X): `df[['PetalLengthCm', 'PetalWidthCm']]` — only petal length & width (2D)
  - No target used for fitting (unsupervised); `Species` column is loaded but not used for clustering.

## Preprocessing steps
- [cell 2] Subset to 2 columns: `X = df[['PetalLengthCm', 'PetalWidthCm']]`
- No scaling/normalization applied (markdown [cell 1] recommends normalization, but code does not scale).

## Method(s) demonstrated
- **K-means clustering** — sklearn. `from sklearn.cluster import KMeans` [cell 2]
- **Silhouette score** evaluation — `from sklearn.metrics import silhouette_score` [cell 8]
- Plotting: `import matplotlib.pyplot as plt` [cell 2]
- No from-scratch implementation. WCSS and silhouette formulas are markdown theory [cells 3, 5].

## Hyperparameters set
- [cell 2] `KMeans(n_clusters=3, random_state=0, n_init=10)` — fit on `X`; init=`default` ('k-means++'), max_iter=`default` (300)
- [cell 8] loop over k in `range(2, 10)`: `KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)`
- [cell 10] `KMeans(n_clusters=2, random_state=0, n_init=10)` — the chosen "optimal" k

## Plots produced
- [cell 2] Scatter `PetalLengthCm` vs `PetalWidthCm` colored by `kmeans.labels_` (3 clusters). Title "K-means Clustering of Iris Flowers".
- [cell 8] Two line plots: (1) WCSS (`kmeans.inertia_`) vs number of clusters (k=2..9); (2) Silhouette score vs number of clusters (k=2..9).
- [cell 10] Scatter `PetalLengthCm` vs `PetalWidthCm` colored by `kmeans.labels_` (2 clusters). Title "K-means Clustering of Iris Flowers".

## What is left as an exercise to the student
- None explicitly numbered. Markdown [cell 7] frames the elbow + silhouette plots (cell 8) as the task; cell 10 then applies the "optimal" k. The student is implicitly invited to read off the elbow/silhouette to confirm k.

## Key cell indices for code idiom extraction
- "[cell 2]: `KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)`" then color scatter by `kmeans.labels_`
- "[cell 8]: elbow + silhouette loop — `for k in range(2,10): km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(X); wcss.append(km.inertia_); silhouette_scores.append(silhouette_score(X, km.labels_))`"
- "[cell 8]: WCSS access via `kmeans.inertia_`"

## Notes / [VERIFY] flags
- Markdown formulas: WCSS $WCSS = \sum(\sum((X - centroid)^2))$ [cell 3]; Silhouette coefficient $s = \frac{(b - a)}{\max(a, b)}$ [cell 5].
- **CODE/PROSE MISMATCH [cell 10]:** the comment says `# Apply k-means clustering with 3 clusters (one for each species of iris)` but the call uses `n_clusters=2`. The surrounding markdown [cell 9] correctly calls this the "optimal" number — and for petal length/width the elbow/silhouette typically favors k=2 (Setosa vs the rest), so the comment is stale/copied from cell 2, not the actual k.
- Conceptual note: iris has 3 species but clustering on only 2 petal features with the silhouette criterion leads the notebook to pick k=2 — a deliberate teaching point that "optimal k" need not equal the true class count. Not flagged as such in the notebook.
- No scaling despite markdown advising normalization (petal length and width are already on similar cm scales, so impact is minor here).
