# Notebook extract: 11_K_means.ipynb

**Source path:** course_materials\Lecture notebooks\11_K_means.ipynb
**Cell count:** 12 (cell-0 through cell-11)

## Dataset(s) loaded
- Iris dataset loaded from URL: `'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'` (cell-2)
  - Columns assigned (header=None): `['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']`
  - Features used: `X = df[['PetalLengthCm', 'PetalWidthCm']]` — only petal length and width (cell-2)
  - `Species` column is in the data but NOT used as input features or to define K; K=3 chosen to match known species count

## Preprocessing steps
- No scaling or encoding applied. Raw numeric features used directly.
- Species column loaded but only used implicitly to motivate K=3.

## Method(s) demonstrated
- `from sklearn.cluster import KMeans` — sklearn, NOT from scratch (cell-2)
- `from sklearn.metrics import silhouette_score` — sklearn (cell-8)
- Elbow method (WCSS via `kmeans.inertia_`) implemented manually in a loop (cell-8)
- Silhouette method implemented manually in a loop (cell-8)

## Hyperparameters set
- Initial clustering: `KMeans(n_clusters=3, random_state=0, n_init=10)` (cell-2)
- Elbow/silhouette loop: `KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)` for k in range(2, 10) (cell-8)
- Final "optimal" clustering: `KMeans(n_clusters=2, random_state=0, n_init=10)` — notebook chooses K=2 based on silhouette (cell-10)

## Plots produced
- [cell-2]: Scatter — `PetalLengthCm` vs `PetalWidthCm`, coloured by `kmeans.labels_` (K=3); xlabel "Petal Length (cm)", ylabel "Petal Width (cm)"; title "K-means Clustering of Iris Flowers"
- [cell-8]: Line plot — WCSS (`kmeans.inertia_`) vs number of clusters (range 2–9); title "Within-Cluster Sum of Squares (WCSS) vs. Number of Clusters"
- [cell-8]: Line plot — silhouette score vs number of clusters (range 2–9); title "Silhouette Score vs. Number of Clusters"
- [cell-10]: Scatter — `PetalLengthCm` vs `PetalWidthCm`, coloured by `kmeans.labels_` (K=2); same labels; title "K-means Clustering of Iris Flowers"

## What is left as an exercise to the student
- No explicit TODO/exercise cells. Notebook is fully worked.

## Key cell indices for code idiom extraction
- "[cell-2]: `kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)` — fit-in-one-line idiom; access labels with `kmeans.labels_`"
- "[cell-8]: `for k in range(2, 10): kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0); kmeans.fit(X); wcss.append(kmeans.inertia_); silhouette_scores.append(silhouette_score(X, kmeans.labels_))` — elbow + silhouette loop"

## Notes / [VERIFY] flags
- WCSS formula from cell-3: $WCSS = \sum\left(\sum((X - centroid)^2)\right)$
- Silhouette coefficient formula from cell-5: $s = \frac{(b - a)}{\max(a, b)}$ where $a$ = mean intra-cluster distance, $b$ = mean nearest-cluster distance
- The notebook selects K=2 as "optimal" (cell-10) based on silhouette, even though the dataset has 3 known species — illustrates that unsupervised optimal K may differ from ground truth.
- `init='k-means++'` is used in the loop (cell-8) but `random_state=0` makes the run deterministic; the initial clustering (cell-2) also uses `random_state=0` but does NOT specify `init` (defaults to `'k-means++'`). [VERIFY: default init in sklearn KMeans is `'k-means++'` — so both are equivalent.]
- `n_init=10` is explicit in all calls — since sklearn>=1.2 changed the default from 10 to `'auto'`, being explicit is pedagogically sound.
- The notebook is labelled "Lecture 6" (cell-0).
