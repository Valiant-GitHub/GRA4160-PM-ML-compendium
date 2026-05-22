# Exercise extract: Customer segmentation (Bank Marketing) (stub + VHL solution)

**Stub path:** `Exercises and solutions(VHL)\06_Customer_segmentation (1).ipynb`   **Solution path:** `Exercises and solutions(VHL)\06_Customer_segmentation_VHL.ipynb`
**Cell counts:** stub=3 (2 markdown, 0 code; trailing empty cell-2), solution=16 (markdown cells-0,1,3,5,7,9,11,13,15; code cells-2,4,6,8,10,12,14)

## What the exercise teaches (1-2 sentences)
Unsupervised customer segmentation on the Bank Marketing dataset: preprocess mixed numeric/categorical data (scale + one-hot), choose the number of clusters via the elbow method (WCSS) and silhouette score, run K-Means, visualize clusters in 2D via PCA, and interpret clusters against the held-aside target `y`. The only end-to-end unsupervised-learning exercise.

## Setup
- **Dataset:** UCI Bank Marketing (Portuguese bank direct-marketing campaigns). Stub describes 4 file variants: `bank-additional-full.csv` (41,188 rows, 20 inputs), `bank-additional.csv` (4,119 rows), `bank-full.csv` (17 inputs, older), `bank.csv`. Source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
- **Exact loader call (solution cell-2):** `pd.read_csv('../../data/bank_marketing_data_set/bank-additional/bank-additional-full.csv', sep=';')` (semicolon separator).
- **Target (held aside, not used for clustering):** `y` (binary `yes`/`no` = subscribed to term deposit).
- **Numeric columns (exact, solution cell-6):** `age, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed`.
- **Categorical columns:** all others (e.g., `job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome`), one-hot encoded.
- **Task:** unsupervised clustering (no labels in training); interpret segments.
- **Expected output:** preprocessed feature shape; histograms + correlation heatmap; WCSS + silhouette per k (2-10); chosen k (=3) and labels; PCA 2D scatter colored by cluster; per-cluster numeric summary stats and target distribution.

## What the student must implement (from the stub)
Stub is markdown-only; tasks span stub cell-0 (data overview/discussion) and cell-1 (5-part exercise):
1. **Data Loading and Preprocessing:** load data; handle missing values; normalize numeric features; encode categoricals.
2. **Data Visualization:** scatter plots / distributions; look for clusters/outliers.
3. **Clustering with K-Means:** apply K-Means; vary K; evaluate via WCSS and silhouette coefficient.
4. **Cluster Visualization:** PCA to 2D; plot first two PCs; visualize clusters.
5. **Cluster Interpretation and Insights:** characterize each cluster; find patterns (e.g., which segments subscribe); discuss marketing implications and limitations.

## Solution walkthrough (from _VHL)
- **cell-2 (Step 1 load):** `import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns`; `%matplotlib inline`. `df = pd.read_csv('../../data/bank_marketing_data_set/bank-additional/bank-additional-full.csv', sep=';')`; print shape; `df.head()`.
- **cell-4 (missing values):** `df.replace('unknown', np.nan, inplace=True)`; `df.dropna(inplace=True)` (treats the literal string `'unknown'` as missing, then drops those rows).
- **cell-6 (feature prep):** `target = df['y']`; `df_features = df.drop('y', axis=1)`. Define `numeric_cols` (the 10 above); `categorical_cols = [col for col in df_features.columns if col not in numeric_cols]`. `from sklearn.preprocessing import StandardScaler`; `scaler = StandardScaler()`; `df_features[numeric_cols] = scaler.fit_transform(df_features[numeric_cols])`. One-hot: `df_features_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=True)`. `X = df_features_encoded`.
- **cell-8 (Step 2 viz):** histograms of numeric features (`X_numeric.hist(bins=30, figsize=(15,10))`); correlation heatmap via `sns.heatmap(X_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f')`.
- **cell-10 (Step 3 K-Means):** `from sklearn.cluster import KMeans`; `from sklearn.metrics import silhouette_score`. `K_range = range(2, 11)`; for each k: `kmeans = KMeans(n_clusters=k, random_state=42)`; `kmeans.fit(X)`; collect `kmeans.inertia_` (WCSS) and `silhouette_score(X, kmeans.labels_)`; print and plot elbow + silhouette. Then `optimal_k = 3`; `kmeans_optimal = KMeans(n_clusters=3, random_state=42)`; `cluster_labels = kmeans_optimal.fit_predict(X)`.
- **cell-12 (Step 4 PCA):** `from sklearn.decomposition import PCA`; `pca = PCA(n_components=2, random_state=42)`; `X_pca = pca.fit_transform(X)`; `sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=cluster_labels, palette='Set1', s=50)`.
- **cell-14 (Step 5 interpret):** `df['cluster'] = cluster_labels`; per cluster `display(df[df['cluster']==c][numeric_cols].describe())` and `df[df['cluster']==c]['y'].value_counts()` to see subscription rates per segment.
- **Methods/classes:** `sklearn.preprocessing.StandardScaler`; `pandas.get_dummies`; `sklearn.cluster.KMeans`; `sklearn.metrics.silhouette_score`; `sklearn.decomposition.PCA`.
- **Hyperparameters (exact):** `StandardScaler()` defaults; `get_dummies(drop_first=True)`; `KMeans(n_clusters=k, random_state=42)` for k in 2..10 (all other params default: `n_init`/`init` default, `max_iter` default); chosen `optimal_k = 3`; `PCA(n_components=2, random_state=42)`. No numeric outputs (shapes, WCSS, silhouette) are saved in this export (cells show no output).

## Common pitfalls (inferred from the solution / data)
- The CSV uses `;` separator and `'unknown'` for missing — forgetting either breaks loading or leaves missing values uncleaned.
- `dropna()` after replacing `'unknown'` can drop a large fraction of rows (many columns use 'unknown'); silently shrinks the dataset.
- `duration` is included in clustering but the data description warns it leaks the outcome (only known after the call) — should arguably be excluded for honest segmentation.
- One-hot encoding the categoricals massively widens X (the encoded shape >> 20 columns); K-Means on high-dimensional one-hot + scaled numeric mixes very different feature geometries (Euclidean distance on 0/1 dummies vs standardized continuous).
- K-Means is unsupervised, so `y` must be dropped before clustering and only re-attached for interpretation (done via `target`/`df['cluster']`).
- `KMeans(random_state=42)` with default `n_init` — results depend on initialization; the elbow at k=3 is a judgment call, not automatic.

## What this exercise teaches that the others don't
- The only **unsupervised learning** exercise: K-Means clustering, cluster-count selection via **elbow (WCSS/inertia) + silhouette score**, and **PCA for visualization** of high-dimensional clusters. Only exercise combining one-hot encoding of many categoricals with scaled numerics for distance-based modeling, and the only one interpreting clusters against a held-aside label.

## Method page(s) it links to
- k-means (primary), PCA (dimensionality reduction for viz). Touches preprocessing/standardization and one-hot encoding.

## Notes / [VERIFY] flags
- [VERIFY: dataset path] Solution loads `../../data/bank_marketing_data_set/bank-additional/bank-additional-full.csv`; the local `data\` folder does NOT contain this file (only Titanic and house-prices present locally). Path/file must be supplied for the build.
- No cell outputs (shapes, WCSS, silhouette, chosen-cluster stats) are saved in the VHL export, so exact numbers can't be cited.
- Stub cell-2 is empty.
- `display(...)` (cell-14) is an IPython built-in (Jupyter), used without explicit import.
