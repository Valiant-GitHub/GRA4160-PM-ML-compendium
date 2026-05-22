# Lecture 6 — Unsupervised Learning (PCA & K-Means)

**Source:** `lecture6 (2).pdf` (19 PDF pages = 19 logical slides; cite `n/19`). Date: February 11th, 2026. Instructor: Vegard Høghaug Larsen.

## Topic
Unsupervised learning: dimensionality reduction, Principal Component Analysis (PCA), and K-Means clustering.

## Key concepts taught
- **Outline (slide 2/19):** What is Unsupervised Learning?; Dimensionality Reduction; PCA; K-Means Clustering.
- **What is Unsupervised Learning (slide 3/19):** ML paradigm where the model learns patterns from data that does NOT have labeled responses; focus is discovering inherent structures rather than predicting an outcome. No pre-assigned labels needed; goal is to uncover hidden patterns, clusters, or associations; widely used for exploratory data analysis.
- **Key Concepts (slide 4/19):** Dimensionality Reduction (reduce #variables while preserving essential info; e.g. PCA, t-SNE); Clustering (group points by similarity; e.g. K-Means, Hierarchical); Density Estimation (model data distribution to find high/low concentration); Anomaly Detection (identify rare items differing significantly from majority).
- **Comparison with Supervised Learning (slide 5/19):** Unsupervised = unlabeled data, pattern discovery, clustering/dim-reduction/anomaly detection. Supervised = labeled data, prediction & classification, regression/classification/forecasting.
- **Intro to Dimensionality Reduction (slide 6/19):** What = reducing number of variables/features. Why = the "Curse of Dimensionality" (overfitting, computation, visualization). Benefits = simpler models, faster computation, visualization.
- **(slide 7/19):** Dimensionality-reduction figure ("Source: Link"); no formula.
- **PCA: Concept (slide 9/19):** linear dimensionality reduction technique; finds new axes (principal components) that maximize variance; a decomposition keeping the most important information; orthogonal components (each captures remaining variance); unsupervised (ignores class labels).
- **PCA: Applications (slide 10/19):** data visualization (2D/3D projections — identify clusters, trends, outliers); preprocessing (feature reduction & noise filtering — mitigates curse of dimensionality, filters noise by discarding low-variance components); compression (image compression, Eigenfaces — represents facial images using a reduced set of features).
- **Limitations of PCA (slide 11/19):** Linearity (limited for non-linear data); Feature Scaling (standardization crucial for unbiased results); Interpretability (hard with many features); Information Loss (use explained variance to decide acceptable loss); Outliers (can significantly alter results).
- **K-Means: Concept (slide 13/19):** goal = partition data into k clusters so points within a cluster are as similar as possible and across clusters as dissimilar as possible. Centroids = mean of points assigned to a cluster; the centroid is the point in feature space minimizing squared distance to all points in its cluster. Assignment Rule: each point assigned to nearest centroid; distance typically Euclidean. Hard Clustering: each point assigned exclusively to one cluster (no soft membership).
- **K-Means: Objective (slide 14/19):** minimize Within-Cluster Sum of Squares (WCSS). Partition data into k clusters `{S_1, S_2, ..., S_k}` minimizing:
  - `min_{S_1,...,S_k}  Σ_{i=1}^{k} Σ_{x ∈ S_i} ‖x − μ_i‖²`
  - where `μ_i` is the centroid (mean) of cluster `S_i`.
  - Centroid definition: `μ_i = (1 / |S_i|) Σ_{x ∈ S_i} x`. Represents the "center" of the cluster, reference point for assignment.
- **K-Means: Practical Considerations (slide 15/19):** Initialization sensitivity — initial centroids greatly affect convergence/final clusters; use **k-means++** for well-spaced init; run multiple times with different random seeds and pick best by objective. Choosing k — use the **Elbow Method** (plot WCSS vs k, find diminishing returns) and **Silhouette Analysis** (measure how similar an object is to its own cluster vs others). Outliers can distort centroids; consider outlier detection or robust clustering.
- **K-Means: Applications (slide 16/19):** market segmentation; document clustering; image segmentation/compression; biology and medicine; anomaly detection (points far from centroids); general exploratory analysis.
- **Limitations of K-Means (slide 17/19):** choosing k (requires prior knowledge of structure); cluster shape/size (assumes spherical clusters of same size); sensitivity to initialization; outliers can alter cluster centers.
- **Combining PCA and K-Means (slide 18/19):** PCA as preprocessing for K-Means — reduces dimensionality and noise, speeds up K-Means, can improve cluster quality, facilitates visualization. Example: image clustering. Caution: don't reduce dimensions too much.
- **Conclusion (slide 19/19):** PCA = dimensionality reduction, maximizes variance; K-Means = clustering, minimizes within-cluster variance. Use PCA for high-dimensional data, visualization, noise reduction; use K-Means for finding groups, exploratory analysis. Powerful combination: PCA + K-Means.

## Notation
- `k` = number of clusters in K-Means. **Collision warning:** in Lecture 5 `k` was #parameters / #folds; here `k` = #clusters. Add to notation_table.
- `S_i` = the i-th cluster (set of points); `|S_i|` = number of points in cluster i.
- `μ_i` = centroid (mean vector) of cluster `S_i`.
- `‖x − μ_i‖²` = squared Euclidean distance from point x to centroid μ_i. WCSS = sum of these over all clusters.
- `x` = data point (feature vector); PCA "principal components" = orthogonal variance-maximizing axes (no symbol assigned on slides).

## R9 cross-check flags (vs ESL/ISL)
- K-Means WCSS objective `Σ_i Σ_{x∈S_i} ‖x − μ_i‖²` and centroid `μ_i = (1/|S_i|) Σ x`: cross-check against ISL §12.4.1 (ISL writes within-cluster variation `W(C_k) = (1/|C_k|) Σ_{i,i'∈C_k} Σ_j (x_ij − x_i'j)²`, the pairwise form). **Flag: slide uses the centroid/squared-distance form; ISL gives the equivalent pairwise form — note they are equal up to a factor of 2|C_k|.**
- PCA "maximize variance / orthogonal components": cross-check against ISL §12.2 / ESL §14.5. No explicit PCA eigenvalue/SVD formula on these slides — **flag: PCA math (covariance eigen-decomposition, explained-variance ratio) NOT given on slides; supply from ISL §12.2 if needed and mark as supplemental.**

## Professor emphasis cues
- Standardization for PCA stressed as "crucial for unbiased results" (limitation slide).
- Outliers flagged as a problem for BOTH PCA and K-Means (appears on PCA limitations slide 11 and K-Means slides 15 & 17).
- k-means++ and multiple random restarts explicitly recommended.
- Elbow Method + Silhouette Analysis given as the two tools for choosing k.
- "Don't reduce dimensions too much" caution when combining PCA + K-Means.

## Companion materials
No lecture-notebook or exercise filenames printed on these slides. Slide 7 and slide 11 reference external figure sources ("Source: Link") without notebook names.

## Cross-refs
→ `methods/pca.qmd` (PCA concept/applications/limitations), `methods/kmeans.qmd` (WCSS, centroids, k-means++, elbow, silhouette), `methods/unsupervised_overview.qmd`. PCA cross-links to `methods/lda.qmd` (Lecture 3 contrast: PCA unsupervised vs LDA supervised). PCA-as-preprocessing feeds the PCA+K-Means combined-pipeline page.
