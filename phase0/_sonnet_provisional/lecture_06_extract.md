# Lecture 6 — Unsupervised Learning

**Source:** `lecture6 (2).pdf` (19 PDF pages = 19 logical slides; cite `n/19`). Date: February 11, 2026.

## Topic
Unsupervised learning overview; dimensionality reduction and Principal Component Analysis (PCA); K-Means clustering (objective, algorithm, practical considerations); combining PCA and K-Means.

## Key concepts taught — Unsupervised Learning Overview

- **Definition (slide 3/19):** Unsupervised learning is a ML paradigm in which the model learns patterns from data without labeled responses. Focus is on discovering inherent structures within the data. No pre-assigned labels needed; goal is to uncover hidden patterns, clusters, or associations; widely used for exploratory data analysis.
- **Key concept types (slide 4/19):**
  - **Dimensionality Reduction:** reducing number of variables while preserving essential information (e.g., PCA, t-SNE).
  - **Clustering:** grouping data points based on similarity (e.g., K-Means, Hierarchical clustering).
  - **Density Estimation:** modeling distribution of data to identify high/low concentration areas.
  - **Anomaly Detection:** identifying rare observations that differ significantly from the majority.
- **Comparison with supervised learning (slide 5/19):** unsupervised uses unlabeled data, focuses on pattern discovery (clustering, dim reduction, anomaly detection); supervised uses labeled data, focuses on prediction and classification (regression, classification, forecasting).

## Key concepts taught — Dimensionality Reduction / PCA

- **Motivation (slide 6/19):** reducing number of variables/features. Reason: "Curse of Dimensionality" (overfitting, computation, visualization). Benefits: simpler models, faster computation, visualization.
- **Slide 7/19:** diagram of dimensionality reduction (image/visual only, no additional formulas).
- **PCA concept (slide 9/19):** linear dimensionality reduction technique. Finds new axes (principal components) that maximize variance. Can be thought of as decomposition keeping most important information. Components are orthogonal (each captures remaining variance). Unsupervised: ignores class labels.
- **PCA applications (slide 10/19):**
  - Data visualization: reduces to 2D/3D, identifies clusters/trends/outliers.
  - Preprocessing: reduces features, mitigates curse of dimensionality, filters out noise (low-variance components).
  - Compression: image compression via Eigenfaces representation; reduces file size while maintaining essential characteristics.
- **PCA limitations (slide 11/19):**
  - Linearity: limited in capturing non-linear data.
  - Feature scaling: standardization is crucial for unbiased results.
  - Interpretability: difficult with many features.
  - Information loss: use explained variance to determine acceptable loss level.
  - Outliers: can significantly alter results.
- Note: No explicit mathematical formula for PCA (eigendecomposition of covariance matrix, or SVD) is shown in this slide deck. [VERIFY: whether PCA math was covered in a companion notebook or exercise.]

## Key concepts taught — K-Means Clustering

- **Concept (slide 13/19):** partition data into k clusters such that within-cluster points are as similar as possible, between-cluster points as dissimilar as possible.
  - **Centroid:** each cluster represented by its mean; minimizes squared distance to all assigned points.
  - **Assignment rule:** each point assigned to the cluster with the nearest centroid; distance = Euclidean.
  - **Hard clustering:** each point assigned exclusively to one cluster (no soft membership or overlapping).
- **Objective — WCSS (slide 14/19):** minimize the Within-Cluster Sum of Squares:
  - `min_{S_1,...,S_k} Σ_{i=1}^{k} Σ_{x ∈ S_i} ‖x − µ_i‖²`
  - where `µ_i` is the centroid (mean) of cluster `S_i`.
  - **Centroid formula:** `µ_i = (1 / |S_i|) Σ_{x ∈ S_i} x`
- **Practical considerations (slide 15/19):**
  - **Initialization sensitivity:** choice of initial centroids greatly affects convergence and final clusters. Use **k-means++** to select well-spaced initial centroids. Run multiple times with different seeds; choose best by objective.
  - **Choosing k:** use **Elbow Method** (plot WCSS vs. k, find point of diminishing returns); apply **Silhouette Analysis** (measure how similar an object is to its own cluster vs. others).
  - **Outliers:** can distort centroid positions and degrade quality. Consider outlier detection or robust clustering as preprocessing.
- **Applications (slide 16/19):** market segmentation, document clustering, image segmentation/compression, biology and medicine, anomaly detection (points far from centroids), general exploratory analysis.
- **Limitations (slide 17/19):** requires prior knowledge of k; assumes spherical clusters of same size; sensitive to initialization; outliers alter cluster centers.

## Key concepts taught — Combining PCA and K-Means

- **PCA as preprocessing for K-Means (slide 18/19):** benefits: reduces dimensionality and noise, speeds up K-Means, can improve cluster quality, facilitates visualization. Example: image clustering. Caution: don't reduce dimensions too much.
- **Conclusion summary (slide 19/19):** PCA = dimensionality reduction, maximizes variance. K-Means = clustering, minimizes within-cluster variance. Powerful combination: PCA + K-Means.

## Notation

- `k` = number of clusters in K-Means (also = number of folds in CV from Lecture 5 — **symbol collision**, note in notation_table).
- `S_i` = the i-th cluster (set of assigned points); `µ_i` = centroid of cluster `S_i`; `|S_i|` = number of points in cluster.
- `‖·‖²` = squared Euclidean distance (L2 norm squared).
- WCSS = Within-Cluster Sum of Squares.
- No PCA-specific notation (e.g., principal component directions/loadings) introduced in this deck.

## R9 cross-check flags (vs ESL/ISL)

- K-Means WCSS objective: matches ISL §12.4.1 and ESL §14.3.6. [VERIFY: ESL uses a slightly different form with factor 1/|C_k| — confirm whether slide's form is equivalent.]
- k-means++ initialization: mentioned but not derived. [VERIFY: not in ESL/ISL; standard reference is Arthur & Vassilvitskii (2007).]
- PCA: no covariance eigendecomposition formula shown — [VERIFY: whether slides in a later lecture or companion notebook derive the math (V^T Σ V = Λ, etc.).]
- Elbow method and Silhouette analysis mentioned without formulas. [VERIFY: Silhouette coefficient formula `s(i) = (b(i) − a(i)) / max(a(i), b(i))` — confirm whether given in exercises.]

## Professor emphasis cues

- Both PCA and K-Means presented with explicit "limitations" slides — signals exam will test knowing when NOT to use each method.
- k-means++ and the Elbow Method flagged as practical tools for choosing k — likely in exam scenarios.
- The combination PCA + K-Means highlighted in both slide 18/19 and the conclusion (slide 19/19) — treat as key workflow.
- Slide 2/19 (outline) omits Density Estimation and Anomaly Detection from the main agenda, suggesting those from slide 4/19 are background taxonomy only, not deep coverage.

## Companion materials

No specific notebook filenames named on slides. Slide 7/19 references an external image via "Source: Link" (URL not captured in text extraction).

## Cross-refs

→ `methods/pca.qmd`, `methods/kmeans.qmd`, `methods/unsupervised_overview.qmd`, `methods/lda.qmd` (PCA comparison from Lecture 3), `methods/regularization.qmd` (curse of dimensionality context).
