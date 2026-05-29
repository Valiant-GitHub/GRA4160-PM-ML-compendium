# Phase 0.8 — Concept Inventory (per-method build plan)

For each of the 16 method pages: which lectures cover it, which notebooks implement it
(and how — sklearn vs from-scratch), which datasets, which exam assignment references it,
and which sibling methods it compares to (feeds "When NOT to use", showdowns, decision
dashboard). This is the primary planning artifact for Phase 2.

Family colors (Dimension A): Linear `#2E86AB`, Tree `#06A77D`, Neural `#7B2CBF`,
Unsupervised `#F18F01`, Cross-cutting `#3D405B`.

---

### 1. OLS (`methods/ols.qmd`) — Linear — Tier 1
- **Lectures:** L2 (slide 7, MSE/BLUE/Gauss-Markov), L1 (workflow), L3 (recap).
- **Notebooks:** nb02 — OLS three ways: normal equation `(XᵀX)⁻¹Xᵀy` [cell 8], OOP `class OLS` via `np.linalg.solve` [cell 10], sklearn `LinearRegression` cross-check [cell 13]. **From-scratch + sklearn.**
- **Datasets:** synthetic (nb02, 100×1); **Hitters** (exam/showdown), house-prices (exercise 03).
- **Exam:** Asgmt 1 (baseline vs Ridge/Lasso on Hitters).
- **Compare to:** Ridge/Lasso (regularization), kNN (parametric vs non-param), trees (linear vs non-linear).

### 2. kNN (`methods/knn.qmd`) — Linear-adjacent/instance → label "Cross-cutting"? Keep with supervised baselines — Tier 1
- **Lectures:** L2 (slide 8), L3 (recap), L10.
- **Notebooks:** nb03 — from-scratch (euclidean + `argsort` + `bincount`, k=3) [cell 7] + sklearn `KNeighborsClassifier(n_neighbors=3)` [cell 12]. **From-scratch + sklearn.**
- **Datasets:** Iris (nb03), make_blobs (viz); digits (exercise 04), Iris (exercise 05).
- **Exam:** Asgmt 4 candidate classifier.
- **Compare to:** logistic/LDA (parametric boundaries), trees, OLS (parametric vs non-param). Curse of dimensionality (Mode B vs PCA).
- *Note: kNN has no natural family color; treat as supervised non-parametric — use Cross-cutting slate or Linear blue; decide in Phase 1 (lean Linear-family grouping for "baseline supervised").*

### 3. Logistic Regression (`methods/logistic.qmd`) — Linear — Tier 1
- **Lectures:** L4 (logit/sigmoid/odds, slides 6-8), L8 (derived from ADALINE via cross-entropy, slides 23-28).
- **Notebooks:** nb06 — sklearn `LogisticRegression(solver='lbfgs', max_iter=500)` on Titanic; nb15 — from-scratch `LogisticRegressionGD` (sigmoid+BCE, `eta=0.3, n_iter=1000`), incl. 3-class One-vs-All. **sklearn + from-scratch.**
- **Datasets:** Titanic (nb06), Iris (nb15), digits (exercise 04), Iris (exercise 05).
- **Exam:** Asgmt 4 candidate.
- **Compare to:** LDA (both linear; LDA assumes Gaussian/shared cov), kNN, ADALINE (same GD update), trees. Variations (Tier 1): class_weights, Pipeline, regularization path, CV.

### 4. LDA (`methods/lda.qmd`) — Linear — Tier 1
- **Lectures:** L3 (core, slides 6-11: scatter matrices, eigenproblem, `w∝S_W⁻¹(m₁−m₂)`, assumptions), L4 (recap).
- **Notebooks:** nb04 — sklearn `LinearDiscriminantAnalysis` (n_components 1 & 2) + from-scratch `LDAFromScratch` (eigendecomp of `S_W⁻¹S_B`). **sklearn + from-scratch.**
- **Datasets:** make_classification (nb04), Iris (nb04).
- **Exam:** — (not examined 2025 → R8: state explicitly).
- **Compare to:** logistic (L3 slide 11 explicit table), QDA, PCA (supervised vs unsupervised dim-reduction), kNN.

### 5. Adaline (`methods/adaline.qmd`) — Linear/Neural-bridge — Tier 1
- **Lectures:** L8 (slides 14-20: net input, MSE loss `(1/2n)Σ(y−z)²`, GD update).
- **Notebooks:** nb14 — from-scratch `AdalineGD` (identity activation, MSE, batch GD; eta 0.01/0.1/0.0001, n_iter 15/1000; standardized variant). **From-scratch (numpy).**
- **Datasets:** Iris (setosa vs versicolor, 2 features).
- **Exam:** —.
- **Compare to:** logistic (L8: same GD update, CE vs MSE, sigmoid vs linear), perceptron, OLS (linear activation = linear regression on labels). Bridge to NN.

### 6. Regularization — Ridge/Lasso/Elastic Net (`methods/regularization.qmd`) — Linear — Tier 1
- **Lectures:** L3 (core, slides 13-24: objectives, geometry, λ-selection), L4 (recap), L9 (L2 in NN context).
- **Notebooks:** nb05 — sklearn `Ridge(alpha=0.8)`, `Lasso(alpha=0.1)`, `ElasticNet(alpha=0.1, l1_ratio=0.5)`, `lars_path`; nb08 — `LassoLarsIC`, `LassoCV(cv=10)`, `LassoLarsCV(cv=20)`. **sklearn.**
- **Datasets:** synthetic sparse DGP (nb05, known coeffs), diabetes+noise (nb08); **Hitters** (exam), house-prices (exercise 03).
- **Exam:** Asgmt 1 (core).
- **Compare to:** OLS (λ=0 limit), each other (L1 vs L2 vs mix), PCA (dim reduction alt). **Trap callout:** sklearn `alpha`=λ ≠ Elastic-Net `α`. See disagreements [R9-12], [R9-13].

### 7. Bias-Variance (`methods/bias_variance.qmd`) — Cross-cutting — Tier 1
- **Lectures:** L1, L2 (slide 5), L3 (recap), L5 (slide 10, `Total Error=Bias²+Var`).
- **Notebooks:** nb09 — polynomial degree 1-20, `Pipeline([PolynomialFeatures, LinearRegression(fit_intercept=False)])`, `cross_val_score(cv=10)`. **sklearn.** ([NB-2]: returns R² not MSE.)
- **Datasets:** synthetic noisy sine (nb09).
- **Exam:** Asgmt 1 (interpretation).
- **Compare to:** all (universal lens). Disagreement [R9-2]: σ² irreducible error.

### 8. CV / Info Criteria (`methods/cv_info_criteria.qmd`) — Cross-cutting — Tier 1
- **Lectures:** L5 (core: AIC/BIC, k-fold, stratified, LOOCV, repeated), L3 (λ via CV).
- **Notebooks:** nb08 — `LassoLarsIC` (AIC/BIC), `LassoCV`, `LassoLarsCV`. **sklearn.**
- **Datasets:** diabetes+noise (nb08); Iris (exercise 05, `GridSearchCV`).
- **Exam:** all assignments (tuning).
- **Compare to:** info criteria vs CV (no hold-out vs hold-out). Disagreements [R9-2], [R9-3].

### 9. Decision Trees (`methods/decision_trees.qmd`) — Tree — Tier 1
- **Lectures:** L4 (Gini/entropy/info-gain, slides 9-11), L7 (recap).
- **Notebooks:** nb07_trees — sklearn `DecisionTreeClassifier` + `plot_tree`. **sklearn.** ([NB-1]: leaf=10 vs comment 5.)
- **Datasets:** Iris (nb07); digits (exercise 04), Iris (exercise 05, `GridSearchCV` on tree), WDBC (exam).
- **Exam:** Asgmt 3.
- **Compare to:** RF/ensembles (single vs many), logistic/LDA (axis-aligned vs linear boundary), kNN.

### 10. Random Forests (`methods/random_forests.qmd`) — Tree — Tier 1
- **Lectures:** L7 (core: bootstrap, √p, decorrelation, extra trees, slides 18-21).
- **Notebooks:** nb13 — from-scratch RF (bootstrap + √p subset + majority vote) + sklearn `RandomForestClassifier(n_estimators=100, max_depth=3)` + `ExtraTreesClassifier`. **From-scratch + sklearn.** ([NB-6] caveats.)
- **Datasets:** Titanic (nb13); WDBC (exam).
- **Exam:** Asgmt 3 (core).
- **Compare to:** single tree (variance reduction), gradient boosting (bagging vs boosting; exam Asgmt 3.8 asks RF-vs-GB), extra trees, bagging.

### 11. Ensembles (`methods/ensembles.qmd`) — Tree — Tier 1
- **Lectures:** L7 (core: bagging/boosting/BMA/stacking, AdaBoost/XGBoost, bootstrap 0.632).
- **Notebooks:** nb12 — `VotingClassifier(voting='soft')` (Iris), `BaggingClassifier` (breast cancer, n_estimators 100-1000); nb07_income — `GradientBoostingClassifier` setup (Adult, no fit). **sklearn.** ([NB-5] soft-vs-hard comment.)
- **Datasets:** Iris, breast cancer (nb12), Adult income (nb07_income), WDBC (exam GB).
- **Exam:** Asgmt 3 (gradient boosting).
- **Compare to:** single tree, RF, stacking vs voting vs bagging vs boosting. Disagreements [R9-6], [R9-8].

### 12. PCA (`methods/pca.qmd`) — Unsupervised — Tier 1
- **Lectures:** L6 (concept/applications/limitations, slides 9-11; no eigen math).
- **Notebooks:** nb10 — sklearn `PCA(n_components=2)`, pipelines with `StandardScaler`+`GaussianNB`. **sklearn.** ([NB-3] scatter mislabel.)
- **Datasets:** seeds (nb10), wine (nb10); **WholesaleCustomers** (exam, exercise 06).
- **Exam:** Asgmt 2 (90% variance).
- **Compare to:** LDA (unsup vs sup), K-means (PCA+KMeans pipeline), regularization (dim reduction). Disagreement [R9-5] (supply eigen/SVD from ESL/ISL).

### 13. K-means (`methods/kmeans.qmd`) — Unsupervised — Tier 1
- **Lectures:** L6 (core: WCSS, centroids, k-means++, elbow, silhouette, slides 13-18).
- **Notebooks:** nb11 — sklearn `KMeans` + elbow (`inertia_`) + `silhouette_score`. **sklearn.** ([NB-4] 2-vs-3 clusters.)
- **Datasets:** Iris (nb11); **WholesaleCustomers**, seeds (exam/showdown, exercise 06).
- **Exam:** Asgmt 2 (elbow+silhouette).
- **Compare to:** PCA (combine), hierarchical, GaussianNB. Disagreement [R9-4].

### 14. NN basics (`methods/nn_basics.qmd`) — Neural — Tier 2
- **Lectures:** L9 (architecture, forward pass, fitting, backprop, dropout/L2), L8 (backprop).
- **Notebooks:** nb17 — custom autodiff `Neuron/Layer/MLP`, `MLP(3,[3,3,1])` tanh, SSE, lr=1.0, 200 iters. **Custom autodiff (from scratch).**
- **Datasets:** toy 4-sample (nb17).
- **Exam:** —.
- **Compare to:** logistic (single neuron), Adaline, deeper nets.

### 15. Autodiff (`methods/autodiff.qmd`) — Neural — Tier 2
- **Lectures:** L8 (computational graphs, chain rule, slides 6-10, 29 demo).
- **Notebooks:** nb16 — `Value` class (micrograd-style), topological backward, GD on toy quadratic (lr=0.1, 100 iters). **Custom engine (from scratch).**
- **Datasets:** none (toy loss).
- **Exam:** —.
- **Compare to:** PyTorch autograd (nb18/19), manual backprop equations (L8/L9).

### 16. NN with PyTorch (`methods/nn_pytorch.qmd`) — Neural — Tier 2
- **Lectures:** L9.
- **Notebooks:** nb18 — low-level PyTorch tensors (4→8→3, ReLU via clamp, manual softmax+CE, manual SGD, lr=0.01, 500 epochs). **PyTorch (low-level).**
- **Datasets:** Iris (nb18).
- **Exam:** —.
- **Compare to:** from-scratch autodiff (nb16/17), high-level `nn.Module` (nb19).

### 17. Build a NN (`methods/build_a_nn.qmd`) — Neural — Tier 2
- **Lectures:** L9 (dropout/L2).
- **Notebooks:** nb19 — high-level PyTorch `nn.Module` (784→128→64→10, ReLU, CrossEntropyLoss, SGD lr=0.1, 5 epochs, batch 32) + manual L2 (λ=1e-4) + manual dropout (p=0.5). Metrics: deep 97.50%, shallow 92.26%, L2 96.95%, dropout 97.13%. **PyTorch (high-level).**
- **Datasets:** MNIST (nb19).
- **Exam:** —.
- **Compare to:** low-level PyTorch (nb18), regularization (L2/dropout). Disagreement [R9-9].

---

## Drill-only topics (no method page; Phase 5)
- **Naive Bayes** — L2/L3 + exercise 02 (`MultinomialNB` + from-scratch). Dataset [DATA-1] missing.
- **Data preprocessing** — L1 + nb01 + exercise 01 (Titanic). Feeds `big_picture/reading_the_data.qmd`.
- **Customer segmentation** — exercise 06 (KMeans+PCA on bank data [DATA-2] missing).

## Common showdown datasets (R5)
- 2D classification boundary: Titanic-derived 2-feature OR Iris 2-feature (logistic/LDA/kNN/tree/RF).
- Regression: **Hitters** (OLS/Ridge/Lasso/Tree/RF).
- Clustering w/ ground truth: **seeds** (variety label) and Iris.
- Unsupervised on real data: **WholesaleCustomers**.
