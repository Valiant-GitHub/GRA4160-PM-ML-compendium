# Lecture 2 — Machine Learning Basics and Supervised Learning

**Source:** `course_materials/Lecture slides/lecture2 (2).pdf` (32 PDF pages = 10 logical slides; progressive-build, cite `n/10`). Date: Jan 14 2026.

## Topic
ML paradigms, bias-variance, linear regression (OLS), kNN, parametric vs non-parametric, Naive Bayes.

## Key concepts taught
- **Three ML types (slide 3/10):** Supervised (labeled; e.g. digit recognition), Unsupervised (unlabeled; e.g. customer segmentation), Reinforcement (reward-based; robot navigation).
- **Supervised learning (slide 4/10):** input-output pair training. Key algorithms: linear regression (continuous), kNN (similarity), decision trees, neural networks.
- **Bias-Variance trade-off (slide 5/10):** Bias = systematic deviation of predictions from true values. Variance = sensitivity to fluctuations in training set. High bias → underfit (oversimplified, consistent but inaccurate). High variance → overfit (well on train, poorly on new). Goal: balance minimizing overall error.
- **Linear Regression (slide 7/10):** predicts continuous values; useful for **inference** (how features influence outcome); parametric, fixed parameters (weights + intercept); minimizes **MSE**; **OLS** finds best-fit line; under **Gauss–Markov assumptions** (linearity, no perfect multicollinearity, homoscedastic errors) OLS is **BLUE** (best linear unbiased estimator). Caveat: linear form may not capture real-world complexity → diagnostics crucial.
- **kNN (slide 8/10):** classification (label) + regression (value). Core: find k closest points, predict from them. k too small → overfit (high variance); too large → underfit (high bias). Distance metric (e.g. **Euclidean**) matters. Often **non-parametric** (no distributional assumption). Computationally expensive for large data; good baseline. Choose k via **cross-validation**.
- **Parametric vs Non-parametric (slide 9/10):** Parametric (linear reg): assume functional form, fixed #params, fast to train/interpret, misleading if form wrong. Non-parametric (kNN): fewer assumptions, #params grows with data, flexible but expensive, can overfit without tuning.
- **Naive Bayes (slide 10/10):** Bayes' theorem `P(A|B) = P(B|A)·P(A) / P(B)`. Used for spam/text classification. "Naive" = features conditionally independent given the class. Strong assumption often untrue but performs surprisingly well; fast, easy, good first model.

## Notation
- MSE (mean squared error) named as the OLS objective (formula not on slide — see notebook 02 / math appendix).
- Bayes: `P(A|B) = P(B|A) × P(A) / P(B)` (slide 10/10).

## Professor emphasis cues
- OLS framed dually as **prediction + inference** (BI is a business school — inference/interpretation matters).
- kNN & Naive Bayes positioned as **baselines / first models**.
- Bias-variance is "a core concept in ML" — recurs L1, L2, L3, dedicated notebook 09.

## Companion materials
Lecture notebooks `02_OLS`, `03_Supervised_learning_with_kNN`; exercise `02_Spam_filtering_with_naive_bayes`.

## Cross-refs
OLS → `methods/ols.qmd`; kNN → `methods/knn.qmd`; Naive Bayes → drill `02_Spam_filtering` (no dedicated method page — exercise-only topic); bias-variance → `methods/bias_variance.qmd`.
