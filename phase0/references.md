# Phase 0.5 — Reference Texts

Both Tier-2 reference texts are present at the working-directory root.

| Text | File | Role |
|---|---|---|
| **ISL** — *An Introduction to Statistical Learning* (Python ed., James, Witten, Hastie, Tibshirani, 2021/2023) | `An Introduction to Statistical Learning Python.pdf` (~19.1 MB) | **Primary for Mode A** (course-level depth). The course lecturer explicitly cites ISL 2021 on Lecture 9 slide 9 for the activation-function figure, and the single-hidden-layer model matches **ISL Eq. 10.1**. ISL is the "gentle introduction" named on Lecture 1 slide 3. |
| **ESL** — *The Elements of Statistical Learning* (2nd ed., Hastie, Tibshirani, Friedman, 2009) | `The Elements of Statistical Learning 2.E..pdf` (~12.7 MB) | **Primary for Mode B** (full rigor) and the formal notation anchor (GR1). Named as the **primary textbook** on Lecture 1 slide 3. Lecture 3 slide 16 attributes the Ridge/Lasso geometry figure to "Hastie, Tibshirani & Friedman (2009)" = ESL. |

**Both present** → per the prompt's rule: ISL primary for Mode A, ESL primary for Mode B. Neither read end-to-end in Phase 0 (per 0.5); they are queried per-method during Phase 3.

## Section pointers gathered during Phase 0 (to query in Phase 3)
From the lecture extracts' R9 cross-check flags — the ESL/ISL sections to consult per method:
- OLS / Gauss–Markov / BLUE → ESL §3.2, ISL §3.1–3.2
- Ridge/Lasso/Elastic Net → ESL §3.4, ISL §6.2 (geometry: ESL Fig 3.11)
- LDA (Fisher criterion, `w ∝ S_W⁻¹(m₁−m₂)`, Gaussian derivation) → ESL §4.3, ISL §4.4
- Logistic regression (logit, MLE, gradient) → ESL §4.4, ISL §4.3
- Decision trees (Gini, entropy/deviance, info gain) → ESL §9.2, ISL §8.1 (note ISL uses natural-log cross-entropy; course slide uses log₂)
- Model assessment, AIC/BIC, bias-variance (incl. irreducible error σ²) → ESL §7.2–7.7, ISL §2.2/§6.1
- Cross-validation (k-fold, LOOCV) → ESL §7.10, ISL §5.1
- Bagging, OOB, 0.632 bootstrap → ESL §7.11, §8.7
- Random forests (`m≈√p` class, `p/3` regr.) → ESL §15, ISL §8.2
- Boosting / AdaBoost (`α_m = ½ln((1−err)/err)`) / gradient boosting → ESL §10
- PCA (covariance eigendecomposition / SVD, explained variance) → ESL §14.5, ISL §12.2
- K-means (WCSS; ISL pairwise form §12.4.1) → ESL §14.3, ISL §12.4
- Neural nets / backprop (δ-recursion, weight decay) → ESL §11.3–11.5, ISL §10.1–10.7
- Naive Bayes (exercise topic, no method page) → ESL §6.6.3, ISL §4.4.4

These are pointers only; exact section numbers to be confirmed when the books are queried in Phase 3.
