# Lecture 4 — Classification Analysis

**Source:** `lecture4 (2).pdf` (24 PDF pages = 11 logical slides; cite `n/11`). Date: January 28, 2026.

## Topic
Classification methods: Logistic Regression (intro, sigmoid function, log-odds derivation); Decision Trees (impurity measures, training algorithm). Slides 3–5 are recap of LDA and regularized regression from Lecture 3.

## Key concepts taught — Logistic Regression

- **Overview (slide 6/11):** Binary classification model (yes/no, success/failure). Assumes a linear relationship between predictors x and the log-odds of the outcome.
- **Log-odds (logit) model (slide 6/11):**
  - `log(p / (1−p)) = β_0 + β_1 x_1 + ··· + β_k x_k`
  - The logit transformation ensures predicted probabilities stay in [0, 1].
  - Fit by **maximum likelihood** (minimizing negative log-likelihood).
  - Can be extended to multi-class via **One-vs-All** or **Multinomial/Softmax**.
- **Sigmoid function (slide 7/11):** Maps ℝ → [0, 1].
  - `f(z) = 1 / (1 + e^{−z})`
  - `p = e^z / (1 + e^z)` (equivalent form shown on slide)
  - As z → +∞, f(z) → 1; as z → −∞, f(z) → 0.
- **Log-odds / probability / sigmoid chain (slide 8/11):**
  - `log(p / (1−p)) = z = β_0 + β_1 x_1 + ··· + β_k x_k`
  - `p / (1−p) = e^z  ⟹  p = e^z / (1 + e^z) = 1 / (1 + e^{−z})` (the sigmoid)
  - **Odds interpretation:** changing z by 1 unit multiplies the odds `p/(1−p)` by `e^1 ≈ 2.718`.

## Key concepts taught — Decision Trees

- **Overview (slide 9/11):** Non-parametric method for classification and regression. Data repeatedly split into subsets based on feature thresholds; each split chosen to maximize purity (minimize impurity). Advantages: interpretability, handles mixed feature types, no feature scaling needed. Drawbacks: overfitting if unpruned, instability (high variance) with small data.
- **Training algorithm — recursive construction (slide 10/11):**
  1. Choose a feature + threshold that best splits the data (based on impurity).
  2. Split data into two (or more) subsets.
  3. Repeat on each subset until stopping criterion met.
  - **Stopping criteria:** maximum depth reached; fewer samples than minimum required; impurity improvement below threshold.
- **Impurity measures (slide 11/11):**
  - **Gini Index (G):**
    - `G = Σ_{i=1}^{C} p_i (1 − p_i) = 1 − Σ_{i=1}^{C} p_i²`
    - Measures chance of misclassification if label picked at random. G = 0 → perfectly pure.
  - **Entropy (H):**
    - `H = −Σ_{i=1}^{C} p_i log_2(p_i)`
    - Based on information-theoretic uncertainty. H = 0 → perfectly pure subset.
  - **Information Gain:** reduction in impurity from a split (introduced slide 11/11 as concept, formula not expanded).

## Key concepts taught — Recap slides (from Lecture 3)

- **LDA recap (slides 3/11 and 4/11):** objective `max_w (w^T S_B w) / (w^T S_W w)`; assumes Gaussian classes with equal covariance; practical tips: standardize, use CV with confusion matrices; if covariances differ → QDA.
- **Regularized regression recap (slide 5/11):** Ridge (L2), Lasso (L1), Elastic Net (L1+L2); two hyperparameters for Elastic Net: overall strength λ and mixing ratio α; choose λ by CV; standardize before penalizing.

## Notation

- `p` = predicted probability; `C` = number of classes; `p_i` = proportion of class i in a node (Gini/Entropy); `z` = linear predictor (net input); `β_0, β_1, …, β_k` = logistic regression coefficients.
- **Symbol collision:** `p` is used both as "number of predictors" (Lecture 3, LDA) and as "predicted probability" in logistic regression on this lecture. Note in notation_table.
- `G` = Gini index; `H` = entropy; `C` = number of classes in impurity formulas.

## R9 cross-check flags (vs ESL/ISL)

- Logistic sigmoid and log-odds derivation: matches ISL §4.3 and ESL §4.4. The step-by-step chain `log-odds → odds → sigmoid` shown explicitly — confirm presentation matches ISL Figure 4.2.
- Gini index formula `G = 1 − Σ p_i²`: matches ISL §8.1 CART criterion. [VERIFY: ISL uses Gini for splits differently from class purity formula — confirm both forms are equivalent.]
- Entropy formula: matches ESL §9.2. [VERIFY: some sources define entropy using natural log `ln` rather than `log_2`; slide uses `log_2` explicitly.]

## Professor emphasis cues

- The progressive-build structure on slide 6/11 (four bullets building up the logistic regression definition) signals this is a key conceptual structure, not a throwaway slide.
- The full algebraic chain on slide 8/11 (log-odds → odds → sigmoid) is presented step-by-step — treat as exam derivation target.
- The odds interpretation (`+1 unit in z → ×e ≈ 2.718 in odds`) called out explicitly — likely tested.
- Decision trees: advantages vs drawbacks stated as a balanced comparison — feeds the "when to use which" decision framework.
- Impurity measures presented with clean formulas on the final slide — both Gini and Entropy given; Information Gain defined as concept.

## Companion materials

Exercise: `Recognizing handwritten digits` (named in Plan for Today, slide 2/11). No separate notebook filenames given on slides.

## Cross-refs

→ `methods/logistic_regression.qmd`, `methods/decision_trees.qmd`, `methods/lda.qmd` (recap), `methods/regularization.qmd` (recap), `methods/ensemble.qmd` (trees are base learners in Lecture 7).
