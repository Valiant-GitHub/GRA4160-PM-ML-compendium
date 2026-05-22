# Lecture 4 — Classification Analysis

**Source:** `lecture4 (2).pdf` (24 PDF pages = 11 logical slides; cite `n/11`). Date: January 28, 2026. Instructor: Vegard Høghaug Larsen.

## Topic
Classification analysis: Logistic Regression and Decision Trees (with recap of LDA and Regularized Regression from Lecture 3).

## Key concepts taught
- **Plan / scope (slide 2/11):** Recap LDA + Regularized Regression; Logistic Regression; Decision Trees; Exercise: Recognizing handwritten digits.
- **Recap LDA (slide 3/11):** Purpose = supervised classification & dimensionality reduction; projects data onto lower-dim space to maximize class separation. When to use: 2+ well-separated classes with similar covariance matrices; need for interpretable linear boundaries + reduced dimensionality. Scatter matrices `S_B` (between-class), `S_W` (within-class). Objective: `max_w (wᵀ S_B w) / (wᵀ S_W w)`.
- **Recap LDA assumptions (slide 4/11):** 1. each class approximately normal (Gaussian); 2. all classes share the same covariance matrix; 3. features ideally uncorrelated, mild correlations acceptable. If covariances differ significantly → QDA. Robust to mild normality violations. Tips: scale/standardize; use CV (confusion matrices).
- **Recap Regularized Regression (slide 5/11):** Ridge (L2) shrinks coefficients rarely to exactly zero, handles multicollinearity; Lasso (L1) can set coefficients to zero (feature selection), good high-dim; Elastic Net (L1 + L2) balances Ridge and Lasso, two hyperparameters: overall strength `λ` + mixing ratio `α`. Choose `λ` via CV; standardize before penalty.
- **Logistic Regression — definition (slide 6/11):** binary classification model (e.g. yes/no, success/failure). Assumes linear relationship between predictors x and the **log-odds** of the outcome:
  - `log( p / (1−p) ) = β_0 + β_1 x_1 + ··· + β_k x_k`
  - The log-odds (logit) transformation ensures predicted probabilities are always in [0, 1]. Often fit by maximum likelihood (minimizing negative log-likelihood). Extends to multi-class via One-vs-All or Multinomial/Softmax.
- **Logistic (Sigmoid) Function (slide 7/11):** maps ℝ to [0, 1]:
  - `f(z) = 1 / (1 + e^{−z})`
  - As z → ∞, f(z) → 1; as z → −∞, f(z) → 0. Relates to log-odds: `p = 1 / (1 + e^{−z}) = e^z / (1 + e^z)`.
- **Relation log-odds / probability / sigmoid (slide 8/11):**
  - `log( p / (1−p) ) = z = β_0 + β_1 x_1 + ··· + β_k x_k`
  - `p / (1−p) = e^z  ⇒  p = e^z / (1 + e^z) = 1 / (1 + e^{−z})` (the sigmoid).
  - Interpretation: changing z by 1 unit multiplies the odds `p/(1−p)` by `e^1 ≈ 2.718`.
- **Decision Trees — definition (slide 9/11):** non-parametric method for classification and regression. Data repeatedly split into subsets based on feature thresholds, growing a tree structure. Each split chosen to maximize purity of resulting subsets (minimize impurity). Advantages: interpretability, handles mixed feature types, no feature scaling needed. Drawbacks: can overfit if not pruned, can be unstable (high variance) for small data.
- **Training a Decision Tree (slide 10/11):** constructed recursively — choose feature + threshold to best split data (based on impurity); split into two (or more) subsets; repeat until stopping criterion met. Stopping criteria: maximum depth reached; fewer samples in a node than minimum required for split; impurity improvement below threshold.
- **Impurity Measures: Gini vs. Entropy (slide 11/11):**
  - Gini Index: `G = Σ_{i=1}^{C} p_i (1 − p_i) = 1 − Σ_{i=1}^{C} p_i²`. Measures chance of misclassification if you pick a label at random. G = 0 means perfectly pure (all samples in one class).
  - Entropy: `H = − Σ_{i=1}^{C} p_i log₂(p_i)`. Based on information theory (uncertainty). H = 0 means a perfectly pure subset.
  - Information Gain: the reduction in impurity from a split.

## Notation
- `p` = probability of positive outcome (logistic). **Collision warning:** `p` here is a probability; in Lecture 7/9 `p` denotes the number of predictors/features. Flag for notation_table.
- `z` = linear predictor / net input `= β_0 + β_1 x_1 + ··· + β_k x_k` (logistic).
- `k` = number of predictors `x_1...x_k` in logistic regression slide. **Collision warning:** `k` overloaded — used here as predictor count; in Decision Tree Gini/Entropy the class index runs `i=1..C`; in later lectures `k`/`K` are hidden units, fold count, cluster count.
- `C` = number of classes (Gini/Entropy sums). `p_i` = class proportion in node for class i.
- `G` = Gini index; `H` = entropy; `f(z)` = sigmoid; `S_B`, `S_W` = scatter matrices (LDA recap); `λ`, `α` = regularization strength and Elastic Net mixing ratio (recap).

## R9 cross-check flags (vs ESL/ISL)
- Logistic regression logit form `log(p/(1−p)) = β_0 + Σ β_j x_j`: matches ISL §4.3 / ESL §4.4. Agree (uses `k` for #predictors instead of ISL's `p`).
- Sigmoid `f(z) = 1/(1+e^{−z})` and `p = e^z/(1+e^z)`: standard. Agree.
- Gini `G = Σ p_i(1−p_i) = 1 − Σ p_i²` and entropy `H = −Σ p_i log₂ p_i`: cross-check against ISL §8.1 (ISL uses natural log / cross-entropy `−Σ p̂_mk log p̂_mk`; this slide explicitly uses **log₂** for entropy). **Flag base-of-log difference (log₂ vs natural log).**

## Professor emphasis cues
- Progressive build on logistic regression slide (6/11 repeats 5 pages) accumulating: definition → log-odds linear form → [0,1] guarantee → MLE → multi-class extension. Indicates careful step-by-step framing.
- Odds interpretation stressed: "changing z by 1 unit multiplies the odds by e ≈ 2.718."
- Decision tree advantages/drawbacks given as explicit pro/con list → feeds "when to use" framing.
- Threshold "REMEMBER" style emphasis appears later (Lecture 8) not here.

## Companion materials
Exercise named on slide 2/11: **"Recognizing handwritten digits"** (exercise notebook for digits classification; exact notebook filename not given on slides). No lecture-notebook filenames printed in this deck.

## Cross-refs
→ `methods/logistic.qmd` (logistic regression core), `methods/decision_trees.qmd` (Gini/entropy/info gain), `methods/lda.qmd` (recap), `methods/regularization.qmd` (recap). Decision-tree impurity feeds `methods/random_forests.qmd` (Lecture 7). Sigmoid + cross-entropy feeds `methods/backprop_gradient_descent.qmd` (Lecture 8) and `methods/neural_networks.qmd` (Lecture 9).
