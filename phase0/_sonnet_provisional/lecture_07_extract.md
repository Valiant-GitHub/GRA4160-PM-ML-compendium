# Lecture 7 — Ensemble Methods

**Source:** `lecture7_260521_184935.pdf` (25 PDF pages = 21 logical slides; cite `n/21`). Date: February 18, 2026.

## Topic
Ensemble methods taxonomy (Bagging, Boosting, BMA, Stacking); Bootstrap theory; Bagging mechanism; Boosting (AdaBoost, XGBoost); Decision tree recap; Random Forests (feature randomness, decorrelation); Extra Trees.

## Key concepts taught — Ensemble Methods Overview

- **Definition (slide 3/21):** family of techniques involving training multiple models and combining predictions to form a more accurate and robust model. Can be used with linear models, decision trees, or neural networks. Combines models to typically reduce variance and boost performance. Trade-off: more powerful but less interpretable than single models.
- **Types of ensemble methods (slide 4/21, built progressively across pages 4–7):**
  - **Bagging (Bootstrap Aggregating):** examples: Random Forests, Extra Trees. Reduces variance by training multiple models on bootstrap samples and averaging predictions.
  - **Boosting (Sequential Learning):** examples: Gradient Boosting, AdaBoost, XGBoost. Trains models sequentially, focusing on misclassified examples to reduce bias.
  - **Bayesian Model Averaging (BMA):** trains multiple models, assigns weights based on posterior probabilities. Can dynamically update model weights as new data becomes available. Useful when model uncertainty must be explicitly accounted for.
  - **Stacking (Stacked Generalization):** combines multiple models by training a meta-model on their outputs. Often uses diverse base models (decision trees, SVMs, neural networks). Example: a neural network trained on outputs of logistic regression, SVM, and random forest.

## Key concepts taught — Bagging

- **Definition (slide 6/21):** Bootstrap AGGregatING. Train multiple instances of a base model on bootstrap samples; combine via majority vote (classification) or averaging (regression). Subsets created by sampling with replacement. Effective at reducing variance and mitigating overfitting for high-variance base learners (e.g., large decision trees).
- **Bootstrap in ML (slide 7/21):** resampling technique to approximate the sampling distribution of a statistic. Repeatedly sample with replacement to create new datasets of same size as original. **About 63% of unique samples from the original dataset appear in each bootstrap sample.** Used to train multiple models (bagging) or provide performance estimates (OOB error). Especially useful for small datasets or when data is expensive to collect.
- **Slide 8/21:** diagram of bootstrap sampling (visual only, no additional formulas).
- **Bagging procedure (slide 9/21):**
  1. Create multiple bootstrap samples of the training data.
  2. Train a base model (e.g., a decision tree) on each bootstrap sample.
  3. Use each trained model to make predictions.
  4. Combine predictions (majority vote for classification; averaging for regression).
  5. Final prediction has lower variance and improved generalization.

## Key concepts taught — Boosting

- **Definition (slide 11/21):** combines multiple weak learners sequentially. Each subsequent model focuses more on samples that previous models misclassified (higher weight). Typically reduces bias but can overfit if not regularized. Often uses simpler base learners (shallow trees) incrementally improved.
- **Boosting procedure (slide 12/21):**
  1. Initialize weights of each training example equally.
  2. Train a base model on training data using current weights.
  3. Evaluate performance on training data.
  4. Increase weights of misclassified examples; decrease weights of correctly classified.
  5. Repeat until stopping criterion (e.g., max iterations or minimal improvement).
  6. Final prediction is a **weighted combination** of all models' predictions.
  7. **Learning rate (η)** often controls contribution of each weak learner.
- **AdaBoost and XGBoost (slide 13/21):**
  - **AdaBoost:** Adaptive Boosting — one of the earliest boosting algorithms.
  - **XGBoost:** eXtreme Gradient Boosting — popular gradient boosting framework.
  - Both typically use decision trees as base learners; differ in objective and optimization strategies. Widely used for image classification, ranking, and structured data.
- **XGBoost features (slide 14/21):**
  1. Regularization: includes L1 and L2 terms to prevent overfitting.
  2. Tree pruning: prunes during construction to improve generalization.
  3. Handling missing values: built-in mechanism.
  4. Cross-validation: integrated for hyperparameter tuning.
  5. Parallelization and out-of-core computation: scales efficiently.

## Key concepts taught — Decision Tree Recap

- **Recap (slide 15/21):** built by recursively splitting data on feature thresholds providing best information gain. Leaf nodes contain final predictions (class labels or numeric values). Advantages: easy to interpret, handles mixed data types, minimal preprocessing. Disadvantages: prone to overfitting if unpruned, unstable w.r.t. small data changes.
- **Slides 16/21 and 17/21:** visualization of decision trees via `https://github.com/parrt/dtreeviz` (visual only).

## Key concepts taught — Random Forests

- **Motivation (slide 18/21):** bagging multiple decision trees alone → highly correlated trees. Random Forest adds **feature randomness** to reduce correlation among trees. In addition to bootstrapping samples, each split considers only a **random subset of features**.
- **De-correlated trees procedure (slide 19/21):**
  1. Each tree built from a bootstrapped subset of training data.
  2. At each split, randomly select a set of predictors — **`√p` out of p** (the canonical default).
  3. From these predictors, choose the best predictor and threshold.
  4. Result: lower correlation among trees, often higher accuracy.
- **Random forests summary (slide 20/21):** each tree trained on unique bootstrap sample; only a subset of features considered for splits; final classification by **majority vote**, regression by **averaging**; good out-of-the-box performance; handles large feature spaces.

## Key concepts taught — Extra Trees

- **Extra Trees (slide 21/21):** Extremely Randomized Trees. Similar to Random Forest (multiple trees on bootstrap samples; random feature subset per split). **Key difference: picks random splitting thresholds** rather than searching for the optimal threshold. This adds additional randomness compared to Random Forest.

## Notation

- `η` = learning rate in boosting (controls contribution of each weak learner). Same symbol used in gradient descent (Lectures 8, 9) — consistent usage, not a collision.
- `p` = total number of features/predictors; `√p` = number of features considered per split in Random Forest.
- OOB = Out-Of-Bag (error estimate from bootstrap samples not included in each tree's training set — concept mentioned, not formalized in this lecture).
- No explicit formulas for AdaBoost weights or gradient boosting objective given in this deck.

## R9 cross-check flags (vs ESL/ISL)

- ~63% unique samples in a bootstrap: follows from `1 − (1 − 1/n)^n → 1 − e^{−1} ≈ 0.632` as n → ∞. [VERIFY: not stated as a formula on the slide — confirm whether the derivation is expected.]
- Random Forest feature selection at splits: slide states `√p`; ISL §8.2.2 gives `√p` for classification and `p/3` for regression as defaults. [VERIFY: slide does not distinguish classification vs regression default — note in methods page.]
- XGBoost L1/L2 regularization: not formalized in this deck. [VERIFY: XGBoost objective includes `γ` (tree complexity) and `λ` (L2 on leaf weights) — confirm whether these are covered in companion notebook.]
- Extra Trees: not in ESL/ISL standard editions. [VERIFY: Geurts et al. (2006) is original reference; confirm any citation given in notebook.]

## Professor emphasis cues

- The four types of ensemble methods (Bagging, Boosting, BMA, Stacking) are listed on slide 4/21 but only Bagging and Boosting are given full treatment — BMA and Stacking are taxonomy-level knowledge.
- The 63% bootstrap fact is stated as a memorable rule of thumb — likely tested.
- The `√p` rule for Random Forest feature selection is presented as the canonical design choice (slide 19/21).
- Extra Trees' key difference (random thresholds vs. optimal threshold search) explicitly called out as the defining distinction from RF — flag as exam comparison point.
- XGBoost features listed as a numbered 5-point list — the kind of structure profs test with "name 3 features of XGBoost."

## Companion materials

No specific notebook filenames named on slides. Decision tree visualization tool referenced: `https://github.com/parrt/dtreeviz` (slide 16/21).

## Cross-refs

→ `methods/ensemble.qmd`, `methods/decision_trees.qmd` (from Lecture 4), `methods/random_forest.qmd`, `methods/boosting.qmd`, `methods/cv_info_criteria.qmd` (OOB error as CV alternative).
