# Lecture 7 — Ensemble Methods

**Source:** `lecture7_260521_184935.pdf` (25 PDF pages = 21 logical slides; cite `n/21`. Page 25 is blank). Date: February 18th, 2026. Instructor: Vegard Høghaug Larsen.

## Topic
Ensemble methods: bagging, boosting, bootstrap, random forests, extra trees (with decision-tree recap).

## Key concepts taught
- **Plan (slide 2/21):** intro to ensemble methods; bagging; boosting; recap decision trees; random forests; extra trees.
- **Ensemble methods (slide 3/21):** family of techniques training multiple models and combining predictions for a more accurate, robust model. Can be used with various base models (linear models, decision trees, neural networks). Combining typically reduces variance and boosts performance. Trade-off: more powerful but less interpretable than single models.
- **Types of Ensemble Methods (slide 4/21):** progressive build listing four families:
  - Bagging (Bootstrap Aggregating) — examples Random Forests, Extra Trees; reduces variance by training multiple models on bootstrap samples and averaging predictions.
  - Boosting (Sequential Learning) — examples Gradient Boosting, AdaBoost, XGBoost; trains models sequentially, focusing on misclassified examples to reduce bias.
  - Bayesian Model Averaging (BMA) — trains multiple models, assigns weights by posterior probabilities; can dynamically update weights with new data; useful when model uncertainty must be explicitly accounted for.
  - Stacking (Stacked Generalization) — combines models by training a meta-model on their outputs; uses diverse base models; example: a neural network trained on outputs of logistic regression, SVM, and a random forest.
- **Bagging (slide 6/21):** short for Bootstrap AGGregatING; train multiple instances of a base model on bootstrap samples; combine predictions via majority vote (classification) or averaging (regression); subsets created by sampling with replacement; effective at reducing variance and mitigating overfitting for high-variance base learners (e.g. large decision trees).
- **Bootstrap (slide 7/21):** a resampling technique to approximate the sampling distribution of a statistic; repeatedly sample WITH replacement to create new datasets of the same size as the original; **about 63% of unique samples from the original dataset typically appear in each bootstrap sample**; used to train multiple models (bagging) or provide performance estimates (e.g. OOB error); useful for small/expensive datasets.
- **(slide 8/21):** bootstrap-sampling figure ("Source"); no formula.
- **How bagging works (slide 9/21):** create multiple bootstrap samples; train a base model (e.g. decision tree) on each; predict with each; combine (majority vote for classification, averaging for regression); final prediction has lower variance and improved generalization.
- **Boosting (slide 11/21):** combines multiple weak learners sequentially; each subsequent model focuses more on samples previous models misclassified (higher weight); typically reduces bias but can overfit if not regularized; often uses simpler base learners (shallow trees) improved incrementally.
- **How boosting works (slide 12/21):** 1. initialize weights of each training example equally; 2. train a base model using current weights; 3. evaluate performance on training data; 4. increase weights of misclassified examples, decrease weights of correctly classified; 5. repeat until stopping criterion (max iterations or minimal improvement); 6. final prediction is a weighted combination of all models; 7. **learning rate (η)** often controls the contribution of each weak learner.
- **AdaBoost and XGBoost (slide 13/21):** AdaBoost = Adaptive Boosting (one of earliest boosting algorithms); XGBoost = eXtreme Gradient Boosting (popular gradient boosting framework); both typically use decision trees as base learners with different objectives/optimization; widely used for image classification, ranking, structured data.
- **XGBoost features (slide 14/21):** 1. Regularization (L1 and L2 terms to prevent overfitting); 2. Tree pruning during construction; 3. Handling missing values (built-in mechanism); 4. Cross-validation (integrated for hyperparameter tuning); 5. Parallelization and out-of-core computation (scale efficiently).
- **Recap: Decision trees (slide 15/21):** built by recursively splitting data on a feature threshold giving best information gain; leaf nodes contain final predictions (class labels or numeric values); advantages = easy to interpret, handle mixed data types, minimal preprocessing; disadvantages = prone to overfitting if unpruned, unstable w.r.t. small data changes.
- **(slides 16/21, 17/21):** decision-tree visualization (dtreeviz, `https://github.com/parrt/dtreeviz`); figures only.
- **Decision trees → Random forests (slide 18/21):** bagging trees alone can produce highly correlated trees; Random Forest adds **feature randomness** to reduce correlation; in addition to bootstrapping samples, each split considers only a random subset of features; reduces variance by decorrelating trees and improves performance.
- **De-correlated trees (slide 19/21):** 1. each tree built from a bootstrapped subset of training data; 2. at each split, randomly select a set of predictors (e.g. **√p**) out of p; 3. from these predictors, choose the best predictor and threshold; 4. result: lower correlation among trees, often higher accuracy.
- **Random forests (slide 20/21):** each tree trained on a unique bootstrap sample; within each tree only a subset of features considered for splits; final classification by majority vote, regression often averaged; good out-of-the-box performance, handles large feature spaces.
- **Extra trees (slide 21/21):** Extremely Randomized Trees; similar to Random Forest (multiple trees on bootstrap samples; random subset of features at each node); KEY DIFFERENCE: Extra Trees picks **random splitting thresholds** rather than searching for the optimal threshold.

## Notation
- `η` = learning rate in boosting (controls weak-learner contribution). **Collision warning:** Lecture 8 also uses `η` for the gradient-descent learning rate, but Lecture 8 ALSO uses `α` for the same role (slide 8-12/29) — flag the η-vs-α learning-rate inconsistency across lectures in notation_table.
- `p` = total number of predictors/features; `√p` = number of predictors randomly considered at each split in a random forest. **Collision warning:** Lecture 4 used `p` as a probability. Same letter, different meaning — flag.
- OOB = out-of-bag (error estimate); ~63% = expected fraction of unique original samples in a bootstrap sample.

## R9 cross-check flags (vs ESL/ISL)
- "~63% of unique samples appear in each bootstrap sample": cross-check against ESL §7.11 (the limit is `1 − (1 − 1/N)^N → 1 − e^{−1} ≈ 0.632`). **Flag: confirm the 63% / 0.632 bootstrap figure matches ESL 0.632 bootstrap.**
- Random forest split feature count `√p`: cross-check against ISL §8.2.2 / ESL §15.2 (ISL default `m ≈ √p` for classification, `m ≈ p/3` for regression). **Flag: slide gives only √p; ESL/ISL note p/3 for regression — record the regression default as supplemental.**
- Random forest "decorrelate trees" rationale: matches ESL §15.2. Agree.
- No explicit AdaBoost weight-update formula or gradient-boosting additive-model equation on slides — **flag: AdaBoost `α_m = ½ ln((1−err)/err)` and the boosting additive model are NOT on slides; supply from ESL §10.1 if needed and mark supplemental.**

## Professor emphasis cues
- Variance-reduction (bagging) vs bias-reduction (boosting) contrast stressed throughout.
- Feature randomness as the distinguishing feature of Random Forest vs plain bagged trees (slide 18 emphasis).
- Extra Trees distinguished by random thresholds (slide 21 "Key difference").
- Interpretability trade-off of ensembles mentioned in opening (slide 3).
- XGBoost features given as a numbered 5-point list — likely exam-relevant enumeration.

## Companion materials
No lecture-notebook or exercise filenames printed on these slides. External links shown: dtreeviz (`github.com/parrt/dtreeviz`) for tree visualization; a bootstrap-sampling figure source.

## Cross-refs
→ `methods/random_forests.qmd` (RF, √p, decorrelation), `methods/bagging.qmd` (bootstrap, OOB), `methods/boosting.qmd` (AdaBoost/XGBoost, learning rate), `methods/extra_trees.qmd`, `methods/decision_trees.qmd` (recap, Lecture 4). Bootstrap links to `methods/cv_info_criteria.qmd` (resampling). Boosting learning rate `η` cross-links to `methods/backprop_gradient_descent.qmd` (Lecture 8). Feeds Lecture 10 "Traditional ML vs DL" ensemble-vs-DL comparison.
