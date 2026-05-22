# Lecture 5 — Model Selection, Evaluation, and Assessment

**Source:** `lecture5_260521_184934.pdf` (11 PDF pages = 11 logical slides; cite `n/11`). Date: February 4, 2026.

## Topic
Model selection vs. model evaluation; confusion matrix and classification metrics (Accuracy, Precision, Recall, F1, ROC/AUC); information criteria (AIC, BIC); k-fold cross-validation variants; bias-variance trade-off.

## Key concepts taught — Selection vs Evaluation

- **Distinction (slide 3/11):**
  - **Model Selection:** choosing the best model/hyperparameters from candidates. Tools: validation sets, information criteria. Goal: select model structure.
  - **Model Evaluation:** measuring how well the final model performs on unseen data. Tools: test sets, cross-validation. Goal: estimate real-world error.
  - **The Pitfall:** overusing a single validation set for selection leads to overfitting that specific subset.

## Key concepts taught — Classification Metrics

- **Confusion matrix (slide 4/11):** 2×2 table comparing predicted vs. actual labels.
  - TP: correctly identified positive; TN: correctly identified negative.
  - FP (Type I error): false alarm; FN (Type II error): missed detection.
  - Layout: rows = Actual (Pos/Neg), columns = Predicted (Pos/Neg); cells = TP, FN (row 1), FP, TN (row 2).
- **Metrics (slide 5/11):**
  - **Accuracy:** `Accuracy = (TP + TN) / (TP + TN + FP + FN)`. Caution: misleading for imbalanced datasets.
  - **Precision:** `Precision = TP / (TP + FP)`. Purity of positive predictions; vital when FP costly (e.g., spam filters).
  - **Recall (Sensitivity):** `Recall = TP / (TP + FN)`. Ability to find positives; vital when FN dangerous (e.g., cancer detection).
  - **F1-Score:** harmonic mean of Precision and Recall. Use when balance between the two is needed. [VERIFY: formula not shown on slide — `F1 = 2 × (Precision × Recall) / (Precision + Recall)`.]
  - **ROC & AUC:** plots TPR vs. FPR; AUC = Area Under Curve, summarizes performance across all thresholds.

## Key concepts taught — Information Criteria

- **Goal (slide 6/11):** select a model that explains data well without being overly complex. Penalize likelihood based on number of parameters k.
- **AIC:**
  - `AIC = 2k − 2 ln(L)`
  - Estimates out-of-sample prediction error; penalizes complexity via `2k`.
- **BIC:**
  - `BIC = ln(n) k − 2 ln(L)`
  - Penalizes complexity more heavily when n is large; tends to favor simpler models than AIC.
- **Rule:** lower values are better; compare only on the same dataset.

## Key concepts taught — Cross-Validation

- **k-fold CV procedure (slide 7/11):**
  1. Shuffle the dataset.
  2. Split into k groups (folds).
  3. For each fold: hold out the fold as test; use remaining k−1 folds as training; fit and evaluate.
  4. Summarize model skill from the k evaluation scores.
- **Advanced CV variants (slide 9/11):**
  - **Stratified k-fold:** each fold maintains same class distribution as the full dataset. Essential for imbalanced classes.
  - **Leave-One-Out (LOOCV):** k = n; uses all but one observation for training. Pro: nearly unbiased. Con: computationally expensive.
  - **Repeated CV:** repeats the k-fold process with different splits to reduce variance in the estimate.
- Slide 8/11 is a diagram of the k-fold procedure (no additional formulas).

## Key concepts taught — Bias-Variance Trade-off

- **Bias (underfitting) (slide 10/11):** error from erroneous assumptions (e.g., assuming linearity when quadratic). Misses relevant relations. Fix: increase complexity, add features.
- **Variance (overfitting) (slide 10/11):** error from sensitivity to small fluctuations in training set. Models random noise. Fix: regularization, more data, simplify model.
- **Goal:** minimize `Total Error = Bias² + Variance`. [VERIFY: slide states `Bias² + Variance` but full decomposition also includes irreducible noise; check if prof includes that term.]
- **Summary framing (slide 11/11):** "Machine Learning is largely the art of balancing Bias and Variance."

## Notation

- `k` = number of parameters in AIC/BIC formulas (also used as number of folds in CV — **symbol collision**: `k` means model complexity AND fold count; note in notation_table).
- `L` = likelihood (maximized); `n` = number of observations; `ln(·)` = natural log.
- TP, TN, FP, FN = confusion matrix cells (Type I = FP, Type II = FN).
- TPR = True Positive Rate = Recall; FPR = False Positive Rate = FP / (FP + TN) [VERIFY: FPR formula not shown on slide].

## R9 cross-check flags (vs ESL/ISL)

- AIC formula `2k − 2 ln(L)`: standard form; matches ESL §7.5. [VERIFY: some sources write AIC with a different scaling, e.g., using deviance = −2 ln(L) directly.]
- BIC formula `ln(n)k − 2 ln(L)`: matches ESL §7.7. Note that BIC penalizes more heavily than AIC for n > 7 (since ln(7) > 2). Flag for comparison note.
- Bias² + Variance decomposition: ESL §7.3 gives full decomposition including irreducible error σ²; slide states `Bias² + Variance` without σ² term — check whether this was intentional simplification.

## Professor emphasis cues

- The pitfall of overusing a single validation set called out explicitly in a dedicated "Pitfall" box on slide 3/11.
- "Don't use your test set to tune parameters!" repeated in the summary (slide 11/11) — high exam relevance.
- Stratified k-fold labeled "gold standard for performance estimation" in summary.
- Precision vs. Recall trade-off framed with real-world motivations (spam filters / cancer detection) — mnemonic-worthy examples likely to appear in exam context.

## Companion materials

No specific notebook filenames named on slides. Plan for Today (slide 2/11) lists: Selection & Evaluation, Information Criteria, Cross-Validation, Bias-Variance Trade-off — all four covered within this single deck.

## Cross-refs

→ `methods/cv_info_criteria.qmd`, `methods/metrics_classification.qmd`, `methods/bias_variance.qmd`, `methods/regularization.qmd` (λ selection via CV from Lecture 3), `methods/ensemble.qmd` (OOB error in Lecture 7).
