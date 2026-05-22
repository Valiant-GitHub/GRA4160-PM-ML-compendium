# Lecture 5 — Model Selection, Evaluation, and Assessment

**Source:** `lecture5_260521_184934.pdf` (11 PDF pages = 11 logical slides; cite `n/11`). Date: February 4th, 2026. Instructor: Vegard Høghaug Larsen.

## Topic
Model selection vs. evaluation; classification metrics (confusion matrix); information criteria (AIC/BIC); cross-validation; bias-variance trade-off.

## Key concepts taught
- **Plan (slide 2/11):** Selection & Evaluation; Information Criteria; Cross-Validation; Bias-Variance Trade-off.
- **Model Selection vs. Evaluation (slide 3/11):** Model Selection = choosing the best model (or hyperparameters) from candidates; goal = select model structure; tool = validation sets, information criteria. Model Evaluation = measuring how well the final model performs on unseen data; goal = estimate real-world error; tool = test sets, cross-validation. Pitfall: overusing a single validation set for selection can lead to overfitting that subset.
- **Confusion Matrix (slide 4/11):** summarizes classification performance comparing predicted vs. actual. TP = correctly identified positive; TN = correctly identified negative; FP (Type I) = false alarm; FN (Type II) = missed detection. Layout: rows = Actual (Pos/Neg), columns = Predicted (Pos/Neg); cells Actual-Pos×Pred-Pos = TP, Actual-Pos×Pred-Neg = FN, Actual-Neg×Pred-Pos = FP, Actual-Neg×Pred-Neg = TN.
- **Common Metrics (slide 5/11):**
  - Accuracy `= (TP + TN) / (TP + TN + FP + FN)` — fraction correct (caution: misleading in imbalanced datasets).
  - Precision `= TP / (TP + FP)` — purity of positive predictions; vital when FPs costly (e.g. spam filters).
  - Recall (Sensitivity) `= TP / (TP + FN)` — ability to find positives; vital when FNs dangerous (e.g. cancer detection).
  - F1-Score: the harmonic mean of Precision and Recall; balance between the two. (No closed-form printed on slide; described as harmonic mean.)
  - ROC & AUC: plots TPR vs. FPR; AUC (Area Under Curve) summarizes performance across all thresholds.
- **Information Criteria (slide 6/11):** goal = select a model that explains data well without being overly complex; penalize the likelihood based on number of parameters `k`.
  - AIC: `AIC = 2k − 2 ln(L)`. Estimates out-of-sample prediction error; penalizes complexity (`2k`).
  - BIC: `BIC = ln(n) k − 2 ln(L)`. Penalizes complexity more heavily when n is large; tends to favor simpler models than AIC.
  - Rule of thumb: lower values are better; compare only on the same dataset.
- **k-Fold Cross-Validation (slide 7/11):** procedure — 1. shuffle dataset; 2. split into k groups (folds); 3. for each group: take it as hold-out/test, take the rest as training, fit model on training and evaluate on test; 4. summarize model skill using the sample of evaluation scores.
- **(slide 8/11):** illustrative diagram of k-fold CV (figure only, no text).
- **Advanced CV Techniques (slide 9/11):** Stratified k-fold — ensures each fold maintains the same class distribution as the full dataset; essential for imbalanced classes. Leave-One-Out (LOOCV) — `k = n`, uses all but one observation for training; pro: nearly unbiased; con: computationally expensive. Repeated CV — repeats k-fold multiple times with different splits to reduce variance in the estimate.
- **Bias-Variance Trade-off (slide 10/11):** Bias (underfitting) = error from erroneous assumptions (e.g. assuming linear when quadratic); misses relevant relations; fix = increase complexity, add features. Variance (overfitting) = error from sensitivity to small fluctuations in training set; models random noise; fix = regularization, more data, simplify model. Goal: find sweet spot where Total Error `= Bias² + Variance` is minimized.
- **Summary / takeaways (slide 11/11):** don't use the test set to tune parameters; accuracy not enough — use Precision/Recall/AUC; AIC/BIC allow comparison without a hold-out set; cross-validation (especially stratified) is the gold standard; ML is largely balancing bias and variance.

## Notation
- `TP, TN, FP, FN` = confusion-matrix counts. `FP` = Type I error; `FN` = Type II error.
- `k` in AIC/BIC = **number of model parameters**. **Collision warning:** same `k` is the **number of folds** in k-fold CV on slides 7–9 (and `k = n` for LOOCV). Two distinct meanings within one lecture — flag prominently for notation_table.
- `n` = number of observations (sample size) in BIC `ln(n)` and LOOCV `k = n`.
- `L` = likelihood (AIC/BIC use `ln(L)`, the log-likelihood).
- Total Error decomposition uses `Bias²` and `Variance` (irreducible-error term not written on slide).

## R9 cross-check flags (vs ESL/ISL)
- AIC `= 2k − 2 ln(L)` and BIC `= ln(n) k − 2 ln(L)`: cross-check against ESL §7.5/§7.7. ESL writes AIC/BIC in terms of `d` parameters and `−2·loglik`; **note BIC factor**: ESL BIC `= −2 loglik + (log n)·d` matches this slide. ISL §6.1 defines AIC/BIC up to scaling/constants for linear models (AIC `= (1/nσ̂²)(RSS + 2dσ̂²)`) — **flag that ISL's linear-model AIC differs in form** from the general likelihood form here.
- Bias-variance: `Total Error = Bias² + Variance`. ESL §7.3 includes an additional **irreducible error σ²** term: `Err = σ² + Bias² + Var`. **Flag: slide omits irreducible error term.**
- Precision/Recall/F1/Accuracy: standard definitions, agree.

## Professor emphasis cues
- Strong "don't use test set to tune" warning repeated (slide 3 pitfall + slide 11 takeaway).
- "Accuracy is not enough" emphasized (imbalanced-data caution stated twice).
- Stratified CV called "the gold standard."
- "ML is largely the art of balancing Bias and Variance" — framing statement.

## Companion materials
No lecture-notebook or exercise filenames printed on these slides.

## Cross-refs
→ `methods/cv_info_criteria.qmd` (CV, AIC/BIC, LOOCV, stratified, repeated), `methods/metrics.qmd` (confusion matrix, precision/recall/F1/ROC-AUC), `methods/bias_variance.qmd`. Feeds λ-selection cross-refs in `methods/regularization.qmd` (Lecture 3/4) and hyperparameter tuning across all method pages. AIC/BIC link to model-selection workflow.
