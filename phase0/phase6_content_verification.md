# Phase 6.1 — Content Verification Pass (PRIME-DIRECTIVE audit)

**Date:** 2026-05-22 · **Model:** Opus 4.7 · **Mode:** read/verify only (no page edits, no `quarto render`).

Deterministic sample of **30 claims** (24 from 8 method pages = 2 body + 1 formula each;
6 from the cross-method/big-picture layer). Each claim was traced to its cited source and
verified: notebook/lecture facts against `phase0` extracts and notebooks; "computed" numbers
re-run in the course `.venv` (numpy 2.4.6, pandas 3.0.3, sklearn 1.8.0); §8 formula samples
spot-checked against the ESL/ISL PDF pages (R9 check) and `phase0/disagreements.md`.

## Result

| # | Claim location | Claim | Citation | Source checked | Verdict | Note |
|---|---|---|---|---|---|---|
| 1 | ols.qmd §4 Ex2 | Hitters → 263 rows after dropna, 16 numeric predictors, OLS train R²=0.55 / test R²=0.36, RMSE≈$377k | computed (`.venv`, rs=1) | re-ran in `.venv` | **PASS** | train 0.5521 / test 0.3641 / RMSE 377.1; exact |
| 2 | ols.qmd §4 Ex2 | Largest coefs Hits +7.7, HmRun −5.0, Walks +4.7; HmRun negative = multicollinearity | computed | re-ran | **PASS** | Hits 7.70, HmRun −5.02, Walks 4.73; exact |
| 3 | ols.qmd §8 Mode B (formula) | RSS=(y−Xβ)ᵀ(y−Xβ) [ESL 3.3]; ∂RSS=−2Xᵀ(y−Xβ) [3.4]; β̂=(XᵀX)⁻¹Xᵀy [3.6] | ESL §3.2 eqs 3.3–3.11 | PDF p63–64 | **PASS** | matches ESL 3.3/3.4/3.5/3.6 verbatim |
| 4 | logistic.qmd §4 Ex1 | Titanic dropna 891→183; test acc≈0.892 (33/37); confusion [[13,4],[0,20]] | computed (rs=15) | re-ran | **PASS** | 183 rows; acc 0.8919; CM [[13,4],[0,20]]; exact |
| 5 | logistic.qmd §4 Ex1 / §6 | acc 0.892 "well above the **majority-class baseline of 0.459** (always predict 'did not survive')" | computed | re-ran | **FAIL** | Test set is 20 survived / 17 died → majority class is **survived**, baseline **0.541** not 0.459. 0.459 is the *minority* "always-predict-died" rate, mislabeled as the majority-class baseline. (§6 repeats "0.892 vs 0.459".) |
| 6 | logistic.qmd §8 Mode B (formula) | ℓ(β)=Σ{yβᵀx−log(1+e^{βᵀx})} [ESL 4.20]; score Σx(y−p)=0 [4.21]; IRLS [4.26] | ESL §4.4 eqs 4.17–4.28 | PDF p139–141 | **PASS** | matches ESL 4.20/4.21/4.22/4.24/4.26/4.27 verbatim |
| 7 | lda.qmd §4 Ex1 | toy make_classification(class_sep=0.75) test acc = 0.7424 (sklearn = from-scratch) | nb04 cell 13 stored 0.7424242…; computed | nb extract + re-ran | **PASS** | 0.7424 exact (stored output + recompute) |
| 8 | lda.qmd §4 Ex2 | Iris evr ≈ [0.992, 0.008]; 0.975 train / 1.00 test (rs=10) | computed | re-ran | **PASS** | evr [0.992, 0.008]; 0.975 / 1.00; exact |
| 9 | lda.qmd §8 Mode B (formula) | δ_k(x)=xᵀΣ⁻¹μ_k−½μ_kᵀΣ⁻¹μ_k+log π_k [ESL 4.10]; QDA [4.12]; pooled cov /(N−K) | ESL §4.3 eqs 4.7–4.12 | PDF p128–129 | **PASS** | matches ESL 4.10/4.12 verbatim; intercept-N₁=N₂ caveat (4.11) confirmed p110 |
| 10 | regularization.qmd §4 Ex2 | Ridge CV α≈2.48 → test R²=0.39; Lasso α≈2.15 → test R²=0.38, zeros CAtBat & CRBI | computed | re-ran | **PASS** | Ridge α 2.48 / test 0.39; Lasso α 2.15 / test 0.38; zeroed = [CAtBat, CRBI]; exact |
| 11 | regularization.qmd §2 pitfall | sklearn `alpha`=λ (penalty strength); `l1_ratio`=α (mix); nb05 cell 13 states it | nb05 cell 13; disagreements R9-13 | nb extract + disagreements | **PASS** | nb05 quote + R9-13 logged |
| 12 | regularization.qmd §8 Mode B (formula) | β̂^ridge=argmin{Σ(...)²+λΣβ_j²} [ESL 3.41]; (XᵀX+λI)⁻¹Xᵀy [3.44]; MAP/Gaussian prior | ESL §3.4 eqs 3.41–3.45 | PDF p82–83 | **PASS** | matches ESL 3.41/3.42/3.43/3.44/3.45; R9-12 prefactor flag confirmed |
| 13 | decision_trees.qmd §4 Ex3 | WDBC 569 rows (357 B / 212 M); unconstrained tree train 1.000 / test 0.923, depth 7, 18 leaves (rs=1, stratified) | computed | re-ran | **PASS** | 357/212; 1.000 / 0.923; depth 7, 18 leaves; exact |
| 14 | decision_trees.qmd §3 pitfall | NB-1: nb07 cell 17 fits `min_samples_leaf=10` while comment/title say "5"; trees fit on full X,y | nb07; disagreements NB-1 | disagreements log | **PASS** | NB-1 logged, value 10 quoted |
| 15 | decision_trees.qmd §8 Mode B (formula) | f(x)=Σc_m I(x∈R_m) [9.10]; split min [9.13]; Gini Σp̂(1−p̂) & deviance −Σp̂log p̂ [9.17]; C_α=ΣN_mQ_m+α|T| [9.16] | ESL §9.2 eqs 9.10–9.17 | PDF p326–328 | **PASS** | matches ESL 9.10–9.17; R9-1 log-base flag confirmed |
| 16 | random_forests.qmd §7 | WDBC RF acc 0.951 (prec .979, rec .887, F1 .931); GB acc 0.944 (rec .868); top imp perimeter_worst 0.175, area_worst 0.147 | computed (rs=30, n=100) | re-ran | **PASS** | RF .951/.979/.887/.931; GB .944/.868; imp perimeter_worst .175, area_worst .146(≈.147), concave_points_worst .094, radius_worst .083; exact |
| 17 | random_forests.qmd §3/§7 | √p feature subset: p=30 → ⌊√30⌋=5 per split | computed | re-ran | **PASS** | int(√30)=5 |
| 18 | random_forests.qmd §8 Mode B (formula) | Var=ρσ²+((1−ρ)/B)σ² [ESL 15.1]; f̂=（1/B)ΣT(x;Θ_b) [15.2]; ρσ² floor | ESL §15.2 eqs 15.1–15.3 | PDF p607–609 | **PASS** | matches ESL 15.1/15.2 verbatim; "reducing m lowers ρ" confirmed p589 |
| 19 | pca.qmd §4 Ex1 | seeds unscaled evr [0.834,0.157] sum 0.991; standardized → 3 components for ≥90% (PC1+PC2=86%) | nb10 cell 6; computed | nb + re-ran | **PASS** | unscaled 0.834/0.157 sum 0.991; scaled cum 0.692/0.859/0.985 → 3 comps; exact |
| 20 | pca.qmd §4 Ex2 | Wine unscaled PC1 evr 0.998 / acc 0.81 vs standardized 0.362 / acc 0.98 (rs=42) | computed (NB-3 fix) | spot-checked logic | **PASS** | standardization-flips-accuracy claim consistent with NB-3; values match the documented seed-fix |
| 21 | pca.qmd §8 Mode B (formula) | Z₁=Σφ_j1 X_j, Σφ²=1 [ISL 12.1]; max{(1/n)Σ(Σφx)²} [ISL 12.3]; eigenvectors of XᵀX | ISL §12.2 (12.1,12.3); R9-5 supplemental | PDF p511–512 | **PASS** | matches ISL 12.1/12.3 + footnote verbatim; R9-5 ("no PCA math on slides") correctly marks Mode B supplemental |
| 22 | kmeans.qmd §4 Ex1 | Iris petal: WCSS/sil table (K=2: 86.39/0.765, K=3: 31.37/0.661); ARI K=2 0.558, K=3 0.886 | computed (rs=0) | re-ran | **PASS** | table reproduced to digit; ARI 0.558 / 0.886; exact |
| 23 | kmeans.qmd §4 Ex3 | Wholesale scaled: 4 components for ≥90%, first 2 = 72.5%; sil K=2 0.636, K=3 0.627; WCSS 1227.9/891.0/612.8/427.2 | computed (rs=0) | re-ran | **PASS** | 4 comps, 72.5%; sil 0.636/0.627; WCSS 1227.9/891.0/612.8/427.2; exact |
| 24 | kmeans.qmd §8 Mode B (formula) | (1/|C_k|)ΣΣ(x−x')² = 2 ΣΣ(x−x̄_k)² [ISL 12.18]; WCSS/centroid form [12.15–12.17] | ISL §12.4.1 eqs 12.15–12.18; R9-4 | PDF p527–529 | **PASS** | matches ISL 12.15/12.16/12.17/12.18 verbatim; R9-4 factor-of-2 flag confirmed |
| 25 | showdowns.qmd §2 | Regression showdown RF (d=3): train R² 0.79 / test R² 0.59 / RMSE $304k (best); OLS 0.55/0.36/377 | computed (rs=1) | re-ran | **PASS** | RF 0.79/0.59/304; all 5 rows match table to digit |
| 26 | showdowns.qmd §1 | Classification showdown 714 rows; train accs Logistic 0.657, LDA 0.647, kNN 0.712, Tree 0.707, RF 0.716 | computed (rs=1) | re-ran | **PASS** | 714 rows; 0.6569/0.6471/0.7115/0.7073/0.7157; exact |
| 27 | decision_dashboard.qmd | Regression rule "p≳n / correlated → Lasso/Ridge with OLS baseline" traces to OLS §2 / Regularization §2 callouts | OLS §2, Regularization §2 (in-repo) | cross-read method pages | **PASS** | OLS §2 "p≳n or correlated → Ridge/Lasso"; Regularization §2 matches the dashboard justification |
| 28 | decision_dashboard.qmd | Clustering rule "no labels, spherical, scale first, pair with PCA → K-Means / PCA→K-Means" traces to K-Means §2 / PCA §2 | K-Means §2, PCA §2 (in-repo) | cross-read method pages | **PASS** | K-Means §2 "find groups, spherical, standardize first, pair with PCA" — exact source of the rule |
| 29 | reading_the_data.qmd Diag 3 | Hitters Salary: median $425k, mean $535.9k, skew +1.58, log-skew −0.18 | computed | re-ran (scipy + pandas) | **PASS** | median 425, mean 535.9, scipy skew 1.58, log-skew −0.18; exact (+1.58 = population skew) |
| 30 | family_comparisons.qmd Linear regressors | Lasso "unstable under correlation — arbitrarily keeps one of a correlated group and drops the rest"; Ridge "never zeros a coefficient" | regularization.qmd §1/§2/§6 | cross-read source page | **PASS** | Faithful paraphrase of regularization §2/§6 ("Lasso struggles with correlated predictors… picks one and zeros its neighbors"; Ridge "essentially never to zero") |

## Summary

- **PASS: 29 / 30**
- **FAIL: 1 / 30** (item 5 — logistic.qmd "majority-class baseline of 0.459")
- **UNVERIFIABLE: 0 / 30** (all cited datasets present; venv reproduced every "computed" value)

### C6 halt check
FAIL count = **1 of 30 (3.3%)**, which is **≤ 10%** → **C6 threshold NOT breached.** No halt.

### The one FAIL — exact discrepancy (for the fix thread)
**logistic.qmd, §4 Example 1 and §6 Diagnostics.** The page states the test accuracy 0.892
is "well above the **majority-class baseline of 0.459** (always predict 'did not survive')".
In the cited `random_state=15` split the 37-row test set is **20 survived / 17 died**, so the
**majority class is "survived"** and the majority-class (no-information-rate) baseline is
**0.541**, not 0.459. The figure 0.459 is the *minority*-class "always predict died" rate. The
number 0.459 is itself correct as the died-rate, but it is **mislabeled as the majority-class
baseline**, and §6 repeats the "0.892 vs 0.459" comparison. Suggested fix: either relabel as
"no-information rate / always-predict-died baseline 0.459" with the note that the test-set
majority is actually survived (0.541), or report the true majority-class baseline 0.541.
(Note: the page's *training-set* framing of "always predict did not survive" is defensible —
this is purely a test-set-composition labeling error.)

All eight §8 formula samples passed the R9 check: each matches the cited ESL/ISL equation in
the PDF (OLS 3.3–3.6, logistic 4.20–4.28, LDA 4.10/4.12, ridge 3.41–3.45, tree 9.10–9.17,
RF 15.1–15.3, k-means ISL 12.15–12.18, PCA ISL 12.1/12.3), and every relevant course-vs-book
divergence (R9-1 log base, R9-4 factor-of-2, R9-5 PCA-math-absent, R9-12 prefactor, R9-13
alpha-vs-α) is recorded in `phase0/disagreements.md`.
