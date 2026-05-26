# Phase 0.9 — Disagreements & Flags Log

Per **GR4** (conflict resolution), **R9** (formula cross-check vs ESL/ISL), **R10**
(structural typo flagging), and **C7** (skip-and-log). Each entry records the source
locations, the discrepancy, and the resolution. **Nothing here is silently "fixed"** —
the course version stays primary in rendered output; the book/standard version is shown
alongside as context. Methods pages cite the relevant entry by ID.

Legend: **[R9]** course-vs-book formula; **[R10]** structurally suspect, no book equiv;
**[NB]** notebook code-vs-prose mismatch (implementation note); **[DATA]** dataset
availability (C7); **[GR4-A]** slide↔notebook.

---

## A. Course-source ↔ reference-text formula differences (R9)

**[R9-1] Decision-tree entropy: log base.** L4 slide 11/11 writes entropy `H = −Σ p_i log₂(p_i)`. ISL §8.1 writes cross-entropy with natural log `D = −Σ p̂_mk log p̂_mk`. *Difference:* base of log (2 vs e) — a constant factor `ln2`, does not change the argmin split. *Resolution:* keep slide's log₂ as primary (course convention); note ISL's natural-log form alongside. Tree page math callout.

**[R9-2] Bias-variance: irreducible error omitted.** L5 slide 10/11 writes `Total Error = Bias² + Variance`. ESL §7.3 includes the irreducible term: `Err = σ² + Bias² + Var`. *Resolution:* primary = slide form (what the course grades); add callout that the full decomposition has the `σ²` floor. bias_variance + cv_info_criteria pages.

**[R9-3] AIC/BIC form.** L5 slide 6/11: `AIC = 2k − 2ln(L)`, `BIC = ln(n)k − 2ln(L)` (general likelihood form). ESL §7.5/7.7 matches (`−2·loglik + 2d`, `−2·loglik + (log n)d`). ISL §6.1 gives a *linear-model-specific* form `AIC = (1/nσ̂²)(RSS + 2dσ̂²)`. *Resolution:* slide's general form primary; note ISL's linear-model form differs by scaling/constants. cv_info_criteria page.

**[R9-4] K-means objective form.** L6 slide 14/19: centroid/squared-distance WCSS `Σ_i Σ_{x∈S_i} ‖x−μ_i‖²`. ISL §12.4.1 uses the equivalent *pairwise* within-cluster form `W(C_k)=(1/|C_k|)Σ_{i,i'∈C_k}Σ_j(x_ij−x_i'j)²`. *Equal up to a factor `2|C_k|`.* *Resolution:* slide's centroid form primary (also matches notebook `inertia_`); note ISL's pairwise form. kmeans page.

**[R9-5] PCA math absent on slides.** L6 slides give PCA only conceptually (maximize variance, orthogonal components) — no covariance eigendecomposition / SVD / explained-variance formula. *Resolution:* supply the eigen/SVD math from ESL §14.5 / ISL §12.2 in Mode B, **clearly marked supplemental** (not from course slides). Notebook 10 also leaves PCA math as prose only. pca page.

**[R9-6] Bootstrap 63% vs 0.632.** L7 slide 7/21: "~63% of unique samples". Exact limit `1−(1−1/N)^N → 1−e⁻¹ ≈ 0.632` (ESL §7.11). *Resolution:* agree; show the derivation in Mode B. random_forests/ensembles page.

**[R9-7] RF feature-subset default.** L7 slides 19–20: only `√p` given. ISL §8.2.2 / ESL §15.2: `m≈√p` (classification), `m≈p/3` (regression). *Resolution:* slide's `√p` primary; add regression `p/3` as supplemental. random_forests page.

**[R9-8] Boosting/AdaBoost formulas absent.** L7 slides describe boosting procedurally; no AdaBoost stage-weight `α_m = ½ln((1−err)/err)` and no gradient-boosting additive-model equation. *Resolution:* supply from ESL §10.1 in Mode B, marked supplemental. ensembles page.

**[R9-9] NN L2 penalty scope.** L9 slide 16/17: `L_reg = ½Σ(y−f)² + λ Σ_{k,j} w_kj²` — sums **only over hidden weights `w_kj`**, not the output weights `β_k`. ESL §11.5 weight decay penalizes all weights. *Resolution:* keep slide form primary; **[VERIFY] flag** whether `β` was intended to be included; note ESL penalizes all. nn pages. (Notebook 19 implements manual L2 as `lambda_reg * sum(p.pow(2))` over all params — i.e. the notebook DOES penalize all weights, mildly diverging from the slide's written scope → cross-link [GR4-A] flavor.)

**[R9-10] Backprop sign/δ convention.** L8 slides 11–12 & L9 slide 14: `∂L_i/∂w_kj = −(y_i−f_i)β_k g'(z_ik)x_ij`. ESL §11.4 uses the δ-recursion (eqs 11.12–11.13). Structure agrees; ESL's `δ_k ≡ −(y−f)β_k g'(z_k)`. *Resolution:* present slide chain-rule form primary; map to ESL δ-notation in Mode B. backprop + nn pages.

**[R9-11] ADALINE→logistic "gradient step unchanged".** L8 slides 27–28 claim the GD update is identical for ADALINE (MSE+linear) and logistic (CE+sigmoid): both reduce to `∂L/∂w_j = −(1/n)Σ(y−activation)x_j`. True by construction (the activation differs: `z` vs `σ(z)`). *Resolution:* present as the course's key pedagogical link; the derivation (an "optional home exercise" per slide 27) goes in Mode B. logistic + backprop pages.

**[R9-12] Ridge/Lasso prefactor.** L3 slides 14–15 use `1/(2n)` on the SSE term with explicit intercept `β_0`. ESL §3.4 writes RSS without `1/(2n)` and centers data so the intercept drops. *Equivalent up to scaling of `λ`.* *Resolution:* slide form primary (matches sklearn's mean-based objective); note the scaling. regularization page.

**[R9-13] sklearn `alpha` ≠ Elastic-Net `α`.** Not a book disagreement but a notation trap: L3 Elastic Net uses `α` for the L1/L2 mix; sklearn `Ridge/Lasso/ElasticNet(alpha=)` is the penalty strength `λ`; sklearn ElasticNet's mix is `l1_ratio`. *Resolution:* explicit callout on regularization page; cross-ref notation_table item 3.

## B. Notebook code ↔ prose / structural mismatches (NB / R10)

**[NB-1] nb07_Decision_trees:** cell 17 fits with `min_samples_leaf=10` but the title/comment says 5; several trees fit on full `X,y` rather than the train split. *Use:* lift the idiom but quote the **actual** value (10) per R3/R5; note the discrepancy.

**[NB-2] nb09_Bias_variance:** `cross_val_score(...)` called with no `scoring=`, so it returns the estimator's default (R²), but prose calls it "MSE". *Use:* when lifting, state it's R² (or set `scoring='neg_mean_squared_error'` and label the change).

**[NB-3] nb10_PCA:** cell 12 scatter plots raw features 0/1 but is titled "after PCA"; `train_test_split` has no `random_state` (non-reproducible). Path uses `../data/seeds.csv` vs repo `course_materials/Data/seeds.csv`. *Use:* fix labeling in the study-site figure; add a seed; reconcile path.

**[NB-4] nb11_K_means:** cell 10 comment says "3 clusters" but call uses `n_clusters=2`; notebook selects K=2 despite Iris having 3 species. *Use:* note the elbow/silhouette led to 2; flag the 3-species ground truth as a teaching point.

**[NB-5] nb12_Ensembles:** cell 4 comment says "hard voting" but uses `voting='soft'`; a comment says "RandomForestClassifier" for what is actually a `DecisionTreeClassifier`. *Use:* quote actual code (`voting='soft'`).

**[NB-6] nb13_Random_forests:** `max_depth=3` defined (cell 7) but never passed to the from-scratch trees (cell 5 `DecisionTreeClassifier(max_features=None, random_state=1)`), so scratch trees grow full-depth while sklearn RF/ExtraTrees use depth 3 — **not apples-to-apples**. Feature subsetting is per-tree, not per-split (differs from a true RF). From-scratch and sklearn RF print identical metrics (coincidental on the small post-dropna set). Dead imports (`scipy.stats.mode`, `BaseEstimator`). *Use:* present the scratch RF as a teaching scaffold; explicitly note these caveats; rely on sklearn version for the canonical idiom.

**[NB-7] nb01_Working_with_data:** a cell prints the pandas version but labels it "scikit-learn version". *Use:* cosmetic; don't propagate the mislabel.

**[NB-8] nb05_Regularised_regressions:** the markdown Elastic-Net objective parameterization differs slightly from sklearn's exact `ElasticNet` objective. *Use:* present slide L3 form as primary; note sklearn's exact objective when showing the code.

**[NB-9] nb15_LogisticRegression:** a plot has swapped axis labels. *Use:* correct in any reproduced figure.

**[NB-10] nb04_LDA:** from-scratch accuracy not stored in outputs ([VERIFY] if a number is quoted, recompute or omit).

## C. Dataset availability (C7) & path reconciliation (C8)

**[DATA-1] Spam dataset missing.** Exercise 02 (naive Bayes) loads `SMSSpamCollection.csv` — **not present locally**. *Resolution (C7):* drill page documents the method/idiom from the `_VHL` solution; mark dataset unavailable; do NOT fabricate outputs. No method page depends on it (naive Bayes is exercise-only).

**[DATA-2] Bank-marketing dataset missing.** Exercise 06 (customer segmentation) loads `bank-additional-full.csv` — **not present locally**. The course's *lecture* K-means/PCA (nb10/nb11) and the exam use `seeds.csv` / `WholesaleCustomers.csv`, which ARE present. *Resolution (C7):* drill page documents idiom; for the kmeans/pca *method* pages and showdowns use `WholesaleCustomers.csv` + `seeds.csv` (present) per R5.

**[DATA-3] sklearn built-ins.** Many notebooks load `load_iris`, `load_digits`, `load_wine`, `load_breast_cancer`, `load_diabetes`, `fetch`/UCI-URL Iris/Adult. These need the venv at build time if any `{python}` cell recomputes; for embedded JS data (C8) prefer the local CSVs (Titanic, house-prices, seeds, Hitters, WholesaleCustomers, wdbc).

**[DATA-4] Notebook paths vs repo layout.** Notebooks reference `../data/titanic/train.csv`, `../data/seeds.csv`, `../data/house-prices/...`. Repo has `course_materials/Data/Titanic data/train.csv`, `course_materials/Data/seeds.csv`, `course_materials/Data/house-prices data/...` (spaces, different case). *Resolution:* the `data/` JS-export step (C8) reads the **real** repo paths from `file_manifest.md`, not the notebook paths.

## D. Slide ↔ notebook (GR4-A)
No hard contradictions found: notebooks implement what slides describe. Complementary pairings (not conflicts), noted for code precedence (notebooks win for code/hyperparameters):
- Logistic: slides L4/L8 (theory) ↔ nb06 (sklearn `LogisticRegression(solver='lbfgs', max_iter=500)`) ↔ nb15 (from-scratch `LogisticRegressionGD`, `eta=0.3, n_iter=1000`). Use nb06 for the canonical sklearn idiom, nb15 for the from-scratch teaching version.
- LDA: slide L3 theory ↔ nb04 (both sklearn `LinearDiscriminantAnalysis` and from-scratch `LDAFromScratch`).
- RF: slide L7 theory ↔ nb13 (scratch + sklearn) — see [NB-6].
- L2 reg scope: slide L9 (hidden weights only) vs nb19 (all params) — see [R9-9].

---

**Halt check (C6):** no source-verification failure; no missing *Tier-1* source; conflicts are all loggable/resolvable. **No halt triggered.** Dataset gaps [DATA-1/2] are C7 skip-and-log items affecting only two drill pages, not method coverage.
