# Phase 0.6 — Notation Analysis (anchor for `appendix/notation_table.qmd`)

**GR1 — Notation precedence:** ESL conventions are the formal anchor. Where the
course (slides/notebooks) uses a different symbol for the same concept, the chosen
convention below is used consistently across all pages, with a "see also" note.
This file is the data; `appendix/notation_table.qmd` (Phase 1.5) renders it.

The course decks reuse a handful of letters for very different quantities. The
collisions below are real and appear *within single lectures* — they must be
disambiguated by context on every page, and the page must state which meaning is in play.

## Master symbol table (chosen convention = ESL-anchored)

| Symbol | Chosen meaning (ESL anchor) | Course usage & collisions | Resolution on pages |
|---|---|---|---|
| `n`, `N` | number of observations (sample size) | `n` throughout; `N`, `N_i` for class counts in LDA (L3) | use `n` for sample size; `N_i` only for LDA class sizes |
| `p` | number of predictors/features | L7 `√p`, L9 `X=(X_1..X_p)`; **BUT L4 uses `p` = P(positive class)** | use `p` = #predictors globally; logistic probability written `π` (ESL convention) with a note "course slides write this as `p`" |
| `k` (lower) | generic index / count, context-set | **heavily overloaded:** #predictors (L4 logistic `x_1..x_k`), #parameters (L5 AIC/BIC), #folds & `k=n` LOOCV (L5), #clusters (L6 K-means), #neighbors (kNN, L2/L10), hidden-unit index (L8) | NEVER leave bare. Use: `K` neighbors (kNN), `d` parameters (AIC/BIC, ESL), `V` folds (CV), `K` clusters (K-means), `k=1..K` hidden-unit index (NN). State the local meaning in each page's math header. |
| `K` (upper) | number of classes / clusters / hidden units (ESL: classes) | L9 = #hidden units; L6 = #clusters (`{S_1..S_k}`); Gini/entropy use `C` for #classes (L4) | `K` = #hidden units on NN pages; `K` = #clusters on K-means page; `C` = #classes on classification pages (matches L4 slide). Each page states it. |
| `η` (eta) | learning rate | L7 boosting `η`; L8 uses `η` (ADALINE/logistic) **and `α`** (generic backprop) for the SAME role; L9 settles on `η` | adopt `η` for learning rate everywhere; footnote on backprop/NN pages: "Lecture 8 also writes this as `α`." |
| `α` | (i) Elastic-Net mixing ratio; (ii) sklearn penalty strength | L3 Elastic Net `α∈[0,1]`; sklearn `Ridge(alpha=)`/`Lasso(alpha=)` = penalty strength `λ`; L8 `α` = learning rate (collision) | `α` = Elastic-Net mix only in math; flag that **sklearn's `alpha` argument = `λ` (penalty strength), NOT the Elastic-Net `α`** — a notorious trap; called out on regularization page. |
| `λ` | regularization strength | L3 Ridge `λ_r`, Lasso `λ_ℓ`, Elastic Net `λ_e`; L9 L2 `λ`; **also LDA eigenvalue `λ` (L3 generalized eigenproblem)** | `λ` = penalty strength; LDA eigenvalues written `λ` only within the LDA eigenproblem with explicit local scope note. |
| `w` | weight vector | OLS/logistic/ADALINE/NN weights; **LDA projection direction `w`** (different role); `w_kj` NN hidden weights, `w_k0` bias | `w` = weights generally; LDA's `w` labeled "discriminant direction" locally. |
| `β` | coefficients (ESL) | regression `β_j`, intercept `β_0`; NN output-layer weights `β_k` (L9) | `β` = linear-model coefficients AND NN output-layer weights (matches ISL Ch.10); page states which. |
| `z` | linear predictor / net input `wᵀx+b` | L4 logit `z`, L8 net input `z`, `z_k`/`z_ik` NN pre-activation | consistent; keep `z`. |
| `σ` | sigmoid `1/(1+e^{−z})` AND error std-dev | L4/L8/L9 sigmoid `σ(z)`; ESL irreducible error `σ²` | `σ(·)` = sigmoid (with argument); `σ²` = irreducible-error variance (no argument). Disambiguated by argument. |
| `g(·)` | generic activation function | L9 `g(z)` (sigmoid/ReLU/softmax) | keep `g` for generic activation. |
| `δ` | backprop error signal (ESL §11.4) | **L9 slide 11 uses `δ` = the parameter update step**, NOT the ESL error signal | reserve `δ` for ESL's error signal on the math-Mode-B backprop derivation; rename L9's update step to `Δθ` with a note. |
| `L` | loss function | loss everywhere; **L5 AIC/BIC use `L` = likelihood** (`ln L`) | `L` = loss; likelihood written `ℒ` (script) in AIC/BIC with note "slide writes `L`". |
| `m` | (context) | L3 LDA class mean `m_i`; L8 `m` = #weights index; ISL RF feature subset `m≈√p` | LDA mean `m_i` kept; RF subset size written `m`; #weights uses `p`. |
| `S_B`, `S_W` | LDA between-/within-class scatter | L3 | keep (ESL §4.3 uses `B`, `W`). |
| `S_i`, `μ_i` | K-means cluster set & centroid | L6 | keep; note `μ_i` is a centroid, distinct from a distribution mean. |
| `A`, `A_k`, `A_ik` | NN hidden activations | L9 | keep (matches ISL Eq. 10.x). |
| `θ` | full parameter vector | L8/L9 | keep. |
| `1_n` | length-n ones vector | L9 matrix forward pass | keep. |

## Highest-priority disambiguations to surface site-wide
1. **`k`/`K`** — the worst offender; resolved by role-specific letters (`K` neighbors, `d` params, `V` folds, `K` clusters, `K` hidden units). Every math header names its local `k`.
2. **`p` = probability vs #predictors** — use `π` for probability (note course writes `p`).
3. **sklearn `alpha` = `λ` penalty strength, NOT Elastic-Net `α`** — explicit trap callout on the regularization page.
4. **`η` vs `α` for learning rate** — standardize on `η`.
5. **`δ` update-step vs error-signal** — standardize on ESL error-signal `δ`; update step `Δθ`.

## Concepts where the course is SILENT on a symbol (supply from ESL/ISL in Phase 3)
- PCA principal-component vectors / eigenvalues (no symbol on L6 slides) → use ESL §14.5 (`v_j`, `d_j`).
- AdaBoost stage weight `α_m` (not on L7 slides) → ESL §10.1.
- RF regression feature-subset default `p/3` (only `√p` on slides) → ISL §8.2.
