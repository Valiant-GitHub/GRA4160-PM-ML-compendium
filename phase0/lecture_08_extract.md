# Lecture 8 — Backpropagation and Gradient Descent

**Source:** `lecture8_260521_184937.pdf` (33 PDF pages = 29 logical slides; cite `n/29`. Page 14 is blank). Date: February 25th, 2026. Instructor: Vegard Høghaug Larsen.

## Topic
How neural networks learn: gradient descent (variants), computational graphs, the chain rule, the backpropagation algorithm; worked through ADALINE and Logistic Regression; live-coded automatic differentiation engine.

## Key concepts taught
- **Plan (slide 2/29):** topic = backprop & gradient descent (how NNs learn). Key concepts: GD variants, computational graphs, chain rule, backprop algorithm. Approach: first principles then live-code a simple autodiff engine. Goal: intuition on how gradients are computed and used. Example: Logistic regression.
- **What is Gradient Descent (slide 3/29):** optimization algorithm minimizing a loss by iteratively moving in the direction of steepest descent (negative gradient). Update rule for a parameter w:
  - `w ← w − η ∇_w L`, where η is the learning rate (step size).
- **What is the gradient (slide 4/29):** gradient = partial derivatives of loss w.r.t. parameters. For parameter w_i: `∇_{w_i} L = ∂L/∂w_i`. Gradient vector (two params): `∇_w L = ( ∂L/∂w_1 , ∂L/∂w_2 )`. The gradient points in the direction of steepest ASCENT of the loss.
- **GD Variants (slide 5/29):** Batch GD — gradient on entire dataset before each update (stable but slow for large data). Stochastic GD (SGD) — gradient on a single example at a time (faster, noisier). Mini-Batch GD — gradient on a small batch (e.g. 32 or 64) (compromise: faster than full-batch, more stable than pure SGD). In practice mini-batches most common (e.g. 32 samples) for vectorized computation.
- **Computational Graphs (slide 6/29):** directed graph representing flow of data through operations; nodes = operations/variables, edges = dependencies. Example expression: `L(w_1, w_2) = ((w_1 − 1)² + (w_2 − 5)²) × 0.5` representable as a graph of subtraction, square, addition, multiply.
- **(slide 7/29):** computational-graph figure for that loss (nodes w1, w2, constants 1 and 5, − , square, + , ×0.5 → L).
- **Autodiff (slide 8/29):** NNs are large computational graphs (input flows forward through layers; trace how each weight influences final loss). Frameworks (PyTorch, TensorFlow) build graphs during forward pass. Automatic differentiation = compute gradient of a function w.r.t. its inputs automatically using the chain rule.
- **Chain Rule (slide 9/29):** if `y = f(u)` and `u = g(x)` then `dy/dx = (dy/du)·(du/dx)`. For a composition of many functions, gradients multiply along the chain. Local gradients: each operation knows how its output changes w.r.t. its inputs; backprop uses these local derivatives + chain rule to get global gradients. Key idea: backprop = applying the chain rule repeatedly on a graph of computations.
- **What is Backpropagation (slide 10/29):** backward propagation of errors — computes gradient of loss w.r.t. all parameters; efficient application of the chain rule on the computational graph. Forward pass = compute outputs and loss; backward pass = starting from loss, propagate error gradient backward through each operation to find `∂L/∂w` for every weight. Result = gradient vector `∇_θ L` (partial derivatives collected in vector θ).
- **Backprop step-by-step — network setup (slide 11/29):**
  - Network: `f_θ(x) = σ( Σ_k β_k h_k(x) + b^{(2)} )` with `h_k(x) = σ( Σ_j w_kj x_j + b_k^{(1)} )`.
  - Single point (x, y), loss `L = ½ (y − f_θ(x))²`. σ = sigmoid.
  - 1. Forward Pass: compute prediction f_θ(x), store intermediates (h_k(x) and `z_k = Σ_j w_kj x_j + b_k^{(1)}`).
  - 2. Output Gradient: initialize `∂L/∂L = 1`; `∂L/∂f_θ(x) = −(y − f_θ(x))`.
  - 3. Backprop through Output Layer: `∂L/∂β_k = ∂L/∂f_θ(x) · h_k(x)`.
- **Backprop step-by-step continued (slide 12/29):**
  - 4. Backprop through Hidden Layer: with `z_k = Σ_j w_kj x_j + b_k^{(1)}` and `h_k(x) = σ(z_k)`,
    - `∂L/∂w_kj = ∂L/∂f_θ(x) · β_k · σ'(z_k) · x_j`.
  - 5. Extend to Deeper Layers: apply chain rule recursively, propagating error backwards.
  - 6. Gradient Updates: with full gradient `∇_θ L`, update `θ ← θ − α ∇_θ L`.
- **Training with Backprop + GD (slide 13/29):** per epoch (and per batch): forward pass (predictions + loss); backward pass (gradients via backprop); update weights `w ← w − α ∇_w L`. Batch GD = average gradient over all data, one update/epoch; SGD = update per example; Mini-batch GD = update per small batch (e.g. 32). Iterate many epochs; backprop nudges each weight in the direction that most reduces error.
- **ADALINE (slides 14/29, 16/29):** ADAptive LInear NEurons for binary classification (section header + figure).
- **Linear decision boundary (slide 15/29):** binary classification with classes 0 and 1; construct a linear decision boundary; use a threshold function; decision function takes a linear combination of features: `z = wᵀx + b` (net input).
- **ADALINE rule — Step 1 updating weights (slide 17/29):** 1. initialize weights and bias; 2. compute NET INPUT `z^{(i)} = wᵀx^{(i)} + b`; 3. compute error in training step by `y^{(i)} − z^{(i)}`, where `y^{(i)} ∈ {0,1}` but `z^{(i)} ∈ ℝ`. Want to minimize `y^{(i)} − z^{(i)}`: for y=1 push z toward 1; for y=0 push z toward 0.
- **Step 2 threshold function (slide 18/29):** at end of training z^{(i)} not exactly 0/1; transform via threshold function: `ŷ^{(i)} = 1 if z^{(i)} ≥ 0.5, else 0`. Emphasis: THE THRESHOLD FUNCTION IS ONLY USED TO MAKE THE FINAL PREDICTION.
- **ADALINE optimization via GD (slide 19/29):** uses MSE loss with net inputs:
  - `L(w, b) = (1/2n) Σ_{i=1}^{n} (y^{(i)} − z^{(i)})²` (residuals between observations and net inputs).
  - Step in opposite direction of gradient; step size from learning rate × slope.
- **The rule — gradients (slide 20/29):**
  - `w = w + Δw` with `Δw = −η ∇_w L(w, b)`; `b = b + Δb` with `Δb = −η ∇_b L(w, b)`.
  - `∇_w L(w, b) = ( ∂L/∂w_1, ∂L/∂w_2, ..., ∂L/∂w_m )ᵀ` with `∂L/∂w_j = −(1/n) Σ_{i=1}^{n} (y^{(i)} − z^{(i)}) x_j^{(i)}`.
  - `∇_b L(w, b) = ∂L/∂b = −(1/n) Σ_{i=1}^{n} (y^{(i)} − z^{(i)})`.
- **Logistic Regression — sigmoid (slide 23/29):** `σ(z) := 1 / (1 + e^{−z})` where `z = wᵀx + b` (net input). For z → +∞, σ → 1; z → −∞, σ → 0; `σ : ℝ → [0,1]` with `σ(0) = 0.5`.
- **Threshold (same as Adaline) (slide 24/29):** predicted probability → binary outcome via `ŷ = 1 if σ(z) ≥ 0.5, else 0`. Remark: often interested in both class label and class-membership probability (sigmoid output before threshold). Examples: weather forecasting (chance of rain), medicine (chance of disease).
- **Learning process / cross-entropy (slide 25/29):** maximize likelihood ≡ minimize CROSS-ENTROPY loss:
  - `L(w, b) := Σ_{i=1}^{n} [ −y^{(i)} log(σ(z^{(i)})) − (1 − y^{(i)}) log(1 − σ(z^{(i)})) ]`.
  - For n = 1: `L = −y^{(1)} log(σ(z^{(1)})) − (1 − y^{(1)}) log(1 − σ(z^{(1)}))`, i.e. `−log(σ(z^{(1)}))` if y=1, `−log(1 − σ(z^{(1)}))` if y=0.
- **Cross-entropy explained (slide 26/29):** loss → 0 if correctly predict class 1 (blue line) or class 0 (orange dashed); if prediction wrong, loss → ∞. Penalizes wrong predictions with increasingly larger loss.
- **The algorithm (slides 27/29):** logistic regression obtained from ADALINE by 1. substituting MSE with cross-entropy; 2. changing linear activation with sigmoid. The parameter updates via gradient descent are EQUAL to ADALINE's — the gradient step is unchanged (optional home exercise).
- **The gradient step (unchanged) (slide 28/29):**
  - `w = w + Δw` with `Δw = −η ∇_w L(w, b)`; `b = b + Δb` with `Δb = −η ∇_b L(w, b)`.
  - `∂L/∂w_j = −(1/n) Σ_{i=1}^{n} (y^{(i)} − σ(z^{(i)})) x_j^{(i)}` (note: `σ(z^{(i)})` replaces `z^{(i)}` from ADALINE).
  - `∇_b L = ∂L/∂b = −(1/n) Σ_{i=1}^{n} (y^{(i)} − σ(z^{(i)}))`.
- **Coding Demo (slide 29/29):** build a simple autodiff engine from scratch; constructs computational graph in forward pass, computes gradients via backprop (chain rule) in backward pass; optimize a small example by gradient descent.

## Notation
- `η` (eta) = learning rate in the GD update rule `w ← w − η ∇_w L` (slides 3, 19, 20, 28). BUT `α` (alpha) = learning rate in the backprop update rule `θ ← θ − α ∇_θ L` (slides 12, 13). **SAME ROLE, TWO SYMBOLS within one lecture** — η for ADALINE/logistic, α for the generic backprop section. Flag prominently in notation_table.
- `z` = net input `wᵀx + b`; superscript `^{(i)}` indexes training examples; `z_k` = pre-activation of hidden unit k (`Σ_j w_kj x_j + b_k^{(1)}`).
- `σ` = sigmoid; `σ'(z_k)` = sigmoid derivative (used in hidden-layer gradient).
- `β_k` = output-layer weights; `w_kj` = hidden-layer weight (input j → hidden unit k); `b^{(1)}`, `b^{(2)}` = layer biases.
- `θ` = vector of all parameters; `∇_θ L` = full gradient.
- `n` = number of training examples; `m` = number of weights/features in `∂L/∂w_j` index (j = 1..m).
- `L` overloaded: loss function AND in `∂L/∂L = 1` self-derivative initialization (slide 11).

## R9 cross-check flags (vs ESL/ISL)
- Hidden-layer gradient `∂L/∂w_kj = −(y − f_θ(x)) · β_k · σ'(z_k) · x_j`: cross-check against ESL §11.4 backprop equations (the δ-recursion). Structure matches (chain rule output→hidden). **Flag: confirm sign convention and that ESL's `δ_k` corresponds to `−(y−f)·β_k·σ'(z_k)`.**
- Cross-entropy loss `Σ [−y log σ(z) − (1−y) log(1−σ(z))]`: matches ESL §4.4 / standard binary cross-entropy. Agree (uses natural log).
- ADALINE/logistic gradient `∂L/∂w_j = −(1/n) Σ (y − ŷ) x_j`: for logistic, replacing z by σ(z), this is the standard logistic-regression gradient — **note: claim "gradient step unchanged from ADALINE" is exactly true only because both reduce to `−(1/n)Σ(y − activation)x_j`; ADALINE uses MSE with linear activation, logistic uses cross-entropy with sigmoid. Verify the algebra holds (the slide labels it an optional home exercise).** [VERIFY: that MSE-linear and CE-sigmoid yield identical update form — true by construction here.]
- MSE prefactor `1/(2n)` (slide 19) vs no `1/2` in some texts: scaling-of-η difference only.

## Professor emphasis cues
- "REMEMBER THAT THE THRESHOLD FUNCTION IS ONLY USED TO MAKE THE FINAL PREDICTION" — all-caps emphasis (slide 18).
- "the gradient step is unchanged!" — repeated emphasis (slides 27, 28); central pedagogical point linking ADALINE → logistic regression.
- Backprop framed as "just the chain rule applied repeatedly on a graph" (slide 9 key idea).
- Mini-batch (32) repeatedly cited as the practical default.
- Live-coding an autodiff engine "from scratch" is the capstone demo (slide 29).

## Companion materials
No lecture-notebook or exercise filenames printed on slides; the live-coded "automatic differentiation engine from scratch" (slide 29) is the in-class demo. Optional home exercise: derive that the logistic gradient step equals ADALINE's (slide 27).

## Cross-refs
→ `methods/backprop_gradient_descent.qmd` (GD variants, chain rule, backprop equations), `methods/logistic.qmd` (sigmoid, cross-entropy, gradient — links Lecture 4), `methods/neural_networks.qmd` (Lecture 9 reuses the same backprop derivation with `g(z)` generic activation). ADALINE/net-input `z = wᵀx + b` feeds the perceptron/linear-classifier page. Cross-entropy cross-links to `methods/metrics.qmd` (log loss).
