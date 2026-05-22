# Lecture 8 — Backpropagation and Gradient Descent

**Source:** `lecture8_260521_184937.pdf` (33 PDF pages = 29 logical slides; cite `n/29`). Date: February 25, 2026.

## Topic
Gradient descent (variants, update rule, gradient definition); computational graphs and automatic differentiation; chain rule as the engine of backpropagation; ADALINE (adaptive linear neuron) for binary classification; logistic regression derived from ADALINE; cross-entropy loss; backpropagation algorithm step-by-step; live-code demo of an automatic differentiation engine.

## Key concepts taught — Gradient Descent

- **Definition (slide 3/29):** optimization algorithm to minimize a loss function by iteratively moving in the direction of steepest descent (negative gradient).
  - **Update rule:** `w ← w − η ∇_w L`  where η is the learning rate (step size).
  - Objective: find parameters that minimize loss on training data by continuously updating weights opposite to the gradient.
- **Gradient definition (slide 4/29):**
  - Gradient = partial derivatives of the loss with respect to the parameters.
  - For parameter `w_i`: `∇_{w_i} L = ∂L / ∂w_i`
  - **Gradient vector** (two parameters `w_1, w_2`): `∇_w L = (∂L/∂w_1 , ∂L/∂w_2)^⊤`
  - The gradient vector points in the direction of **steepest ascent** of the loss function (so gradient descent moves in the opposite direction).
- **Gradient descent variants (slide 5/29):**
  - **Batch Gradient Descent:** gradient computed on entire dataset before each update. Stable but very slow for large data.
  - **Stochastic Gradient Descent (SGD):** gradient computed on a single example at a time. Much faster updates but noisier (updates fluctuate).
  - **Mini-Batch Gradient Descent:** gradient on small batch (e.g., 32 or 64 examples). Compromise: faster convergence than full-batch, more stability than pure SGD.
  - In practice: **mini-batches are most common** (e.g., 32 samples per update) to leverage vectorized computation and achieve good convergence.

## Key concepts taught — Computational Graphs

- **Definition (slide 6/29):** directed graph representing the flow of data through a sequence of operations. Nodes = operations or variables; edges = dependencies.
  - **Example function:** `L(w_1, w_2) = ((w_1 − 1)² + (w_2 − 5)²) × 0.5` expressed as graph of simpler operations (subtraction, square, addition, multiply).
- **Slide 7/29:** diagram of the above loss function as a computational graph (visual only).
- **Automatic differentiation (slide 8/29):** deep learning frameworks (PyTorch, TensorFlow) build these graphs during the forward pass, recording operations and inputs/outputs. Sets the stage for autodiff. **Automatic differentiation:** technique to compute the gradient of a function w.r.t. its inputs automatically using the chain rule.

## Key concepts taught — Chain Rule and Backpropagation

- **Chain Rule (slide 9/29):** if `y = f(u)` and `u = g(x)`, then `dy/dx = (dy/du) · (du/dx)`. For compositions of many functions, gradients multiply along the chain.
  - **Local gradients:** each operation knows how its output changes w.r.t. its inputs. Backpropagation uses local derivatives + chain rule to get global gradients.
  - **Key idea:** backpropagation = applying the chain rule repeatedly on a graph of computations.
- **What is backpropagation (slide 10/29):** backward propagation of errors. Core algorithm that computes gradient of the loss w.r.t. all parameters.
  - Efficient application of the chain rule on the computational graph of the forward pass.
  - **Forward pass:** compute outputs and loss using current weights.
  - **Backward pass:** starting from the loss, propagate error gradient backward through each operation to find `∂L/∂w` for every weight w.
  - Result: gradient vector `∇_θ L` (partial derivatives of loss w.r.t. each parameter collected in vector θ). Used to update weights via gradient descent.
- **Backpropagation step-by-step (slides 11/29 and 12/29):** for the two-layer network:
  - `f_θ(x) = σ(Σ_k β_k h_k(x) + b^{(2)})` with `h_k(x) = σ(Σ_j w_{kj} x_j + b^{(1)}_k)`
  - Loss for a single data point: `L = (1/2)(y − f_θ(x))²`; σ = sigmoid function.
  - **Step 1 — Forward Pass:** compute prediction `f_θ(x)`; store intermediate values `h_k(x)` and `z_k = Σ_j w_{kj} x_j + b^{(1)}_k`.
  - **Step 2 — Output Gradient:** initialize with `∂L/∂L = 1`; compute `∂L/∂f_θ(x) = −(y − f_θ(x))`.
  - **Step 3 — Backprop through Output Layer:** for each output weight `β_k`:
    - `∂L/∂β_k = (∂L/∂f_θ(x)) · h_k(x)`
  - **Step 4 — Backprop through Hidden Layer (slide 12/29):** for each hidden weight `w_{kj}` (connecting input j to hidden unit k), using `z_k = Σ_j w_{kj} x_j + b^{(1)}_k` and `h_k(x) = σ(z_k)`:
    - `∂L/∂w_{kj} = (∂L/∂f_θ(x)) · β_k · σ'(z_k) · x_j`
  - **Step 5:** extend to deeper layers by applying chain rule recursively.
  - **Step 6 — Gradient Updates:** `θ ← θ − α ∇_θ L`
- **Training loop (slide 13/29):** for each epoch (and each batch):
  1. Forward pass: compute predictions and loss.
  2. Backward pass: compute gradients via backpropagation.
  3. Update weights: `w ← w − α ∇_w L` for all weights.
  - Batch GD: average gradient over all data, update once per epoch.
  - SGD: update per example.
  - Mini-batch GD: update per small batch (e.g., 32 samples).
  - Iterate for many epochs; loss typically decreases.

## Key concepts taught — ADALINE

- **ADALINE — ADAptive LInear NEurons (slides 14/29 onward):** model for binary classification using a linear decision boundary.
  - **Net input:** `z = w^⊤ x + b`
  - **ADALINE rule (slide 17/29):**
    1. Initialize weights and bias.
    2. Compute net input: `z^{(i)} = w^⊤ x^{(i)} + b`
    3. Compute error in training step: `y^{(i)} − z^{(i)}`, where `y^{(i)} ∈ {0, 1}` but `z^{(i)} ∈ ℝ`.
    - For `y^{(i)} = 1`: push z towards 1; for `y^{(i)} = 0`: push z towards 0.
  - **Threshold function for prediction (slide 18/29):**
    - `ŷ^{(i)} = 1 if z^{(i)} ≥ 0.5;  ŷ^{(i)} = 0 if z^{(i)} < 0.5`
    - **The threshold function is ONLY used to make the final prediction** (capitalized emphasis on slides).
  - **MSE loss function (slide 19/29):**
    - `L(w, b) = (1/2n) Σ_{i=1}^{n} (y^{(i)} − z^{(i)})²`
    - Residuals computed between observations and net inputs (not predictions).
  - **Gradient descent update rule (slide 20/29):**
    - `w = w + Δw`  with  `Δw = −η ∇_w L(w, b)`
    - `b = b + Δb`  with  `Δb = −η ∇_b L(w, b)`
    - **Weight gradient:**
      - `∇_w L(w, b) = (∂L/∂w_1, ∂L/∂w_2, …, ∂L/∂w_m)^⊤`  with  `∂L/∂w_j = −(1/n) Σ_{i=1}^{n} (y^{(i)} − z^{(i)}) x^{(i)}_j`
    - **Bias gradient:**
      - `∇_b L(w, b) = ∂L/∂b = −(1/n) Σ_{i=1}^{n} (y^{(i)} − z^{(i)})`

## Key concepts taught — Logistic Regression (from gradient descent perspective)

- **Slide 21/29:** logistic regression section header (visual only, with diagram).
- **Sigmoid function (slide 23/29):**
  - `σ(z) := 1 / (1 + e^{−z})` where `z = w^⊤ x + b` is the net input.
  - σ(0) = 0.5; σ: ℝ → [0, 1].
- **Threshold function (slide 24/29):** same as ADALINE but applied to σ(z):
  - `ŷ = 1 if σ(z) ≥ 0.5;  ŷ = 0 if σ(z) < 0.5`
- **Cross-entropy loss (slide 25/29):**
  - `L(w, b) := Σ_{i=1}^{n} [ −y^{(i)} log(σ(z^{(i)})) − (1 − y^{(i)}) log(1 − σ(z^{(i)})) ]`
  - For n = 1, a single example: `L = −y^{(1)} log(σ(z^{(1)})) − (1 − y^{(1)}) log(1 − σ(z^{(1)}))`
  - Which simplifies to: `L = −log(σ(z^{(1)})) if y^{(1)} = 1;  L = −log(1 − σ(z^{(1)})) if y^{(1)} = 0`
- **Cross-entropy explained (slide 26/29):** loss → 0 for correct prediction; loss → ∞ for wrong prediction. Wrong predictions penalized with increasingly larger loss.
- **Logistic regression from ADALINE (slides 27/29 and 28/29):**
  - Logistic regression = ADALINE with two changes:
    1. Substitute MSE loss → cross-entropy loss.
    2. Change linear activation → sigmoid function.
  - **Key result:** the gradient for the optimization step is **unchanged** from ADALINE:
    - `∂L/∂w_j = −(1/n) Σ_{i=1}^{n} (y^{(i)} − σ(z^{(i)})) x^{(i)}_j`
    - `∂L/∂b = −(1/n) Σ_{i=1}^{n} (y^{(i)} − σ(z^{(i)}))`
  - Same weight/bias update formulas as ADALINE (slides 20/29 and 28/29 are structurally identical, with `z^{(i)}` replaced by `σ(z^{(i)})`).
- **Coding demo (slide 29/29):** build a simple automatic differentiation engine from scratch to illustrate backpropagation in action: builds computational graph during forward pass, computes gradients via backprop in backward pass, uses it to optimize a small example by gradient descent.

## Notation

- `η` = learning rate (gradient descent step size); also `α` used on slide 12/29 for the same quantity — **symbol collision within lecture**: slides 3/29 use η, slide 12/29 uses α for the update rule. Note in notation_table.
- `w` = weight vector; `b` = bias scalar; `z` = net input (`w^⊤ x + b`); `σ(·)` = sigmoid function.
- `L` = loss (scalar); `∇_w L` = gradient vector of loss w.r.t. weights; `∂L/∂w_j` = j-th component.
- `θ` = combined parameter vector (all weights and biases).
- `b^{(1)}_k` = bias for hidden unit k in layer 1; `b^{(2)}` = output layer bias.
- `β_k` = output-layer weights (connecting hidden unit k to output); `w_{kj}` = hidden-layer weight (input j to hidden unit k).
- `σ'(z_k)` = derivative of sigmoid at `z_k`.

## R9 cross-check flags (vs ESL/ISL)

- Cross-entropy loss: slide writes summed (not averaged) cross-entropy `Σ_i [...]`. Some references use `(1/n) Σ_i [...]`. [VERIFY: sklearn's `LogisticRegression` uses averaged log-loss; confirm whether slide convention matches or differs.]
- ADALINE vs. Logistic gradient equivalence: the claim that the gradient step is "unchanged" after switching from MSE+linear to cross-entropy+sigmoid is a non-trivial result. [VERIFY: the derivation is left as an "optional exercise at home" on slide 27/29 — confirm correctness: ∂L_{CE}/∂w_j = −(1/n) Σ(y^{(i)} − σ(z^{(i)})) x^{(i)}_j, which matches ADALINE's ∂L_{MSE}/∂w_j structurally.]
- MSE loss on ADALINE: slide uses `(1/2n) Σ (y − z)²` — matches sklearn convention with the `1/(2n)` normalization (same as Lecture 3 regularized regression).
- Backprop formula for hidden weights: `∂L/∂w_{kj} = −(y − f_θ(x)) · β_k · σ'(z_k) · x_j` — matches ESL §11.4 derivation.

## Professor emphasis cues

- "REMEMBER THAT THE THRESHOLD FUNCTION IS ONLY USED TO MAKE THE FINAL PREDICTION!" — all-caps on slide 18/29, extremely strong exam signal.
- The equivalence of ADALINE and logistic regression gradients is flagged as an "optional exercise at home" — suggests the prof considers it derivable from course material; may appear on exam.
- Mini-batch gradient descent described as "most common in practice" — exam context answers should prefer mini-batch.
- The "from first principles" approach (slide 2/29) and the live coding demo of an autodiff engine signals the prof wants conceptual depth, not just formula recall.
- Learning rate η controls contribution of each weak learner in boosting (Lecture 7) AND step size in gradient descent here — same symbol, same role.

## Companion materials

Live coding session: building an automatic differentiation engine from scratch (mentioned on slides 2/29 and 29/29). No notebook filename given on slides. Example task: logistic regression (mentioned on slide 2/29 plan).

## Cross-refs

→ `methods/gradient_descent.qmd`, `methods/backpropagation.qmd`, `methods/logistic_regression.qmd` (cross-entropy loss from first principles), `methods/neural_networks.qmd` (Lecture 9 builds on this), `methods/regularization.qmd` (L2 in neural networks, Lecture 9).
