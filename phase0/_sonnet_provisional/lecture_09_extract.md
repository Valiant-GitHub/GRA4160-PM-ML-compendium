# Lecture 9 — Neural Networks

**Source:** `lecture9_260521_184938.pdf` (43 PDF pages = 17 logical slides; cite `n/17`). Date: March 4, 2026.

## Topic
Feed-forward neural networks: architecture (single hidden layer), forward pass as matrix operations, activation functions (sigmoid, ReLU, softmax); fitting via nonlinear least squares and gradient descent; backpropagation chain-rule derivation; stochastic gradient descent; L2 regularization and dropout in neural networks.

## Key concepts taught — Introduction to Neural Networks

- **Definition (slide 3/17):** computational system inspired by the structure and function of the human brain. Consists of many interconnected processing units (neurons) that work together to solve complex problems. Each neuron receives input, processes it, and produces output sent to other neurons. By combining and processing information, a network can learn to recognize patterns and make predictions.

## Key concepts taught — Single-Layer (Feed-Forward) Neural Network

- **Architecture (slide 4/17):** a model taking p input variables `X = (X_1, X_2, …, X_p)`. The NN is a nonlinear function `f(X)` mapping inputs to output Y. Weights `w_1, w_2, …, w_p` are parameters controlling the strength of relationships. Given nonlinear activation function `g(z)`, the model is written as a sum of K activation functions:
  - `Y = f(X) = β_0 + Σ_{k=1}^{K} β_k g(w_{k0} + Σ_{j=1}^{p} w_{kj} X_j)`
- **Network diagram (slide 5/17):** input layer (X_1, …, X_p, +1 bias); hidden layer (A_1, …, A_K, +1 bias); output Y.
  - Hidden unit weights: `w_{kj}` (input j to hidden unit k); bias `w_{k0}`.
  - Output layer weights: `β_k`; bias `β_0`.
  - Activation: `A_k = g(w_{k0} + Σ_{j=1}^{p} w_{kj} X_j)`
- **Two-step construction (slide 6/17):**
  1. **Hidden layer:** specify K hidden units and activation function `g(X)`. Compute K activations:
     - `A_k = h_k(X) = g(w_{k0} + Σ_{j=1}^{p} w_{kj} X_j)`
  2. **Output layer:** K activations feed into a linear combination:
     - `f(X) = β_0 + Σ_{k=1}^{K} β_k A_k`

## Key concepts taught — Forward Pass as Matrix Multiplications

- **Matrix form (slides 7/17):** let X be the n × p input matrix.
  1. **Hidden layer:** compute n × K activation matrix:
     - `A_{ik} = g(w_{k0} + Σ_{j=1}^{p} w_{kj} X_{ij})`  equivalently:  `A = g(1_n w_0^⊤ + X W^⊤)`
     - where W is the K × p weight matrix; `w_0` is the K × 1 bias vector; `g(·)` applied element-wise.
  2. **Output layer:** compute n × 1 output vector:
     - `Y_i = β_0 + Σ_{k=1}^{K} β_k A_{ik}`  equivalently:  `Y = β_0 1_n + A β`
     - where `β = (β_1, …, β_K)^⊤`
  - **Key insight (slide 7/17):** each layer = matrix multiplication followed by activation function. This is what makes neural networks efficient to compute on GPUs.

## Key concepts taught — Activation Functions

- **Common activation functions (slides 8/17 and 9/17):**
  1. **Sigmoid:**  `g(z) = 1 / (1 + e^{−z})`  (same as used for logistic regression)
  2. **ReLU (Rectified Linear Unit):**  `g(z) = max(0, z)`
  3. **Softmax** (output layer for multi-class classification):
     - `g(z_i) = exp(z_i) / Σ_{j=1}^{K} exp(z_j)`
- **Slide 9/17:** Sigmoid vs ReLU comparison diagram. Source: *An Introduction to Statistical Learning, James, Witten, Hastie and Tibshirani, 2021* (ISL, 2nd edition).
- **Slide 8/17:** empty slide (visual/diagram only, no additional text).

## Key concepts taught — Fitting a Neural Network

- **Objective — nonlinear least squares (slides 10/17):**
  - `min_{w_k, β} (1/2) Σ_{i=1}^{n} (y_i − f(x_i))²`
  - where `f(X) = β_0 + Σ_{k=1}^{K} β_k g(w_{k0} + Σ_{j=1}^{p} w_{kj} X_j)`
  - This is a **nonlinear optimization problem** (not straightforward to solve analytically).
- **Gradient descent for NN (slides 11/17 and 12/17):** collect all parameters in vector θ; define cost:
  - `L(θ) = (1/2) Σ_{i=1}^{n} (y_i − f_θ(x_i))²`
  - **Iterative update:** let θ_t be parameters at iteration t with initial guess θ_0.
    - Update: `θ_{t+1} = θ_t + δ` where δ is a small change such that `L(θ_{t+1}) < L(θ_t)`.
    - Repeat until convergence.
  - **How to compute δ (slide 12/17):** compute gradient:
    - `∇L(θ_t) = ∂L(θ)/∂θ |_{θ=θ_t} = ∂/∂θ [(1/2) Σ_{i=1}^{n} (y_i − f_θ(x_i))²] |_{θ=θ_t}`
    - The gradient gives the directions to move θ to decrease L(θ).
  - **Update rule with learning rate η:**
    - `θ_{t+1} = θ_t − η ∇L(θ_t)`
  - **Gradient vector:** `∇L = [∂L/∂β_k , ∂L/∂w_{kj}]^⊤`

## Key concepts taught — Backpropagation (Neural Network derivation)

- **Setup (slides 13/17):** for observation i, define intermediate quantities:
  - `z_{ik} = w_{k0} + Σ_{j=1}^{p} w_{kj} x_{ij}` (pre-activation for hidden unit k)
  - `A_{ik} = g(z_{ik})` (activation for hidden unit k)
  - `f_i = β_0 + Σ_{k=1}^{K} β_k A_{ik}` (network output)
  - `L_i = (1/2)(y_i − f_i)²` (loss for observation i)
  - Total loss: `L(θ) = Σ_{i=1}^{n} L_i`. Need `∂L_i/∂β_k` and `∂L_i/∂w_{kj}`.
  - **Key idea:** apply chain rule layer by layer, working backwards from loss to parameters.
- **Chain rule derivation (slides 14/17):**
  - **Step 1:** derivative of loss w.r.t. output:  `∂L_i/∂f_i = −(y_i − f_i)`
  - **Step 2:** derivative w.r.t. output-layer weights β_k (one link):
    - `∂L_i/∂β_k = (∂L_i/∂f_i) · (∂f_i/∂β_k) = −(y_i − f_i) · A_{ik}`
  - **Step 3:** derivative w.r.t. hidden-layer weights w_{kj} (three links):
    - `∂L_i/∂w_{kj} = (∂L_i/∂f_i) · (∂f_i/∂A_{ik}) · (∂A_{ik}/∂z_{ik}) · (∂z_{ik}/∂w_{kj})`
    - Substituting: `= −(y_i − f_i) · β_k · g'(z_{ik}) · x_{ij}`
  - **Interpretation:** each parameter receives a fraction of the residual `(y_i − f_i)` propagated back through the network — hence "backpropagation."

## Key concepts taught — Stochastic Gradient Descent

- **SGD (slides 15/17):** method for minimizing cost function by iteratively moving in the direction of steepest descent (negative gradient). Contrasts with batch GD (computes gradient over entire dataset): SGD computes gradient on **a single training example at a time**. Much faster than batch GD but much noisier. Noise can be reduced by averaging gradients over several training examples (→ mini-batch).

## Key concepts taught — Regularization in Neural Networks

- **L2 regularization (slide 16/17):** adds penalty term to cost function to prevent overfitting. Penalty proportional to sum of squares of weights. Regularized cost:
  - `L_reg(θ) = (1/2) Σ_{i=1}^{n} (y_i − f_θ(x_i))² + λ Σ_{k,j} w_{kj}²`
  - where λ is the regularization parameter controlling the trade-off between fit and complexity.
  - Note: penalty applies to hidden-layer weights `w_{kj}` only (not output weights `β_k` or biases, though slide uses `Σ_{k,j} w_{kj}²` notation without explicit exclusion).
- **Dropout (slide 17/17):** regularization technique that randomly sets some neurons to zero during training.
  - Prevents overfitting to training data.
  - Can be seen as an **ensemble** of all networks formed by dropping different neurons.
  - **Typical dropout rates: 0.2 to 0.5.**
  - Dropout applied **only during training**, not at inference time.

## Notation

- `K` = number of hidden units; `p` = number of input features; `n` = number of observations.
- `w_{kj}` = weight connecting input j to hidden unit k; `w_{k0}` = bias for hidden unit k.
- `β_k` = weight connecting hidden unit k to output; `β_0` = output bias.
- `A_k` (or `A_{ik}`) = activation of hidden unit k (for observation i).
- `z_{ik}` = pre-activation (linear combination before applying activation function) for hidden unit k, observation i.
- `g(·)` = activation function; `g'(·)` = its derivative.
- `θ` = full parameter vector `[β_k, w_{kj}]^⊤`; `η` = learning rate.
- `λ` = L2 regularization parameter (same symbol as regularization in Lecture 3 — consistent use).
- **Notation conflict:** `w_{kj}` in this lecture corresponds to the same conceptual quantity as in Lecture 8's backprop derivation, but Lecture 8 uses `b^{(1)}_k` for bias (layer notation) while Lecture 9 uses `w_{k0}` (absorbing bias into weight notation). Note in notation_table.
- `1_n` = n-dimensional column vector of ones (matrix formula in slide 7/17).
- `W` = K × p weight matrix (capital W = matrix; lowercase w = vector or scalar).

## R9 cross-check flags (vs ESL/ISL)

- Single-layer NN formula: `f(X) = β_0 + Σ_k β_k g(w_{k0} + Σ_j w_{kj} X_j)` — matches ISL §10.1 Equation 10.1 exactly (same notation). Source cited on sigmoid vs ReLU slide 9/17 = ISL (2021).
- Backprop derivation: `∂L_i/∂w_{kj} = −(y_i − f_i) β_k g'(z_{ik}) x_{ij}` — matches ESL §11.4 Equation 11.16 (modulo sign convention). [VERIFY: ESL uses a "delta" notation δ_k for intermediate error signals — confirm whether prof uses delta notation in any companion material.]
- L2 regularization on NN: `L_reg = (1/2) Σ(y_i − f_θ)² + λ Σ_{k,j} w_{kj}²` — matches ISL §10.7 "weight decay" formulation. [VERIFY: whether λ convention matches sklearn's `alpha` parameter in `MLPClassifier`/`MLPRegressor`.]
- Softmax formula `g(z_i) = exp(z_i) / Σ_j exp(z_j)`: standard. [VERIFY: denominator sum is over K classes; slide indices `j=1 to K` — confirm notation consistency.]
- Dropout rates 0.2–0.5: practical guideline; not in ESL (older than dropout). [VERIFY: ISL 2nd ed. §10.7.2 covers dropout — check whether rate range matches slide.]

## Professor emphasis cues

- "Each layer is a matrix multiplication followed by an activation function — this is what makes neural networks efficient to compute on GPUs." (slide 7/17) — explicit conceptual emphasis, likely exam-relevant framing.
- ISL explicitly cited as the source for sigmoid vs ReLU comparison diagram (slide 9/17) — signals that ISL Chapter 10 is required reading.
- The backpropagation derivation on slide 14/17 is fully worked out step-by-step with labeled partial derivatives — treat as exam derivation target (mirrors the three-step chain rule derivation style).
- Dropout applied "only during training, not at inference time" — specific operational detail, commonly tested.
- Typical dropout rates 0.2–0.5 given as a specific quantitative guideline.

## Companion materials

No specific notebook filenames named on slides. Plan for Today (slide 2/17) covers: Introduction to neural networks, Single layer (feed forward) neural networks, Fitting a Neural Network.

## Cross-refs

→ `methods/neural_networks.qmd`, `methods/backpropagation.qmd` (chain rule from Lecture 8), `methods/gradient_descent.qmd` (Lecture 8 foundations), `methods/regularization.qmd` (L2 / weight decay), `methods/logistic_regression.qmd` (sigmoid link), `methods/traditional_vs_dl.qmd` (Lecture 10 context).
