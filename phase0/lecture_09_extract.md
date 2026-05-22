# Lecture 9 — Neural Networks

**Source:** `lecture9_260521_184938.pdf` (43 PDF pages = 17 logical slides; cite `n/17`. Page 21 is blank). Date: March 4th, 2026. Instructor: Vegard Høghaug Larsen.

## Topic
Neural networks: introduction, single-layer (feed-forward) networks with one hidden layer, the forward pass as matrix multiplications, activation functions, fitting via gradient descent, backpropagation, SGD, and regularization (L2 + dropout).

## Key concepts taught
- **Plan (slide 2/17):** intro to neural networks; single-layer (feed-forward) NNs; fitting a NN.
- **Intro to neural networks (slide 3/17):** computational system inspired by structure/function of the human brain; many interconnected processing units (neurons) working together; each neuron receives input from others, processes it, produces an output sent to other neurons; by combining/processing information, the network learns to recognize patterns and make predictions.
- **Single-layer feed-forward NN (slide 4/17):** takes p input variables `X = (X_1, X_2, ..., X_p)`; the NN is a nonlinear function `f(X)` mapping inputs to output Y; weights `w_1, w_2, ..., w_p` control the strength of relationships. Given a nonlinear activation `g(z)`, the model is a sum of K activation functions:
  - `Y = f(X) = β_0 + Σ_{k=1}^{K} β_k g( w_k0 + Σ_{j=1}^{p} w_kj X_j )`.
- **Network diagram (slide 5/17):** input layer X_1..X_p (+1 bias) → hidden layer A_1..A_K (+1 bias) → output Y. Weights `w_kj` (input→hidden), `β_k` (hidden→output), biases `w_k0`, `β_0`. `A_k = g( w_k0 + Σ_{j=1}^{p} w_kj X_j )`.
- **Hidden units and output layer (slide 6/17):** two steps — 1. specify number of hidden units K and activation g(X); compute K activations `A_k = h_k(X) = g( w_k0 + Σ_{j=1}^{p} w_kj X_j )`. 2. the K activations feed the output layer, a linear function of the activations: `f(X) = β_0 + Σ_{k=1}^{K} β_k A_k`.
- **Forward pass as matrix multiplications (slide 7/17):** let X be the n×p input matrix.
  - 1. Hidden layer (n×K activation matrix): `A_ik = g( w_k0 + Σ_{j=1}^{p} w_kj X_ij )  ⟺  A = g( 1_n w_0ᵀ + X Wᵀ )`, where W is the K×p weight matrix, w_0 is the K×1 bias vector, g(·) applied element-wise.
  - 2. Output layer (n×1 output vector): `Y_i = β_0 + Σ_{k=1}^{K} β_k A_ik  ⟺  Y = β_0 1_n + A β`, where `β = (β_1, ..., β_K)ᵀ`.
  - Each layer = matrix multiplication followed by an activation function (efficient on GPUs).
- **Activation functions (slide 8/17):** introduce non-linearity, enable learning complex patterns.
  - 1. Sigmoid: `g(z) = 1 / (1 + e^{−z})` (same as logistic regression).
  - 2. ReLU (Rectified Linear Unit): `g(z) = max(0, z)`.
  - 3. Softmax (output layer, multi-class): `g(z_i) = exp(z_i) / Σ_{j=1}^{K} exp(z_j)`.
- **(slide 9/17):** Sigmoid vs ReLU figure. Source explicitly cited: "An Introduction to Statistical Learning, James, Witten, Hastie and Tibshirani, 2021" = **ISL**.
- **Fitting a Neural Network (slide 10/17):** fit parameters by minimizing a cost function (nonlinear least squares):
  - `min_{w_k, β}  ½ Σ_{i=1}^{n} (y_i − f(x_i))²` where `f(X) = β_0 + Σ_{k=1}^{K} β_k g( w_k0 + Σ_{j=1}^{p} w_kj X_j )`.
  - This is a nonlinear optimization problem, not straightforward to solve.
- **Gradient descent (slide 11/17):** put all parameters in vector θ; cost `L(θ) = ½ Σ_{i=1}^{n} (y_i − f_θ(x_i))²`. Iterative update: 1. start with initial guess θ_0; 2. update `θ_{t+1} = θ_t + δ`, where δ reflects a small change such that `L(θ_{t+1}) < L(θ_t)`; 3. repeat until convergence.
- **What is δ (slide 12/17):** compute the gradient of the cost w.r.t. θ:
  - `∇L(θ_t) = ∂L(θ)/∂θ |_{θ=θ_t} = ∂/∂θ [ ½ Σ_{i=1}^{n} (y_i − f_θ(x_i))² ] |_{θ=θ_t}`.
  - `∇L(θ_t)` gives directions to move θ to decrease L(θ). With learning rate η: `θ_{t+1} = θ_t − η ∇L(θ_t)`.
  - The gradient packs all partials: `∇L = [ ∂L/∂β_k , ∂L/∂w_kj ]ᵀ`.
- **Backpropagation: setup (slide 13/17):** intermediate quantities for observation i:
  - `z_ik = w_k0 + Σ_{j=1}^{p} w_kj x_ij` (pre-activation for hidden unit k); `A_ik = g(z_ik)` (activation); `f_i = β_0 + Σ_{k=1}^{K} β_k A_ik` (network output); `L_i = ½ (y_i − f_i)²` (loss for obs i).
  - Total loss `L(θ) = Σ_{i=1}^{n} L_i`. Need `∂L_i/∂β_k` and `∂L_i/∂w_kj`. Key idea: apply chain rule layer by layer, working backwards from the loss.
- **Backpropagation: the chain rule (slide 14/17):**
  - Step 1: `∂L_i/∂f_i = −(y_i − f_i)`.
  - Step 2 (output-layer weights β_k): `∂L_i/∂β_k = (∂L_i/∂f_i)·(∂f_i/∂β_k) = −(y_i − f_i) · A_ik` (since ∂f_i/∂β_k = A_ik).
  - Step 3 (hidden-layer weights w_kj, three links): `∂L_i/∂w_kj = (∂L_i/∂f_i)·(∂f_i/∂A_ik)·(∂A_ik/∂z_ik)·(∂z_ik/∂w_kj) = −(y_i − f_i) · β_k · g'(z_ik) · x_ij` (links: −(y_i−f_i), β_k, g'(z_ik), x_ij).
  - Each parameter receives a fraction of the residual `(y_i − f_i)` propagated back — hence "backpropagation."
- **Stochastic gradient descent (slide 15/17):** minimizes cost by moving in direction of steepest descent (negative gradient); unlike batch GD (gradient over entire dataset), SGD computes the gradient on a single training example at a time; faster but noisier; noise reduced by averaging gradients over several examples.
- **L2 regularization (slide 16/17):** adds a penalty proportional to the sum of squares of the weights to prevent overfitting:
  - `L_reg(θ) = ½ Σ_{i=1}^{n} (y_i − f_θ(x_i))² + λ Σ_{k,j} w_kj²`, where λ controls the fit-vs-complexity trade-off.
- **Dropout (slide 17/17):** randomly drops out (sets to zero) some neurons during training to prevent overfitting; can be seen as an ensemble of all networks formed by dropping different neurons; typical dropout rates **between 0.2 and 0.5**; applied only during training, not at inference.

## Notation
- `K` = number of hidden units (capital K). **Collision warning:** Lecture 5 used lowercase `k` for #parameters/#folds, Lecture 6 used `k` for #clusters, Lecture 8 used `k` to index hidden units. Here capital `K` = hidden-unit count; lowercase `k` = the hidden-unit index (k = 1..K). Flag the k/K family in notation_table.
- `p` = number of input variables/features (X has p columns). Consistent with Lecture 7's predictor count; conflicts with Lecture 4's probability `p`.
- `β_0, β_k` = output-layer bias and weights; `w_k0, w_kj` = hidden-unit bias and input→hidden weights; `θ` = all parameters stacked.
- `A_k`/`A_ik` = activation of hidden unit k (for obs i); `z_ik` = pre-activation; `g(·)` = generic activation; `g'(z_ik)` = its derivative.
- `η` = learning rate (slide 12) — **note: Lecture 8 used both η and α for this; Lecture 9 settles on η.** `λ` = L2 regularization strength (slide 16).
- `δ` = small parameter-change vector (slide 11) — NOT the backprop error-signal δ used in some texts; here δ is just the update step. Flag potential confusion.
- `1_n` = column vector of ones (length n) used in matrix forward pass.

## R9 cross-check flags (vs ESL/ISL)
- Single-hidden-layer model `f(X) = β_0 + Σ_k β_k g(w_k0 + Σ_j w_kj X_j)`: directly matches ISL §10.1 (ISL Eq. 10.1 / 10.2). **Slide 9 explicitly cites ISL (James, Witten, Hastie, Tibshirani, 2021) for the activation-function figure — record ISL Ch.10 as the primary source for this lecture.**
- Backprop gradients `∂L_i/∂β_k = −(y_i−f_i)A_ik` and `∂L_i/∂w_kj = −(y_i−f_i)β_k g'(z_ik)x_ij`: cross-check against ESL §11.4 (back-propagation equations 11.12–11.13). Structure agrees. **Flag: confirm sign/δ convention matches ESL.**
- L2 regularization `+ λ Σ_{k,j} w_kj²` (penalizes hidden-layer weights only as written): ESL §11.5 weight decay penalizes all weights. **Flag: slide sums only over `w_kj` (hidden weights), not β — verify whether output weights β are intended to be penalized too.** [VERIFY: whether L2 sum should include β_k.]
- Softmax `g(z_i) = exp(z_i)/Σ_j exp(z_j)`: standard, matches ISL §10.x. Agree.
- Cost uses `½ Σ (y−f)²` (squared-error / nonlinear least squares for regression NN) — ISL uses squared error for quantitative response. Agree.

## Professor emphasis cues
- Forward pass framed explicitly as matrix multiplication "this is what makes neural networks efficient to compute on GPUs" (slide 7).
- Backprop chain rule annotated link-by-link with underbraces (each factor labeled) — strong pedagogical scaffolding (slide 14); same three-link structure as Lecture 8.
- "Each parameter receives a fraction of the residual propagated back — hence the name backpropagation" (slide 14 punchline).
- Two regularization tools highlighted: L2 weight penalty AND dropout (0.2–0.5, train-only).
- Direct continuity with Lecture 8 (same loss ½(y−f)², same backprop derivation, now with generic g).

## Companion materials
No lecture-notebook or exercise filenames printed on slides. Cited textbook figure: ISL (James, Witten, Hastie, Tibshirani, 2021), Sigmoid vs ReLU (slide 9).

## Cross-refs
→ `methods/neural_networks.qmd` (architecture, forward pass, fitting, backprop, dropout/L2), `methods/backprop_gradient_descent.qmd` (Lecture 8 — shared chain-rule derivation), `methods/activation_functions.qmd` (sigmoid/ReLU/softmax). L2 penalty cross-links `methods/regularization.qmd` (Lecture 3). SGD links `methods/backprop_gradient_descent.qmd`. Feeds Lecture 10 "Traditional ML vs Deep Learning" (NN vs traditional comparisons). Sigmoid + softmax cross-link `methods/logistic.qmd`.
