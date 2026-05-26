# Notebook extract: 18_NN_with_PyTorch.ipynb

**Source path:** course_materials/Lecture notebooks/18_NN_with_PyTorch.ipynb
**Cell count:** 9 cells (ids cell-0 ... cell-8; cell-3 and cell-8 empty).

## Dataset(s) loaded
- **Iris dataset** from scikit-learn (cell-2): `from sklearn import datasets`; `iris = datasets.load_iris()`; `X = iris.data` shape (150, 4); `y = iris.target` shape (150,), 3 classes (0,1,2).
- (Imports in cell-1: `torch`, `numpy as np`, `from sklearn import datasets`, `from sklearn.model_selection import train_test_split`, `from sklearn.preprocessing import StandardScaler`.)

## Preprocessing steps
- [cell-2] Seeds: `np.random.seed(42)`, `torch.manual_seed(42)`.
- [cell-2] Standardize: `scaler = StandardScaler(); X = scaler.fit_transform(X)`.
- [cell-2] To tensors: `X = torch.tensor(X, dtype=torch.float32)`; `y = torch.tensor(y, dtype=torch.long)`.
- [cell-2] Split: `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)` → 120 train / 30 test.

## Method(s) demonstrated
- **2-layer feed-forward neural network in PyTorch — but with MANUALLY managed parameters and a hand-written forward/softmax/cross-entropy** (PyTorch tensors + autograd, NOT `nn.Module`). This is the "PyTorch low-level" style.
- **Architecture** (cell-4): input_size=4, hidden_size=8, output_size=3. One hidden layer.
  - Parameters manually created with `requires_grad=True`:
    - `W1 = torch.randn(input_size, hidden_size, ...)` (4x8), `b1 = torch.randn(hidden_size, ...)` (8,)
    - `W2 = torch.randn(hidden_size, output_size, ...)` (8x3), `b2 = torch.randn(output_size, ...)` (3,)
    - all `dtype=torch.float32, requires_grad=True`.
  - **Forward** (cell-5, `forward(x)`): `z1 = torch.mm(x, W1) + b1`; `a1 = z1.clamp(min=0)` (**ReLU** via clamp); `logits = torch.mm(a1, W2) + b2`.
  - **`softmax(logits)`** (cell-5): `exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0]); probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)` (max-subtraction for numerical stability).
  - **`cross_entropy_loss(logits, y_true)`** (cell-5): `probs = softmax(logits); n_samples = logits.shape[0]; correct_logprobs = -torch.log(probs[range(n_samples), y_true]); loss = torch.sum(correct_logprobs) / n_samples`.
  - **Optimizer**: manual SGD inside `with torch.no_grad():` (cell-6): `W1 -= learning_rate * W1.grad` (and b1, W2, b2), then `W1.grad.zero_()` etc. No `torch.optim`.
- **Activation**: ReLU (hidden), softmax (output, inside loss). **Loss**: manual multi-class cross-entropy. **Batch**: full-batch (entire `X_train` per epoch — no DataLoader/mini-batches).

## Hyperparameters set
- `input_size = 4`, `hidden_size = 8`, `output_size = 3` (cell-4).
- `learning_rate = 0.01` (cell-6).
- `num_epochs = 500` (cell-6).
- Batch size: full batch (all 120 training rows each step).
- `test_size=0.2`, `random_state=42` (split); seeds 42 (numpy + torch).
- Param init: `torch.randn` (standard normal), no manual scaling.

## Plots produced
- **None.** No matplotlib. Output is printed loss every 50 epochs and final test accuracy.

## What is left as an exercise to the student
- No explicit exercise/TODO. cell-0 markdown states the goal is to keep it simple and not use advanced PyTorch features. (Full self-contained worked example.)

## Key cell indices for code idiom extraction
- "[cell-2]: sklearn → torch pipeline — `StandardScaler().fit_transform`, `torch.tensor(..., dtype=torch.float32/long)`, `train_test_split(test_size=0.2, random_state=42)`"
- "[cell-4]: manual parameter tensors `torch.randn(in,out, requires_grad=True)` for W1/b1/W2/b2"
- "[cell-5]: hand-written forward `z1 = torch.mm(x,W1)+b1; a1 = z1.clamp(min=0); logits = torch.mm(a1,W2)+b2` (ReLU via clamp)"
- "[cell-5]: numerically-stable softmax `torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0]) / sum(...)`"
- "[cell-5]: manual cross-entropy `-torch.log(probs[range(n), y_true])` summed / n"
- "[cell-6]: manual SGD with `torch.no_grad()` block, `W -= lr*W.grad`, then `W.grad.zero_()`"
- "[cell-7]: eval with `torch.no_grad()`, `torch.argmax(probs, dim=1)`, accuracy `(predictions == y_test).float().mean()`"

## Notes / [VERIFY] flags
- Softmax/cross-entropy formulas transcribed: softmax = exp(z_i - max z) / sum exp(z - max z); cross-entropy = -(1/N) sum log p[correct class]. Matches code and standard definitions.
- ReLU implemented as `z1.clamp(min=0)` (no `nn.ReLU` / `torch.relu`).
- This notebook does NOT use `nn.Module`, `nn.Linear`, or `torch.optim` — contrast with nb 19 which uses all three. Parameters and the optimization step are fully manual.
- [VERIFY] No saved execution outputs in the provided cells for the loss prints or final "Test Accuracy" line — accuracy value not captured. (cf. nb 19 which DOES have saved accuracy outputs.)
- Seeds set (42 for both numpy and torch) → training is reproducible.
