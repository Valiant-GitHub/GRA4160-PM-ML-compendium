# Notebook extract: 18_NN_with_PyTorch.ipynb

**Source path:** `Lecture notebooks/18_NN_with_PyTorch.ipynb`
**Cell count:** 8 cells (markdown + code, plus one empty)

---

## Dataset(s) loaded

- **Iris dataset** — loaded via `sklearn.datasets.load_iris()` (cell `cell-2`).
  - `X = iris.data` — shape: (150, 4); 4 features.
  - `y = iris.target` — shape: (150,); 3 classes (0, 1, 2).
  - After preprocessing: converted to `torch.float32` (X) and `torch.long` (y).

---

## Preprocessing steps

- **Standardisation** (cell `cell-2`):  
  `scaler = StandardScaler()` ; `X = scaler.fit_transform(X)`  
  (fit and transform on full dataset — no separate scaler fit on train only)
- **Seeds** (cell `cell-2`):  
  `np.random.seed(42)` ; `torch.manual_seed(42)`
- **Tensor conversion** (cell `cell-2`):  
  `X = torch.tensor(X, dtype=torch.float32)` ; `y = torch.tensor(y, dtype=torch.long)`
- **Train/test split** (cell `cell-2`):  
  `train_test_split(X, y, test_size=0.2, random_state=42)` → 120 train, 30 test

---

## Method(s) demonstrated

**PyTorch — manually managed weights (no `nn.Module`), 2-layer MLP**

This notebook does **not** use `nn.Module`, `nn.Linear`, or `torch.optim`. All parameters are raw tensors with `requires_grad=True`.

### Architecture (cell `cell-4`)

```python
input_size  = 4   # four Iris features
hidden_size = 8   # neurons in hidden layer
output_size = 3   # three Iris classes

W1 = torch.randn(input_size, hidden_size, dtype=torch.float32, requires_grad=True)
b1 = torch.randn(hidden_size, dtype=torch.float32, requires_grad=True)
W2 = torch.randn(hidden_size, output_size, dtype=torch.float32, requires_grad=True)
b2 = torch.randn(output_size, dtype=torch.float32, requires_grad=True)
```

**Layer structure:** input(4) → hidden(8, ReLU) → output(3, logits)

### Forward pass (cell `cell-5`)

```python
def forward(x):
    z1 = torch.mm(x, W1) + b1          # linear: (n, 4) @ (4, 8) → (n, 8)
    a1 = z1.clamp(min=0)                # ReLU activation
    logits = torch.mm(a1, W2) + b2     # linear: (n, 8) @ (8, 3) → (n, 3)
    return logits
```

**Activation:** ReLU implemented as `.clamp(min=0)` — not `torch.relu` or `nn.ReLU`.

### Loss (cell `cell-5`)

Manual softmax + cross-entropy:

```python
def softmax(logits):
    exp_logits = torch.exp(logits - torch.max(logits, dim=1, keepdim=True)[0])  # numerically stable
    probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)
    return probs

def cross_entropy_loss(logits, y_true):
    probs = softmax(logits)
    n_samples = logits.shape[0]
    correct_logprobs = -torch.log(probs[range(n_samples), y_true])
    loss = torch.sum(correct_logprobs) / n_samples
    return loss
```

### Training loop (cell `cell-6`)

```python
learning_rate = 0.01
num_epochs = 500
for epoch in range(num_epochs):
    logits = forward(X_train)
    loss = cross_entropy_loss(logits, y_train)
    loss.backward()
    with torch.no_grad():
        W1 -= learning_rate * W1.grad
        b1 -= learning_rate * b1.grad
        W2 -= learning_rate * W2.grad
        b2 -= learning_rate * b2.grad
    W1.grad.zero_(); b1.grad.zero_(); W2.grad.zero_(); b2.grad.zero_()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```

Manual gradient zeroing with `.grad.zero_()`.

### Evaluation (cell `cell-7`)

```python
with torch.no_grad():
    test_logits = forward(X_test)
    test_probs = softmax(test_logits)
    predictions = torch.argmax(test_probs, dim=1)
    accuracy = (predictions == y_test).float().mean()
```

---

## Hyperparameters set

| Parameter | Value |
|---|---|
| `input_size` | 4 |
| `hidden_size` | 8 |
| `output_size` | 3 |
| `learning_rate` | 0.01 |
| `num_epochs` | 500 |
| Weight init | `torch.randn(...)` (no explicit scale, unit normal) |
| Optimizer | vanilla gradient descent (manual, no momentum) |
| Loss | cross-entropy (manual softmax + NLL) |
| Activation | ReLU (`z.clamp(min=0)`) |

---

## Plots produced

None.

---

## What is left as an exercise to the student

Not explicitly stated. The notebook ends with the test accuracy printout (cell `cell-7`). The introductory markdown (cell `cell-0`) notes the intent to "keep the example simple" — more advanced PyTorch features (`nn.Module`, optimizers, DataLoader) are introduced in notebook 19.

---

## Key cell indices for code idiom extraction

- `[cell cell-4]`: Raw `torch.randn(..., requires_grad=True)` parameter initialisation without `nn.Module`.
- `[cell cell-5]`: Manual softmax with max-subtraction stability trick; manual NLL loss indexing `probs[range(n), y_true]`; ReLU as `.clamp(min=0)`.
- `[cell cell-6]`: Training loop pattern — `loss.backward()` → `with torch.no_grad(): param -= lr * param.grad` → `.grad.zero_()`.
- `[cell cell-2]`: `StandardScaler().fit_transform(X)` + `torch.tensor(..., dtype=...)` + `train_test_split`.

---

## Notes / [VERIFY] flags

- This is the **transition notebook** between scratch numpy (NB 14–17) and full PyTorch `nn.Module` (NB 19). Parameters are PyTorch tensors with autograd, but no high-level API is used.
- Weight init is `torch.randn` with no scale factor (unit normal σ=1) — unlike notebooks 14/15 which use `rgen.normal(scale=0.01)`. This could produce large initial logits for a hidden size of 8; [VERIFY: whether this causes any convergence issues in practice].
- The `StandardScaler` is fit on the **full** 150-sample dataset before splitting — technically a data leakage issue for real evaluation. [VERIFY: whether this is intentional for simplicity or an oversight].
- `sklearn.model_selection.train_test_split` and `sklearn.preprocessing.StandardScaler` are both imported (cell `cell-1`), making this notebook a hybrid sklearn-preprocessing + PyTorch-training setup.
- Loss printed every 50 epochs (not every epoch).
