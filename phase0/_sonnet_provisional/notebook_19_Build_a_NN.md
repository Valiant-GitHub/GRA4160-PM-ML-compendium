# Notebook extract: 19_Build_a_NN.ipynb

**Source path:** `Lecture notebooks/19_Build_a_NN.ipynb`
**Cell count:** 30 cells (markdown + code, includes markdown pseudocode cell)

---

## Dataset(s) loaded

- **MNIST** — loaded via `torchvision.datasets.MNIST` (cell `cell-8`):
  ```python
  train_dataset = torchvision.datasets.MNIST(root='./data', train=True,  transform=transform, download=True)
  test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
  ```
  - 60,000 training images, 10,000 test images.
  - Images: 28×28 pixels, grayscale, flattened to 784-dim vectors in the training loop.
  - Classes: 10 (digits 0–9).

---

## Preprocessing steps

- **Transform** (cell `cell-8`): `transform = transforms.ToTensor()` — converts PIL image to float tensor, scales pixels to [0, 1].
- **Flattening** in training loop (cells `cell-12`, `cell-16`, etc.): `images = images.view(images.size(0), -1)` — reshapes `(batch, 1, 28, 28)` → `(batch, 784)`.
- **No additional normalisation** (no mean/std subtraction beyond [0,1] scaling).
- **DataLoader** (cell `cell-8`):
  - `batch_size = 32`
  - `train_loader`: `shuffle=True`
  - `test_loader`: `shuffle=False`

---

## Method(s) demonstrated

### Method 1 — PyTorch `nn.Module`: `SingleLayerNet` (cell `cell-2`)

- **Architecture:** input(784) → output(10) — single `nn.Linear(784, 10)`, no hidden layers (logistic regression).
- **Activation:** none in forward (raw logits; softmax inside `CrossEntropyLoss`).

### Method 2 — PyTorch `nn.Module`: `TwoHiddenLayerNet` (cell `cell-2`)

```python
class TwoHiddenLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, activation_fn=nn.ReLU):
        ...
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, output_size)
        self.activation = activation_fn()
    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        return self.output_layer(x)
```

**Architecture (instantiated values, cell `cell-2`):**
- `input_size = 784` (28×28 flattened MNIST)
- `hidden_size1 = 128`
- `hidden_size2 = 64`
- `output_size = 10`
- Activation: `nn.ReLU` (default; `nn.Tanh` and `nn.Sigmoid` mentioned as alternatives)
- Output: raw logits (no activation on output layer)

### Method 3 — PyTorch `nn.Module`: `TwoHiddenLayerNetWithDropout` (cell `cell-24`)

Same architecture as `TwoHiddenLayerNet` (784→128→64→10, ReLU) with **manually implemented inverted dropout** on both hidden layers:

```python
class TwoHiddenLayerNetWithDropout(nn.Module):
    def __init__(self, ..., dropout_prob=0.5):
        ...
        self.dropout_prob = dropout_prob
    def forward(self, x):
        x = self.activation(self.hidden1(x))
        if self.training:
            mask = (torch.rand_like(x) > self.dropout_prob).float()
            x = x * mask / (1.0 - self.dropout_prob)   # inverted dropout
        x = self.activation(self.hidden2(x))
        if self.training:
            mask = (torch.rand_like(x) > self.dropout_prob).float()
            x = x * mask / (1.0 - self.dropout_prob)
        return self.output_layer(x)
```

Dropout is NOT applied to the output layer.

---

## Hyperparameters set

### Main deep model training (cells `cell-10`, `cell-12`)

| Parameter | Value |
|---|---|
| `input_size` | 784 |
| `hidden_size1` | 128 |
| `hidden_size2` | 64 |
| `output_size` | 10 |
| Activation | `nn.ReLU` |
| Loss | `nn.CrossEntropyLoss()` |
| Optimizer | `torch.optim.SGD(model.parameters(), lr=0.1)` |
| `learning_rate` | 0.1 |
| `num_epochs` | 5 |
| `batch_size` | 32 |

### Shallow model comparison (cell `cell-16`)

| Parameter | Value |
|---|---|
| Optimizer | `torch.optim.SGD(shallow_model.parameters(), lr=0.1)` (same `learning_rate`) |
| `num_epochs` | 5 |

### L2-regularised model (cells `cell-19`)

| Parameter | Value |
|---|---|
| Architecture | same as deep model |
| Optimizer | `torch.optim.SGD(model_reg.parameters(), lr=0.1)` |
| `lambda_reg` | 1e-4 |
| `num_epochs` | 5 |
| L2 implementation | manual: `l2_loss += torch.sum(param.pow(2))` for all params; `loss = base_loss + lambda_reg * l2_loss` |

### Dropout model (cells `cell-24`, `cell-26`)

| Parameter | Value |
|---|---|
| Architecture | 784→128→64→10, ReLU, manual inverted dropout |
| `dropout_prob` | 0.5 |
| Optimizer | `torch.optim.SGD(model_dropout.parameters(), lr=0.1)` |
| `num_epochs` | 5 |

---

## Plots produced

None (all results printed to stdout as epoch loss/accuracy strings).

---

## What is left as an exercise to the student

The notebook is explicitly described as an exercise from the previous week (cell `cell-0`): "we will work on the exercise from last week." Specific exercise items stated in cell `cell-0`:
1. Build a deeper neural network with multiple hidden layers vs. shallow; try Tanh, Sigmoid, ReLU activations.
2. Implement mini-batch gradient descent.
3. Train on MNIST dataset with proper preprocessing for 10-class classification.
4. Add L2 regularization (weight decay) to the loss.
5. Manually implement Dropout regularization.

These are the five topics the notebook covers — the notebook *is* the exercise solution.

---

## Key cell indices for code idiom extraction

- `[cell cell-2]`: `SingleLayerNet` and `TwoHiddenLayerNet` `nn.Module` definitions; `activation_fn` as constructor argument pattern.
- `[cell cell-8]`: `torchvision.datasets.MNIST` + `DataLoader` with `shuffle`, `batch_size`; `transforms.ToTensor()`.
- `[cell cell-10]`: `nn.CrossEntropyLoss()` + `torch.optim.SGD` initialisation.
- `[cell cell-12]`: Full mini-batch training loop — `model.train()`, `images.view(..., -1)`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`, `torch.max(outputs, 1)` for accuracy.
- `[cell cell-14]`: Evaluation loop — `model.eval()`, `torch.no_grad()`, `torch.max(outputs, 1)`.
- `[cell cell-19]`: Manual L2 regularisation — `for param in model_reg.parameters(): l2_loss += torch.sum(param.pow(2))`.
- `[cell cell-24]`: Manual inverted dropout — `mask = (torch.rand_like(x) > p).float()` ; `x = x * mask / (1-p)` ; gated by `self.training`.

---

## Notes / [VERIFY] flags

- **Reported training/test accuracy (from cell outputs):**
  - Deep model (no regularisation): training 98.27% after 5 epochs; test 97.50%.
  - Shallow model: training 92.10% after 5 epochs; test 92.26%.
  - L2-regularised deep model: test 96.95%.
  - Dropout deep model: training 94.34%; test 97.13%.
- The comment in cell `cell-10` notes: "PyTorch's `torch.optim.SGD` isn't strictly stochastic gradient descent by definition. Whether it's 'batch,' 'mini-batch,' or 'stochastic' depends on how much data is provided per step."
- The `TwoHiddenLayerNet` uses a single `self.activation` instance shared across both hidden layers. This works for `ReLU`/`Tanh`/`Sigmoid` (stateless), but would fail for stateful activation modules.
- `nn.CrossEntropyLoss` applies softmax internally — the model output must be raw logits (not post-softmax).
- L2 regularisation is applied to **all** parameters including biases (the code loops `for param in model_reg.parameters()`). A comment in cell `cell-19` acknowledges this: "We'll include everything for simplicity."
- Mini-batch training pseudocode appears in a markdown code block (cell `cell-5`) — not executable but documents the pattern.
- `lambda_reg = 1e-4` — the variable name `lambda_reg` avoids conflict with Python's `lambda` keyword.
- Device detection: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` — output shown as `Using device: cpu`.
