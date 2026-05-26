# Notebook extract: 19_Build_a_NN.ipynb

**Source path:** course_materials/Lecture notebooks/19_Build_a_NN.ipynb
**Cell count:** 31 cells (ids cell-0 ... cell-30; cell-30 empty). Title: "Beginner's Guide to Deep Learning in PyTorch: Multi-Layer Networks, Mini-Batches, MNIST, L2 Regularization, and Dropout."

## Dataset(s) loaded
- **MNIST** via torchvision (cell-8): `torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)` and `train=False` for test.
  - 60,000 training images, 10,000 test images (printed output cell-8: "Number of training samples: 60000 / Number of test samples: 10000").
  - Each image 28x28 grayscale; flattened to 784 in the training loop. 10 classes (digits 0-9).
- `transform = transforms.ToTensor()` (cell-8) — scales pixels to [0,1].

## Preprocessing steps
- [cell-8] `transform = transforms.ToTensor()` (image → tensor, pixels scaled to [0,1]).
- [cell-8] `DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)`; test loader `shuffle=False`.
- [cell-12 / cell-16 etc.] In each training loop: `images = images.view(images.size(0), -1)` to flatten (batch, 1, 28, 28) → (batch, 784).
- (No StandardScaler / mean-std normalization applied; markdown mentions optional 0.1307/0.3081 normalization but it is NOT used.)

## Method(s) demonstrated
- **PyTorch using `nn.Module`, `nn.Linear`, `nn.CrossEntropyLoss`, `torch.optim.SGD`, `DataLoader`** (high-level PyTorch). Imports cell-2: `torch`, `torch.nn as nn`, `torch.optim as optim`, `from torchvision import datasets, transforms`, `from torch.utils.data import DataLoader`. Device: `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` → printed "Using device: cpu".
- **Models defined** (cell-2):
  - **`SingleLayerNet(input_size, output_size)`**: one `nn.Linear(input_size, output_size)`, no hidden layer, no activation (logistic-regression-equivalent). `forward`: `out = self.linear(x)`.
  - **`TwoHiddenLayerNet(input_size, hidden_size1, hidden_size2, output_size, activation_fn=nn.ReLU)`**: `hidden1 = nn.Linear(input_size, hidden_size1)`, `hidden2 = nn.Linear(hidden_size1, hidden_size2)`, `output_layer = nn.Linear(hidden_size2, output_size)`, `self.activation = activation_fn()`. `forward`: hidden1 → activation → hidden2 → activation → output_layer (no activation on output). activation_fn options noted: `nn.ReLU` (default), `nn.Tanh`, `nn.Sigmoid`.
  - **`TwoHiddenLayerNetWithDropout(input_size, hidden_size1, hidden_size2, output_size, dropout_prob=0.5)`** (cell-24): same 3 linear layers + `nn.ReLU`; **manual inverted dropout** in forward (only when `self.training`): `mask = (torch.rand_like(x) > self.dropout_prob).float(); x = x * mask; x = x / (1.0 - self.dropout_prob)` after each hidden activation. No dropout on output layer.
- **Architecture (instantiated, cell-2)**: `input_size = 784`, `output_size = 10`, `hidden_size1 = 128`, `hidden_size2 = 64`. Confirmed by printed model repr:
  - SingleLayerNet: Linear(784→10).
  - TwoHiddenLayerNet: Linear(784→128) → ReLU → Linear(128→64) → ReLU → Linear(64→10).
- **Loss**: `nn.CrossEntropyLoss()` (applies softmax internally) — cell-10, cell-16, cell-19, cell-26.
- **Optimizer**: `torch.optim.SGD(model.parameters(), lr=learning_rate)` (no momentum, no weight_decay set in optimizer) — cell-10 and per-model.
- **Training**: mini-batch GD via DataLoader; standard loop `optimizer.zero_grad(); loss.backward(); optimizer.step()` (cell-12, 16, 19, 26).
- **L2 regularization (weight decay), manual** (cell-19): `lambda_reg = 1e-4`; per batch `l2_loss = sum(torch.sum(param.pow(2)) for param in model_reg.parameters())`; `loss = base_loss + lambda_reg * l2_loss` (biases included). Logs `base_loss.item()` only.
- **Dropout, manual inverted dropout** (cell-24, p=0.5).

## Hyperparameters set
- `input_size=784`, `output_size=10`, `hidden_size1=128`, `hidden_size2=64` (cell-2).
- `batch_size = 32` (cell-8).
- Deep model training (cell-10/12): `criterion = nn.CrossEntropyLoss()`, `learning_rate = 0.1`, `optimizer = torch.optim.SGD(model.parameters(), lr=0.1)`, `num_epochs = 5`.
- Shallow model (cell-16): `SGD(lr=learning_rate)` = 0.1, `num_epochs = 5`.
- L2-reg model (cell-19): `optimizer_reg = torch.optim.SGD(model_reg.parameters(), lr=0.1)`, `lambda_reg = 1e-4`, `num_epochs = 5`.
- Dropout model (cell-26): `dropout_prob = 0.5`, `optimizer_do = torch.optim.SGD(model_dropout.parameters(), lr=0.1)`, `num_epochs = 5`.
- activation_fn default `nn.ReLU` for all TwoHiddenLayerNet instances.
- Mini-batch training-loop skeleton in markdown (cell-5) shows `num_epochs=5`, `batch_size=32` example.

## Plots produced
- **None.** No matplotlib. All reporting is printed loss/accuracy per epoch + model repr strings.

## What is left as an exercise to the student
- Markdown (cell-0) frames the whole notebook as "the exercise from last week" worked out. Specific student-tunable knobs called out as experiments (not blank TODOs): try Tanh/Sigmoid in place of ReLU (cell-2/3); vary `batch_size` 16/32/64 (cell-6/8); tune `lambda_reg` among 1e-3/1e-4/1e-5 (cell-19/22); combine L2 + dropout (cell-29). No empty starter cells are left for the student.

## Key cell indices for code idiom extraction
- "[cell-2]: `nn.Module` subclasses — SingleLayerNet (one nn.Linear) and TwoHiddenLayerNet (Linear→act→Linear→act→Linear) with `activation_fn=nn.ReLU` param"
- "[cell-8]: torchvision MNIST load + `transforms.ToTensor()` + `DataLoader(batch_size=32, shuffle=True/False)`"
- "[cell-12]: canonical PyTorch training loop — flatten `images.view(images.size(0), -1)`, `optimizer.zero_grad(); loss.backward(); optimizer.step()`, running-loss + accuracy via `torch.max(outputs, 1)`"
- "[cell-14]: eval idiom — `model.eval()`, `with torch.no_grad():`, accuracy accumulation"
- "[cell-19]: manual L2 penalty `l2_loss = sum(torch.sum(p.pow(2)) for p in model.parameters()); loss = base_loss + lambda_reg*l2_loss`"
- "[cell-24]: manual inverted dropout `mask = (torch.rand_like(x) > p).float(); x = x*mask; x = x/(1-p)` gated by `if self.training`"
- "[cell-10]: `nn.CrossEntropyLoss()` + `torch.optim.SGD(model.parameters(), lr=0.1)`"

## Plots produced / reported metrics (saved execution outputs)
- **Deep TwoHiddenLayerNet (ReLU, lr=0.1, 5 epochs)** — cell-12 train: epoch1 loss 0.3664 acc 89.13%; epoch2 0.1316/96.01%; epoch3 0.0890/97.20%; epoch4 0.0675/97.98%; epoch5 0.0553/98.27%. cell-14 **test: loss 0.0773, accuracy 97.50%**.
- **SingleLayerNet (shallow, lr=0.1, 5 epochs)** — cell-16 train acc: 88.89 / 91.17 / 91.76 / 91.95 / 92.10%; **shallow test accuracy 92.26%**.
- **L2-regularized deep model (lambda=1e-4)** — cell-19 base-loss per epoch: 0.3698 / 0.1357 / 0.0950 / 0.0715 / 0.0586; cell-21 **test accuracy with L2: 96.95%**.
- **Dropout deep model (p=0.5)** — cell-26 train: epoch1 loss 0.2216 acc 93.61%; ... epoch5 0.1975/94.34%; cell-28 **test accuracy with Dropout: 97.13%**.

## Notes / [VERIFY] flags
- Output layer has NO activation in all models; `nn.CrossEntropyLoss` applies softmax internally (markdown cells 3, 11 state this explicitly). Correct.
- L2 implementation includes biases (markdown cell-20 notes one could skip biases via `param.ndim > 1`, but code sums all params). Minor convention note, not a bug.
- Dropout uses **inverted dropout**: scale surviving activations by `1/(1-p)` at train time, nothing at eval (markdown cell-23 transcribes the rationale correctly). `if self.training` gating verified in cell-24.
- Device printed as **cpu** (no CUDA in this run).
- Reported metrics above are the actual saved cell outputs (load-bearing, transcribed verbatim). Deep net (97.50%) > shallow (92.26%) confirms the lecture's expected ordering; L2 (96.95%) and Dropout (97.13%) are within noise of the unregularized 97.50% on this already-non-overfitting setup — markdown acknowledges this.
- The mini-batch training loop in cell-5 is a markdown code block (illustrative, references `model_deep`/`criterion`/`optimizer` before they are defined for the real run) — not executed; the real loops are cells 12/16/19/26.
