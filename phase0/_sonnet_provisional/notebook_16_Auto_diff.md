# Notebook extract: 16_Auto_diff.ipynb

**Source path:** `Lecture notebooks/16_Auto_diff.ipynb`
**Cell count:** 10 cells (markdown + code)

---

## Dataset(s) loaded

None. This notebook uses a toy synthetic function with no external dataset.

- **Synthetic objective:** $\text{loss} = 0.5 \times ((w_1 - 1)^2 + (w_2 - 5)^2)$ (cell `cell-1`, `cell-3`, `cell-7`)
- Initial values: $w_1 = 2.0$, $w_2 = 3.0$

---

## Preprocessing steps

None.

---

## Method(s) demonstrated

**FROM-SCRATCH (pure Python/numpy scalars) — Minimal Automatic Differentiation Engine**

Inspired by [micrograd](https://github.com/karpathy/micrograd) (cited in cell `cell-0`).

### `Value` class (cell `cell-2`)

A scalar computational graph node with:

| Method | Forward computation | Backward (gradient) rule |
|---|---|---|
| `__add__` | `self.data + other.data` | `self.grad += out.grad` ; `other.grad += out.grad` |
| `__mul__` | `self.data * other.data` | `self.grad += other.data * out.grad` ; `other.grad += self.data * out.grad` |
| `__pow__(k)` | `self.data ** k` | `self.grad += k * (self.data ** (k-1)) * out.grad` |
| `__sub__` | defined as `self + (-1 * other)` | — (uses add + mul) |
| `__radd__` / `__rmul__` | commutativity wrappers | — |

- **`backward()` method:** topological sort (DFS `build_topo`) then reverse traversal calling `node._backward()` for each node; initialises `self.grad = 1.0` at the output node.

### Gradient descent optimisation loop (cell `cell-7`)

```python
w1 = Value(2.0)
w2 = Value(3.0)
params = [w1, w2]
learning_rate = 0.1
for i in range(100):
    loss = 0.5 * ((w1 - 1)**2 + (w2 - 5)**2)
    loss.backward()
    for p in params:
        p.data -= learning_rate * p.grad
        p.grad = 0.0   # reset gradient for next iteration
```

---

## Hyperparameters set

| Parameter | Value |
|---|---|
| `learning_rate` | 0.1 |
| iterations | 100 |
| initial `w1` | 2.0 |
| initial `w2` | 3.0 |

---

## Plots produced

None. Convergence is tracked by `print` statements only (loss printed each iteration in cell `cell-7`).

---

## What is left as an exercise to the student

Not explicitly stated. Implicit extensions:
- Adding more operations (e.g., `exp`, `log`, `tanh`) to the `Value` class — these are added in notebook 17.
- Applying the engine to a real dataset or multi-layer network — done in notebook 17.

---

## Key cell indices for code idiom extraction

- `[cell cell-2]`: Full scalar autodiff `Value` class with `__add__`, `__mul__`, `__pow__`, `__sub__`, `__radd__`, `__rmul__`, and `backward()` with topological sort.
- `[cell cell-7]`: Canonical gradient descent loop using the `Value` engine — forward pass, `.backward()`, manual `p.data -= lr * p.grad`, gradient zero reset.

---

## Notes / [VERIFY] flags

- The markdown walkthrough (cell `cell-4`) traces the full forward and backward pass analytically for $w_1=2.0$, $w_2=3.0$:
  - Forward: $a=1.0$, $b=1.0$, $c=-2.0$, $d=4.0$, $s=5.0$, $\text{loss}=2.5$
  - Backward: $\partial\text{loss}/\partial w_1 = 1.0$ ; $\partial\text{loss}/\partial w_2 = -2.0$
  - Confirmed by code output (cell `cell-5`): `∂loss/∂w1 = 1.0`, `∂loss/∂w2 = -2.0`.
- The `Value` class here does **not** include `tanh` or `exp` — those are added in notebook 17 (the extended version from Karpathy's series).
- `_backward` is initialised as `lambda: None` so that leaf nodes (with no children) do nothing on backward — correct pattern for leaf variables.
- Gradients accumulate with `+=` (not `=`) in each `_backward` to handle shared nodes correctly in a DAG.
