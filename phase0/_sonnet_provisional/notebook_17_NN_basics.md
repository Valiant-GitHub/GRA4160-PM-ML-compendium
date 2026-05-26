# Notebook extract: 17_Neural_nets_basics.ipynb

**Source path:** `course_materials/Lecture notebooks/17_Neural_nets_basics.ipynb`
**Cell count:** 16 cells (markdown + code)

---

## Dataset(s) loaded

Toy synthetic dataset — 4 training examples, manually defined (cell `cell-13`):

```python
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets (binary: ±1)
```

No external dataset is loaded.

---

## Preprocessing steps

None.

---

## Method(s) demonstrated

### Part 1 — FROM-SCRATCH (pure Python scalars): Extended `Value` autodiff engine (cell `cell-3`)

Extended version of the engine from notebook 16, adding:

| Method | Forward | Backward |
|---|---|---|
| `__truediv__` | `self * other**-1` | (via mul + pow) |
| `__neg__` | `self * -1` | — |
| `__radd__` | commutativity | — |
| `tanh()` | $t = \frac{e^{2x}-1}{e^{2x}+1}$ | `self.grad += (1 - t**2) * out.grad` |
| `exp()` | `math.exp(x)` | `self.grad += out.data * out.grad` |

`backward()` method: same topological-sort + reverse traversal as notebook 16.

### Part 2 — FROM-SCRATCH (pure Python scalars): Single-neuron backprop example (cells `cell-5`, `cell-6`, `cell-7`, `cell-8`)

Manual computation of one neuron:
```python
x1 = Value(2.0)  # input
x2 = Value(0.0)  # input
w1 = Value(-1.0) # weight
w2 = Value(1.0)  # weight
b  = Value(5)    # bias
# net input: x1*w1 + x2*w2 + b = (-2 + 0 + 5) = 3
# output: tanh(3) = o
o = (x1*w1 + x2*w2 + b).tanh()
o.backward()
```
Graphviz visualisation of computational graph (cell `cell-7` using `draw_dot`).

### Part 3 — FROM-SCRATCH (pure Python scalars): Multi-layer Perceptron (MLP) (cell `cell-10`)

Three classes built on top of `Value`:

**`Neuron`:**
- Weights: `[Value(random.uniform(-1,1)) for _ in range(nin)]`
- Bias: `Value(random.uniform(-1,1))`
- Forward: `act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)` ; output = `act.tanh()`
- Activation: **tanh**

**`Layer`:**
- `nin` inputs → `nout` neurons; returns scalar if `nout==1`, else list.

**`MLP`:**
- Constructor: `MLP(nin, nouts)` where `nouts` is a list of layer widths.
- Layers: `[Layer(sz[i], sz[i+1]) for i in range(len(nouts))]`
- Forward: sequential layer application.

**Network instantiated (cell `cell-11`):**
```python
n = MLP(3, [3, 3, 1])
```
- **Architecture:** 3 inputs → hidden layer (3 neurons, tanh) → hidden layer (3 neurons, tanh) → output layer (1 neuron, tanh)
- **Total layers:** 3 (two hidden + one output)
- **Activation:** tanh at every neuron (including output)

**Training loop (cell `cell-14`):**
```python
for k in range(200):
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    for p in n.parameters():
        p.data += -1.0 * p.grad
    print(k, loss.data)
```
- **Loss:** Sum of squared errors (SSE) — not MSE; sum over 4 training examples.
- **Optimizer:** Vanilla gradient descent (no momentum, no weight decay).
- **Gradient reset:** explicit `p.grad = 0.0` before each backward (not `p.grad = 0.0` after, but before — correct).

---

## Hyperparameters set

| Parameter | Value |
|---|---|
| Architecture | MLP(3, [3, 3, 1]) |
| Activation | tanh (all neurons) |
| Loss | Sum of squared errors |
| Learning rate | 1.0 (hard-coded: `p.data += -1.0 * p.grad`) |
| Epochs/iterations | 200 |
| Weight init | `random.uniform(-1, 1)` |
| Bias init | `random.uniform(-1, 1)` |

---

## Plots produced

1. **Computational graph (Graphviz SVG)** — single neuron forward pass, showing all Value nodes with data and grad values — cell `cell-7`.
2. **Computational graph (Graphviz SVG)** — full MLP(3,[3,3,1]) forward pass graph — cell `cell-12`.

No matplotlib loss-vs-epoch plots; loss is printed to stdout via `print(k, loss.data)`.

---

## What is left as an exercise to the student

Not explicitly stated. The notebook ends after training the MLP (cell `cell-15` shows `ypred`). Implicit extensions:
- Varying architecture (different `nouts` list).
- Using a proper learning rate instead of hard-coded 1.0.
- Replacing SSE with MSE.

---

## Key cell indices for code idiom extraction

- `[cell cell-3]`: Extended `Value` class with `tanh`, `exp`, `__truediv__`, `__neg__`, full `backward()`.
- `[cell cell-10]`: `Neuron`, `Layer`, `MLP` classes — full scalar-based MLP from scratch.
- `[cell cell-14]`: Training loop — SSE loss, gradient zero before backward, GD update with `p.data += -lr * p.grad`.
- `[cell cell-4]`: `draw_dot` / `trace` functions for Graphviz computational graph visualisation.

---

## Notes / [VERIFY] flags

- Credit: notebook explicitly cites Andrej Karpathy's [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) repository (cell `cell-2`).
- The learning rate is implicitly 1.0 (`p.data += -1.0 * p.grad`) — this is unusual but works on the tiny 4-sample toy problem.
- The loss is SSE (sum, not mean): `sum((yout - ygt)**2 ...)` over 4 samples.
- The `tanh` output neuron means targets ±1 are natural; unlike notebooks 14/15 where the threshold is 0.5 on [0,1] sigmoid output.
- `random` (Python stdlib) is used for weight initialisation, not `numpy.random` — important for reproducibility: no seed is set.
- Graphviz (`from graphviz import Digraph`) must be installed; the `draw_dot` function renders SVG inline in the notebook.
