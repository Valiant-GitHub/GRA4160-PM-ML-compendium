# Notebook extract: 17_Neural_nets_basics.ipynb

**Source path:** course_materials/Lecture notebooks/17_Neural_nets_basics.ipynb
**Cell count:** 17 cells (ids cell-0 ... cell-16; cell-16 empty). Course header (cell-0): "Neural net basics / Lecture 9 / GRA 4160 / Lecturer: Vegard H. Larsen".

## Dataset(s) loaded
- **None external.** Two tiny hard-coded toy datasets:
  - Single-neuron demo inputs (cell-6): `x1=2.0, x2=0.0`, weights `w1=-1.0, w2=1.0`, bias `b=5`.
  - MLP training set (cell-13): `xs` = 4 samples of 3 features each; `ys = [1.0, -1.0, -1.0, 1.0]` (desired targets, in {-1, +1}).

## Preprocessing steps
- None (data hard-coded as Python lists/Value objects).

## Method(s) demonstrated
- **Custom autodiff engine + MLP from scratch (CUSTOM AUTODIFF, micrograd-style, pure Python).** Markdown (cell-2) attributes it to Karpathy nn-zero-to-hero; scalar-based (not tensors).
- **`Value` class** (cell-3) — richer than nb 16. Adds `label` attribute and more ops:
  - Ops with `_backward` closures: `__add__`, `__mul__`, `__pow__` (int/float only, asserted), `__rmul__`, `__truediv__` (= `self * other**-1`), `__neg__` (= `self * -1`), `__sub__` (= `self + (-other)`), `__radd__`.
  - **`tanh()`**: `t = (math.exp(2*x)-1)/(math.exp(2*x)+1)`; backward `self.grad += (1 - t**2) * out.grad`.
  - **`exp()`**: `out = Value(math.exp(x), ...)`; backward `self.grad += out.data * out.grad`.
  - `backward()`: topological sort via `build_topo`, seed `self.grad = 1.0`, reversed traversal calling `node._backward()`.
- **Graph visualization** (cell-4): `trace(root)` + `draw_dot(root)` using `from graphviz import Digraph` (rankdir LR, record nodes showing label/data/grad).
- **Neuron / Layer / MLP classes** (cell-10) — FROM SCRATCH on top of the custom autodiff `Value`:
  - **`Neuron(nin)`**: `self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]`, `self.b = Value(random.uniform(-1,1))`. `__call__(x)`: `act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)` then `out = act.tanh()`. `parameters()` returns `self.w + [self.b]`. **Activation = tanh.**
  - **`Layer(nin, nout)`**: `self.neurons = [Neuron(nin) for _ in range(nout)]`. `__call__(x)` returns list of neuron outputs (or single scalar if nout==1). `parameters()` flattens neuron params.
  - **`MLP(nin, nouts)`**: `sz = [nin] + nouts`; `self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]`. `__call__(x)` chains layers. `parameters()` flattens all.
- **NN architecture instantiated** (cell-11): `n = MLP(3, [3, 3, 1])` → input size 3, hidden layer 1 = 3 neurons, hidden layer 2 = 3 neurons, output layer = 1 neuron. **All neurons use tanh activation** (including output). Total = 3 layers (two hidden of size 3, one output of size 1).
  - **Loss**: sum of squared errors — `loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))` (cell-14). No averaging.
  - **Optimizer**: manual gradient descent, **learning rate effectively 1.0** (update is `p.data += -1.0 * p.grad`).
  - Training loop: `for k in range(200)` → 200 iterations (cell-14). Each step: forward all 4 samples, zero grads (`p.grad = 0.0`), `loss.backward()`, update.
  - **Batch**: full-batch (all 4 samples summed into one loss per step).

## Hyperparameters set
- MLP shape: `MLP(3, [3, 3, 1])` (cell-11).
- Neuron weight/bias init: `random.uniform(-1, 1)` (cell-10).
- Training iterations: `range(200)` (cell-14).
- Learning rate: `-1.0` multiplier in update `p.data += -1.0 * p.grad` (i.e. lr = 1.0).
- Loss: SSE (no `mean`), targets in {-1, +1}.
- Single-neuron demo (cell-6): x1=2.0, x2=0.0, w1=-1.0, w2=1.0, b=5; output `o = n.tanh()`.

## Plots produced
- No matplotlib plots. **Computation-graph diagrams** via graphviz:
  - [cell-7] `draw_dot(o)` — graph of the single-neuron forward computation (x1*w1 + x2*w2 + b → tanh).
  - [cell-12] `draw_dot(n(x))` — graph of full MLP(3,[3,3,1]) forward pass for input `[2.0, 3.0, -1.0]`.
- `import matplotlib.pyplot as plt` is imported (cell-1) but never used for a plot.

## What is left as an exercise to the student
- No explicit exercise/TODO in this notebook (it is the teaching/demo notebook; the assignment "build a NN" is followed up in nb 19). cell-13/14 set up and run the training directly.

## Key cell indices for code idiom extraction
- "[cell-3]: `Value.tanh()` with backward `self.grad += (1 - t**2)*out.grad`; `Value.exp()` with `self.grad += out.data*out.grad`"
- "[cell-10]: Neuron `__call__` idiom `sum((wi*xi for wi,xi in zip(self.w, x)), self.b).tanh()`"
- "[cell-10]: Layer/MLP `parameters()` flatten comprehensions `[p for neuron in self.neurons for p in neuron.parameters()]`"
- "[cell-14]: training loop — `loss = sum((yout-ygt)**2 ...)`, zero grads, `loss.backward()`, `p.data += -1.0 * p.grad`"
- "[cell-4]: `draw_dot`/`trace` graphviz computation-graph visualization"
- "[cell-6]: manual single-neuron graph construction with `.label` assignments"

## Notes / [VERIFY] flags
- Output activation is **tanh** for all neurons (including the final output neuron) because `Neuron.__call__` always applies `.tanh()` and `MLP` uses only `Neuron`/`Layer`. Targets are in {-1,+1}, matching tanh range — consistent.
- Loss is **summed (not averaged)** SSE across 4 samples.
- Effective learning rate = 1.0 (`p.data += -1.0 * p.grad`), unusually large but works on this 4-sample toy.
- Weights randomly initialized with `random.uniform(-1,1)` and **no seed set** for `random` → results non-deterministic across runs. (`np` imported but RNG not seeded either.) [VERIFY: no reproducibility seed.]
- No saved numeric outputs (loss prints per iteration, final `ypred`) captured in provided cells. [VERIFY: reported final predictions — expected to approach `ys=[1,-1,-1,1]`.]
- The richer `Value` here (vs nb 16) adds `tanh`, `exp`, `__neg__`, `__truediv__`, and the `label` field.
