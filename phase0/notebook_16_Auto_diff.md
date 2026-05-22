# Notebook extract: 16_Auto_diff.ipynb

**Source path:** Lecture notebooks/16_Auto_diff.ipynb
**Cell count:** 11 cells (ids cell-0 ... cell-10; cell-10 empty).

## Dataset(s) loaded
- **None.** No external dataset. Works on a hand-built scalar expression / synthetic toy loss.

## Preprocessing steps
- None (no data). Parameters are hard-coded scalars `w1 = Value(2.0)`, `w2 = Value(3.0)`.

## Method(s) demonstrated
- **Custom automatic-differentiation engine (CUSTOM AUTODIFF, micrograd-style, from scratch, pure Python).** Markdown (cell-0) cites it as "similar to micrograd" (Karpathy). Reverse-mode autodiff over a scalar computation graph.
- **`Value` class** (cell-2). Scalar node in a computation graph.
  - Attributes: `data` (scalar value), `grad` (init 0.0), `_prev` (set of child/parent nodes from `_children`), `_op` (string op name), `_backward` (closure, default `lambda: None`).
  - Supported ops (each defines a local `_backward` implementing the chain rule):
    - `__add__` (`+`): out grad flows 1.0 to each input → `self.grad += out.grad; other.grad += out.grad`. Coerces scalar to `Value`.
    - `__radd__`: `return self + other` (commutative add for scalar+Value).
    - `__mul__` (`*`): `self.grad += other.data * out.grad; other.grad += self.data * out.grad`. Coerces scalar to `Value`.
    - `__rmul__`: `return self * other`.
    - `__sub__` (`-`): defined as `self + (-1 * other)`.
    - `__pow__` (`** k`, k int/float only, asserted): `self.grad += exponent * (self.data ** (exponent - 1)) * out.grad`.
  - `backward()`: builds topological order via recursive `build_topo` over `_prev`, sets `self.grad = 1.0`, then calls `node._backward()` for each node in reversed topo order.
  - `__repr__` → `f"Value(data={self.data})"`.
  - **NOT supported in this notebook's Value:** no `tanh`, no `exp`, no `__neg__`, no `__truediv__` (those appear in nb 17's richer Value). Subtraction here is built only from add + scalar multiply.
- **No neural network here** — pure autodiff demo + manual gradient-descent on a quadratic loss.

## Hyperparameters set
- `learning_rate = 0.1` (cell-7).
- Optimization loop: `for i in range(100)` → 100 iterations (cell-7).
- Initial params: `w1 = Value(2.0)`, `w2 = Value(3.0)` (cell-3 and re-init cell-7).
- Loss function (toy): `loss = 0.5 * ((w1 - 1)**2 + (w2 - 5)**2)` — quadratic with minimum at w1=1, w2=5.
- Update rule: `p.data -= learning_rate * p.grad`; gradients reset each step `p.grad = 0.0`.

## Plots produced
- **None.** No matplotlib/graphviz in this notebook. Output is printed scalars only (loss per iteration, final `w1`, `w2`, gradient prints).

## What is left as an exercise to the student
- No explicit exercise. Notebook is a worked demonstration (forward pass verified analytically in markdown cell-4, then verified numerically in code).

## Key cell indices for code idiom extraction
- "[cell-2]: full `Value` autodiff class — `_backward` closures per op, topological-sort `backward()` with `self.grad=1.0` seed and reversed traversal"
- "[cell-2]: chain-rule snippets — add (grad +=1), mul (grad += other.data*out.grad), pow (grad += k*data**(k-1)*out.grad)"
- "[cell-3]: building an expression graph manually `a = w1 - 1; b = a**2; c = w2 - 5; d = c**2; s = b + d; loss = s*0.5`"
- "[cell-7]: manual gradient-descent loop — forward, `loss.backward()`, `p.data -= lr*p.grad`, `p.grad = 0.0` reset"

## Notes / [VERIFY] flags
- Loss formula (markdown cell-1 and code): `loss = ((w1-1)^2 + (w2-5)^2) * 0.5`. Matches code.
- Markdown cell-4 transcribes the full forward+backward by hand: forward gives a=1.0, b=1.0, c=-2.0, d=4.0, s=5.0, loss=2.5; backward gives `d loss/d w1 = 1.0`, `d loss/d w2 = -2.0`. Code comments in cell-5 confirm the same expected values (`w1.grad`=1.0, `w2.grad`=-2.0). Internally consistent.
- `backward()` accumulates gradients (`+=`), so the GD loop explicitly zeroes `p.grad = 0.0` after each update — important idiom highlighted by a comment.
- No saved execution outputs were captured in the cells provided (loss prints, final w values). [VERIFY: numeric convergence — analytically should converge toward w1=1.0, w2=5.0 over 100 steps at lr=0.1.]
- `__sub__` relies on `-1 * other` (scalar*Value via `__rmul__`), so `other` must support multiplication — works only because `other` is a `Value` or coerced.
