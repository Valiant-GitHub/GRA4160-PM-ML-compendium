# Notebook extract: 15_LogisticRegression.ipynb

**Source path:** `Lecture notebooks/15_LogisticRegression.ipynb`
**Cell count:** 30 cells (markdown + code)

---

## Dataset(s) loaded

- **Iris dataset** — loaded via `pd.read_csv` from UCI URL or local fallback (cell `c4af8a90`).
  - Full shape: (150, 5).
  - **Example 1 & 2:** rows 0–99, features col 0 (sepal length) + col 2 (petal length) → `X.shape = (100, 2)`.
  - **Example 2:** rows 0–99, all 4 features → `X_all.shape = (100, 4)`.
  - **Example 3:** all 150 rows, all 4 features → `X_all2.shape = (150, 4)`.
  - Target: 0 = Iris-setosa, 1 = Iris-versicolor (binary); 0/1/2 all three classes (Example 3).

---

## Preprocessing steps

- **Binary label encoding, 2-class** (cell `d60ccbaa`):  
  `y = np.where(y == 'Iris-setosa', 0, 1)`
- **3-class label encoding** (cell `d6708f6b`):  
  ```python
  y_all = np.where(y_all == 'Iris-setosa', 0, y_all)
  y_all = np.where(y_all == 'Iris-versicolor', 1, y_all)
  y_all = np.where(y_all == 'Iris-virginica', 2, y_all)
  ```
- No standardization is applied in this notebook (unlike notebook 14).

---

## Method(s) demonstrated

### Method 1 — FROM-SCRATCH (numpy): `LogisticRegressionGD` (cell `a342d4e2`)

- **Architecture:** single neuron with sigmoid activation, no hidden layers.
- **Net input:** `np.dot(X, self.w_) + self.b_`
- **Activation (sigmoid):**
  ```python
  z_clipped = np.clip(z, -250, 250)
  return 1. / (1. + np.exp(-z_clipped))
  ```
- **Loss:** Cross-entropy (binary):
  ```python
  output_clipped = np.clip(output, 1e-10, 1.0 - 1e-10)
  loss = -y.dot(np.log(output_clipped)) - (1 - y).dot(np.log(1 - output_clipped))
  loss /= X.shape[0]
  ```
- **Weight update (vectorized):**
  ```python
  self.w_ += self.eta * X.T.dot(errors) / X.shape[0]
  self.b_ += self.eta * errors.mean()
  ```
- **Weight init:** `rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])` ; bias `b_ = 0.0`
- **Prediction threshold:** 0.5 — `np.where(... >= 0.5, 1, 0)`

### Method 2 — FROM-SCRATCH (numpy): One-vs-All (OvA) multi-class (cells `f4432549`, `b896bc6b`, `d5cac4b4`, `07ba1fb6`)

- Three separate `LogisticRegressionGD` classifiers, one per class.
- Probabilities stored in `probabilities` array of shape `(150, 3)`.
- Final class: `np.argmax(probabilities, axis=1)`.
- Note: probabilities do not sum to 1 (separate binary classifiers — not softmax).

### Method 3 — FROM-SCRATCH (numpy): `OneModel` (cell `835b264a`)

- Minimal, concise re-implementation of the same logistic regression GD — functionally identical to `LogisticRegressionGD`. Included for reference.

---

## Hyperparameters set

| Experiment | `eta` | `n_iter` | `random_state` |
|---|---|---|---|
| Example 1 (2 features) | 0.3 | 1000 | 1 |
| Example 2 (4 features) | 0.3 | 1000 | 1 |
| OvA (each of 3 classifiers) | 0.3 | 1000 | 1 |

(cells `50ee34c8`, `a16d3295`, `f4432549`)

---

## Plots produced

1. **Scatter plot** — sepal length vs petal length, setosa red / versicolor blue — cell `73078bbc`.
2. **Scatter + prediction overlay** — true class colours + black `.` (pred 0) or `x` (pred 1) markers — cell `08c79e90`.
3. **Decision region plot** — `plot_decision_regions`, 2-feature model — cell `44ecc01c`.
4. **Scatter + prediction overlay** — 4-feature model, plotted on 2 selected features (0 and 2) — cell `9f10168e`.

No loss/epoch convergence plot is produced in this notebook.

---

## What is left as an exercise to the student

Not explicitly stated as exercises, but:
- Standardization is omitted (contrast with notebook 14) — implicitly left for students to try.
- Extending OvA to use softmax regression is mentioned but not implemented (cell `0db01570`): "if we wanted a valid set of 'class probabilities' that sum to 1, we might consider softmax regression."

---

## Key cell indices for code idiom extraction

- `[cell a342d4e2]`: Full `LogisticRegressionGD` class — vectorized weight update `X.T.dot(errors) / n`, sigmoid with clip, binary cross-entropy loss.
- `[cell f4432549]`: OvA training loop — `for k in range(3)` — binary label construction + fit + probability extraction.
- `[cell 07ba1fb6]`: `np.argmax(probabilities, axis=1)` for final class assignment.
- `[cell fa0b4810]`: `plot_decision_regions` helper (identical pattern to notebook 14, worth noting as canonical idiom).

---

## Notes / [VERIFY] flags

- Key difference from Adaline (notebook 14): the activation is sigmoid (not identity) and the loss is cross-entropy (not MSE), but the gradient update formula is structurally identical — this is by design and demonstrated explicitly in the notebook title ("Adaline-like approach").
- The vectorized update `X.T.dot(errors) / X.shape[0]` replaces the loop-over-j used in Adaline.
- No train/test split; all samples used for training and same samples for evaluation in all three examples.
- `output_clipped = np.clip(output, 1e-10, 1.0 - 1e-10)` is used only inside the loss computation; `self.activation` already clips `z` to `[-250, 250]` before sigmoid — double protection against numerical overflow.
