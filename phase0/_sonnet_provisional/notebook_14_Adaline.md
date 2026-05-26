# Notebook extract: 14_Adaline.ipynb

**Source path:** `course_materials/Lecture notebooks/14_Adaline.ipynb`
**Cell count:** 27 cells (markdown + code)

---

## Dataset(s) loaded

- **Iris dataset** — loaded via `pd.read_csv` from UCI URL or local `iris.data` fallback (cell `e53f1686`).
  - Full shape: (150, 5); columns 0–3 are numeric features, column 4 is class label string.
  - Subset used: rows 0–99 (Iris-setosa vs Iris-versicolor), features: column 0 (sepal length) and column 2 (petal length) → `X.shape = (100, 2)`.
  - Target `y`: binary, 0 = Iris-setosa, 1 = Iris-versicolor (`np.where`).

---

## Preprocessing steps

- **Label encoding** (cell `100854b8`):  
  `y = np.where(y == 'Iris-setosa', 0, 1)`
- **Feature extraction** (cell `100854b8`):  
  `X = df.iloc[0:100, [0, 2]].values`
- **Standardization** (cell `07f941ef`):  
  ```python
  X_std = np.copy(X)
  X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
  X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
  ```
  Formula in markdown (cell `56451c18`): $x_{\text{std}} = \frac{x - \mu}{\sigma}$

---

## Method(s) demonstrated

**FROM-SCRATCH (numpy) — Adaline with batch gradient descent**

Class `AdalineGD` (cell `0d71e0f7`):

- **Architecture:** single neuron — linear activation (identity function), no hidden layers.
- **Net input:** $z = X \cdot w + b$  (`np.dot(X, self.w_) + self.b_`)
- **Activation:** identity — `return X` (the net input itself)
- **Loss:** Mean Squared Error (MSE): `(errors ** 2).mean()`
- **Prediction threshold:** 0.5 — `np.where(... >= 0.5, 1, 0)`
- **Weight init:** `rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])` ; bias `b_ = 0.0`
- **Gradient update rule (per epoch, per weight j):**
  ```python
  self.w_[j] += self.eta * (X[:, j] * errors).mean()
  self.b_  += self.eta * errors.mean()
  ```
  (batch gradient descent — full dataset used each epoch)

---

## Hyperparameters set

| Experiment | `eta` (learning rate) | `n_iter` (epochs) | `random_state` |
|---|---|---|---|
| Main run (unscaled) | 0.01 | 15 | 1 (default) |
| Comparison run 1 | 0.1 | 15 | 1 (default) |
| Comparison run 2 | 0.0001 | 1000 | 1 (default) |
| Standardized run | 0.1 | 15 | 1 (default) |

(cell `44873e7a`, `c77b191c`, `1886dd9a`)

---

## Plots produced

1. **Scatter plot** — sepal length vs petal length, coloured by class (setosa red, versicolor blue) — cell `100854b8`.
2. **Line plot** — MSE loss vs epochs, learning rate 0.01, unscaled — cell `58261aed`.
3. **Line plot** — misclassified count vs epochs, learning rate 0.01, unscaled — cell `ba5b2a36`.
4. **Decision region plot** — `plot_decision_regions`, unscaled features, eta=0.01 — cell `0bca99de`.
5. **2-panel figure** — MSE loss vs epochs for eta=0.1 and eta=0.0001 (unscaled) — cell `c77b191c`.
6. **2-panel figure** — misclassified count vs epochs for eta=0.1 and eta=0.0001 — cell `408aa8ea`.
7. **Decision region plot** — eta=0.1, unscaled — cell `2a2a0f26`.
8. **Line plot** — MSE loss vs epochs, eta=0.1, standardized features — cell `1886dd9a`.
9. **Line plot** — misclassified count vs epochs, eta=0.1, standardized — cell `85f85d5f`.
10. **Decision region plot** — standardized features, eta=0.1 — cell `f91035c3`.

---

## What is left as an exercise to the student

Mentioned in the concluding markdown (cell `6d2dd11a`) as "next steps":
- Try different feature subsets.
- Add regularization.
- Experiment with stochastic gradient descent (SGD variant of Adaline).

---

## Key cell indices for code idiom extraction

- `[cell 0d71e0f7]`: Full `AdalineGD` class — weight init, batch GD update loop, MSE loss, identity activation, 0.5-threshold predict.
- `[cell 07f941ef]`: Manual standardization with numpy (mean/std per column).
- `[cell 44efc974]`: `plot_decision_regions` helper — meshgrid + `contourf` + scatter overlay.
- `[cell c77b191c]`: Side-by-side subplot comparison of two learning rates.

---

## Notes / [VERIFY] flags

- The gradient update is written as a Python `for` loop over weights (not vectorized): `for j in range(self.w_.shape[0]): self.w_[j] += self.eta * (X[:, j] * errors).mean()`. This is correct batch GD but not the vectorized form used in the later logistic regression notebook.
- `missed_` list tracks misclassifications (via `.predict()`) at each epoch; `losses_` tracks MSE — both are logged simultaneously.
- No train/test split is performed; all 100 samples are used for training and the same samples for evaluation.
- The notebook uses `df.tail()` (cell `e53f1686`) and `df.shape` (cell `ea56b3e2`) for EDA inspection.
