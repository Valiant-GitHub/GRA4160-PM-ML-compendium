# Notebook extract: 14_Adaline.ipynb

**Source path:** course_materials/Lecture notebooks/14_Adaline.ipynb
**Cell count:** 28 cells (cell ids: 5eb5ea15, 4bbbb7fc, 399d5e4c, e53f1686, ea56b3e2, 100854b8, 13cf7108, 598c94f5, 44efc974, dab6b4ad, 0d71e0f7, 72811869, 44873e7a, bc0dfdff, 58261aed, ba5b2a36, 0bca99de, ae48d679, c77b191c, 408aa8ea, 2a2a0f26, 3394230d, 56451c18, 07f941ef, 1886dd9a, 85f85d5f, f91035c3, 6d2dd11a, 5d52fa76 — last cell empty). Numbered references below use sequential cell index 0-28.

## Dataset(s) loaded
- **Iris dataset**, loaded with `pd.read_csv` (cell 3, id e53f1686). Primary source URL `https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data`, `header=None`, `encoding='utf-8'`; fallback to local `'iris.data'` in `except Exception`.
- Full shape (cell 4, id ea56b3e2): `df.shape` → 150 rows x 5 columns (4 features + 1 label string column). [VERIFY: shape value not printed in saved output, but Iris is 150x5.]
- Subset used (cell 5, id 100854b8): first 100 rows = Iris-setosa + Iris-versicolor (50 each).
- Features: column 0 = sepal length, column 2 = petal length → `X = df.iloc[0:100, [0, 2]].values`, shape (100, 2) (confirmed cell 7, id 598c94f5).
- Target: column 4 (class string), encoded `y = np.where(y == 'Iris-setosa', 0, 1)` → 0=Setosa, 1=Versicolor.

## Preprocessing steps
- [cell 5, id 100854b8] `y = df.iloc[0:100, 4].values` then `y = np.where(y == 'Iris-setosa', 0, 1)` (label encoding 0/1).
- [cell 5, id 100854b8] `X = df.iloc[0:100, [0, 2]].values` (select sepal length + petal length).
- **Standardization** (cell 23, id 07f941ef), applied only in the later "feature scaling" section:
  - `X_std = np.copy(X)`
  - `X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()`
  - `X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()`
  - Formula in markdown (cell 22, id 56451c18): `x_std = (x - mu) / sigma`.

## Method(s) demonstrated
- **Adaline (Adaptive Linear Neuron)** — FROM SCRATCH (numpy). Class `AdalineGD` (cell 10, id 0d71e0f7). Single-layer linear neuron trained by full-batch gradient descent on MSE.
- Architecture: single linear unit. Input = 2 features (or all features if passed), no hidden layer, 1 linear output.
  - Net input: `net_input(X) = np.dot(X, self.w_) + self.b_`.
  - **Activation**: identity / linear — `activation(X)` returns `X` unchanged.
  - **Loss**: Mean Squared Error — `loss = (errors ** 2).mean()` where `errors = (y - output)`.
  - **Optimizer**: manual full-batch gradient descent. Weight update done per-feature in a loop:
    `self.w_[j] += self.eta * (X[:, j] * errors).mean()` for each j; `self.b_ += self.eta * errors.mean()`.
  - **Prediction**: unit step at threshold 0.5 — `np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)`.
  - Weight init: `rgen = np.random.RandomState(self.random_state)`, `self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])`, `self.b_ = 0.0`.
  - Tracks `self.losses_` (MSE per epoch) and `self.missed_` (misclassification count per epoch via `np.sum(np.abs(y - self.predict(X)))`).
- `plot_decision_regions(X, y, classifier, resolution=0.02)` helper (cell 8, id 44efc974) — meshgrid + `classifier.predict` + `plt.contourf`.

## Hyperparameters set
- `AdalineGD.__init__` defaults (cell 10): `eta=0.01`, `n_iter=50`, `random_state=1`.
- Weight init scale: `scale=0.01`, `loc=0.0` (default within class).
- First training run (cells 12-13): `eta = 0.01`, `AdalineGD(n_iter=15, eta=eta)` → n_iter=15, eta=0.01, random_state=default(1).
- Learning-rate comparison (cell 18, id c77b191c):
  - `ada1 = AdalineGD(n_iter=15, eta=0.1)`
  - `ada2 = AdalineGD(n_iter=1000, eta=0.0001)`
- Standardized run (cell 24, id 1886dd9a): `ada_gd = AdalineGD(n_iter=15, eta=0.1)` on `X_std`.

## Plots produced
- [cell 5, id 100854b8] Scatter: sepal length (x) vs petal length (y), Setosa red 'o' / Versicolor blue 's'.
- [cell 14, id 58261aed] Line plot: MSE (`ada.losses_`) vs Epochs, eta=0.01.
- [cell 15, id ba5b2a36] Line plot: Misclassified labels (`ada.missed_`) vs Epochs, eta=0.01.
- [cell 16, id 0bca99de] Decision regions (unscaled features), sepal length vs petal length.
- [cell 18, id c77b191c] 1x2 subplots: MSE vs Epochs for eta=0.1 (left) and eta=0.0001 (right).
- [cell 19, id 408aa8ea] 1x2 subplots: Misclassified labels vs Epochs for eta=0.1 and eta=0.0001.
- [cell 20, id 2a2a0f26] Decision regions for `ada1` (eta=0.1, unscaled).
- [cell 24, id 1886dd9a] Line plot: MSE vs Epochs (standardized features, eta=0.1).
- [cell 25, id 85f85d5f] Line plot: Misclassified labels vs Epochs (standardized, eta=0.1).
- [cell 26, id f91035c3] Decision regions on `X_std` (standardized axes).

## What is left as an exercise to the student
- No explicit "exercise" prompt. The closing markdown (cell 27, id 6d2dd11a) suggests as next steps: trying different feature subsets, adding regularization, and experimenting with alternative optimization (e.g. stochastic gradient descent). No starter/TODO code provided.

## Key cell indices for code idiom extraction
- "[cell 5, id 100854b8]: `y = np.where(y == 'Iris-setosa', 0, 1)` — binary label encoding from string classes"
- "[cell 10, id 0d71e0f7]: full AdalineGD class — net_input/activation(identity)/predict(step@0.5) + per-feature GD update `self.w_[j] += self.eta * (X[:, j] * errors).mean()`"
- "[cell 10, id 0d71e0f7]: weight init `rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])` with `np.random.RandomState`"
- "[cell 23, id 07f941ef]: manual per-column standardization `(X[:,j] - X[:,j].mean()) / X[:,j].std()`"
- "[cell 8, id 44efc974]: `plot_decision_regions` meshgrid idiom — `np.meshgrid(np.arange(...))`, `classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)`, `plt.contourf`"

## Notes / [VERIFY] flags
- Markdown net-input formula (cell 9, id dab6b4ad): `z = X · w + b`; loss = MSE; activation = identity. Matches code.
- Standardization formula (cell 22): `x_std = (x - mu)/sigma`. Matches code.
- Labels are 0/1 (not the more common Adaline -1/+1). The `predict` threshold is 0.5 accordingly — consistent with 0/1 encoding.
- The per-feature gradient loop (`for j in range(self.w_.shape[0])`) is mathematically equivalent to a vectorized `self.w_ += self.eta * X.T.dot(errors)/n` only because `.mean()` divides by n; note Adaline here uses `.mean()` (average gradient), whereas notebook 15's LogisticRegressionGD uses `X.T.dot(errors)/X.shape[0]`.
- No numeric loss/accuracy values are saved in outputs (notebook cells show no execution output for the training/plot cells). [VERIFY: reported convergence behavior described only in prose, no stored metrics.]
- `random_state` passed only via `__init__`; the comparison models `ada1`/`ada2` use default random_state=1.
