# Notebook extract: 15_LogisticRegression.ipynb

**Source path:** course_materials/Lecture notebooks/15_LogisticRegression.ipynb
**Cell count:** 31 cells (ids: 13ae256f, be029d55, 5a80f92a, c4af8a90, 5697a510, a342d4e2, f14d7969, fa0b4810, dde959c6, d60ccbaa, e05e5377, 73078bbc, bc676da2, 50ee34c8, 8affaafa, 08c79e90, 8f5cca9f, 44ecc01c, ee8326af, 5c284f92, a16d3295, d059cb11, 5f0d2408, 1132a520, fd1dd9c4, 9f10168e, 886ee02c, a75fd8c6, d6708f6b, 976cd468, f4432549, 6b3b3496, b896bc6b, d5cac4b4, 07ba1fb6, 0db01570, 587787f5, 835b264a, e6b31df0... — final cell empty). Sequential indices used below.

## Dataset(s) loaded
- **Iris dataset** via `pd.read_csv` (cell id c4af8a90). URL `https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data`, `header=None`, `encoding='utf-8'`; fallback to local `'iris.data'` on `HTTPError`.
- Example 1 (id d60ccbaa): first 100 rows (Setosa+Versicolor); `X = df.iloc[0:100, [0, 2]].values` (sepal length, petal length), shape (100,2); `y = np.where(df.iloc[0:100,4].values == 'Iris-setosa', 0, 1)`.
- Example 2 (id 5c284f92): `X_all = df.iloc[0:100, :4].values` — same 2 classes, ALL 4 features (shape (100,4)); same binary y.
- Example 3 (ids d6708f6b): full 150 rows, all 3 classes; `X_all2 = df.iloc[:, :4].values` (150,4); `y_all` encoded 0=setosa, 1=versicolor, 2=virginica via chained `np.where`.

## Preprocessing steps
- [id d60ccbaa] `y = np.where(y == 'Iris-setosa', 0, 1)` (binary encode).
- [id d6708f6b] 3-class encode:
  - `y_all = np.where(y_all == 'Iris-setosa', 0, y_all)`
  - `y_all = np.where(y_all == 'Iris-versicolor', 1, y_all)`
  - `y_all = np.where(y_all == 'Iris-virginica', 2, y_all)`
- No standardization/scaling applied anywhere in this notebook (raw features used).

## Method(s) demonstrated
- **Logistic Regression via gradient descent** — FROM SCRATCH (numpy). Class `LogisticRegressionGD` (id a342d4e2). Adaline-style single linear unit with sigmoid activation + cross-entropy loss.
  - Architecture: single linear unit, no hidden layer, 1 output. `net_input(X) = np.dot(X, self.w_) + self.b_`.
  - **Activation**: logistic sigmoid — `activation(z)`: `z_clipped = np.clip(z, -250, 250); return 1./(1.+np.exp(-z_clipped))`.
  - **Loss**: binary cross-entropy (per-epoch logging):
    `output_clipped = np.clip(output, 1e-10, 1.0-1e-10)`,
    `loss = -y.dot(np.log(output_clipped)) - (1-y).dot(np.log(1-output_clipped)); loss /= X.shape[0]`.
  - **Optimizer**: full-batch GD. `self.w_ += self.eta * X.T.dot(errors) / X.shape[0]`; `self.b_ += self.eta * errors.mean()`; with `errors = (y - output)`.
  - Weight init: `rgen = np.random.RandomState(self.random_state)`, `self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])`, `self.b_ = 0.0`.
  - **Prediction**: `np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)`.
- **One-vs-All (OvA) multiclass** (id f4432549): loop `for k in range(3)`, build binary `y_k = np.where(y_all == k, 1, 0)`, fit a separate `LogisticRegressionGD`, store `probabilities[:, k] = lrgd_k.activation(lrgd_k.net_input(X_all2))`; final class = `np.argmax(probabilities, axis=1)`.
- `OneModel` class (id 835b264a): a minimal duplicate of `LogisticRegressionGD` (same fit/net_input/activation/predict) provided "for reference"; not trained/used.
- `plot_decision_regions(X, y, classifier, resolution=0.02)` helper (id fa0b4810) — identical idiom to notebook 14.

## Hyperparameters set
- `LogisticRegressionGD.__init__` defaults: `eta=0.01`, `n_iter=50`, `random_state=1`.
- Example 1 (id 50ee34c8): `LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)`.
- Example 2 (id a16d3295): `LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)` on `X_all`.
- Example 3 OvA (id f4432549): each of 3 models `LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)`.
- Weight-init scale `0.01`, sigmoid clip `[-250, 250]`, loss prob clip `[1e-10, 1-1e-10]`.
- `OneModel` defaults: `eta=0.01`, `n_iter=50`, `random_state=1` (never instantiated with overrides).

## Plots produced
- [id 73078bbc] Scatter: sepal length vs petal length, Setosa red 'o' / Versicolor blue 's'.
- [id 08c79e90] Scatter of data + overlaid predicted labels as black markers (`{0:'.', 1:'x'}`), Example 1.
- [id 44ecc01c] Decision regions for `lrgd` (Example 1). Note axis labels swapped: `xlabel='Petal length [cm]'`, `ylabel='Sepal length [cm]'` (mismatch with data — see VERIFY).
- [id 9f10168e] Scatter (features 0 vs 2) + predicted-label overlay for the 4-feature model (Example 2). No decision-region plot (model is 4D).
- (Example 3 produces printed output only — see below; no plots.)

## What is left as an exercise to the student
- No explicit exercise/TODO. Closing markdown (id 0db01570) notes that for true probabilities summing to 1 one would use *softmax regression* rather than OvA — framed as conceptual remark, not an assignment.

## Key cell indices for code idiom extraction
- "[id a342d4e2]: LogisticRegressionGD full class — sigmoid `1./(1.+np.exp(-np.clip(z,-250,250)))`, vectorized GD update `self.w_ += self.eta * X.T.dot(errors)/X.shape[0]`, cross-entropy loss"
- "[id a342d4e2]: numerically-safe cross-entropy `-y.dot(np.log(clip)) - (1-y).dot(np.log(1-clip))` / n"
- "[id f4432549]: One-vs-All loop building per-class binary labels and stacking sigmoid outputs into `probabilities[:,k]`, then `np.argmax(..., axis=1)`"
- "[id d6708f6b]: chained `np.where` to encode 3 string classes to 0/1/2"
- "[id 1132a520]: `lrgd.activation(lrgd.net_input(X_all))` to recover class-1 probabilities"

## Notes / [VERIFY] flags
- Markdown (id 13ae256f) frames this explicitly as a small modification of Adaline GD: difference = sigmoid activation + cross-entropy loss (vs identity + MSE in nb 14).
- Gradient-update note: here `X.T.dot(errors)/X.shape[0]` (vectorized, divided by n) vs nb 14's per-feature `(X[:,j]*errors).mean()`. Both average over samples; consistent.
- [VERIFY: axis-label mismatch] cell id 44ecc01c sets xlabel='Petal length' / ylabel='Sepal length', but X columns are [sepal length, petal length] so the labels are reversed relative to the plotted data. Flagged code/prose inconsistency.
- OvA caveat stated in markdown (id 6b3b3496): the 3 per-class sigmoid outputs do NOT sum to 1 (independent binary detectors). Accurate.
- No accuracy is computed/printed anywhere; outputs that exist are raw `predict` arrays and probability arrays (cells 5f0d2408, 1132a520, b896bc6b, d5cac4b4, 07ba1fb6). [VERIFY: no stored execution outputs / metrics in the saved notebook for these prints.]
- `lrgd.fit(X, y);` uses trailing semicolon to suppress repr (id 50ee34c8, a16d3295).
