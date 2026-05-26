# Exercise extract: Recognising handwritten digits (stub + VHL solution)

**Stub path:** `course_materials\Exercises and solutions(VHL)\04_Recognising_handwritten_digits.ipynb`   **Solution path:** `course_materials\Exercises and solutions(VHL)\04_Recognising_handwritten_digits_VHL.ipynb`
**Cell counts:** stub=2 (2 markdown, 0 code), solution=32 (markdown intro/Q&A cells interleaved; code cells-2,3,6,8,10,12,15,16,19,21,23,25,28,30)

## What the exercise teaches (1-2 sentences)
Multiclass image classification on the sklearn `digits` (8x8) dataset: load/inspect, visualize images, train a multinomial logistic regression baseline, predict on a random test image, then benchmark Decision Tree, kNN, Naive Bayes, and SVM, finishing with manual hyperparameter loops for the tree (`max_depth`, `min_samples_split`) and kNN (`n_neighbors`).

## Setup
- **Dataset:** scikit-learn `digits` — 1797 samples, 10 classes (digits 0-9), 8x8 images = 64 pixel features.
- **Exact loader call (solution cell-2):** `from sklearn.datasets import load_digits`; `digits = load_digits()`.
- **Feature/target objects:** `digits.data` (1797, 64), `digits.target` (1797,), `digits.images` (8x8 per sample).
- **Flattening (cell-12):** `X = digits.images.reshape((len(digits.images), -1))`; `y = digits.target`.
- **Target:** `digits.target` (integers 0-9).
- **Task:** 10-class classification; metric = accuracy via `model.score`.
- **Expected output:** data/target shapes; grid of first 10 images with labels; logistic-regression accuracy; a random test image with predicted label; accuracies for DT/kNN/NB/SVM; tuning tables.

## What the student must implement (from the stub)
Stub markdown (cells-0,1) gives a 7-part task list (identical in solution cells-0,1):
1. Load digits; print shapes of data and target.
2. Display first ten training images with their target values.
3. Split into train/test; train a (multinomial) logistic regression; evaluate accuracy.
4. Draw one random test image, display it, print predicted digit.
5. Train alternative classifiers and compare to logistic regression: Decision Tree, K-Nearest Neighbors, Naive Bayes, Support Vector Machine.
6. Tune hyperparameters (e.g., `max_depth` for DT, `n_neighbors` for kNN) to improve accuracy.
7. (Optional) Try the larger MNIST dataset (70,000 images).

## Solution walkthrough (from _VHL)
- **cell-2 (Sol 1):** `load_digits()`; print `np.shape(digits.data)` → (1797, 64) and `digits.target` → (1797,). Imports `numpy as np`, `matplotlib.pyplot as plt`.
- **cell-3/6/8:** inspect `digits.data[0]`, `digits.target`, `digits.images[2]` (8x8 array).
- **cell-10 (Sol 2):** `plt.subplots(nrows=1, ncols=10, figsize=(15,3))`; loop `zip(axes, digits.images, digits.target)`; `ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")`; titles "Label: %i".
- **cell-12 (Sol 3):** `from sklearn.linear_model import LogisticRegression`; `from sklearn.model_selection import train_test_split`. Flatten X; `train_test_split(X, y, test_size=0.2, random_state=42)`. `clf_lr = LogisticRegression(solver="lbfgs", max_iter=10000, random_state=42)` (multinomial is the default for lbfgs); fit; `acc_lr = clf_lr.score(X_test, y_test)` → **0.975**.
- **cell-15/16 (Sol 4):** `import random`; `random_idx = random.choice(range(len(X_test)))`; reshape to 8x8 and `plt.imshow(...)`; `clf_lr.predict(X_test[random_idx,:].reshape(1,-1))` prints predicted label.
- **cell-19 (Sol 5a):** `from sklearn.tree import DecisionTreeClassifier`; `clf_dt = DecisionTreeClassifier(random_state=42)` (other params default); accuracy → **0.8417**.
- **cell-21 (Sol 5b):** `from sklearn.neighbors import KNeighborsClassifier`; `clf_knn = KNeighborsClassifier(n_neighbors=5)`; accuracy → **0.9861**.
- **cell-23 (Sol 5c):** `from sklearn.naive_bayes import MultinomialNB`; `clf_nb = MultinomialNB()` (defaults, `alpha=1.0`); accuracy → **0.9111**.
- **cell-25 (Sol 5d):** `from sklearn.svm import SVC`; `clf_svm = SVC(random_state=42)` (default RBF kernel); accuracy → **0.9861**.
- **cell-28 (Sol 6 — DT tuning):** grid `max_depth ∈ [10,20,None]`, `min_samples_split ∈ [2,6,10]`; loop with `dtc.set_params(...)`. Best around `max_depth=10, min_samples_split=6` → 0.8583.
- **cell-30 (kNN tuning):** `n_neighbors ∈ [3,5,7,9,11,60]`; best `n_neighbors=7` → 0.9889; `n_neighbors=60` drops to 0.9472.
- **Methods/classes:** `sklearn.datasets.load_digits`; `sklearn.model_selection.train_test_split`; `sklearn.linear_model.LogisticRegression`; `sklearn.tree.DecisionTreeClassifier`; `sklearn.neighbors.KNeighborsClassifier`; `sklearn.naive_bayes.MultinomialNB`; `sklearn.svm.SVC`.
- **Hyperparameters (exact):** split `test_size=0.2, random_state=42`; `LogisticRegression(solver="lbfgs", max_iter=10000, random_state=42)` (multinomial default, `C` default 1.0); `DecisionTreeClassifier(random_state=42)` (all else default); `KNeighborsClassifier(n_neighbors=5)`; `MultinomialNB()` defaults; `SVC(random_state=42)` (kernel default 'rbf', `C` default 1.0). Tuning grids as above.

## Common pitfalls (inferred from the solution / data)
- Must flatten `digits.images` to (n, 64) before fitting; passing 3D `images` to sklearn fails.
- `random.choice` (no fixed seed in cell-15) gives a different image each run; the saved output happened to land on index 233 / label 1.
- `MultinomialNB` works on digits only because pixel intensities are non-negative counts (0-16); it would be inappropriate for centered/negative features (use `GaussianNB` instead) — flagged in the notebook's Q&A.
- SVM is scale-sensitive; here raw 0-16 pixels work but the notebook notes preprocessing/scaling could help.
- Very large `n_neighbors` (e.g., 60) over-smooths and lowers accuracy (0.9472), illustrating bias rising as k grows.
- Logistic regression needs a high `max_iter` (10000) to converge on 64 features.

## What this exercise teaches that the others don't
- The only **multiclass (10-class) image** problem and the broadest **model bake-off**: logistic vs tree vs kNN vs naive Bayes vs SVM on one dataset, with manual hyperparameter sweeps. Strong intuition pump for bias-variance via `n_neighbors` and `max_depth`. Only exercise that visualizes raw input images and does single-instance prediction.

## Method page(s) it links to
- logistic regression (multinomial baseline), kNN, trees, naive bayes. Also SVM (noted as not covered in detail in the course). Touches hyperparameter tuning / model selection.

## Notes / [VERIFY] flags
- Dataset is built into scikit-learn (`load_digits`) — no external file path, so no path-mismatch risk.
- Accuracy values above are the exact saved outputs in the VHL notebook (logistic 0.975, DT 0.8417, kNN 0.9861, NB 0.9111, SVM 0.9861).
- Stub has only the 2 intro markdown cells (no blanks/TODO code) — the whole exercise is the prompt list.
