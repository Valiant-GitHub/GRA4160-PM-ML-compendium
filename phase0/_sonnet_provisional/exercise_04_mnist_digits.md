# Exercise extract: Recognising handwritten digits (stub + VHL solution)

**Stub path:** `Exercises and solutions(VHL)/04_Recognising_handwritten_digits.ipynb`
**Solution path:** `Exercises and solutions(VHL)/04_Recognising_handwritten_digits_VHL.ipynb`
**Cell counts:** stub=2 (cells 0-1 markdown); solution=31 cells (cells 0-1 markdown, cells 2-30 code/markdown interleaved)

## What the exercise teaches (1-2 sentences)

Benchmarks four classifier families — multinomial logistic regression, decision tree, kNN, Naive Bayes, and SVM — on the sklearn digits dataset (8×8 pixel images) and introduces manual hyperparameter tuning by looping over candidate values, building intuition for the bias-variance trade-off across methods.

## Setup

- **Dataset:** sklearn built-in `digits` dataset
  - **Exact loader call** (solution cell-2):
    ```python
    from sklearn.datasets import load_digits
    digits = load_digits()
    ```
  - **Shape:** `digits.data` → (1797, 64); `digits.target` → (1797,)
  - **Features:** 64 pixel-intensity values (flattened from 8×8 images); range 0–16
  - **Target:** integer digit label 0–9 (10 classes)
  - **No external file** — fully in-memory via sklearn
- **Task:** 10-class image classification
- **Expected outputs:** accuracy scores for 5 classifiers; random test-image display with predicted label; hyperparameter grid results for Decision Tree and kNN

## What the student must implement (from the stub)

All 7 tasks are in **stub cell-1** (markdown). No code scaffolding:

1. Load `digits`; print shape of data and target
2. Display first 10 images with their labels (using `matplotlib`)
3. Split; train multinomial logistic regression; evaluate accuracy
4. Draw one random image from test set; display it; print predicted label
5. Train and compare: Decision Tree, kNN, Naive Bayes, SVM
6. Tune hyperparameters (`max_depth` for DT, `n_neighbors` for kNN)
7. (Optional) Explore the full MNIST dataset

## Solution walkthrough (from _VHL)

**Cell-2 — Load and inspect** (task 1):
```python
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
digits = load_digits()
print("Shape of digits.data:", np.shape(digits.data))    # (1797, 64)
print("Shape of digits.target:", np.shape(digits.target)) # (1797,)
```

**Cell-10 — Display 10 images** (task 2):
```python
_, axes = plt.subplots(nrows=1, ncols=10, figsize=(15, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Label: %i" % label)
    ax.axis('off')
```

**Cell-12 — Multinomial logistic regression** (task 3):
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X = digits.images.reshape((len(digits.images), -1))   # flatten: (1797, 64)
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
clf_lr = LogisticRegression(solver="lbfgs", max_iter=10000, random_state=42)
clf_lr.fit(X_train, y_train)
acc_lr = clf_lr.score(X_test, y_test)
# Result: 0.975
```
- `LogisticRegression`: import `sklearn.linear_model.LogisticRegression`
- `solver="lbfgs"` (explicit); `max_iter=10000` (explicit — default is 100, far too few for convergence); `C=1.0` (default, not overridden); `multi_class` defaults to `'auto'` which uses multinomial for lbfgs
- `test_size=0.2, random_state=42`

**Cells-15 to 16 — Random test-image display + prediction** (task 4):
```python
import random
random_idx = random.choice(range(len(X_test)))
img_to_disp = X_test[random_idx, :].reshape(8, 8)
plt.imshow(img_to_disp, cmap=plt.cm.gray_r, interpolation="nearest")
prediction = clf_lr.predict(X_test[random_idx, :].reshape(1, -1))
print("Predicted Label:", prediction[0])
```

**Cell-19 — Decision Tree baseline** (task 5a):
```python
from sklearn.tree import DecisionTreeClassifier
clf_dt = DecisionTreeClassifier(random_state=42)   # max_depth=None (default)
clf_dt.fit(X_train, y_train)
acc_dt = clf_dt.score(X_test, y_test)
# Result: 0.8417
```

**Cell-21 — kNN** (task 5b):
```python
from sklearn.neighbors import KNeighborsClassifier
clf_knn = KNeighborsClassifier(n_neighbors=5)   # n_neighbors=5 (default)
clf_knn.fit(X_train, y_train)
acc_knn = clf_knn.score(X_test, y_test)
# Result: 0.9861
```

**Cell-23 — Naive Bayes** (task 5c):
```python
from sklearn.naive_bayes import MultinomialNB
clf_nb = MultinomialNB()    # alpha=1.0 (default)
clf_nb.fit(X_train, y_train)
acc_nb = clf_nb.score(X_test, y_test)
# Result: 0.9111
```

**Cell-25 — SVM** (task 5d):
```python
from sklearn.svm import SVC
clf_svm = SVC(random_state=42)   # kernel='rbf' (default), C=1.0 (default)
clf_svm.fit(X_train, y_train)
acc_svm = clf_svm.score(X_test, y_test)
# Result: 0.9861
```

**Cell-28 — Decision Tree hyperparameter loop** (task 6):
```python
param_grid_dt = {'max_depth': [10, 20, None], 'min_samples_split': [2, 6, 10]}
dtc = DecisionTreeClassifier(random_state=42)
for max_depth in param_grid_dt['max_depth']:
    for min_samples_split in param_grid_dt['min_samples_split']:
        dtc.set_params(max_depth=max_depth, min_samples_split=min_samples_split)
        dtc.fit(X_train, y_train)
        score = dtc.score(X_test, y_test)
```
- Best DT result in grid: `max_depth=10, min_samples_split=6` → 0.8583

**Cell-30 — kNN hyperparameter loop** (task 6):
```python
param_grid_knn = {'n_neighbors': [3, 5, 7, 9, 11, 60]}
knn = KNeighborsClassifier()
for n_neighbors in param_grid_knn['n_neighbors']:
    knn.set_params(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
```
- Best kNN: `n_neighbors=7` → 0.9889

**Accuracy summary (solution runtime outputs):**

| Classifier | Accuracy |
|---|---|
| Logistic Regression (multinomial) | 0.975 |
| Decision Tree (default) | 0.8417 |
| kNN (k=5) | 0.9861 |
| Naive Bayes (MultinomialNB) | 0.9111 |
| SVM (RBF kernel) | 0.9861 |
| kNN (k=7, tuned) | 0.9889 |

## Common pitfalls (inferred from the solution / data)

- Not reshaping the images before passing to the classifier: `digits.images` has shape (1797, 8, 8) — must be flattened to (1797, 64) with `.reshape((len(digits.images), -1))`
- Setting `max_iter` too low for `LogisticRegression`; the default 100 iterations will not converge, causing a `ConvergenceWarning` and lower accuracy
- Using `GaussianNB` instead of `MultinomialNB`; pixel intensities are non-negative counts so Multinomial is more appropriate, though Gaussian will also run
- Forgetting `random_state` on the train/test split — results will differ across runs, making comparisons misleading
- Manual hyperparameter loop uses the same held-out test set for tuning, which is technically test-set snooping; proper approach would use `GridSearchCV` with cross-validation

## What this exercise teaches that the others don't

The only exercise that explicitly benchmarks five different classifier families on the same dataset and displays images. It introduces multi-class classification (10 classes) via logistic regression and makes clear that the sklearn `score()` method reports accuracy. The manual hyperparameter loop demonstrates the concept of grid search before the `GridSearchCV` abstraction is introduced in Exercise 05.

## Method page(s) it links to

- **logistic** (primary — multinomial extension)
- **kNN** (primary)
- **trees** (Decision Tree)
- **naive bayes** (MultinomialNB)
- touches: SVM (noted as "not covered in detail")

## Notes / [VERIFY] flags

- [VERIFY] `MultinomialNB` requires non-negative features; raw pixel values (0–16) satisfy this. Confirm pixels are not normalised before passing to MultinomialNB — solution does not apply `StandardScaler` here.
- [VERIFY] The stub mentions "multinomial extension" of logistic regression — in sklearn ≥1.0, `LogisticRegression(solver='lbfgs')` uses multinomial automatically for multi-class; no `multi_class='multinomial'` kwarg is needed.
- Solution interleaves Questions/Extensions markdown cells (cells 4, 13, 17, 20, 22, 24, 26, 29, 31) — these are discussion prompts, not code, and should be preserved in the drill section as reflection questions.
