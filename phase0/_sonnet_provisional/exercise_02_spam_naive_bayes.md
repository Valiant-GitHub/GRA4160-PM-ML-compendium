# Exercise extract: Spam filtering with naive Bayes (stub + VHL solution)

**Stub path:** `course_materials/Exercises and solutions(VHL)/02_Spam_filtering_with_naive_bayes (1).ipynb`
**Solution path:** `course_materials/Exercises and solutions(VHL)/02_Spam_filtering_with_naive_bayes_VHL.ipynb`
**Cell counts:** stub=4 (cells 0-3 markdown + cell-4 empty); solution=21 (cells 0-20, cells 5-20 are code)

## What the exercise teaches (1-2 sentences)

Builds a Naive Bayes spam classifier from scratch — deriving class and conditional word probabilities manually with log-likelihoods — and then replicates the result with sklearn's `MultinomialNB`, making the connection between probabilistic theory and library implementation explicit.

## Setup

- **Dataset:** SMS Spam Collection (`SMSSpamCollection.csv`)
  - **Exact loader call** (solution cell-7):
    ```python
    df = pd.read_csv('../../data/smsspamcollection/SMSSpamCollection.csv',
                     sep='\t', header=None, names=['label', 'message'])
    ```
  - **Columns:** `label` (string: `'spam'` / `'ham'`), `message` (raw SMS text)
  - **Label encoding** (cell-7): `df['label'] = df['label'].map({'spam': 0, 'ham': 1})`
    — Spam = 0, Ham = 1
  - **Size:** 5,572 SMS messages
  - **Source:** https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
- **Task:** Binary text classification (spam vs. ham)
- **Expected outputs:** Class probability P(spam) ≈ reported at runtime; top-5 words per class; custom classifier accuracy; sklearn `MultinomialNB` accuracy

## What the student must implement (from the stub)

All 9 tasks are in **stub cell-3** (markdown); cells-0 to -2 are theory exposition. No code scaffolding is provided.

1. (cell-3 task 1) Load `smsspamcollection` with `pd.read_csv`, tab-separated, no header; map labels to `{spam:0, ham:1}`
2. (cell-3 task 2) `train_test_split` — split messages and labels
3. (cell-3 task 3) Build count matrix with `CountVectorizer` (one feature = one word)
4. (cell-3 task 4) Compute class priors P(C=spam) and P(C=ham)
5. (cell-3 task 5) Compute conditional word probabilities P(x_i | C_1) and P(x_i | C_2)
6. (cell-3 task 6) Print the 5 most frequent words in spam and ham messages
7. (cell-3 task 7) Compute log-posterior: `log P(C|x) ∝ log P(C) + Σ log P(x_i|C)`
8. (cell-3 task 8) Write a from-scratch classifier using `argmax`; evaluate accuracy vs. true labels
9. (cell-3 task 9) Train `sklearn.naive_bayes.MultinomialNB`; compare accuracy to hand-rolled classifier

## Solution walkthrough (from _VHL)

**Cell-5:** `import numpy as np, pandas as pd`

**Cell-7 — Data loading** (task 1):
```python
df = pd.read_csv('../../data/smsspamcollection/SMSSpamCollection.csv',
                 sep='\t', header=None, names=['label', 'message'])
df['label'] = df['label'].map({'spam': 0, 'ham': 1})
```

**Cell-8 — Train/test split** (task 2):
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.33, random_state=1)
```

**Cell-9 — CountVectorizer** (task 3):
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train).toarray()
X_test  = vectorizer.transform(X_test).toarray()
```
- `fit_transform` on train only; `.toarray()` converts sparse to dense numpy array
- Vocabulary size = shape `X_train` column count (runtime output: `np.shape(X_train)`)

**Cell-12 — Class priors** (task 4):
```python
class_probs = {}
class_probs[0] = (y_train == 0).mean()   # P(spam)
class_probs[1] = (y_train == 1).mean()   # P(ham)
```

**Cell-13 — Conditional probabilities** (task 5):
```python
total_words_in_class_0 = X_train[y_train == 0].sum()
total_words_in_class_1 = X_train[y_train == 1].sum()
count_x_0 = X_train[y_train == 0].sum(axis=0) + 0.1   # Laplace-style additive smoothing α=0.1
count_x_1 = X_train[y_train == 1].sum(axis=0) + 0.1
cond_probs_0 = count_x_0 / total_words_in_class_0
cond_probs_1 = count_x_1 / total_words_in_class_1
```
- Smoothing constant: **0.1** (not 1 — non-standard Laplace; avoids log(0))

**Cells-15 to 17 — Top words** (task 6):
```python
top5_spam_words = pd.DataFrame(cond_probs_0).sort_values(by=0, ascending=False).head(10)
def idx_to_word(idx):
    print(list(vectorizer.vocabulary_.keys())[
          list(vectorizer.vocabulary_.values()).index(idx)])
for i in top5_spam_words.index: idx_to_word(i)
```
- Note: `.head(10)` is used despite task saying "5 most frequent" — solution inspects 10

**Cell-18 — Log-posterior computation** (task 7):
```python
log_probs = {}
log_probs[0] = np.log(class_probs[0]) + np.log(cond_probs_0).dot(X_test.T)
log_probs[1] = np.log(class_probs[1]) + np.log(cond_probs_1).dot(X_test.T)
```
- Matrix dot product `(vocab_size,) @ (vocab_size, n_test)` — vectorised over all test messages at once

**Cell-19 — From-scratch classifier** (task 8):
```python
y_pred = np.argmax([log_probs[0], log_probs[1]], axis=0)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
```

**Cell-20 — sklearn MultinomialNB** (task 9):
```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()   # alpha=1.0 default
model.fit(X_train, y_train)
y_pred_sklearn = model.predict(X_test)
print(accuracy_score(y_test, y_pred_sklearn))
```
- `MultinomialNB`: import path `sklearn.naive_bayes.MultinomialNB`; `alpha=1.0` (default, not overridden)

## Common pitfalls (inferred from the solution / data)

- Using `vectorizer.fit_transform` on the test set instead of `vectorizer.transform` — leaks test vocabulary into the model
- Forgetting smoothing: `log(0)` causes `-inf` in the log-posterior, crashing the argmax step
- Dividing `count_x` by the number of training examples instead of the total word count in that class
- `CountVectorizer` default lowercases and strips punctuation; students may not realize non-word characters are excluded
- `np.argmax([log_probs[0], log_probs[1]], axis=0)` — the axis must be 0 (over classes), not 1
- sklearn's `MultinomialNB` uses `alpha=1.0` (full Laplace); the hand-rolled solution uses `0.1` — the two accuracies will differ slightly

## What this exercise teaches that the others don't

The only exercise that requires building the full probabilistic model by hand before using the sklearn equivalent. It exposes the internal mechanics of Naive Bayes — prior estimation, smoothed conditional word probabilities, and the log-sum prediction rule — in a way that no other exercise does. It also introduces `CountVectorizer` as the standard text-to-feature-matrix tool.

## Method page(s) it links to

- **naive bayes** (primary)
- touches: feature extraction (bag-of-words), log-likelihood

## Notes / [VERIFY] flags

- [VERIFY] Exact file path: solution cell-7 uses `'../../data/smsspamcollection/SMSSpamCollection.csv'` — confirm this path relative to the notebook location resolves correctly in the site build
- [VERIFY] `test_size=0.33, random_state=1` in stub solution; stub markdown does not specify these values — confirm they are canonical for the exercise
- The stub notebook has cell-4 as an empty markdown cell; this is intentional (placeholder)
- The solution prints `.head(10)` for "top 5 words" — minor discrepancy from task wording; students may use `.head(5)`
