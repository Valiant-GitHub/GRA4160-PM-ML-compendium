# Exercise extract: Spam filtering with naive Bayes (stub + VHL solution)

**Stub path:** `Exercises and solutions(VHL)\02_Spam_filtering_with_naive_bayes (1).ipynb`   **Solution path:** `Exercises and solutions(VHL)\02_Spam_filtering_with_naive_bayes_VHL.ipynb`
**Cell counts:** stub=5 (5 markdown, 0 code; cell-4 is empty), solution=22 (5 markdown intro cells-0..4, then code cells-5..20, trailing empty cell-21)

## What the exercise teaches (1-2 sentences)
Builds a multinomial naive Bayes spam classifier on SMS text, first **from scratch** (manual class priors, conditional word probabilities, log-likelihood prediction rule) and then with scikit-learn's `MultinomialNB`, comparing accuracies. Reinforces Bayes' theorem, the conditional-independence assumption, Laplace-style smoothing, and log-space computation to avoid underflow.

## Setup
- **Dataset:** SMS Spam Collection — 5,572 labeled English SMS messages, tab-separated `label` + `message`, labels `ham`/`spam`. Source cited: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
- **Exact loader call (solution cell-7):**
  `pd.read_csv('../../data/smsspamcollection/SMSSpamCollection.csv', sep='\t', header=None, names=['label', 'message'])`
- **Columns (exact):** `label`, `message`. Label remapped via `df['label'].map({'spam': 0, 'ham': 1})` → spam=0, ham=1.
- **Target:** `label` (binary 0=spam, 1=ham).
- **Task:** binary text classification.
- **Expected output:** probability an SMS is spam; top-5 (printed top-10) words per class; from-scratch test accuracy; sklearn `MultinomialNB` test accuracy.

## What the student must implement (from the stub)
The stub is markdown-only (no code). The 9-part task list appears in stub cell-3 (and is repeated in solution cell-4):
1. Load `smsspamcollection`; inspect; map label to numeric `[0,1]`; hint: `pd.read_csv` with tab separator.
2. Train/test split via `from sklearn.model_selection import train_test_split`.
3. Build count feature matrix via `from sklearn.feature_extraction.text import CountVectorizer`.
4. Compute class probabilities P(C1=Spam), P(C2=No spam); report P(spam).
5. Compute conditional probabilities P(xi|C1), P(xi|C2); report probability of a word in a spam message.
6. Print the five most frequent words in spam vs not-spam messages.
7. Compute log P(C1|x) ∝ log P(C1) + Σ log P(xi|C1) and the analog for C2.
8. Write a **from-scratch** classifier (no sklearn) using the prediction rule; evaluate vs true labels.
9. Train sklearn `MultinomialNB`; evaluate on the test set.

## Solution walkthrough (from _VHL)
- **cell-5:** `import numpy as np`, `import pandas as pd`.
- **cell-6:** `ls ../../` (directory listing helper).
- **cell-7 (Sol 1):** load via `pd.read_csv(..., sep='\t', header=None, names=['label','message'])`; `df['label'] = df['label'].map({'spam': 0, 'ham': 1})`.
- **cell-8 (Sol 2):** `train_test_split(df['message'], df['label'], test_size=0.33, random_state=1)`.
- **cell-9 (Sol 3):** `from sklearn.feature_extraction.text import CountVectorizer`; `vectorizer = CountVectorizer()` (all params default); `X_train = vectorizer.fit_transform(X_train).toarray()`; `X_test = vectorizer.transform(X_test).toarray()`. (Note: reassigns `X_train`/`X_test` from text Series to dense arrays.)
- **cell-10/11:** inspect `np.shape(X_train)` and the sparse count matrix.
- **cell-12 (Sol 4):** `class_probs[0] = (y_train == 0).mean()`; `class_probs[1] = (y_train == 1).mean()`; prints P(spam) = `class_probs[0]`.
- **cell-13 (Sol 5):** `total_words_in_class_0 = X_train[y_train == 0].sum()`; same for class 1. Per-word counts with **additive smoothing of 0.1**: `count_x_0 = X_train[y_train == 0].sum(axis=0) + 0.1` (and `count_x_1`). `cond_probs_0 = count_x_0 / total_words_in_class_0` (and `cond_probs_1`). (Smoothing only added to numerator counts, not the denominator total — see pitfalls.)
- **cell-15..17 (Sol 6):** rank words via `pd.DataFrame(cond_probs_0).sort_values(by=0, ascending=False).head(10)` (top-10, not 5) for spam and ham; `idx_to_word(idx)` helper maps a column index back to the word using `vectorizer.vocabulary_`.
- **cell-18 (Sol 7):** log posteriors in log space: `log_probs[0] = np.log(class_probs[0]) + np.log(cond_probs_0).dot(X_test.T)`; analog for class 1. (Uses dot product of log conditional probs with the test count matrix.)
- **cell-19 (Sol 8):** `y_pred = np.argmax([log_probs[0], log_probs[1]], axis=0)`; `from sklearn.metrics import accuracy_score`; `print(accuracy_score(y_test, y_pred))`.
- **cell-20 (Sol 9):** `from sklearn.naive_bayes import MultinomialNB`; `model = MultinomialNB()` (all params default, incl. `alpha=1.0`); `model.fit(X_train, y_train)`; `y_pred_sklearn = model.predict(X_test)`; `print(accuracy_score(y_test, y_pred_sklearn))`.
- **Methods/classes (import path + class):** `sklearn.model_selection.train_test_split`; `sklearn.feature_extraction.text.CountVectorizer`; `sklearn.naive_bayes.MultinomialNB`; `sklearn.metrics.accuracy_score`; `numpy.argmax`.
- **Hyperparameters:** `train_test_split(test_size=0.33, random_state=1)`; `CountVectorizer()` = all defaults; manual smoothing constant = `0.1`; `MultinomialNB()` = all defaults (`alpha=1.0`, `fit_prior=True`). No printed numeric accuracy values are stored in this export (cells show no output).

## Common pitfalls (inferred from the solution / data)
- The from-scratch smoothing is applied as `counts + 0.1` in the numerator but divides by the **unsmoothed** total word count, so the conditional "probabilities" don't sum to 1 per class. It still works for argmax classification but is not a strict probability (a deliberate hack to avoid `log(0)`).
- `X_train`/`X_test` are reassigned from text Series to dense arrays in cell-9; running cells out of order breaks the pipeline.
- `.toarray()` densifies a large sparse matrix — memory-heavy for big vocabularies.
- Label mapping spam=0 / ham=1 is non-obvious (spam is the positive class index 0); `class_probs[0]` is P(spam).
- `CountVectorizer` must be `fit` on train only and `transform` on test (done correctly here) to avoid leakage.
- The from-scratch and sklearn label conventions match because both use the same `y` mapping, but `MultinomialNB` smoothing (`alpha=1.0`) differs from the manual `0.1`, so accuracies can differ slightly.

## What this exercise teaches that the others don't
- Only exercise using **text / NLP features** (`CountVectorizer` bag-of-words) and the only one that builds a **probabilistic classifier from scratch** (manual priors, conditional probabilities, log-likelihood, argmax) before validating against the library. Deepest treatment of Bayes' theorem and log-space numerical stability.

## Method page(s) it links to
- naive bayes (primary). Also touches: feature extraction (bag-of-words / CountVectorizer) and basic classification evaluation (accuracy).

## Notes / [VERIFY] flags
- [VERIFY: dataset path] Solution loads `../../data/smsspamcollection/SMSSpamCollection.csv`; the local `data\` folder does NOT contain this file (only Titanic and house-prices data are present locally), so the path may need adjustment for the build.
- Stub cell-4 is empty; solution cell-21 is empty.
- Prompt asks for "five most frequent" words but the solution prints `.head(10)`.
- No cell outputs (accuracy numbers) are saved in the VHL export.
