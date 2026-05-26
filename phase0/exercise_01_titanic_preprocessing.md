# Exercise extract: Data preprocessing (Titanic) (stub + VHL solution)

**Stub path:** `course_materials\Exercises and solutions(VHL)\01_Data_preprocessing_titanic.ipynb`   **Solution path:** none (stub-only; no `_VHL` file exists for Topic 01)
**Cell counts:** stub=1 (1 markdown, 0 code), solution=0

## What the exercise teaches (1-2 sentences)
Hands-on data preprocessing on the Titanic dataset: loading/inspecting data, handling missing values, treating outliers, encoding categorical variables, and computing simple conditional survival probabilities. It is a pure preprocessing/EDA drill with no model fitting required.

## Setup
- **Dataset:** `train.csv` (Titanic). The stub names only the bare filename `train.csv`; no loader call is given (no code cells). [VERIFY: exact loader path — the local repo has the file at `data\Titanic data\train.csv` (folder name contains a space), so a `pd.read_csv` path is not specified by the notebook itself.]
- **Columns referenced in the prompt (exact names):** `Fare`, `Sex`, `Embarked`, `Survived` (implied by "probability of survival"), `pclass` (note: stub writes it lowercase `pclass`; the standard Titanic column is `Pclass` [VERIFY: case]).
- **Target (implied):** `Survived` (survival probability questions).
- **Task:** preprocessing + descriptive analysis (no train/test split, no estimator).
- **Expected output:** an understanding of the data plus computed survival probabilities by class and by sex; written discussion of `pclass` meaning and why preprocessing matters.

## What the student must implement (from the stub)
All from the single markdown cell (cell-0); there are no code cells, blanks, or TODO markers — the entire exercise is a 7-part prompt:
1. (cell-0) **Load and Inspect:** load `train.csv` into a Pandas DataFrame; use `head()`, `info()`, `describe()`.
2. (cell-0) **Handle Missing Values:** identify columns with missing values; apply an imputation strategy (e.g., fill missing ages with median or mean, drop rows if necessary).
3. (cell-0) **Address Outliers:** inspect distribution of continuous variables, focusing on `Fare`; decide whether to cap, remove, or otherwise handle outliers.
4. (cell-0) **Encode Categorical Variables:** convert `Sex` and `Embarked` to numeric via one-hot or label encoding.
5. (cell-0) **Analyze Survival Probabilities:** survival probability by passenger class (1st/2nd/3rd); survival probability male vs female.
6. (cell-0) **Role of `pclass`:** explain what `pclass` represents and how/why class influences survival.
7. (cell-0) **Importance of Preprocessing:** explain why handling missing values and outliers matters before fitting an ML model, and impact on performance and interpretability.

## Solution walkthrough (from _VHL)
None — there is no `_VHL` solution notebook for this topic. No worked code, methods, or hyperparameters exist to cite.

## Common pitfalls (inferred from the solution / data)
- `Age` and `Cabin` have many missing values; `Embarked` has a few. Choosing mean vs median for `Age` matters because the distribution is skewed (median is the safer default).
- `Fare` is right-skewed with extreme high values; naive removal can drop legitimate first-class fares.
- One-hot encoding `Embarked` produces 3 columns (C/Q/S) plus handling of NaN; label-encoding `Sex` (0/1) vs one-hot is a modeling choice.
- Column-name case: the prompt uses lowercase `pclass`, but the Kaggle Titanic CSV uses `Pclass` (and `Survived`, `Sex`, `Embarked`, `Fare` are capitalized). Code that hard-codes `pclass` will KeyError. [VERIFY against actual `train.csv` header.]
- Computing "survival probability" = `groupby` mean of `Survived` (a 0/1 column); forgetting that `Survived` is the mean of a binary is a common conceptual slip.

## What this exercise teaches that the others don't
- The only exercise focused purely on **raw-data preprocessing and EDA** (missing-value imputation, outlier treatment, categorical encoding) with **no model**. Other exercises assume cleaned data or use sklearn toy datasets. Also the only one asking for hand-computed **conditional probabilities** directly from a real tabular dataset as descriptive statistics.

## Method page(s) it links to
- None of the modeling methods directly. It is foundational/preprocessing. Closest conceptual link is to data-prep that precedes OLS / logistic / trees, but no estimator is named. (Loosely supports: logistic regression and trees as downstream classifiers, but not invoked here.)

## Notes / [VERIFY] flags
- [VERIFY: loader path] Stub gives no `read_csv` call; local file is at `data\Titanic data\train.csv`.
- [VERIFY: column case] Stub uses `pclass` (lowercase); standard Titanic header is `Pclass`.
- [VERIFY] No solution notebook exists; "Solution walkthrough" intentionally empty.
- Stub cell-0 is the entire content; there is also a trailing empty cell in some exports but this file contains a single non-empty markdown cell.
