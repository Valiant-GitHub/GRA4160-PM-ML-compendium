# Exercise extract: Titanic data preprocessing (stub only)

**Stub path:** `Exercises and solutions(VHL)/01_Data_preprocessing_titanic.ipynb`
**Solution path:** none
**Cell counts:** stub=1 (single markdown cell containing all 7 exercise prompts)

## What the exercise teaches (1-2 sentences)

Introduces the full data-preprocessing pipeline — missing-value imputation, outlier handling, categorical encoding, and survival-probability analysis — using the Titanic dataset before any modelling is done.

## Setup

- **Dataset:** `train.csv` (Titanic training split, Kaggle/UCI)
  - Loader call (instructed, not implemented in stub): `pd.read_csv('train.csv')`
  - Relevant columns mentioned: `Age`, `Fare`, `Sex`, `Embarked`, `Pclass`, `Survived`
  - Target variable: `Survived` (binary, used for probability calculations only — no model is trained in this exercise)
- **Task:** Preprocessing + exploratory survival-probability analysis; no prediction step
- **Expected output:** Cleaned DataFrame; survival probabilities by `Pclass` and `Sex`

## What the student must implement (from the stub)

All 7 items are in **cell-0** (single markdown prompt cell — no code scaffolding provided):

1. Load `train.csv`; call `head()`, `info()`, `describe()`
2. Identify columns with missing values; apply imputation strategy (e.g., fill `Age` with median/mean; decide whether to drop rows)
3. Investigate distribution of `Fare`; decide how to handle outliers (cap, remove, or other)
4. Encode `Sex` and `Embarked` as numeric (one-hot or label encoding)
5. Calculate P(Survived | Pclass=1), P(Survived | Pclass=2), P(Survived | Pclass=3)
6. Compute P(Survived | Sex=male) and P(Survived | Sex=female)
7. [Discussion] Explain role of `Pclass` in survival; explain why preprocessing matters before ML fitting

## Solution walkthrough (from _VHL)

No VHL solution file exists for Topic 01. The stub itself provides sufficient method hints:
- Imputation: median for `Age` (hinted: "fill missing ages with the median or mean")
- Outlier handling: capping or removal for `Fare`
- Encoding: one-hot encoding or label encoding for `Sex`, `Embarked`
- Survival probabilities: `df.groupby('Pclass')['Survived'].mean()` pattern implied

[VERIFY: Confirm no `01_Data_preprocessing_titanic_VHL.ipynb` was provided — only stub seen]

## Common pitfalls (inferred from the exercise text)

- Imputing `Age` with the global mean rather than class- or sex-stratified median, which distorts distributions
- Dropping all rows with any NaN (loses too much data) vs. dropping only rows where all key features are missing
- Fitting the imputer/scaler on the full dataset before the train/test split (data leakage — no split is shown here, but students should be aware)
- Encoding `Embarked` before handling its missing values (can introduce a spurious "NaN" category)
- Treating `Pclass` as a continuous variable rather than an ordinal/categorical one

## What this exercise teaches that the others don't

Pure preprocessing focus with no modelling step: it isolates the data-cleaning decisions (imputation strategy, outlier treatment, encoding choices) as first-class learning objectives rather than as a preamble to model fitting. It is the only exercise that asks for manual probability calculations from grouped data rather than sklearn outputs.

## Method page(s) it links to

- No specific ML method; foundational for: OLS, logistic, kNN, regularization, trees, RF, ensembles (all subsequent exercises)

## Notes / [VERIFY] flags

- [VERIFY] Exact file path to `train.csv` — stub says "Load the `train.csv` file" without specifying a relative path. Other exercises load from `../../data/`; likely `../../data/titanic/train.csv` or similar.
- [VERIFY] No solution notebook confirmed. If one is added later, update this extract.
- The stub has exactly 1 cell (all markdown). There are no code cells, so cell-count for any auto-counter should be noted.
