# Notebook extract: 06_Logistic_regression.ipynb

**Source path:** Lecture notebooks/06_Logistic_regression.ipynb
**Cell count:** 17 (cells 0-16)

## Dataset(s) loaded
- **Titanic** training data: `pd.read_csv('../data/titanic/train.csv')` [cell 3]
  - Features actually used (X): `['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']` [cell 5]
  - Target (y): `df['Survived']` [cell 5]
- **Synthetic data** (predict_proba demo) [cell 14]:
  - `X_synth = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])`
  - `y_synth = np.array([0, 0, 1, 1])`
  - New point predicted: `x_new = np.array([[5, 10]])`

## Preprocessing steps
- [cell 5] `df = df.dropna()` — drop all rows with missing values
- [cell 5] `df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)` — encode Sex (male=1, female=0)
- [cell 5] `X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]`; `y = df['Survived']`
- [cell 5] `train_test_split(X, y, test_size=0.2, random_state=15)` — 80/20 split
- [cell 3] `warnings.filterwarnings('ignore')`

## Method(s) demonstrated
- **Logistic Regression** — sklearn. Import: `from sklearn.linear_model import LogisticRegression` [cell 3]
- Helpers: `from sklearn.model_selection import train_test_split`, `from sklearn.metrics import accuracy_score` [cell 3]; `from sklearn.metrics import confusion_matrix, classification_report` [cell 11]
- No from-scratch / numpy implementation. The sigmoid formula appears only in markdown theory [cell 1].

## Hyperparameters set
- [cell 7] `LogisticRegression(solver='lbfgs', max_iter=500)` — penalty=`default` (l2), C=`default` (1.0)
- [cell 14] `LogisticRegression(solver='lbfgs')` — max_iter=`default` (100), all other params `default`
- [cell 5] `train_test_split(test_size=0.2, random_state=15)`

## Plots produced
- None. (No matplotlib import; outputs are printed text + DataFrames only.)

## What is left as an exercise to the student
- [cell 12] Interpret coefficients: which features most influence survival.
- [cell 12] Threshold adjustment: change 0.5 to 0.3 / 0.7; observe effect on confusion matrix / accuracy.
- [cell 12] Feature engineering: e.g. "traveling alone vs with family".
- [cell 15] How probabilities relate to the sigmoid; compare `predict_proba(x)` vs `decision_function(x)`; change input to `[10, 20]`.

## Key cell indices for code idiom extraction
- "[cell 5]: `df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)`" — binary categorical encoding idiom
- "[cell 7]: `log_reg = LogisticRegression(solver='lbfgs', max_iter=500); log_reg.fit(X_train, y_train)`" — canonical fit
- "[cell 7]: baseline accuracy idiom — `no_survival_acc = round((1 - y_test.mean()), 3)`" (majority-class baseline)
- "[cell 9]: coefficient DataFrame — `pd.DataFrame(log_reg.coef_, columns=X_train.columns)` then `coef_df['Intercept'] = log_reg.intercept_`"
- "[cell 11]: `confusion_matrix(y_test, y_pred)` and `classification_report(y_test, y_pred)`"
- "[cell 14]: `clf_synth.predict_proba(x_new)` and `clf_synth.predict(x_new)`"

## Notes / [VERIFY] flags
- Markdown formula [cell 1]: $\hat{y} = \frac{1}{1 + e^{-z}}$ where $z = \beta_0 + \beta_1 x_1 + \ldots + \beta_p x_p$; log-odds $\ln\left(\frac{p}{1-p}\right)$.
- AIC/BIC not in this notebook (that's nb 08).
- No code/prose mismatch found. Markdown [cell 6] correctly says the `lbfgs` solver is used, matching the call.
- Note on data quirk: `dropna()` on the full Titanic frame (including the sparse `Cabin` column) drastically reduces rows, biasing the sample toward 1st-class passengers — not flagged in the notebook itself.
