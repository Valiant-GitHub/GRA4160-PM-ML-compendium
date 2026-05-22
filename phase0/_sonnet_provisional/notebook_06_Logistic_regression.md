# Notebook extract: 06_Logistic_regression.ipynb

**Source path:** Lecture notebooks\06_Logistic_regression.ipynb
**Cell count:** 17 (cell-0 through cell-16)

## Dataset(s) loaded
- `../data/titanic/train.csv` — columns used: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`; target: `Survived`
- Synthetic in-notebook array: `X_synth = np.array([[1,2],[2,4],[3,6],[4,8]])`, `y_synth = np.array([0,0,1,1])` (cell-14)

## Preprocessing steps
- `df = df.dropna()` — drop rows with any missing value (cell-5)
- `df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)` — encode Sex as binary (cell-5)
- Features selected: `X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]` (cell-5)
- `train_test_split(X, y, test_size=0.2, random_state=15)` (cell-5)

## Method(s) demonstrated
- `from sklearn.linear_model import LogisticRegression` — sklearn, NOT from scratch
- `from sklearn.metrics import accuracy_score` (cell-3)
- `from sklearn.metrics import confusion_matrix, classification_report` (cell-11)

## Hyperparameters set
- Titanic model: `LogisticRegression(solver='lbfgs', max_iter=500)` (cell-7)
- Synthetic demo: `LogisticRegression(solver='lbfgs')` — `max_iter` default (cell-14)

## Plots produced
- No matplotlib plots produced in this notebook. Outputs are printed tables (confusion matrix, classification report, coefficient DataFrame).

## What is left as an exercise to the student
- Cell-12 (Questions/Extensions): interpret coefficients; adjust decision threshold (e.g., 0.3 or 0.7) and observe effect on confusion matrix; engineer new features (e.g., travelling alone vs. with family).
- Cell-15 (Questions/Extensions): relate `predict_proba` to sigmoid; compare with `decision_function`; change synthetic input to `[10, 20]` and observe probability shift.

## Key cell indices for code idiom extraction
- "[cell-5]: `df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)` — binary encoding with apply/lambda"
- "[cell-7]: `log_reg = LogisticRegression(solver='lbfgs', max_iter=500); log_reg.fit(X_train, y_train); y_pred = log_reg.predict(X_test); score = accuracy_score(y_test, y_pred)` — canonical fit/predict/score idiom"
- "[cell-9]: `coef_df = pd.DataFrame(log_reg.coef_, columns=X_train.columns); coef_df['Intercept'] = log_reg.intercept_` — extract coefficients to DataFrame"
- "[cell-14]: `probabilities = clf_synth.predict_proba(x_new)` — predict_proba demo"

## Notes / [VERIFY] flags
- Model form: $\hat{y} = \frac{1}{1+e^{-z}}$ where $z = \beta_0 + \beta_1 x_1 + \ldots + \beta_p x_p$ (cell-1)
- Log-odds interpretation: $\ln\left(\frac{p}{1-p}\right)$ (cell-1)
- Default threshold is 0.5; notebook notes it can be adjusted (cell-1)
- Multi-class extensions described: One-vs-All (OvA) and Multinomial (Softmax) Regression (cell-1)
- Baseline accuracy mentioned: "if we predicted no survivors always" = `1 - y_test.mean()` (cell-7)
- `train_test_split` uses `random_state=15` (not the common 42) — cite exactly.
- The notebook is labelled "Lecture 4" in the header (cell-0).
