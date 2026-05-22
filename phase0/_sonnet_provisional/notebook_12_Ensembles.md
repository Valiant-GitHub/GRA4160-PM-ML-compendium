# Notebook extract: 12_Introducing_ensemble_methods.ipynb

**Source path:** Lecture notebooks\12_Introducing_ensemble_methods.ipynb
**Cell count:** 16 (cell-0 through cell-15)

## Dataset(s) loaded
- Iris dataset: `load_iris(return_X_y=True)` — 150 samples, 4 features; 3-class target (cell-4)
  - No named columns referenced; used as raw arrays
- Breast Cancer dataset: `load_breast_cancer()` — 569 samples, 30 features; binary target (cell-9)
  - Accessed as `data.data` and `data.target`; feature names accessed as `data.feature_names` for plotting (cell-14)

## Preprocessing steps
- Iris: `train_test_split(X, y, test_size=0.50, random_state=2)` — 50/50 split (cell-4)
- Breast Cancer: `train_test_split(data.data, data.target, test_size=0.4, random_state=42)` — 60/40 split (cell-9)
- No feature scaling applied.

## Method(s) demonstrated
- `from sklearn.ensemble import VotingClassifier, BaggingClassifier` (cell-1)
- `from sklearn.tree import DecisionTreeClassifier` (cell-1)
- `from sklearn.linear_model import LogisticRegression` (cell-1)

### VotingClassifier (Iris dataset, cell-4):
- `clf1 = LogisticRegression(random_state=10, solver='lbfgs', max_iter=1000)` — sklearn
- `clf2 = DecisionTreeClassifier(random_state=42)` — sklearn
- `VotingClassifier(estimators=[('lr', clf1), ('dt', clf2)], voting='soft')` — sklearn soft voting

### BaggingClassifier (Breast Cancer dataset, cells 11, 13):
- Base estimator: `DecisionTreeClassifier(random_state=42)` — sklearn
- `BaggingClassifier(estimator=tree_clf, n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)` — main demo (cell-11)

### Feature importance (cell-14):
- `tree_clf.fit(X_train, y_train); tree_clf.feature_importances_` — single tree fit on breast cancer training data

## Hyperparameters set
- `LogisticRegression`: `random_state=10`, `solver='lbfgs'`, `max_iter=1000` (cell-4)
- `DecisionTreeClassifier` (VotingClassifier): `random_state=42`; all others `default` (cell-4)
- `VotingClassifier`: `voting='soft'` (cell-4)
- `DecisionTreeClassifier` (BaggingClassifier base): `random_state=42`; all others `default` (cell-11)
- `BaggingClassifier` main: `n_estimators=500`, `max_samples=100`, `bootstrap=True`, `n_jobs=-1`, `random_state=42` (cell-11)
- Hyperparameter tuning comparison — four configurations in cell-13:
  - `bag_clf1`: `n_estimators=100`, `max_samples=10`, `bootstrap=True`, `n_jobs=-1`, `random_state=42`
  - `bag_clf2`: `n_estimators=200`, `max_samples=50`, `bootstrap=True`, `n_jobs=-1`, `random_state=42`
  - `bag_clf3`: `n_estimators=500`, `max_samples=100`, `bootstrap=True`, `n_jobs=-1`, `random_state=42`
  - `bag_clf4`: `n_estimators=1000`, `max_samples=200`, `bootstrap=True`, `n_jobs=-1`, `random_state=42`

## Plots produced
- [cell-14]: Horizontal bar chart — feature importances of single DecisionTree on Breast Cancer data; `plt.barh(range(data.data.shape[1]), feature_importances)`; yticks = `data.feature_names`; xlabel "Feature Importance", ylabel "Feature"

## What is left as an exercise to the student
- No explicit TODO/exercise cells. Notebook is fully worked.

## Key cell indices for code idiom extraction
- "[cell-4]: `voting_clf = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2)], voting='soft'); voting_clf.fit(X_train, y_train); y_pred = voting_clf.predict(X_test)` — soft VotingClassifier idiom"
- "[cell-11]: `bag_clf = BaggingClassifier(estimator=tree_clf, n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1, random_state=42); bag_clf.fit(X_train, y_train)` — BaggingClassifier fit"
- "[cell-13]: four-configuration accuracy comparison loop (manual, not GridSearch) — `bag_clf1.score(X_test, y_test)` etc."
- "[cell-14]: `feature_importances = tree_clf.feature_importances_; plt.barh(range(data.data.shape[1]), feature_importances); plt.yticks(range(data.data.shape[1]), data.feature_names)` — feature importance bar chart"

## Notes / [VERIFY] flags
- VotingClassifier is fit with `voting='soft'` (cell-4) but the markdown (cell-3) introduces it as "hard voting" before switching to soft — [VERIFY: markdown says "hard voting scheme" in description but code uses `voting='soft'`.]
- `np.shape(X_train)` is called with a numpy function on what is an ndarray from sklearn — standard usage (cell-10).
- The base estimator parameter in BaggingClassifier is `estimator=` (not the deprecated `base_estimator=`) — confirms sklearn >= 1.2 API.
- Cell-15 is empty.
- The notebook is labelled "Lecture 7" (cell-0).
