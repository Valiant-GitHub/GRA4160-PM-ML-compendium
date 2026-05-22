# Notebook extract: 12_Introducing_ensemble_methods.ipynb

**Source path:** Lecture notebooks/12_Introducing_ensemble_methods.ipynb
**Cell count:** 16 (cells 0-15; cell 15 is empty)

## Dataset(s) loaded
- **Iris** dataset from sklearn (Voting Classifier demo) [cell 4]:
  - `from sklearn.datasets import load_iris` [cell 1]
  - `X, y = load_iris(return_X_y=True)`
- **Breast cancer** dataset from sklearn (Bagging demo) [cell 9]:
  - `from sklearn.datasets import load_breast_cancer` [cell 1]
  - `data = load_breast_cancer()`; features `data.data`, target `data.target`; `data.feature_names` used for plot labels [cell 14]

## Preprocessing steps
- Iris [cell 4]: `train_test_split(X, y, test_size=0.50, random_state=2)` — 50/50 split. No scaling.
- Breast cancer [cell 9]: `train_test_split(data.data, data.target, test_size=0.4, random_state=42)` — 60/40 split. No scaling.

## Method(s) demonstrated
- **VotingClassifier** — sklearn. `from sklearn.ensemble import VotingClassifier, BaggingClassifier` [cell 1]
- **BaggingClassifier** — sklearn (with DecisionTree base estimator).
- Base learners: `from sklearn.tree import DecisionTreeClassifier`, `from sklearn.linear_model import LogisticRegression` [cell 1]
- Metrics: `from sklearn.metrics import accuracy_score, confusion_matrix, classification_report` [cell 1]
- All sklearn; no from-scratch implementation. Bootstrap/bagging/boosting are markdown theory [cells 2, 5, 6, 7].

## Hyperparameters set
- [cell 4] `clf1 = LogisticRegression(random_state=10, solver='lbfgs', max_iter=1000)`
- [cell 4] `clf2 = DecisionTreeClassifier(random_state=42)`
- [cell 4] `VotingClassifier(estimators=[('lr', clf1), ('dt', clf2)], voting='soft')`
- [cell 11] `tree_clf = DecisionTreeClassifier(random_state=42)`
- [cell 11] `BaggingClassifier(estimator=tree_clf, n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)`
- [cell 13] four bagging configs:
  - `bag_clf1`: n_estimators=100, max_samples=10
  - `bag_clf2`: n_estimators=200, max_samples=50
  - `bag_clf3`: n_estimators=500, max_samples=100
  - `bag_clf4`: n_estimators=1000, max_samples=200
  - (all with `estimator=tree_clf, bootstrap=True, n_jobs=-1, random_state=42`)

## Plots produced
- [cell 14] Horizontal bar chart of `tree_clf.feature_importances_` for the breast cancer features. y-ticks = `data.feature_names`; x = "Feature Importance", y = "Feature". (Only plot in the notebook.)

## What is left as an exercise to the student
- None explicitly stated (no exercise cells). Cell 13 demonstrates manual hyperparameter exploration but is fully worked.

## Key cell indices for code idiom extraction
- "[cell 4]: `VotingClassifier(estimators=[('lr', clf1), ('dt', clf2)], voting='soft')`" — voting ensemble idiom
- "[cell 11]: `BaggingClassifier(estimator=tree_clf, n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)`" — bagging idiom (note `estimator=` keyword, sklearn >=1.2; older `base_estimator=` deprecated)
- "[cell 12]: `accuracy_score`, `confusion_matrix`, `classification_report` triad for evaluation"
- "[cell 14]: `plt.barh(range(n), tree_clf.feature_importances_); plt.yticks(range(n), data.feature_names)`" — feature-importance bar idiom

## Notes / [VERIFY] flags
- **CODE/PROSE MISMATCH [cell 4]:** the comment `# Define the VotingClassifier with hard voting` precedes a call that actually uses `voting='soft'`. Also the comment `# Evaluate the LogisticRegression and RandomForestClassifier` is wrong — the second base learner is a DecisionTree, not a RandomForest.
- `BaggingClassifier` uses the `estimator=` keyword [cells 11, 13], which requires scikit-learn >= 1.2 (older versions use `base_estimator=`). `[VERIFY: target sklearn version supports estimator= keyword]`.
- The feature-importance plot [cell 14] is from a single `DecisionTreeClassifier` (`tree_clf`), not from the bagging ensemble.
- Cell 15 is empty.
