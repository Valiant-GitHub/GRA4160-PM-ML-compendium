# Exercise extract: Model selection, evaluation, and assessment (stub + VHL solution)

**Stub path:** `course_materials\Exercises and solutions(VHL)\05_Model_selection_evaluation_and_assessment.ipynb`   **Solution path:** `course_materials\Exercises and solutions(VHL)\05_Model_selection_evaluation_and_assessment_VHL.ipynb`
**Cell counts:** stub=2 (2 markdown, 0 code), solution=7 (markdown cells-0,1; code cells-2,3,4,5; trailing empty cell-6)

## What the exercise teaches (1-2 sentences)
Model selection and evaluation workflow on the Iris dataset: cross-validation to pick kNN's `k`, comparing several classifiers via accuracy and confusion matrices, k-fold CV across models, and `GridSearchCV` hyperparameter tuning for a Decision Tree. Part 5 (ROC / Precision-Recall) is discussion-only (no code).

## Setup
- **Dataset:** Iris — `from sklearn.datasets import load_iris`; `iris = load_iris()`; `X, y = iris.data, iris.target`. 150 samples, 4 features, 3 classes; `iris.target_names` used as confusion-matrix labels.
- **Target:** `iris.target` (3 species: setosa/versicolor/virginica).
- **Task:** multiclass classification, focused on evaluation/selection rather than building one model.
- **Expected output:** CV-accuracy-vs-k plot for kNN; per-model accuracy + confusion-matrix heatmaps; 5-fold CV accuracy bar chart across models; best Decision Tree hyperparameters, best CV accuracy, and feature importances.

## What the student must implement (from the stub)
Stub is markdown-only; the 5-part task list is stub cell-1 (identical in solution cell-1):
1. **kNN + Cross-Validation:** CV-evaluate kNN on Iris varying k=1..15; discuss best k and bias/variance.
2. **Comparing models via Accuracy + Confusion Matrix:** train kNN, Decision Tree, Logistic Regression, etc.; compare accuracy and confusion matrices.
3. **k-Fold CV with Multiple Models:** k-fold CV comparing kNN, Decision Trees, Logistic Regression, SVM.
4. **Hyperparameter Tuning with Decision Trees:** use `GridSearchCV` over `max_depth`, `min_samples_leaf`, `min_samples_split`.
5. **ROC and Precision-Recall Curves:** review sklearn docs; answer conceptual questions (multiclass adaptation, threshold trade-offs, PR vs ROC, AUC interpretation). Discussion-only.
Plus three "Additional Interpretation" discussion points (class overlap, practical significance, error analysis).

## Solution walkthrough (from _VHL)
- **cell-2 (Task 1):** imports `numpy`, `matplotlib.pyplot`, `load_iris`, `KNeighborsClassifier`, `cross_val_score`. `k_values = range(1, 15)`; for each k: `knn = KNeighborsClassifier(n_neighbors=k)`; `scores = cross_val_score(knn, X, y, cv=5)`; append `scores.mean()`; prints `k=..: CV Accuracy = ..`; then plots accuracy vs k.
- **cell-3 (Task 2):** `from sklearn.model_selection import train_test_split`; `from sklearn.tree import DecisionTreeClassifier`; `from sklearn.linear_model import LogisticRegression`; `from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay`. `train_test_split(X, y, test_size=0.8, random_state=42)` (NOTE: 80% test / 20% train — unusually large test). `models = {'kNN (k=5))': KNeighborsClassifier(n_neighbors=5), 'Decision Tree': DecisionTreeClassifier(random_state=42), 'Logistic Regression': LogisticRegression(max_iter=200, random_state=42)}`. For each: fit, predict, `accuracy_score`, `confusion_matrix`, and `ConfusionMatrixDisplay(..., display_labels=iris.target_names).plot(cmap=plt.cm.Blues)`.
- **cell-4 (Task 3):** `from sklearn.svm import SVC`. `models_cv = {'kNN': KNeighborsClassifier(n_neighbors=10), 'Decision Tree': DecisionTreeClassifier(random_state=42), 'Logistic Regression': LogisticRegression(max_iter=200, random_state=42), 'SVM': SVC(probability=True, random_state=42)}`. For each: `cross_val_score(model, X, y, cv=5).mean()`; bar plot of results.
- **cell-5 (Task 4):** `from sklearn.model_selection import GridSearchCV`. `param_grid = {'max_depth': [None,2,3,4,5], 'min_samples_leaf': [1,2,3], 'min_samples_split': [2,3,4]}`; `dtree = DecisionTreeClassifier(random_state=42)`; `grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy')`; `grid_search.fit(X, y)`; prints `best_params_`, `best_score_`; `best_dtree = grid_search.best_estimator_`; prints `best_dtree.feature_importances_`.
- **Methods/classes:** `sklearn.datasets.load_iris`; `sklearn.neighbors.KNeighborsClassifier`; `sklearn.model_selection.cross_val_score`, `train_test_split`, `GridSearchCV`; `sklearn.tree.DecisionTreeClassifier`; `sklearn.linear_model.LogisticRegression`; `sklearn.svm.SVC`; `sklearn.metrics.accuracy_score`, `confusion_matrix`, `ConfusionMatrixDisplay`.
- **Hyperparameters (exact):** kNN sweep `n_neighbors = 1..14`; comparison `KNeighborsClassifier(n_neighbors=5)`; CV `KNeighborsClassifier(n_neighbors=10)`; `DecisionTreeClassifier(random_state=42)` (else default); `LogisticRegression(max_iter=200, random_state=42)`; `SVC(probability=True, random_state=42)` (kernel default 'rbf'); `train_test_split(test_size=0.8, random_state=42)`; all `cv=5`; `GridSearchCV(scoring='accuracy', cv=5)` over the grid above. No numeric outputs are saved in this export (cells show no output).

## Common pitfalls (inferred from the solution / data)
- `test_size=0.8` in cell-3 trains on only 30 samples (20% of 150) — small training set inflates variance of the accuracy/confusion-matrix comparison; easy to misread.
- `k_values = range(1, 15)` stops at 14 (range is exclusive), despite the prompt saying "k=1 to 15".
- `GridSearchCV` is fit on the full `X, y` (no held-out test), so `best_score_` is the CV estimate, not test accuracy — answering Task 4's "held-out test set OR via CV" with CV.
- SVM uses `probability=True` (needed only if predicting probabilities/ROC); it slows fitting via internal CV calibration.
- The kNN dict key `'kNN (k=5))'` has an unbalanced parenthesis (cosmetic).
- Iris is small and easy; near-perfect accuracies make model differences subtle — the point is the evaluation machinery, not winning accuracy.

## What this exercise teaches that the others don't
- The only exercise centered on **evaluation methodology** rather than building a single model: `cross_val_score`, `GridSearchCV`, confusion matrices with `ConfusionMatrixDisplay`, and conceptual ROC/PR/AUC discussion. Only place tying `k` and tree depth explicitly to the bias-variance tradeoff via CV curves, and the only systematic (grid) hyperparameter search.

## Method page(s) it links to
- kNN, trees, logistic, ensembles (SVM noted but is the SVC here). Strongest link to model-selection / cross-validation / evaluation method pages (CV, GridSearchCV, confusion matrix, ROC/PR/AUC). Touches bias-variance.

## Notes / [VERIFY] flags
- Dataset is built into scikit-learn (`load_iris`) — no external file path.
- No cell outputs (accuracy numbers, best params) are saved in the VHL export.
- Task 5 (ROC/Precision-Recall) is intentionally code-free (review + discussion); no implementation to cite.
- Solution cell-6 is empty.
