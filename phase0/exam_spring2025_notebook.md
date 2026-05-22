# Exam extract: Spring 2025 take-home (representative format)
**Source:** `Lecture slides\Past exams\GRA4160_2025-05-26_kl_09_EP_2\Exam_GRA4160_spring2025.ipynb` (dataset description: `wdbc.names` in the same folder)   **Cell count:** 16

> LENS NOTE: This extract catalogues the representative take-home exam format ONLY to frame the drill section. It is NOT a content source for method pages, and the exercises below are NOT to be solved or copied as drill questions. Do not invent additional exam questions.

## Format & structure
- **Type:** 30-hour take-home exam. Window stated in the notebook header: **26.05.2025 09:00 - 27.05.2025 15:00**.
- **Title:** "Exam in GRA 4160 (Spring 2025) — Predictive modelling with machine learning".
- **Delivery format:** A single Jupyter notebook (`Exam_GRA4160_spring2025.ipynb`) that the student fills in. Each assignment block is followed by a `# Solution cell (add more if necesarry)` code cell and a markdown `#### List of resources used in this section:` cell.
- **Files attached to the exam (datasets provided):**
  1. `Exam_GRA4160_spring2025.ipynb` (the notebook itself)
  2. `Hitters.csv`
  3. `WholesaleCustomers.csv`
  4. `wdbc.data`
  5. `wdbc.names`
  6. `SimulatedData.csv`
- **Structure:** Four sections (assignments), each with multiple numbered exercises:
  1. Predict baseball salaries with regularized regression (`Hitters.csv`) — Exercises 1.1–1.7
  2. Customer segmentation with clustering / unsupervised learning (`WholesaleCustomers.csv`) — Exercises 2.1–2.7
  3. Ensemble methods for breast cancer prediction (`wdbc.data` + `wdbc.names`) — Exercises 3.1–3.8
  4. Predictive classification challenge on simulated data (`SimulatedData.csv`) — Exercises 4.1–4.9
- **Points:** No per-exercise or per-section point values are stated **in the notebook**. `[VERIFY: section weighting]` — the 2024 guidelines file states each of its four assignments counts 25%, but the 2025 notebook itself does not state weights.
- **Stated working instructions:** "At the beginning of each added cell, explicitly state the exercise you are addressing." Cells must be "arranged logically" and "all code cells execute error-free." Code must have "comprehensive comments and explanations." Non-code responses must be "as concise and short as possible" — "Long responses with irrelevant information can lead to a deduction of points." Non-code responses preferably in markdown cells. Save/back up periodically.
- **Allowed tools (from notebook):** All course material and notes; the 3rd-party Python libraries used in class without explanation (NumPy, Pandas, Scikit-learn, PyTorch); any other 3rd-party libraries with explanation; Internet resources (must list URLs in the resources cell); generative AI for code/aid (must explain in own words, justify use, and list tools in the resources cell). NOT allowed: help from other people.
- **Submission:** Submit the completed `.ipynb` via WISEflow before the (strict) deadline; fill in student ID in the designated cell.

## Question types present
- **Code-write (load/preprocess/model/evaluate):** present in nearly every exercise — e.g., 1.1, 1.4, 1.5, 1.6; 2.1, 2.3, 2.4, 2.5; 3.1, 3.2, 3.3, 3.4, 3.5; 4.1, 4.3, 4.4, 4.5, 4.6, 4.8.
- **Code-analyze / compare results:** 1.7 (compare OLS/Ridge/Lasso coefficients & performance), 3.6 (compare three tree models), 4.6/4.7 (compare models, choose final).
- **Interpret / explain in prose (markdown):** EDA interpretation (1.2, 2.2, 4.2), PC interpretation (2.3), cluster characterization (2.5, 2.6), feature-importance interpretation (3.7), reflection (4.9).
- **Conceptual / "explain the concept" (derive-ish, conceptual not algebraic):** 1.3 (explain regularization; Ridge vs Lasso), 3.8 (how Random Forest vs Gradient Boosting are constructed).
- **No T/F items and no algebraic-derivation items observed** in this notebook. `[VERIFY: none present]`.

## Question-by-question outline
(Paraphrased faithfully. NOT solved — catalogue only.)

### Assignment 1 — Regularized regression, dataset: `Hitters.csv`
- **1.1** Load `Hitters.csv`, show first five rows, examine structure (features & types), report #observations and #features, check for missing values and handle them appropriately. *Methods: data loading, missing-value handling.*
- **1.2** EDA: summary stats for numeric features; category distributions for categorical features; argue which variables likely most useful for predicting salary. *Methods: descriptive stats, EDA reasoning.*
- **1.3** Explain regularization in linear regression; key difference between Ridge and Lasso; when one is preferred. *Methods: conceptual — Ridge/Lasso.*
- **1.4** Prepare data: encode categoricals, standardize numerics, train/test split. *Methods: encoding, standardization, train/test split.*
- **1.5** Train OLS linear regression (no regularization); evaluate on test set with appropriate regression metrics; comment. *Methods: OLS, regression metrics.*
- **1.6** Train Ridge and Lasso; use cross-validation on training set to find optimal alpha for each; report chosen alphas; evaluate on test set with same metrics. *Methods: Ridge, Lasso, CV for hyperparameter tuning.*
- **1.7** Compare OLS/Ridge/Lasso; which best on test; discuss differences; examine coefficients (how Ridge/Lasso differ from OLS); did Lasso zero any coefficients (feature selection); implications for feature importance; interpret bias-variance trade-off. *Methods: model comparison, coefficient interpretation, bias-variance.*

### Assignment 2 — Clustering / unsupervised, dataset: `WholesaleCustomers.csv`
- **2.1** Load `WholesaleCustomers.csv`, inspect first rows, report #obs and #features, identify continuous vs categorical features, check/handle missing values (expected none). *Methods: data loading, inspection.*
- **2.2** Summary stats for spending features; which categories have highest mean spend and highest variability; consider value ranges; decide whether scaling is necessary and explain. *Methods: descriptive stats, scaling rationale.*
- **2.3** PCA on the six spending features (scale first if deemed necessary); determine #components to explain ~90% variance; interpret the first two PCs in terms of original features. *Methods: scaling, PCA, variance explained.*
- **2.4** K-Means clustering; use elbow method and/or silhouette score to choose a reasonable k. *Methods: K-Means, elbow/silhouette.*
- **2.5** Run K-Means with chosen k; describe cluster characteristics (mean spend per category), characterize clusters in plain terms; report #customers per cluster. *Methods: K-Means, cluster profiling.*
- **2.6** Relate clusters to provided Region and Channel labels; per-cluster distribution of Region/Channel; discuss whether alignment makes sense (labels not used in clustering — validation/interpretation aid). *Methods: cluster validation against labels.*
- **2.7** Discuss how to evaluate clustering quality; what metrics/methods assess meaningfulness; reflect on evaluating clusters without ground truth; judge whether found clusters are meaningful and justify. *Methods: clustering evaluation, conceptual.*

### Assignment 3 — Ensemble methods, dataset: `wdbc.data` + `wdbc.names`
- **3.1** Load by combining `wdbc.data` and `wdbc.names`; report #obs and #features; give feature names and what the target represents; check class balance (malignant vs benign). *Methods: data loading/joining, class balance.*
- **3.2** Train/test split with class proportion roughly maintained (stratified). *Methods: stratified split.*
- **3.3** Decision tree classifier; use CV on training set to choose a complexity constraint; refit with chosen params on full training set; report test performance. *Methods: decision tree, CV tuning.*
- **3.4** Random Forest classifier; evaluate on test set; compare with decision tree. *Methods: random forest, comparison.*
- **3.5** Gradient Boosting classifier; may tune #estimators / learning rate via CV; evaluate on test set. *Methods: gradient boosting, CV tuning.*
- **3.6** Compare decision tree vs random forest vs gradient boosting on the test set; which best and by how much; discuss via bias-variance trade-off and advantages of ensembles over a single tree. *Methods: model comparison, bias-variance.*
- **3.7** Identify most important features for the Random Forest; list top features and interpret. *Methods: feature importance.*
- **3.8** Explain difference in how Random Forests vs Gradient Boosting are constructed; why ensembles may beat a single tree. *Methods: conceptual — ensembles.*

### Assignment 4 — Predictive classification challenge, dataset: `SimulatedData.csv`
- **4.1** Load `SimulatedData.csv`, examine structure; #features and types; #instances; class distribution (% class 1 vs 0). *Methods: data loading, class distribution.*
- **4.2** EDA wrt target: distributions per numeric feature (optionally split by class), pairwise feature/target relationships, target proportion within categorical levels; summarize patterns (data is simulated, may be non-linear). *Methods: EDA.*
- **4.3** Prepare data: encode categoricals, scale if necessary; discuss choices (trees may not need scaling; distance/gradient models like k-NN or logistic regression do). *Methods: preprocessing decisions.*
- **4.4** Train at least two different classifiers (e.g., a simple baseline like logistic regression or single decision tree, plus a more complex model like random forest, gradient boosting, SVM, or neural network); justify model choices. *Methods: multiple classifiers.*
- **4.5** Hyperparameter tuning per model (CV on training set or hold-out validation subset); describe process and criteria. *Methods: hyperparameter tuning.*
- **4.6** Evaluate with accuracy, precision, recall, F1 (consider beyond accuracy if imbalanced); use CV/validation to estimate unseen performance; identify best model. *Methods: classification metrics.*
- **4.7** Compare models; choose final model to deploy and why (performance, complexity, interpretability); what this implies about the task (e.g., complex decision boundary). *Methods: model selection reasoning.*
- **4.8** Demonstrate using the final model to predict on new data (e.g., simulate loading `SimulatedData_test.csv` and `model.predict`); ensure preprocessing from 4.3 is applied to new data; code must be generalizable (grader runs notebook on a hidden test set). *Methods: prediction pipeline / deployment.*
- **4.9** Reflect on the overall process: what made the problem challenging (non-linear interactions, irrelevant features); how addressed; how to improve with more time/data. *Methods: reflection.*

## Recurring topics / framing patterns
- **Pipeline-style workflow per section:** load -> inspect/EDA -> handle missing values -> encode/scale -> train/test split -> model -> tune via cross-validation -> evaluate -> compare -> interpret.
- **Cross-validation for hyperparameter tuning** recurs (1.6 alpha, 3.3 tree complexity, 3.5 boosting params, 4.5 general).
- **Model comparison + bias-variance trade-off** is an explicit recurring deliverable (1.7, 3.6).
- **"Explain the concept" conceptual prompts** embedded among coding tasks (1.3, 2.7, 3.8).
- **Interpretation of feature importance / coefficients** (1.7, 3.7).
- **Scaling decision discussed, not assumed** (2.2, 2.3, 4.3).
- **Appropriate metrics emphasized** — regression metrics for Assignment 1; classification metrics (accuracy, precision, recall, F1) for Assignments 3–4, with explicit caution about imbalance in 4.6.
- **A hidden hold-out test set** is used by the grader for Assignment 4 — submitted code must generalize/run on unseen data.
- **Concise prose, code commenting, and resource-attribution** are framing constraints throughout.

## Datasets used
- **`Hitters.csv`** (Assignment 1): MLB player stats from the 1986 season plus career totals, with 1987 salaries. 20 features (performance metrics like hits, home runs, RBIs, plus categorical league and division); 322 players. Target = annual salary (some missing). After removing missing salaries, 263 observations remain. (Originally from the ISLR book; Kaggle "Hitters dataset".)
- **`WholesaleCustomers.csv`** (Assignment 2): Annual spending of 440 clients of a wholesale distributor across six product categories — Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen — plus two categorical attributes: Region (1/2/3) and Channel (1 = Horeca [Hotel/Restaurant/Café], 2 = Retail). Source: UCI Wholesale customers data.
- **`wdbc.data` + `wdbc.names`** (Assignment 3) — Wisconsin Diagnostic Breast Cancer (WDBC), from `wdbc.names`:
  - **What it is:** Features computed from a digitized image of a fine-needle aspirate (FNA) of a breast mass, describing cell-nuclei characteristics.
  - **#Instances:** 569.
  - **#Attributes:** 32 total = ID number + diagnosis + **30 real-valued input features**.
  - **Features:** Ten base measurements per nucleus — radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension — each reported as mean, standard error, and "worst" (mean of three largest values) -> 30 features. (E.g., field 3 = Mean Radius, field 13 = Radius SE, field 23 = Worst Radius.)
  - **Target:** Diagnosis — M = malignant, B = benign.
  - **Class distribution:** 357 benign, 212 malignant. Missing values: none. (Note: notebook prose says "212 Malignant, 357 Benign," matching `wdbc.names`.)
- **`SimulatedData.csv`** (Assignment 4): A simulated dataset for binary classification; multiple features and a binary target; non-trivial relationship with non-linear effects and noise. A separate hold-out test set in the same format is withheld for grading. (Exact #features, #instances, and class split are not stated in the notebook — students must determine them in 4.1.)

## Notes / [VERIFY]
- The notebook itself states **no point values / no section weights**. `[VERIFY: section weighting for Spring 2025]` — only the 2024 guidelines document states 25% per assignment; do not assume the same applies to 2025.
- `[VERIFY: exact #features/#instances/class split for SimulatedData.csv]` — not stated in the notebook by design (the student computes these in 4.1); the `SimulatedData.csv` file was not read for this extract.
- Library list in the 2025 notebook names **PyTorch** (not Keras) as a class library; the 2024 guidelines named Keras. Different years, different deep-learning stack.
- This notebook is the **representative** take-home format per the task brief; the three "question paper" PDFs are one-page cover sheets only (see `exam_question_papers.md`).
