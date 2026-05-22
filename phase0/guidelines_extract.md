# Guidelines extract

> Two guideline documents are extracted below. Both pertain to GRA 4160 take-home exams (Spring 2024 and Spring 2023). They describe grading conventions, allowed tools, expected answer format, AI-use rules, and submission. Used as a LENS for the drill section only. Quotes preserve the source wording.

---

## File 1: `phase0\raw_text\exam_exam_gra4160_guidelines.txt`
**(Source: Exam_GRA4160_Guidelines.pdf, 12 pages — Spring 2024, "Predictive modelling with machine learning," 30-hour take-home, 29.05.2024 09:00 – 30.05.2024 15:00)**

### Grading conventions
- **Total scale: 100 points.** Each of the four assignments **"counts for 25% of the total grade."** (Stated explicitly for Assignments 1, 2, 3, and 4.)
- **Suggested grade boundaries (out of 100 points):**
  - 0–40: F
  - 41–60: E
  - 61–70: D
  - 71–80: C
  - 81–90: B
  - 91–100: A
- **Per-exercise point allocations are itemized.** Examples:
  - Assignment 1 (25%): 1.1 = 3 pts, 1.2 = 4 pts, 1.3 = 4 pts, 1.4 = 4 pts, 1.5 = 5 pts, 1.6 = 5 pts.
  - Assignment 2 (25%): Guidelines 1 (data generation) = 10 pts, Guidelines 2 (tree-based models) = 15 pts.
  - Assignment 3 (25%): 3.1–3.5 = 5 pts each.
  - Assignment 4 (25%): 4.1–4.5 = 5 pts each.
- **What graders reward (recurring criteria):** correct loading/display of data; correct identification & handling of missing values **with justification of the chosen method** ("In this dataset missing values appear as zeros"); correct descriptive stats and frequency analysis; **well-reasoned model choices including strengths and weaknesses**; correct train/test split and **correct, well-explained cross-validation**; appropriate metrics ("e.g., accuracy, precision, recall, F1-score"); **comparative presentation of results**; thoughtful interpretation of feature influence; reflection on **limitations or biases**; and depth/critical analysis tied to real-world implications. For the neural-network assignment, points for architecture justification (neurons, activation functions), training-process description (optimizer, epochs, batch size, monitored metrics), and reflection on whether a neural network is the optimal model.

### Allowed tools / resources
- "All course material and your notes."
- "You can use the 3rd party Python libraries we used in class without any explanations (e.g., NumPy, Pandas, Scikit-learn, and **Keras**)."
- "Any other 3rd party Python libraries, but you must explain what they do and why you choose to use them."
- "Resources on the Internet, but if you include code pieces you find on-line you must list the URLs of these resources in the corresponding resources' cell at the end of each section."
- "You are NOT allowed to receive help from other people in solving the assignments."

### Expected answer format
- "At the beginning of each added cell, explicitly state the exercise you are addressing."
- "Ensure the cells are arranged logically and that all code cells execute error-free."
- "Ensure you document your code with comprehensive comments and explanations to keep it clean and readable."
- "Additionally, strive to make your non-code responses as concise and short as possible. **Long responses with irrelevant information can lead to a deduction of points.** Preferably, put your non-code responses in markdown cells."
- For Assignment 4 (Reflection on mini project): "For each of the following 5 questions, write a response of **no more than 300 words.**"
- "Remember to save and back up the notebook periodically."
- Clarifications: email Vegard H. Larsen at vegard.h.larsen@bi.no.

### Academic-integrity / AI-use rules
- "You are allowed to use generative AI tools to generate code snippets, but you must explain the code snippets in your own words."
- "Furthermore, you can use generative AI as an aid for all exercises, but you must explain why you chose to use generative AI tools and how you used them."
- "If you use generative AI tools, you must list the tools you used in the corresponding resources' cell at the end of each section."
- Plus the no-help-from-other-people rule (above).

### Submission format
- "Submit your completed notebook (the `.ipynb` file) via WISEflow before the deadline."
- "Be aware that the deadline is strict, so do not wait until the very last minute to submit."
- "Fill out your student ID at the designated area below."

### How answers are assessed (summary phrasing from the doc)
- Each guideline block tells the grader to **"Check"** technical correctness, **"Evaluate"** appropriateness/justification of methods, and **"Award points"/"Grade"** on depth of reasoning, clarity of presentation, and critical reflection. Justification and interpretation are weighted as heavily as correct execution.

---

## File 2: `phase0\raw_text\exam_exam_gra4160_spring2023_guidlines.txt`
**(Source: Exam_GRA4160_Spring2023_guidlines.pdf, 5 pages — Spring 2023, dated June 9, 2023)**

> This document is a per-assignment **methodological rubric / checklist** (what the solution should contain). Unlike the 2024 file, it does **not** state point totals, percentage weights, or grade boundaries. `[VERIFY: 2023 point/weighting scheme — not present in this file.]`

### Grading conventions / what is assessed
- Organized as four "Assignment Guidelines" sections, each a bulleted list of expected steps and the reasoning the student must show. No numeric points are given.
- Strong, repeated emphasis that **justification is graded alongside the choice itself**:
  - "Remember that justifications for your choices are equally important as the choices themselves. Your reasoning should take into consideration the nature of the data, the objectives of the assignment, and best practices in data handling."
  - "In each step of the process, you should aim to provide clear, concise code and commentary to illustrate your data handling decisions and techniques."
- **Assignment 1** (data prep on heart-disease data spread across four files): load and **concatenate the four files** into one DataFrame; check/report missing values **with counts** and compare completeness across files; justify missing-value handling (delete / mean / median / mode); decide whether to simplify the target to binary; convert categorical features; choose & justify feature scaling (normalization vs standardization); feature engineering (create new features e.g. age groups, plot distributions; transform continuous->categorical with justification); EDA (histograms, box plots, scatter plots; outliers; correlation matrix/heatmap; multicollinearity); feature selection (tree-based feature importance; Lasso/Ridge regularization — note these are for continuous variables, discuss handling a binary target).
- **Assignment 2** (regression & classification): Linear Regression to predict the age attribute — justify feature choice; evaluate with **MSE, RMSE, MAE, R-squared**. Logistic Regression for the target — evaluate with **accuracy, precision, recall, F1 score, ROC-AUC**. Decision Tree Classifier — justify configuration (feature selection, tree depth, splitting criterion, hyperparameters); same classification metrics. Model evaluation must discuss **why** each metric was selected and what it reveals about strengths/weaknesses.
- **Assignment 3** (ensemble methods): implement and compare four ensembles — (1) simple majority-vote ensemble of decision trees on random data subsets; (2) bagging ensemble on bootstrap samples; (3) random forest (random feature subsets) — explain how it differs from bagging; (4) boosting ensemble — explain how it differs from bagging/RF and the core boosting principle. Compare all four with **accuracy, precision, recall, F1 score, ROC-AUC**; discuss strengths/weaknesses and link differences to the **bias-variance trade-off**.
- **Assignment 4** (neural network on Telco churn): load `Telco-Customer-Churn.csv`; clean/preprocess (missing values, encode categoricals, normalize numerics) with rationale; justify train/test split ratio; design a **one-hidden-layer** network (describe activation functions, #neurons, why); describe training (loss function, optimizer, monitored metrics); evaluate with accuracy/precision/recall/F1/ROC-AUC and interpret; reflect on improvements (hyperparameter tuning, architecture changes, class imbalance); discuss whether a neural network is the best model and suggest alternatives.

### Allowed tools / resources
- **Not separately stated in this 2023 guidelines file.** It is a methodological rubric only. `[VERIFY: allowed-aids list for 2023 — see the 2023 cover sheet / its notebook; not in this file.]`

### Expected answer format
- "Throughout your solution, ensure your code is clear, well-annotated, and replicable."
- "Your commentary should reflect a solid understanding of [the methods] and their application to the problem at hand, as well as best practices in data analysis and machine learning."
- Repeated: "provide clear and concise code along with detailed commentary on your observations and decisions."

### Academic-integrity / AI-use rules
- **Not stated in this 2023 guidelines file.** `[VERIFY: AI-use / academic-integrity policy for 2023 — not present here; check the 2023 exam notebook itself.]`

### Submission format
- **Not stated in this 2023 guidelines file.** `[VERIFY: submission channel for 2023 — likely WISEflow `.ipynb` per the recurring pattern, but not in this document.]`

---

## Cross-document notes
- **Common, stable expectations across years:** clear, well-commented, error-free, replicable notebook code; **concise prose** (2024 explicitly penalizes verbosity; 300-word cap on reflection answers); **justification of every methodological choice graded as heavily as the choice**; appropriate, well-explained metrics; cross-validation and train/test discipline; interpretation, comparison, and reflection (incl. bias-variance and limitations/biases).
- **Only the 2024 guidelines** provide explicit point values, 25%-per-assignment weighting, and A–F grade boundaries. Do not assume identical weighting for 2023 or 2025.
- The **2025 take-home notebook** (catalogued separately) carries the same answer-format instructions verbatim (cell-labeling, concise non-code responses, resources cell, WISEflow `.ipynb` submission, allowed-aids and generative-AI rules) but names **PyTorch** rather than Keras as the class deep-learning library, and states no point values in the notebook itself.
