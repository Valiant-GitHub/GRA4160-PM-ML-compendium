# Lecture 1 — Introduction and Working with Data

**Source:** `Lecture slides/lecture1 (2).pdf` (32 PDF pages = 17 logical slides; progressive-build deck, cite by slide number `n/17`). Instructor: Vegard Høghaug Larsen. Date: Jan 7 2026.

## Topic
Course intro, ML workflow, train/val/test, over/underfitting, working with data in Jupyter.

## Key concepts taught
- **Course scope (slide 5/17):** predictive modeling with *traditional (pre-deep-learning)* methods. Core algorithms named: linear/logistic regression, decision trees, ensemble methods. Emphasis on writing code from scratch to understand mechanics; real workflows (preprocessing, training, tuning, evaluation).
- **ML workflow (slide 12/17):** 1. Define & Collect → 2. Preprocessing (clean, transform, encode) → 3. Model Selection → 4. Training & Tuning (fit, optimize hyperparameters) → 5. Evaluation (test on unseen, deploy).
- **Traditional ML vs Deep Learning (slide 13/17):** table — features manual vs automated; data small/medium structured vs large/unstructured; complexity lower (linear/trees/SVM) vs high (DNN); interpretability white-box vs black-box.
- **Over/underfitting (slide 14/17):** Underfitting = high bias, model too simple, misses trends; fix with more complex model / new features. Overfitting = high variance, fits noise; fix with regularization, more data, simpler model.
- **Train/Val/Test (slide 15/17):** Training 60–80% (fit params); Validation 10–20% (hyperparameter tuning & model selection); Test 10–20% (final eval on unseen). Use random splits or **stratification** for representative samples.
- **Why learn the math (slide 16/17):** even with AI + home exam — skill building, detect AI hallucinations, careers need people who can fix AI, exam rigor ("you must justify every step; 'the AI said so' is not valid").

## Notation
None formal this lecture (conceptual). "MSE" referenced later.

## Professor emphasis cues
- Course is a **practical, take-home / home-exam** course ("You have unlimited resources (AI, Home Exam)", slide 16/17) — confirms the build's framing.
- Exam rigor: must **justify every step**. This is the practitioner-thinking center of gravity the build targets.
- Hands-on / from-scratch implementation is a stated goal — explains why notebooks build methods manually (Adaline, autodiff, build-a-NN).

## Companion materials (slide 11/17)
- Lecture notebook `01_Working_with_data_in_jupyter_notebooks.ipynb`
- Exercise notebook `01_Data_preprocessing_titanic.ipynb`

## Cross-refs
Workflow → `big_picture/workflow_patterns.qmd`; over/underfitting & train/val/test → bias-variance, CV pages.
