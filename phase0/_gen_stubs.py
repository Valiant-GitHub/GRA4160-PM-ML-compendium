"""Phase 1 — generate stub .qmd pages for every nav entry (idempotent).

Creates a titled placeholder for each page so the website renders with the full
sidebar/nav intact and no broken links. Will NOT overwrite an existing file
(C2-safe) — so hand-authored pages (index, appendices) and any already-built
method pages are left untouched.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (relative path, title, family-class or "", phase that fills it)
PAGES = [
    ("big_picture/reading_the_data.qmd", "Reading the Data", "fam-cross", "Phase 4.4"),
    ("big_picture/workflow_patterns.qmd", "Workflow Patterns", "fam-cross", "Phase 4.5"),
    ("methods/ols.qmd", "Ordinary Least Squares (OLS)", "fam-linear", "Phase 2/3"),
    ("methods/knn.qmd", "k-Nearest Neighbors (kNN)", "fam-linear", "Phase 2/3"),
    ("methods/logistic.qmd", "Logistic Regression", "fam-linear", "Phase 2/3"),
    ("methods/lda.qmd", "Linear Discriminant Analysis (LDA)", "fam-linear", "Phase 2/3"),
    ("methods/adaline.qmd", "Adaline", "fam-linear", "Phase 2/3"),
    ("methods/regularization.qmd", "Regularization — Ridge, Lasso, Elastic Net", "fam-linear", "Phase 2/3"),
    ("methods/bias_variance.qmd", "Bias–Variance Trade-off", "fam-cross", "Phase 2/3"),
    ("methods/cv_info_criteria.qmd", "Cross-Validation & Information Criteria", "fam-cross", "Phase 2/3"),
    ("methods/decision_trees.qmd", "Decision Trees", "fam-tree", "Phase 2/3"),
    ("methods/random_forests.qmd", "Random Forests", "fam-tree", "Phase 2/3"),
    ("methods/ensembles.qmd", "Ensemble Methods", "fam-tree", "Phase 2/3"),
    ("methods/pca.qmd", "Principal Component Analysis (PCA)", "fam-unsup", "Phase 2/3"),
    ("methods/kmeans.qmd", "K-Means Clustering", "fam-unsup", "Phase 2/3"),
    ("methods/nn_basics.qmd", "Neural Network Basics", "fam-neural", "Phase 2/3"),
    ("methods/autodiff.qmd", "Automatic Differentiation", "fam-neural", "Phase 2/3"),
    ("methods/nn_pytorch.qmd", "Neural Networks with PyTorch", "fam-neural", "Phase 2/3"),
    ("methods/build_a_nn.qmd", "Build a Neural Network", "fam-neural", "Phase 2/3"),
    ("cross_method/decision_dashboard.qmd", "Decision Dashboard", "fam-cross", "Phase 1.4 / 4.1"),
    ("cross_method/family_comparisons.qmd", "Family Comparisons", "fam-cross", "Phase 4.2"),
    ("cross_method/showdowns.qmd", "Showdowns", "fam-cross", "Phase 4.3"),
    ("cross_method/exercise_dependency_map.qmd", "Exercise Dependency Map", "fam-cross", "Phase 4.6"),
    ("drill/index.qmd", "Drill — Index", "fam-cross", "Phase 5.3"),
    ("drill/exam_spring2025.qmd", "Drill: Spring 2025 Take-Home Exam", "fam-cross", "Phase 5.1"),
    ("drill/exercise_01_titanic_preprocessing.qmd", "Drill: Titanic Preprocessing", "fam-cross", "Phase 5.2"),
    ("drill/exercise_02_spam_naive_bayes.qmd", "Drill: Spam Filtering (Naive Bayes)", "fam-cross", "Phase 5.2"),
    ("drill/exercise_03_house_prices.qmd", "Drill: Predicting House Prices", "fam-cross", "Phase 5.2"),
    ("drill/exercise_04_mnist_digits.qmd", "Drill: Recognising Handwritten Digits", "fam-cross", "Phase 5.2"),
    ("drill/exercise_05_model_selection.qmd", "Drill: Model Selection & Evaluation", "fam-cross", "Phase 5.2"),
    ("drill/exercise_06_customer_segmentation.qmd", "Drill: Customer Segmentation", "fam-cross", "Phase 5.2"),
]

created, skipped = [], []
for rel, title, fam, phase in PAGES:
    dest = ROOT / rel
    if dest.exists():
        skipped.append(rel)
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    fm = ["---", f'title: "{title}"']
    if fam:
        fm.append(f'body-classes: {fam}')
    fm.append("---")
    body = (
        "\n".join(fm)
        + f"\n\n::: {{.callout-bigpicture}}\n**Under construction.** This page is filled in during **{phase}** of the build. "
        + "The structure, sources, and cross-links are planned in `phase0/concept_inventory.md`.\n:::\n"
    )
    dest.write_text(body, encoding="utf-8")
    created.append(rel)

print(f"Created {len(created)} stub(s):")
for r in created:
    print("  +", r)
if skipped:
    print(f"Skipped {len(skipped)} existing file(s):")
    for r in skipped:
        print("  =", r)
