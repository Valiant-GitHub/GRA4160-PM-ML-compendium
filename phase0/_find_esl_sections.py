"""ESL has no embedded bookmarks, so build a section -> PDF-page map for the
sections Phase 3 needs (from references.md). Search each heading string; the
front-matter Contents listing appears on low pages, the real section start later
— we report all hits and mark the likely body page (first hit on page > 30).
"""
from pathlib import Path
import fitz

ROOT = Path(__file__).resolve().parent.parent
ESL = ROOT / "The Elements of Statistical Learning 2.E..pdf"

# (label, distinctive heading substring to search for)
NEEDED = [
    ("§3.2 Linear regression / least squares (OLS)", "Linear Regression Models and Least Squares"),
    ("§3.4 Shrinkage Methods (Ridge/Lasso)", "Shrinkage Methods"),
    ("§4.3 Linear Discriminant Analysis", "Linear Discriminant Analysis"),
    ("§4.4 Logistic Regression", "Logistic Regression"),
    ("§7.2 Bias, Variance and Model Complexity", "Bias, Variance and Model Complexity"),
    ("§7.3 Bias-Variance Decomposition", "The Bias"),  # 'The Bias–Variance Decomposition' (en-dash)
    ("§7.5 In-sample prediction error / AIC", "Estimates of In-Sample Prediction Error"),
    ("§7.7 BIC", "The Bayesian Approach and BIC"),
    ("§7.10 Cross-Validation", "Cross-Validation"),
    ("§7.11 Bootstrap Methods", "Bootstrap Methods"),
    ("§8.7 Bagging", "Bagging"),
    ("§9.2 Tree-Based Methods", "Tree-Based Methods"),
    ("§10.1 Boosting / AdaBoost", "Boosting Methods"),
    ("§11.3 Neural Networks", "Neural Networks"),
    ("§11.4 Fitting Neural Networks (backprop)", "Fitting Neural Networks"),
    ("§14.3 Cluster Analysis (K-means)", "Cluster Analysis"),
    ("§14.5 Principal Components", "Principal Components Analysis"),
    ("§15 Random Forests", "Random Forests"),
]

doc = fitz.open(ESL)
n = doc.page_count
lines = [f"# ESL — curated section -> PDF-page map (no embedded bookmarks)",
         f"# file: {ESL.name}  pages: {n}",
         "# 'body p' = first occurrence on a PDF page > 30 (skips the front-matter Contents).",
         "# Read a few pages from 'body p' onward to reach the derivation. Verify the heading on arrival.",
         ""]
for label, needle in NEEDED:
    hits = []
    for i in range(n):
        if doc.load_page(i).search_for(needle):
            hits.append(i + 1)  # 1-based for Read(pages=)
        if len(hits) >= 12:
            break
    body = next((p for p in hits if p > 30), (hits[0] if hits else None))
    lines.append(f"- {label}")
    lines.append(f"    body p≈{body}   |   all hits: {hits[:10]}")
doc.close()
(ROOT / "phase0" / "esl_toc.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
