# Lecture 10 — Traditional ML vs Deep Learning

**Source:** `lecture10 (1).pdf` (15 PDF pages = 15 logical slides; cite `n/15`). Date: March 11th, 2026. Instructor: Vegard Høghaug Larsen.

## Topic
Conceptual comparison of traditional machine learning vs deep learning: overview, application-by-application model comparisons, tools, emerging trends, and a practical decision checklist. (No new formulas — a synthesis/decision lecture.)

## Key concepts taught
- **Plan (slide 2/15):** Traditional ML vs DL overview; practical applications & model comparisons; tools used in practice; emerging trends; feedback/questions on mini project.
- **Traditional ML vs Deep Learning (slide 3/15):** Traditional ML — algorithms k-NN, decision trees, logistic regression, SVMs; manual feature selection required; interpretable, good for structured data and smaller datasets; example: decision trees as a flowchart. Deep Learning — multi-layer neural networks; automatic feature learning from data; excellent with large datasets, images, text, audio; high accuracy, high computational cost, low interpretability; often "black-box."
- **Performance vs Dataset Size (slide 4/15):** traditional ML plateaus quickly (diminishing returns beyond a threshold); DL scales better with data (keeps improving with larger datasets); with small data, traditional ML often outperforms DL (fewer parameters, less overfitting); with large data, DL leverages capacity for richer representations; crossover point depends on task complexity and feature quality. Quote: "More data beats a cleverer algorithm, but only if the algorithm can exploit it."
- **When to Choose Which (slide 5/15):** Traditional ML — limited data; interpretability critical (e.g. credit decisions). Deep Learning — large datasets; complex perceptual tasks (vision, NLP); compute available. Practice: start simple (Logistic, Random Forest) then move to DL if needed.
- **k-NN vs Neural Networks — Recommenders (slide 6/15):** k-NN — simple, memory-based, no explicit training; good for small-scale recommendations; slow at scale, limited to direct similarity. NNs — learn embeddings for user-item interactions; scalable, handle diverse data, highly accurate; used by Netflix, YouTube.
- **Logistic Regression vs DL — Text Classification (slide 7/15):** Logistic Regression — baseline, simple linear model, interpretable; effective with engineered features (e.g. TF-IDF); limited nuance in complex texts. DL (CNNs, LSTMs, Transformers) — surpassed traditional methods on NLP; learns word/context embeddings; more data-intensive, computationally demanding.
- **Decision Trees vs NNs — Medical (slide 8/15):** Trees/Ensembles — highly interpretable, suitable for small datasets, simple diagnostic rules, easy for clinicians to validate. DL — human-level accuracy (X-rays, pathology slides); harder to interpret ("black-box"); explainability (XAI) essential in practice.
- **Random Forests vs CNNs — Images (slide 9/15):** Random Forests — require extensive feature engineering; limited scalability and accuracy. CNNs — revolutionized image recognition; learn hierarchical features directly from pixels; state-of-the-art accuracy, computationally demanding.
- **Ensemble Methods vs DL — Finance (slide 10/15):** Ensembles — good with structured/tabular financial data; robust with smaller datasets; interpretable, less tuning. DL — powerful with large historical datasets; captures subtle complex interactions; harder to interpret, extensive tuning.
- **Summary table (slide 11/15):** Aspect → Traditional ML | Deep Learning:
  - Feature engineering: Manual | Automatic
  - Data requirement: Small–medium | Large
  - Interpretability: High | Low
  - Computational cost: Low | High
  - Structured data: Excellent | Good
  - Unstructured data: Limited | Excellent
  - Training time: Fast | Slow
  - Overfitting risk: Lower | Higher
  - Takeaway: "Neither approach dominates — the best choice depends on the data, the task, and the constraints."
- **Tools in Practice (slide 12/15):** Scikit-learn — traditional ML, quick prototyping, easy; CPU-based, integrates with pandas/numpy. TensorFlow/PyTorch — DL frameworks, GPU acceleration; TensorFlow for production deployments; PyTorch for research, dynamic graphs, HuggingFace integration.
- **Emerging Trends (slide 13/15):** Hybrid Models (DL features + interpretable ML); Transfer/Self-supervised Learning (large pre-trained models); Foundation Models (GPT, Stable Diffusion, general-purpose); AutoML/No-Code ML (automated model building, accessibility).
- **Transfer Learning in Practice (slide 14/15):** idea — take a model pre-trained on a large dataset and adapt to a new smaller task. Workflow: 1. start with a pre-trained model (ResNet for images, BERT for text); 2. freeze early layers (general features); 3. fine-tune later layers on your dataset. Works because early layers learn generic features (edges, textures, word meanings) that transfer; dramatically reduces data/training requirements. Example: fine-tuning BERT for sentiment analysis with a few hundred labeled examples.
- **Practical Decision Checklist (slide 15/15):** 1. Data size: <10k samples → start with traditional ML. 2. Data type: tabular → tree-based; images/text/audio → deep learning. 3. Interpretability: required by regulation/stakeholders → traditional ML or add XAI. 4. Compute budget: limited GPU → traditional ML or smaller DL. 5. Iteration speed: need quick experiments → scikit-learn faster to prototype. 6. Baseline first: always start with a simple model. Rule of thumb: if a gradient boosted tree (XGBoost/LightGBM) works well on tabular data, deep learning will rarely beat it by much.

## Notation
- No new mathematical notation introduced (synthesis lecture). Named algorithms/tools: k-NN, decision trees, logistic regression, SVMs, Random Forests, CNNs, LSTMs, Transformers, ResNet, BERT, GPT, Stable Diffusion; libraries scikit-learn, TensorFlow, PyTorch, HuggingFace, XGBoost, LightGBM.
- "XAI" = explainable AI; "AutoML" = automated machine learning; "TF-IDF" = term frequency–inverse document frequency feature representation.
- Quantitative thresholds stated: "<10k samples → traditional ML" (slide 15).

## R9 cross-check flags (vs ESL/ISL)
- No formulas to cross-check (conceptual lecture). The "data size vs performance" crossover narrative (slide 4) is qualitative; not from ESL/ISL — flag as opinion/heuristic, not a derived result.
- "Gradient boosted tree rarely beaten by DL on tabular data" (slide 15) is an empirical rule-of-thumb, not a theorem — record as professor heuristic, do not cite ESL.

## Professor emphasis cues
- Recurring "start simple / baseline first" mantra (slides 5, 15) — strongly emphasized exam-relevant principle.
- "Neither approach dominates — depends on data, task, constraints" (slide 11) — central thesis.
- Two memorable quotes: "More data beats a cleverer algorithm, but only if the algorithm can exploit it" (slide 4) and the XGBoost/LightGBM tabular rule-of-thumb (slide 15).
- Six-point decision checklist (slide 15) is the practical capstone — likely directly examinable.
- Mini-project feedback flagged on the agenda (slide 2).
- Typo in source: slide 2 reads "queestions" (transcribed as-is from the deck).

## Companion materials
No lecture-notebook or exercise filenames printed on slides. Mini-project feedback/Q&A session noted on the plan (slide 2).

## Cross-refs
→ `methods/traditional_vs_deep.qmd` (this comparison), and synthesizes all prior method pages: `methods/knn.qmd`, `methods/decision_trees.qmd`, `methods/logistic.qmd`, `methods/random_forests.qmd`, `methods/boosting.qmd`, `methods/neural_networks.qmd`. Feeds the decision dashboard / "when to use which" showdowns. Transfer learning + foundation models → `concepts/deep_learning_trends.qmd`. Tools section → `setup/tooling.qmd` (scikit-learn vs TF/PyTorch).
