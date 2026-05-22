# Lecture 10 — Traditional ML vs Deep Learning

**Source:** `lecture10 (1).pdf` (15 PDF pages = 15 logical slides; cite `n/15`). Date: March 11, 2026.

## Topic
Comparative survey: traditional ML vs. deep learning across task types (structured, images, text, medical, finance, recommendations); performance vs. dataset size; practical tools (scikit-learn, TensorFlow, PyTorch); emerging trends (hybrid models, transfer learning, foundation models, AutoML); practical decision checklist for model selection.

## Key concepts taught — Traditional ML vs Deep Learning Overview

- **Characterization (slide 3/15):**
  - **Traditional ML:** algorithms = k-NN, decision trees, logistic regression, SVMs. Manual feature selection required. Interpretable; good for structured data and smaller datasets. Example: decision trees as flowcharts.
  - **Deep Learning:** multi-layer neural networks. Automatic feature learning from data. Excellent with large datasets, images, text, audio. High accuracy, high computational cost, low interpretability. Often "black-box" models.
- **Performance vs dataset size (slide 4/15):**
  - Traditional ML plateaus quickly — adding more data beyond a threshold yields diminishing returns.
  - Deep learning scales better with data: performance continues to improve with larger datasets.
  - With **small data**: traditional ML often outperforms DL (fewer parameters, less overfitting).
  - With **large data**: DL leverages capacity to learn richer representations.
  - Key insight: the crossover point depends on task complexity and feature quality.
  - Quote on slide: *"More data beats a cleverer algorithm, but only if the algorithm can exploit it."*
- **When to choose which (slide 5/15):**
  - **Traditional ML:** limited data; interpretability critical (e.g., credit decisions).
  - **Deep Learning:** large datasets available; complex, perceptual tasks (vision, NLP); computational resources available.
  - **Practice heuristic:** start simple (Logistic Regression, Random Forest), then move to DL if needed.

## Key concepts taught — Head-to-Head Comparisons by Domain

- **k-NN vs Neural Networks (Recommenders) (slide 6/15):**
  - k-NN: simple, memory-based, no explicit training. Good for small-scale recommendations. Slow at scale, limited to direct similarity.
  - Neural Networks: learns embeddings for user-item interactions. Scalable, handles diverse data, highly accurate. Used by Netflix, YouTube.
- **Logistic Regression vs DL (Text Classification) (slide 7/15):**
  - Logistic Regression: baseline, simple linear model, interpretable. Effective with engineered features (e.g., TF-IDF). Limited nuance in complex texts.
  - Deep Learning (CNNs, LSTMs, Transformers): surpassed traditional methods on NLP tasks. Learns word/context embeddings. More data-intensive, computationally demanding.
- **Decision Trees vs Neural Networks (Medical) (slide 8/15):**
  - Decision Trees/Ensembles: highly interpretable, suitable for small datasets. Used for simple diagnostic rules. Easy to validate by clinicians.
  - Deep Learning: human-level accuracy (X-rays, pathology slides). Harder to interpret ("black-box"). Explainability (XAI) essential in practice.
- **Random Forests vs CNNs (Images) (slide 9/15):**
  - Random Forests: requires extensive feature engineering. Limited scalability and accuracy.
  - CNNs: revolutionized image recognition. Learn hierarchical features directly from pixels. State-of-the-art accuracy, computationally demanding.
- **Ensemble Methods vs DL (Finance) (slide 10/15):**
  - Ensemble Methods: good with structured/tabular financial data. Robust with smaller datasets. Interpretable, less tuning required.
  - Deep Learning: powerful with large historical datasets. Captures subtle, complex interactions. Harder to interpret, extensive tuning required.

## Key concepts taught — Summary Comparison Table

- **Summary table (slide 11/15):**

| Aspect | Traditional ML | Deep Learning |
|---|---|---|
| Feature engineering | Manual | Automatic |
| Data requirement | Small–medium | Large |
| Interpretability | High | Low |
| Computational cost | Low | High |
| Structured data | Excellent | Good |
| Unstructured data | Limited | Excellent |
| Training time | Fast | Slow |
| Overfitting risk | Lower | Higher |

  - Closing note: "Neither approach dominates — the best choice depends on the data, the task, and the constraints."

## Key concepts taught — Tools in Practice

- **Scikit-learn (slide 12/15):** traditional ML, quick prototyping, easy to use. CPU-based, integrates with pandas, numpy.
- **TensorFlow/PyTorch (slide 12/15):** deep learning frameworks, GPU acceleration.
  - TensorFlow: production deployments.
  - PyTorch: research, dynamic graphs, HuggingFace integration.

## Key concepts taught — Emerging Trends

- **Trends (slide 13/15):**
  - **Hybrid Models:** combining DL features with interpretable ML.
  - **Transfer/Self-supervised Learning:** leveraging large, pre-trained models.
  - **Foundation Models:** GPT, Stable Diffusion, general-purpose models.
  - **AutoML/No-Code ML:** automated model building, increased accessibility.
- **Transfer Learning in practice (slide 14/15):**
  - **Idea:** take a model pre-trained on a large dataset and adapt it to a new, smaller task.
  - **Workflow:**
    1. Start with a pre-trained model (e.g., ResNet for images, BERT for text).
    2. Freeze early layers that capture general features.
    3. Fine-tune later layers on specific dataset.
  - **Why it works:** early layers learn generic features (edges, textures, word meanings) that transfer across tasks.
  - Dramatically reduces data requirements and training time.
  - Example: fine-tuning pre-trained BERT for sentiment analysis can achieve strong results with just a few hundred labeled examples.

## Key concepts taught — Practical Decision Checklist

- **Decision checklist (slide 15/15):** when choosing between traditional ML and deep learning:
  1. **Data size:** <10k samples? Start with traditional ML.
  2. **Data type:** tabular → tree-based methods; images/text/audio → deep learning.
  3. **Interpretability:** required by regulation or stakeholders? Prefer traditional ML or add XAI methods.
  4. **Compute budget:** limited GPU access? Traditional ML or smaller DL models.
  5. **Iteration speed:** need quick experiments? Scikit-learn is faster to prototype.
  6. **Baseline first:** always start with a simple model — you need a benchmark to justify complexity.
  - **Rule of thumb:** "if a gradient boosted tree (XGBoost/LightGBM) works well on your tabular data, deep learning will rarely beat it by much."

## Notation

- No new mathematical notation introduced in this lecture. Terminology: XAI = Explainable AI; TF-IDF = Term Frequency-Inverse Document Frequency; CNN = Convolutional Neural Network; LSTM = Long Short-Term Memory; BERT = Bidirectional Encoder Representations from Transformers; ResNet = Residual Network; GPT = Generative Pre-trained Transformer.
- k-NN: the `k` here is number of neighbors — consistent with earlier lectures' use but a different concept from k-fold (Lecture 5) and k clusters (Lecture 6). Note in notation_table as a third collision.

## R9 cross-check flags (vs ESL/ISL)

- No formulas in this lecture. Comparative claims are qualitative/empirical.
- [VERIFY: The claim "DL rarely beats XGBoost on tabular data" is a practitioner rule of thumb attributed to the ML community broadly (Shwartz-Ziv & Armon 2022, "Tabular Data: Deep Learning is Not All You Need"). Confirm whether any citation is given in companion materials.]
- Transfer learning workflow (freeze early layers, fine-tune later): standard practice. [VERIFY: ISL 2nd ed. §10.9 covers transfer learning — check whether course reading assignment includes this section.]

## Professor emphasis cues

- The `<10k samples → traditional ML` threshold on slide 15/15 is a concrete, memorable rule — likely to appear as an exam scenario.
- The XGBoost/LightGBM rule of thumb on slide 15/15 is stated as a direct "rule of thumb" — treat as an exam-ready answer for "when would you choose DL over traditional ML on tabular data?"
- "Start simple (Logistic, Random Forest) then move to DL if needed" (slide 5/15) — the default workflow heuristic; reinforces the "baseline first" principle of slide 15/15.
- The 8-row comparison table on slide 11/15 is the kind of structure profs put on exams; memorize all 8 dimensions.
- Slide 2/15 mentions "Feedback and questions on mini project" — signals this lecture falls around a course deliverable deadline.

## Companion materials

No specific notebook filenames named on slides. Plan for Today (slide 2/15) mentions mini project feedback session.

## Cross-refs

→ `methods/traditional_vs_dl.qmd`, `methods/neural_networks.qmd` (Lecture 9), `methods/ensemble.qmd` (Random Forest, XGBoost from Lectures 7), `methods/logistic_regression.qmd` (Lecture 4/8), `methods/decision_trees.qmd` (Lecture 4/7), `methods/transfer_learning.qmd` (if created).
