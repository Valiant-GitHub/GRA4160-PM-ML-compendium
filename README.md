# GRA 4160 — Predictive Modeling with Machine Learning · Interactive Compendium

> An interactive web companion for *GRA 4160 Predictive Modeling with Machine Learning* at BI Norwegian Business School. 36 cross-linked pages covering 17 methods, with two-mode math (course-level and full rigor), a method-selection decision dashboard, hyperparameter sandboxes, decision-boundary showdowns, and a drill section anchored to past exams and exercises.

**[→ Open the live site](https://valiant-github.github.io/GRA4160-PM-ML-compendium/)**

![Landing page — method-selection dashboard](Decision%20Dashboard.png)

---

## Contents

- [What this is](#what-this-is)
- [Why this exists](#why-this-exists)
- [What's covered](#whats-covered)
- [How it's organized](#how-its-organized)
- [How it was built](#how-it-was-built)
- [Known limitations](#known-limitations)
- [Tech stack](#tech-stack)
- [How to view locally](#how-to-view-locally)
- [How to rebuild from source](#how-to-rebuild-from-source)
- [Repo structure](#repo-structure)
- [Usage and attribution](#usage-and-attribution)
- [Author](#author)

---

## What this is

An interactive HTML compendium for the methods taught in GRA 4160. The site treats method selection as the spine — visitors start at a landing-page decision dashboard ("what does your data look like, what's your goal?") and navigate outward to specific method pages, comparison showdowns, or drill walkthroughs.

Every method page follows a fixed eight-section template: Essence → When/When-NOT to use → Code idiom → Worked examples → Visualization → Diagnostics → Past-Exam Lens → Two-mode math (course-level and full rigor). The cross-method layer adds a decision dashboard, family comparison tables, decision-boundary and regression showdowns on common datasets, and an exercise dependency map. The drill section walks past exams and exercise notebooks as practical solution templates.

The site is fully static, offline-capable (once served), and uses Observable JS for interactivity.

## Why this exists

I built this as a personal study companion before the GRA 4160 final exam. The course covers a wide range of supervised, unsupervised, and neural methods, and I wanted a single navigable artifact that made the *connections between methods* explicit — what to reach for when, how families compare, where each technique fails. Most existing references are either textbook-deep (slow to navigate during a time-boxed exam) or cheat-sheet-thin (no real teaching).

I've now built versions of this for a few courses. The cost is real — it takes meaningful time away from studying — but the format is engaging enough to study from in a way passive re-reading isn't, and the side benefit is a durable post-exam reference.

## What's covered

Seventeen methods, organized by family. Each links to its page on the live site:

| Family | Methods |
|---|---|
| **Linear** | [OLS](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/ols.html), [kNN](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/knn.html), [Logistic Regression](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/logistic.html), [LDA](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/lda.html), [Adaline](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/adaline.html), [Regularization (Ridge/Lasso)](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/regularization.html) |
| **Tree-based** | [Decision Trees](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/decision_trees.html), [Random Forests](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/random_forests.html), [Ensembles](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/ensembles.html) |
| **Unsupervised** | [PCA](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/pca.html), [K-Means](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/kmeans.html) |
| **Neural** | [NN Basics](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/nn_basics.html), [Autodiff](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/autodiff.html), [NN with PyTorch](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/nn_pytorch.html), [Build a NN](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/build_a_nn.html) |
| **Cross-cutting** | [Bias–Variance](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/bias_variance.html), [Cross-Validation & Information Criteria](https://valiant-github.github.io/GRA4160-PM-ML-compendium/methods/cv_info_criteria.html) |

## How it's organized

The site has six functional layers:

**1. Landing decision dashboard** — Interactive method-selection tool. Inputs: task type, dataset shape, linearity, interpretability need, presence of categoricals, missing data, class imbalance. Output: ranked method shortlist with one-sentence justifications and links. Each ranking rule traces to a "When NOT to use" callout from the corresponding method page.

**2. Big-picture pages** — `Reading the Data` (diagnostics that drive method choice — data types, dimensionality, missingness, target distribution, class balance, correlations, linearity) and `Workflow Patterns` (seven reusable end-to-end recipes for common situations: classification on a new dataset, regression with many predictors, clustering without ground truth, high-dim → dim-reduction → classifier, imbalanced classes, missing data, model selection and evaluation).

**3. Method pages (17)** — Each on the same eight-section template. Code idioms come from the course notebooks directly (cited cell-by-cell). Worked examples use course datasets (Titanic, Hitters, seeds, Wholesale Customers, MNIST, house prices). Visualizations focus on what static prose can't convey — decision boundaries with hyperparameter sliders, coefficient paths, scree plots, loss curves, etc.

**4. Cross-method layer** — Family comparison tables, decision-boundary showdowns (logistic vs. LDA vs. kNN vs. tree vs. RF on a common 2D dataset), regression showdown (OLS vs. Ridge vs. Lasso vs. Tree vs. RF on Hitters), and an exercise dependency map showing which exercise notebooks build on which method notebooks.

**5. Drill section** — A worked walkthrough of the Spring 2025 take-home exam (solution sketches, not full code dumps) plus walkthroughs of six exercise notebooks (Titanic preprocessing, naive Bayes spam filtering, house-price prediction, MNIST digits, model selection/evaluation, customer segmentation). Filterable index.

**6. Math layer** — Two modes selectable via a global toggle persisted in `localStorage`:
- *Mode A — Course level:* the math the course actually teaches (loss function, optimization, key results, interpretation)
- *Mode B — Full rigor:* full derivations from ESL with section citations

Both modes are populated for every method. A shared math appendix covers seven foundations (linear algebra essentials, gradient descent, MLE/MAP, bias-variance decomposition, information criteria, backprop, KKT conditions) that method pages link into rather than re-derive.

## How it was built

This site was built with AI assistance over several iterations, using your lecture material, the course notebooks, past exams, and the two textbooks as the source basis. The build was deliberately set up to keep AI-generated content grounded: every formula, code snippet, and claim cites the underlying source so it can be verified directly, formulas were cross-checked between the course material and the textbooks during construction, and gaps where a source was unclear were flagged inline rather than filled in by guessing.

AI-generated content is also flagged via the `NOT_GROUND_TRUTH.md` note in the rendered site. The site is offered for personal study, not as authoritative course material.

## Known limitations

Three honest caveats stated upfront:

1. **`nn_pytorch.html` §4 worked-example numbers are qualitative with `[VERIFY]` markers.** PyTorch wasn't installed in the build environment, so the numerical training-loss and accuracy outputs aren't computed; the page references the underlying notebook for real numbers. The marker is intentional — it tells the reader to verify rather than trust.

2. **Two datasets weren't available at build time:** `SMSSpamCollection.csv` (spam-classification exercise) and `bank-additional-full.csv` (customer-segmentation exercise). The corresponding drill pages document the *idiom* (preprocessing → vectorization → model fit) without computed outputs.

3. **The site requires an HTTP origin to function.** Observable JS doesn't run over `file://`. The live GitHub Pages URL works fine; for local viewing, a local web server is required (see *How to view locally* below).

## Tech stack

- **Quarto** for the multi-page site framework
- **Observable JS** (Quarto-native) for interactive widgets — sliders, decision dashboards, exercise dependency map
- **Observable Plot** for most visualizations (smaller bundle than Plotly)
- **KaTeX** for math rendering (vendored locally; no CDN dependency)
- **Custom SCSS theme** with a two-dimensional color system: family color (linear/tree/neural/unsupervised/cross-cutting) × content callout (intuition / math-A / math-B / code / pitfall / exam-lens / cross-link)
- **Python** (venv, isolated at build time) for data extraction, source verification, and the few build-time computations

## How to view locally

The site is live at the GitHub Pages URL above; no local setup is needed to view it. For local viewing:

```powershell
cd docs
python -m http.server 8000
```

Then open `http://localhost:8000/` in any browser. Opening the HTML files directly via `file://` will not work — Observable JS needs an HTTP origin.

## How to rebuild from source

This is only relevant if you want to modify the site and re-render. Requires [Quarto](https://quarto.org/) and Python 3.11+.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-build.txt
$env:QUARTO_PYTHON = ".venv\Scripts\python.exe"
quarto render
```

The render produces output in `docs/`. The Windows `QUARTO_PYTHON` env var step is necessary if Python is installed via the Microsoft Store (whose Python app-aliases break Quarto's default detection).

## Repo structure

```
.
├── _quarto.yml              # Site config (output-dir: docs)
├── index.qmd                # Landing page source (Quarto)
├── docs/                    # Rendered site (served by GitHub Pages)
├── methods/                 # 17 method-page .qmd sources
├── cross_method/            # Decision dashboard, family comparisons, showdowns, exercise map
├── drill/                   # Past-exam and exercise walkthroughs
├── big_picture/             # Reading the data, workflow patterns
├── appendix/                # Notation table, math appendix, sources
├── theme/                   # Custom SCSS + math-mode toggle JS
├── assets/                  # Vendored KaTeX
├── requirements-build.txt   # Python build dependencies
└── .nojekyll                # Disables Jekyll on GitHub Pages
```

The original course material (lecture slides, notebooks, datasets, exam files, ESL and ISL textbooks) is intentionally **not** in the repo — it belongs to the instructor and the textbook authors. Source citations on each page reference these by name; reproducing the build from scratch would require supplying them locally.

## Usage and attribution

This is a derivative work built on materials I don't own (the GRA 4160 course content owned by the instructor; the ESL and ISL textbooks owned by their respective authors and publishers). Accordingly, no permissive license is offered for the code in this repository — **all rights reserved**.

You are welcome to view the live site for personal study. Forking, redistributing, or building on the repository is discouraged, and any derivative work would inherit the same constraints around the underlying materials.

If you're a BI student in a future cohort of GRA 4160 considering whether to build something similar: I'd encourage it. Reach out and I'm happy to share notes on what worked and what didn't.

## Author

Built by **Vilijam Cekov** — MSc student at BI Norwegian Business School.
- LinkedIn: [linkedin.com/in/vilijam-cekov](https://www.linkedin.com/in/vilijam-cekov/)
