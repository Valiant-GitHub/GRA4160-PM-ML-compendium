# Phase 0.0 — File Manifest (authoritative source map)

Recursive enumeration of the workspace, honoring **C11** (skip `__MACOSX/**` and
`._*` AppleDouble files). This manifest is the single source of truth for all
subsequent phases. Where it contradicts the prompt's illustrative filenames, **this
manifest wins**. Builds on `workspace_inventory.md` (setup phase) and re-verified by
recursive glob on 2026-05-22.

Working directory: `C:\Users\V\Downloads\GRA4160 Predictive Mod. w Mach.Learn`

---

## Tier 1 — Lecture slides (`course_materials/Lecture slides/`, 10 PDFs)

Filenames carry download suffixes; lecture number parsed from the `lecture<N>` prefix.

| # | Real path | Size |
|---|---|---|
| 1 | `course_materials/Lecture slides/lecture1 (2).pdf` | 340,414 B |
| 2 | `course_materials/Lecture slides/lecture2 (2).pdf` | 64,422 B |
| 3 | `course_materials/Lecture slides/lecture3_260521_184932.pdf` | 1,127,719 B |
| 4 | `course_materials/Lecture slides/lecture4 (2).pdf` | 85,383 B |
| 5 | `course_materials/Lecture slides/lecture5_260521_184934.pdf` | 572,456 B |
| 6 | `course_materials/Lecture slides/lecture6 (2).pdf` | 548,390 B |
| 7 | `course_materials/Lecture slides/lecture7_260521_184935.pdf` | 1,088,609 B |
| 8 | `course_materials/Lecture slides/lecture8_260521_184937.pdf` | 1,178,973 B |
| 9 | `course_materials/Lecture slides/lecture9_260521_184938.pdf` | 908,812 B |
| 10 | `course_materials/Lecture slides/lecture10 (1).pdf` | 39,396 B |

## Tier 1 — Lecture (method) notebooks (`course_materials/Lecture notebooks/`, 20)

| Real path | Size |
|---|---|
| `course_materials/Lecture notebooks/01_Working_with_data_in_jupyter_notebooks.ipynb` | 203,329 B |
| `course_materials/Lecture notebooks/02_OLS (1).ipynb` | 8,744 B |
| `course_materials/Lecture notebooks/03_Supervised_learning_with_kNN (1).ipynb` | 9,989 B |
| `course_materials/Lecture notebooks/04_Linear_discriminant_analysis.ipynb` | 418,665 B |
| `course_materials/Lecture notebooks/05_Regularised_regressions (1).ipynb` | 562,240 B |
| `course_materials/Lecture notebooks/06_Logistic_regression.ipynb` | 12,225 B |
| `course_materials/Lecture notebooks/07_Decision_trees.ipynb` | 12,803 B |
| `course_materials/Lecture notebooks/07_Predicting_income.ipynb` | 8,129 B |
| `course_materials/Lecture notebooks/08_Information_criteria_and_cross_validation (1).ipynb` | 12,874 B |
| `course_materials/Lecture notebooks/09_Bias_variance_tradeoff.ipynb` | 7,851 B |
| `course_materials/Lecture notebooks/10_PCA.ipynb` | 14,149 B |
| `course_materials/Lecture notebooks/11_K_means.ipynb` | 10,807 B |
| `course_materials/Lecture notebooks/12_Introducing_ensemble_methods.ipynb` | 13,370 B |
| `course_materials/Lecture notebooks/13_Random_forests.ipynb` | 15,911 B |
| `course_materials/Lecture notebooks/14_Adaline.ipynb` | 18,857 B |
| `course_materials/Lecture notebooks/15_LogisticRegression.ipynb` | 23,727 B |
| `course_materials/Lecture notebooks/16_Auto_diff.ipynb` | 10,788 B |
| `course_materials/Lecture notebooks/17_Neural_nets_basics.ipynb` | 12,251 B |
| `course_materials/Lecture notebooks/18_NN_with_PyTorch.ipynb` | 9,122 B |
| `course_materials/Lecture notebooks/19_Build_a_NN.ipynb` | 57,083 B |

> Two `07_` notebooks: `07_Decision_trees` (method) and `07_Predicting_income`
> (applied/exercise-flavored). There are two logistic regression notebooks:
> `06_Logistic_regression` and `15_LogisticRegression` (treat both as logistic source).

## Tier 1 — Exercise / solution notebooks (`course_materials/Exercises and solutions(VHL)/`, 11)

`_VHL` = worked solution; plain name = exercise stub. Six topics.

| Real path | Size |
|---|---|
| `course_materials/Exercises and solutions(VHL)/01_Data_preprocessing_titanic.ipynb` | 2,452 B |
| `course_materials/Exercises and solutions(VHL)/02_Spam_filtering_with_naive_bayes (1).ipynb` | 10,044 B |
| `course_materials/Exercises and solutions(VHL)/02_Spam_filtering_with_naive_bayes_VHL.ipynb` | 16,895 B |
| `course_materials/Exercises and solutions(VHL)/03_Predicting_house_prices.ipynb` | 3,787 B |
| `course_materials/Exercises and solutions(VHL)/03_Predicting_house_prices_VHL.ipynb` | 113,625 B |
| `course_materials/Exercises and solutions(VHL)/04_Recognising_handwritten_digits.ipynb` | 2,397 B |
| `course_materials/Exercises and solutions(VHL)/04_Recognising_handwritten_digits_VHL.ipynb` | 32,847 B |
| `course_materials/Exercises and solutions(VHL)/05_Model_selection_evaluation_and_assessment.ipynb` | 5,320 B |
| `course_materials/Exercises and solutions(VHL)/05_Model_selection_evaluation_and_assessment_VHL.ipynb` | 10,554 B |
| `course_materials/Exercises and solutions(VHL)/06_Customer_segmentation (1).ipynb` | 8,296 B |
| `course_materials/Exercises and solutions(VHL)/06_Customer_segmentation_VHL.ipynb` | 11,787 B |

## Lens — Past exams & guidelines (`course_materials/Lecture slides/Past exams/`)

| Real path | Size | Type |
|---|---|---|
| `course_materials/Lecture slides/Past exams/GRA41601QP.pdf` | 84,518 B | Question paper |
| `course_materials/Lecture slides/Past exams/GRA41602qp.pdf` | 77,729 B | Question paper |
| `course_materials/Lecture slides/Past exams/GRA4160_2025-05-26_kl_09_EP.pdf` | 98,289 B | Exam paper (2025) |
| `course_materials/Lecture slides/Past exams/Exam_GRA4160_Guidelines.pdf` | 111,998 B | Guideline |
| `course_materials/Lecture slides/Past exams/Exam_GRA4160_Spring2023_guidlines.pdf` | 33,831 B | Guideline (sic "guidlines") |
| `course_materials/Lecture slides/Past exams/GRA4160_2025-05-26_kl_09_EP_2.zip` | 125,844 B | Zip (already extracted ↓) |

### Extracted 2025 exam bundle — `course_materials/Lecture slides/Past exams/GRA4160_2025-05-26_kl_09_EP_2/`

| Real path (relative to bundle) | Size | Type |
|---|---|---|
| `Exam_GRA4160_spring2025.ipynb` | 21,560 B | **Spring-2025 take-home exam notebook** |
| `Hitters.csv` | 20,906 B | Exam dataset |
| `SimulatedData.csv` | 102,154 B | Exam dataset |
| `WholesaleCustomers.csv` | 15,021 B | Exam dataset |
| `wdbc.data` | 124,103 B | Exam dataset (Wisconsin Diagnostic Breast Cancer) |
| `wdbc.names` | 4,708 B | Dataset description for wdbc |

> **Ignored (C11):** `GRA4160_2025-05-26_kl_09_EP_2/__MACOSX/` and its `._*` files
> (`._Exam_GRA4160_spring2025.ipynb`, `._Hitters.csv`, `._SimulatedData.csv`,
> `._WholesaleCustomers.csv`). Zip noise — not real sources.

## Datasets

| Real path | Size | Notes |
|---|---|---|
| `course_materials/Data/seeds.csv` | 6,284 B | Seeds (clustering w/ ground-truth variety) |
| `course_materials/Data/Titanic data/train.csv` | 61,194 B | Titanic train (12 cols) |
| `course_materials/Data/Titanic data/test.csv` | 28,629 B | Titanic test |
| `course_materials/Data/Titanic data/gender_submission.csv` | 3,258 B | Titanic sample submission |
| `course_materials/Data/house-prices data/train.csv` | 460,676 B | House prices train (~80 cols) |
| `course_materials/Data/house-prices data/test.csv` | 451,405 B | House prices test |
| `course_materials/Data/house-prices data/sample_submission.csv` | 31,939 B | House prices sample submission |
| `course_materials/Data/house-prices data/data_description.txt` | 13,370 B | House-price feature schema |
| `course_materials/Lecture slides/Past exams/.../Hitters.csv` | 20,906 B | Regression (baseball salaries) |
| `course_materials/Lecture slides/Past exams/.../SimulatedData.csv` | 102,154 B | Simulated |
| `course_materials/Lecture slides/Past exams/.../WholesaleCustomers.csv` | 15,021 B | Unsupervised (real) |
| `course_materials/Lecture slides/Past exams/.../wdbc.data` | 124,103 B | Breast-cancer classification |

## Tier 2 — Reference texts (working-directory root)

| Real path | Size | Role |
|---|---|---|
| `course_materials/An Introduction to Statistical Learning Python.pdf` | ~19.1 MB | **ISL** (Python ed.) — present → Mode A depth |
| `course_materials/The Elements of Statistical Learning 2.E..pdf` | ~12.7 MB | **ESL** 2nd ed. — present → Mode B rigor |

> Per Phase 0.5: ESL/ISL are **not** read end-to-end here. Presence confirmed;
> queried per-method during Phase 3. Both present → ISL primary for Mode A, ESL for Mode B.

---

## Notes on discrepancies vs. prompt's illustrative names

- No clean `lectureN.pdf` names — all suffixed; globbed and renumbered by prefix.
- Notebooks use spaces + parens (`02_OLS (1).ipynb`), not `02_OLS__1_.ipynb`.
- ESL filename has spaces + trailing dot; ISL is the Python edition.
- Numbering runs `02_`–`19_` for methods (not `02`–`19` contiguous topics: `07_` doubled,
  logistic appears at `06_` and `15_`, Adaline at `14_` after the tree/PCA/ensemble block).
- `wdbc` (breast cancer) dataset present in the exam bundle — not in the prompt's
  illustrative dataset list; it is the 2025 exam's classification dataset.
