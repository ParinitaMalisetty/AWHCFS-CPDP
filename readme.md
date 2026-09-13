# AWHCFS for Cross-Project Defect Prediction

## Overview

Cross-Project Defect Prediction (CPDP) aims to predict software defects in a target project using data from other software projects.

This project proposes **AWHCFS (Adaptive Weighted Hybrid Feature Selection)**, a feature-selection approach designed to identify a smaller and more stable set of software metrics for CPDP.

The approach combines three feature-selection techniques:

- Random Forest Feature Importance
- Mutual Information (MI)
- ANOVA-based Feature Relevance

These techniques are combined using **adaptive variance-based weighting** to produce a final feature ranking.

Instead of using all **21 software metrics**, AWHCFS selects the **top 10 features**, reducing the feature space by approximately **52%**.

Cross-Project Defect Prediction (CPDP) aims to predict software defects in a target project using data from other projects. However, differences between projects and the presence of redundant or less informative software metrics can affect prediction performance. This study proposes AWHCFS (Adaptive Weighted Hybrid Feature Selection), a feature-selection approach that combines Random Forest feature importance, Mutual Information, and ANOVA-based feature relevance using adaptive variance-based weighting to select a compact and relevant feature set for CPDP. The approach is evaluated on 14 software projects using a Leave-One-Project-Out (LOPO) strategy across four tree-based classifiers—Random Forest, XGBoost, LightGBM, and CatBoost—and compared against a full-feature baseline and two hybrid feature-selection methods, HCFS and WHCFS. AWHCFS achieved the highest overall Accuracy (0.7686) and F1-score (0.2466) while maintaining competitive Balanced Accuracy (0.5745) and MCC (0.1824), reducing the feature space from 21 metrics to 10. Pairwise Wilcoxon signed-rank tests indicated no statistically significant difference in Balanced Accuracy between AWHCFS and the compared methods; however, AWHCFS achieved this performance with approximately 52% fewer features and demonstrated high feature-selection stability, with eight features selected consistently across all 14 LOPO experiments. To assess the validity and interpretability of the selected features, SHAP explanations were generated across all four classifiers and aggregated into a cross-model consensus ranking. The AWHCFS feature ranking showed a Spearman correlation of 0.8466 and a Kendall correlation of 0.7249 with the SHAP consensus ranking, along with 90% Top-10 feature overlap. These results indicate that AWHCFS provides a compact, stable, and model-aligned feature-selection strategy that maintains competitive predictive performance with substantially reduced dimensionality and strong agreement with model-based feature importance.

---
## Methodology

The project follows a **Leave-One-Project-Out (LOPO)** evaluation strategy.

For each experiment:

1. One project is selected as the target project.
2. The remaining projects are used for training.
3. Feature selection is performed using the training data.
4. The selected features are used to train the classifiers.
5. The target project is used for testing.
6. Results are collected across all projects.

### Machine Learning Models

Four tree-based classifiers are evaluated:

- Random Forest
- XGBoost
- LightGBM
- CatBoost

### Feature Selection Methods

AWHCFS is compared against:

- **Baseline** – all available features
- **HCFS** – Hybrid Correlation-based Feature Selection
- **WHCFS** – Weighted Hybrid Correlation-based Feature Selection
- **AWHCFS** – Adaptive Weighted Hybrid Feature Selection

---

## Dataset

The experiments use **14 software projects** from the CPDP dataset.

The projects include:

`ant-1.7`, `camel-1.0`, `camel-1.6`, `jedit-3.2`, `jedit-4.2`, `log4j-1.1`, `lucene-2.0`, `poi-2.0`, `synapse-1.0`, `synapse-1.2`, `velocity-1.6`, `xalan-2.4`, `xerces-1.2`, `xerces-1.3`

Each experiment evaluates one project as the unseen target project, resulting in **14 LOPO experiments × 4 models = 56 model evaluations per method**.

---

## Final Results

Overall performance across the 56 evaluations:

| Method | Accuracy | Precision | Recall | F1 | Balanced Accuracy | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.7654 ± 0.0900 | **0.4806 ± 0.2486** | **0.2184 ± 0.1666** | 0.2444 ± 0.1188 | **0.5767 ± 0.0614** | **0.1924 ± 0.1040** |
| HCFS | 0.7679 ± 0.0921 | 0.4514 ± 0.2206 | 0.2117 ± 0.1545 | 0.2458 ± 0.1264 | 0.5737 ± 0.0613 | 0.1825 ± 0.1106 |
| WHCFS | 0.7672 ± 0.0936 | 0.4434 ± 0.2089 | 0.2131 ± 0.1528 | 0.2459 ± 0.1246 | 0.5739 ± 0.0605 | 0.1800 ± 0.1085 |
| **AWHCFS** | **0.7686 ± 0.0929** | 0.4477 ± 0.2154 | 0.2130 ± 0.1576 | **0.2466 ± 0.1302** | 0.5745 ± 0.0630 | 0.1824 ± 0.1143 |

### Key Results

- **21 → 10 features**, approximately **52% dimensionality reduction**
- Highest overall **Accuracy: 0.7686**
- Highest overall **F1-score: 0.2466**
- Competitive **Balanced Accuracy: 0.5745**
- Eight features were selected consistently across all **14 LOPO experiments**
- AWHCFS selected a stable and compact feature set across projects

---

## Selected Features

The overall AWHCFS feature-selection frequency was:

| Feature | Selection Count | Selection % | Rank |
|---|---:|---:|---:|
| rfc | 14/14 | 100% | 1 |
| loc | 14/14 | 100% | 2 |
| cam | 14/14 | 100% | 3 |
| amc | 14/14 | 100% | 4 |
| ce | 14/14 | 100% | 5 |
| avg_cc | 14/14 | 100% | 6 |
| max_cc | 14/14 | 100% | 7 |
| wmc | 14/14 | 100% | 8 |
| cbo | 13/14 | 92.86% | 9 |
| lcom3 | 13/14 | 92.86% | 10 |

---

## Explainable AI Validation

SHAP was used to validate whether the features selected by AWHCFS were also considered important by the trained machine learning models.

SHAP explanations were generated for:

- Random Forest
- XGBoost
- LightGBM
- CatBoost

The feature importance results were aggregated into a cross-model SHAP consensus ranking.

### SHAP Validation Results

| Measure | Result |
|---|---:|
| Spearman Correlation | **0.8466** |
| Kendall Correlation | **0.7249** |
| Top-5 Feature Overlap | **80% (4/5)** |
| Top-10 Feature Overlap | **90% (9/10)** |

The strong agreement between AWHCFS and SHAP indicates that the selected features are broadly aligned with the features considered important by the trained models.

---

## Statistical Analysis

Pairwise Wilcoxon signed-rank tests were performed using the 56 paired model/project results.

| Comparison | p-value | Significant (α = 0.05) |
|---|---:|---|
| AWHCFS vs Baseline | 0.484168 | No |
| AWHCFS vs HCFS | 0.987465 | No |
| AWHCFS vs WHCFS | 0.940377 | No |

The results indicate that the performance differences were **not statistically significant at α = 0.05**.

Therefore, the main contribution of AWHCFS is not a statistically significant increase in predictive performance, but rather achieving **comparable predictive performance with substantially fewer and more stable features**.

---

## Project Structure

```text
CPDP/
│
├── data/
│   └── datasets/
│
├── src/
│   ├── feature_selection/
│   ├── models/
│   └── explainability.py
│
├── experiments/
│   ├── lopo_baseline.py
│   ├── lopo_hcfs.py
│   ├── lopo_whcfs.py
│   ├── lopo_awhcfs.py
│   ├── shap_experiment.py
│   └── comparison.py
│
├── results/
│   ├── lopo_baseline_results.csv
│   ├── lopo_awhcfs_results.csv
│   ├── adaptive_weights.csv
│   ├── awhcfs_feature_frequency.csv
│   ├── shap/
│   └── comparison/
│
├── main.py
├── requirements.txt
└── README.md
```
---
## Reproducibility
Requirements
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SHAP
- SciPy
---
Run the Project
```
pip install -r requirements.txt
```
Then run:
```
python main.py
```
---
## The complete pipeline performs:
1. LOPO Baseline evaluation
2. LOPO AWHCFS evaluation
3. SHAP-based explainability analysis
4. AWHCFS vs SHAP validation
5. Statistical comparison
6. Final result generation
---

## Main Contribution
The project demonstrates that AWHCFS can reduce the CPDP feature space from 21 to 10 metrics while maintaining competitive predictive performance and achieving strong agreement with SHAP-based model explanations.
The approach focuses on three key properties:
Compactness + Stability + Interpretability
---
