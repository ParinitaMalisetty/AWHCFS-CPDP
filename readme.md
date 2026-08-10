AWHCFS-CPDP

Adaptive Weighted Hybrid Consensus Feature Selection for Cross-Project Defect Prediction
Cross-Project Defect Prediction (CPDP) aims to predict software defects in a target project using data from other projects. However, differences between projects and the presence of redundant or less informative software metrics can affect prediction performance. This study proposes AWHCFS (Adaptive Weighted Hybrid Feature Selection), a feature-selection approach that combines Random Forest feature importance, Mutual Information, and ANOVA-based feature relevance using adaptive variance-based weighting to select a compact and relevant feature set for CPDP. The approach is evaluated on 14 software projects using a Leave-One-Project-Out (LOPO) strategy across four tree-based classifiers—Random Forest, XGBoost, LightGBM, and CatBoost—and compared against a full-feature baseline and two hybrid feature-selection methods, HCFS and WHCFS. AWHCFS achieved the highest overall Accuracy (0.7686) and F1-score (0.2466) while maintaining competitive Balanced Accuracy (0.5745) and MCC (0.1824), reducing the feature space from 21 metrics to 10. Pairwise Wilcoxon signed-rank tests indicated no statistically significant difference in Balanced Accuracy between AWHCFS and the compared methods; however, AWHCFS achieved this performance with approximately 52% fewer features and demonstrated high feature-selection stability, with eight features selected consistently across all 14 LOPO experiments. To assess the validity and interpretability of the selected features, SHAP explanations were generated across all four classifiers and aggregated into a cross-model consensus ranking. The AWHCFS feature ranking showed a Spearman correlation of 0.8466 and a Kendall correlation of 0.7249 with the SHAP consensus ranking, along with 90% Top-10 feature overlap. These results indicate that AWHCFS provides a compact, stable, and model-aligned feature-selection strategy that maintains competitive predictive performance with substantially reduced dimensionality and strong agreement with model-based feature importance.

This project implements a Cross-Project Defect Prediction (CPDP) framework that predicts software defects in a target project using data from other projects.

The core contribution is Adaptive Weighted Hybrid Consensus Feature Selection (AWHCFS), which combines three feature-ranking techniques:

Random Forest feature importance
Mutual Information (MI)
ANOVA F-score

Instead of using fixed weights, AWHCFS calculates the weights dynamically based on the variance of the feature scores from each ranking method.

The selected features are then used with multiple machine-learning models:

Random Forest
XGBoost
LightGBM
CatBoost

The framework uses Leave-One-Project-Out (LOPO) evaluation, where each project is treated as the unseen target project while the remaining projects are used for training.

The project also includes comparisons with:

Baseline — all available features
HCFS — Hybrid Consensus Feature Selection
WHCFS — Weighted Hybrid Consensus Feature Selection
AWHCFS — Adaptive Weighted Hybrid Consensus Feature Selection

Additionally, SHAP-based explainability is included to analyze the importance of selected features and understand model predictions.

Overall, the project aims to investigate whether adaptive, consensus-based feature selection can improve cross-project defect prediction while maintaining interpretable feature importance.


AWHCFS demonstrates strong alignment with model-based feature importance, achieving 0.8466 Spearman rank correlation, 0.7249 Kendall rank correlation, and 90% Top-10 feature overlap with the cross-model SHAP consensus.


IN PROGRESS...
