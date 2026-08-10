AWHCFS-CPDP

Adaptive Weighted Hybrid Consensus Feature Selection for Cross-Project Defect Prediction

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
