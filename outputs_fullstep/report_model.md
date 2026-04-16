# Binary Parkinson Classifier Report

## Problem Definition
- Positive class: Parkinson (original label=1)
- Negative class: Non-Parkinson (original labels 0 and 2)

## Selected Model
- Best model by balanced accuracy: **hist_gradient_boosting**
- Decision threshold: **0.50**

## Out-of-Fold Metrics (Best Model)
- Balanced Accuracy: **0.7374**
- ROC AUC: **0.8207**
- F1: **0.8041**
- Sensitivity: **0.8478**
- Specificity: **0.6269**

## Cross-Validation Model Comparison
| model | balanced_accuracy_mean | balanced_accuracy_std | roc_auc_mean | f1_mean | sensitivity_mean | specificity_mean |
| --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting | 0.737299 | 0.012888 | 0.820442 | 0.804181 | 0.847826 | 0.626772 |
| logistic_regression | 0.691431 | 0.004221 | 0.784264 | 0.750993 | 0.760870 | 0.621993 |
| random_forest | 0.666493 | 0.020781 | 0.807130 | 0.787660 | 0.913043 | 0.419942 |


## Generated Artifacts
- ROC curve: outputs_fullstep\plot_roc_curve.png
- PR curve: outputs_fullstep\plot_pr_curve.png
- Confusion matrix: outputs_fullstep\plot_confusion_matrix.png
- Permutation importance CSV: outputs_fullstep\explainability_permutation_importance.csv
- Permutation plot: outputs_fullstep\plot_permutation_top20.png
- SHAP CSV (optional): N/A
- SHAP plot (optional): N/A
- SHAP status: outputs_fullstep\explainability_shap_status.txt

## Notes
- OOF means out-of-fold predictions from stratified CV.
- Permutation importance is model-agnostic and directly comparable across all features.