Parkinson Binary Classifier (Option 2)

This repository now contains a complete and reproducible binary machine learning pipeline using:
- Movement smartwatch signals (preprocessed movement binaries)
- Questionnaire responses (30 NMS binary items)
- Demographic and clinical metadata

Target problem:
- Positive class (1): Parkinson
- Negative class (0): Healthy + Other diagnoses

Quick start

1) Install dependencies

Windows PowerShell:
C:/Python313/python.exe -m pip install -r requirements.txt

2) Train the model

C:/Python313/python.exe scripts/train_model.py

3) Inspect model comparison results

C:/Python313/python.exe scripts/show_results.py

4) Predict one subject

C:/Python313/python.exe scripts/predict_subject.py --subject-id 001

All outputs are saved in outputs:
- model_best.joblib
- cv_summary_metrics.csv
- cv_fold_metrics.csv
- oof_predictions_<model>.csv
- X_features.npy
- y_binary.npy
- plot_roc_curve.png
- plot_pr_curve.png
- plot_confusion_matrix.png
- explainability_permutation_importance.csv
- plot_permutation_top20.png
- explainability_shap_status.txt
- report_model.md
- report_model.pdf
- feature_names.json
- subjects_binary_labels.csv

Optional SHAP dependency:

C:/Python313/python.exe -m pip install -r requirements-optional.txt

Detailed documentation

See docs/MODEL_GUIDE.md for full documentation:
- data understanding
- feature engineering details
- model training protocol
- reproducibility guarantees
- interpretation and troubleshooting
