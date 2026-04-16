Model Guide: Binary Parkinson Classifier (Movement + Questionnaire + Demographics)

1. Objective

This project implements a reproducible binary classifier for:
- Parkinson vs Non-Parkinson

Definition used in this pipeline:
- Parkinson = original label 1
- Non-Parkinson = original labels 0 and 2 combined

This design follows the requested practical binary setup while preserving all available non-Parkinson cases.

2. Dataset Context

Data source in this workspace:
- Data/pads-parkinsons-disease-smartwatch-dataset-1.0.0

Pipeline input folder:
- Data/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed

Used files:
- file_list.csv
- movement/NNN_ml.bin
- questionnaire/NNN_ml.bin

Observed class counts from file_list.csv:
- label 0 (Healthy): 79
- label 1 (Parkinson): 276
- label 2 (Other diagnoses): 114

Binary counts used in this model:
- Positive (Parkinson): 276
- Negative (Non-Parkinson): 193

3. Reproducible Pipeline Design

The workflow is deterministic and reproducible due to:
- fixed random seed (default: 42)
- stratified k-fold split with shuffle and fixed random state
- fully scripted data loading, feature extraction, and model training
- persisted artifacts and metrics files

No manual notebook steps are required.

4. Feature Engineering

4.1 Movement features

Each subject movement file is interpreted as:
- 132 channels
- 976 timesteps per channel
- sampling rate: 100 Hz

For each channel, this pipeline computes 12 engineered features:
- mean
- standard deviation
- median
- interquartile range
- RMS
- energy
- mean absolute value
- skewness
- excess kurtosis
- dominant frequency
- band power in 3 to 7 Hz
- spectral entropy

Total movement features:
- 132 channels x 12 features = 1584 features

4.2 Questionnaire features

Each subject has 30 binary questionnaire values.
These are included directly as 30 numeric features.

4.3 Demographic and clinical metadata features

Numeric features:
- age
- height
- weight
- age_at_diagnosis

Boolean features:
- appearance_in_kinship
- appearance_in_first_grade_kinship

Categorical one-hot features:
- gender
- handedness
- effect_of_alcohol_on_tremor

Missing values are handled at model level by median imputation.

5. Models Compared

The pipeline compares three classic tabular models:
- Logistic Regression (with scaling and class_weight balanced)
- Random Forest (balanced_subsample)
- HistGradientBoosting

Selection rule:
- best model is chosen by highest mean balanced accuracy across folds

6. Validation Protocol

Validation method:
- Stratified K-Fold cross-validation
- default folds: 5

Metrics computed per fold and aggregated:
- balanced accuracy
- ROC AUC
- F1 score
- sensitivity (recall for Parkinson)
- specificity (recall for Non-Parkinson)

Why balanced accuracy:
- class distribution is not perfectly balanced
- balanced accuracy is robust for this setting

7. Repository Structure Added

- src/pd_binary_classifier/config.py
- src/pd_binary_classifier/data.py
- src/pd_binary_classifier/features.py
- src/pd_binary_classifier/training.py
- src/pd_binary_classifier/inference.py
- scripts/train_model.py
- scripts/predict_subject.py
- scripts/show_results.py
- requirements.txt
- README.md
- docs/MODEL_GUIDE.md

8. Installation

From project root, run:

C:/Python313/python.exe -m pip install -r requirements.txt

If your python path is different, replace C:/Python313/python.exe accordingly.

9. Training

Default run:

C:/Python313/python.exe scripts/train_model.py

Custom run example:

C:/Python313/python.exe scripts/train_model.py --seed 42 --folds 5 --output-dir outputs

10. Produced Artifacts

After training, outputs folder contains:

- model_best.joblib
  - trained model pipeline
  - feature names
  - configuration and label mapping

- cv_summary_metrics.csv
  - per-model mean and std summary

- cv_fold_metrics.csv
  - per-fold metrics for each model

- oof_predictions_<model>.csv
  - out-of-fold probabilities and thresholded predictions

- X_features.npy
  - final feature matrix used for training

- y_binary.npy
  - binary target vector

- feature_names.json
  - ordered list of feature names

- subjects_binary_labels.csv
  - subject id and binary label mapping

- plot_roc_curve.png
- plot_pr_curve.png
- plot_confusion_matrix.png
- explainability_permutation_importance.csv
- plot_permutation_top20.png
- explainability_shap_status.txt
- explainability_shap_top.csv (if SHAP available)
- plot_shap_top20.png (if SHAP available)
- report_model.md
- report_model.pdf

11. How to Test the Model Quickly

11.1 Check model ranking

C:/Python313/python.exe scripts/show_results.py

11.2 Predict one subject

C:/Python313/python.exe scripts/predict_subject.py --subject-id 001

Expected output fields:
- subject_id
- predicted_label
- probability_parkinson

Decision threshold currently used:
- 0.5

11.3 Optional SHAP setup

C:/Python313/python.exe -m pip install -r requirements-optional.txt

12. Recommended Experimental Extensions

To make this publishable or stronger for thesis work, next experiments are recommended:

1) Compare binary definitions:
- Parkinson vs Healthy only
- Parkinson vs Other diagnoses only
- Parkinson vs (Healthy + Other) current baseline

2) Add threshold optimization:
- choose threshold by Youden index on validation folds
- compare sensitivity-prioritized operating point

3) Add explainability:
- permutation importance (already implemented)
- SHAP on best model (implemented as optional dependency)

4) Robustness checks:
- repeated stratified CV
- confidence intervals via bootstrap on fold scores

5) Fairness diagnostics:
- subgroup performance by gender and age bands

13. Troubleshooting

Problem: ModuleNotFoundError for sklearn or pandas
- Reinstall dependencies with requirements.txt

Problem: Subject not found in prediction
- Ensure id format is valid (for example 1 and 001 both work)
- Verify that subject exists in preprocessed/file_list.csv

Problem: Data shape mismatch
- Confirm original preprocessed files are intact
- Do not mix files from different preprocessing versions

14. Reproducibility Checklist

- Keep the same dataset files
- Keep same python version and package versions
- Keep same random seed
- Keep same number of CV folds
- Keep same feature extraction code

If all conditions above are preserved, results should be reproducible within normal floating point tolerance.
