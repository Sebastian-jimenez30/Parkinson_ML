import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DatasetConfig
from .data import (
    build_binary_target,
    build_demographic_features,
    get_subject_ids,
    load_metadata,
    load_movement_array,
    load_questionnaire_array,
)
from .features import (
    extract_movement_features,
    movement_channel_names,
    movement_feature_names,
    questionnaire_feature_names,
)


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_prob)
    else:
        roc_auc = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)

    return {
        "balanced_accuracy": float(bal_acc),
        "roc_auc": float(roc_auc),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def _save_classification_plots(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    threshold: float = 0.5,
) -> Dict[str, str]:
    y_pred = (y_prob >= threshold).astype(int)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    roc_path = output_dir / "plot_roc_curve.png"
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (OOF)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # Precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_path = output_dir / "plot_pr_curve.png"
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (OOF)")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_path = output_dir / "plot_confusion_matrix.png"
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix (threshold={threshold:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Non-PD", "PD"])
    plt.yticks([0, 1], ["Non-PD", "PD"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    return {
        "roc_plot": str(roc_path),
        "pr_plot": str(pr_path),
        "cm_plot": str(cm_path),
    }


def _save_explainability(
    fitted_pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    seed: int,
) -> Dict[str, str]:
    out: Dict[str, str] = {}

    rng = np.random.default_rng(seed)

    # Use model-driven ranking to keep permutation explainability computationally practical.
    model = fitted_pipeline.steps[-1][1]
    if hasattr(model, "coef_"):
        base_rank = np.abs(np.ravel(model.coef_))
    elif hasattr(model, "feature_importances_"):
        base_rank = np.ravel(model.feature_importances_)
    else:
        base_rank = np.var(X, axis=0)

    max_candidates = min(120, X.shape[1])
    candidate_idx = np.argsort(base_rank)[::-1][:max_candidates]

    baseline_prob = fitted_pipeline.predict_proba(X)[:, 1]
    baseline_metric = balanced_accuracy_score(y, (baseline_prob >= 0.5).astype(int))

    rows = []
    n_repeats = 5
    for j in candidate_idx:
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = Xp[rng.permutation(X.shape[0]), j]
            p = fitted_pipeline.predict_proba(Xp)[:, 1]
            s = balanced_accuracy_score(y, (p >= 0.5).astype(int))
            scores.append(baseline_metric - s)
        rows.append(
            {
                "feature": feature_names[j],
                "importance_mean": float(np.mean(scores)),
                "importance_std": float(np.std(scores, ddof=0)),
            }
        )

    perm_df = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    perm_path = output_dir / "explainability_permutation_importance.csv"
    perm_df.to_csv(perm_path, index=False)
    out["permutation_importance_csv"] = str(perm_path)

    topk = perm_df.head(20).iloc[::-1]
    perm_plot = output_dir / "plot_permutation_top20.png"
    plt.figure(figsize=(8, 7))
    plt.barh(topk["feature"], topk["importance_mean"])
    plt.xlabel("Permutation Importance (mean balanced accuracy drop)")
    plt.title("Top 20 Features by Permutation Importance")
    plt.tight_layout()
    plt.savefig(perm_plot, dpi=150)
    plt.close()
    out["permutation_plot"] = str(perm_plot)

    # Optional SHAP computation.
    shap_status_path = output_dir / "explainability_shap_status.txt"
    try:
        import shap  # type: ignore

        # Transform features as the model sees them.
        if len(fitted_pipeline.steps) > 1:
            preprocess = Pipeline(fitted_pipeline.steps[:-1])
            model = fitted_pipeline.steps[-1][1]
            X_trans = preprocess.transform(X)
        else:
            model = fitted_pipeline.steps[-1][1]
            X_trans = X

        n_sample = min(200, X_trans.shape[0])
        rng = np.random.default_rng(seed)
        idx = rng.choice(np.arange(X_trans.shape[0]), size=n_sample, replace=False)
        X_sample = X_trans[idx]

        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

        values = np.array(shap_values.values)
        if values.ndim == 3:
            values = values[:, :, 1]
        mean_abs = np.mean(np.abs(values), axis=0)

        shap_df = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        shap_csv = output_dir / "explainability_shap_top.csv"
        shap_df.to_csv(shap_csv, index=False)
        out["shap_csv"] = str(shap_csv)

        shap_top = shap_df.head(20).iloc[::-1]
        shap_plot = output_dir / "plot_shap_top20.png"
        plt.figure(figsize=(8, 7))
        plt.barh(shap_top["feature"], shap_top["mean_abs_shap"])
        plt.xlabel("Mean |SHAP value|")
        plt.title("Top 20 Features by SHAP")
        plt.tight_layout()
        plt.savefig(shap_plot, dpi=150)
        plt.close()
        out["shap_plot"] = str(shap_plot)

        shap_status_path.write_text("SHAP computed successfully.", encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        shap_status_path.write_text(
            "SHAP not generated. Install optional dependency or check compatibility.\n"
            f"Detail: {exc}",
            encoding="utf-8",
        )
    out["shap_status"] = str(shap_status_path)
    return out


def _save_markdown_report(
    output_dir: Path,
    best_model_name: str,
    best_metrics: Dict[str, float],
    summary_df: pd.DataFrame,
    artifact_paths: Dict[str, str],
    threshold: float,
) -> str:
    report_path = output_dir / "report_model.md"

    summary_table = "| " + " | ".join(summary_df.columns) + " |\n"
    summary_table += "| " + " | ".join(["---"] * len(summary_df.columns)) + " |\n"
    for _, row in summary_df.iterrows():
        vals = []
        for v in row.values:
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        summary_table += "| " + " | ".join(vals) + " |\n"

    lines = [
        "# Binary Parkinson Classifier Report",
        "",
        "## Problem Definition",
        "- Positive class: Parkinson (original label=1)",
        "- Negative class: Non-Parkinson (original labels 0 and 2)",
        "",
        "## Selected Model",
        f"- Best model by balanced accuracy: **{best_model_name}**",
        f"- Decision threshold: **{threshold:.2f}**",
        "",
        "## Out-of-Fold Metrics (Best Model)",
        f"- Balanced Accuracy: **{best_metrics['balanced_accuracy']:.4f}**",
        f"- ROC AUC: **{best_metrics['roc_auc']:.4f}**",
        f"- F1: **{best_metrics['f1']:.4f}**",
        f"- Sensitivity: **{best_metrics['sensitivity']:.4f}**",
        f"- Specificity: **{best_metrics['specificity']:.4f}**",
        "",
        "## Cross-Validation Model Comparison",
        summary_table,
        "",
        "## Generated Artifacts",
        f"- ROC curve: {artifact_paths.get('roc_plot', 'N/A')}",
        f"- PR curve: {artifact_paths.get('pr_plot', 'N/A')}",
        f"- Confusion matrix: {artifact_paths.get('cm_plot', 'N/A')}",
        f"- Permutation importance CSV: {artifact_paths.get('permutation_importance_csv', 'N/A')}",
        f"- Permutation plot: {artifact_paths.get('permutation_plot', 'N/A')}",
        f"- SHAP CSV (optional): {artifact_paths.get('shap_csv', 'N/A')}",
        f"- SHAP plot (optional): {artifact_paths.get('shap_plot', 'N/A')}",
        f"- SHAP status: {artifact_paths.get('shap_status', 'N/A')}",
        "",
        "## Notes",
        "- OOF means out-of-fold predictions from stratified CV.",
        "- Permutation importance is model-agnostic and directly comparable across all features.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def _save_pdf_report(
    output_dir: Path,
    summary_df: pd.DataFrame,
    best_model_name: str,
    best_metrics: Dict[str, float],
    artifact_paths: Dict[str, str],
) -> str:
    pdf_path = output_dir / "report_model.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1: textual summary
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Binary Parkinson Classifier Report", fontsize=16)
        text = (
            f"Best model: {best_model_name}\n"
            f"Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}\n"
            f"ROC AUC: {best_metrics['roc_auc']:.4f}\n"
            f"F1: {best_metrics['f1']:.4f}\n"
            f"Sensitivity: {best_metrics['sensitivity']:.4f}\n"
            f"Specificity: {best_metrics['specificity']:.4f}"
        )
        fig.text(0.06, 0.78, text, fontsize=11, va="top")

        tbl_ax = fig.add_axes([0.06, 0.12, 0.88, 0.55])
        tbl_ax.axis("off")
        show_df = summary_df.copy()
        table = tbl_ax.table(
            cellText=show_df.round(4).values,
            colLabels=show_df.columns,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        pdf.savefig(fig)
        plt.close(fig)

        # Image pages
        for key in ["roc_plot", "pr_plot", "cm_plot", "permutation_plot", "shap_plot"]:
            img_path = artifact_paths.get(key)
            if not img_path:
                continue
            p = Path(img_path)
            if not p.exists():
                continue
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.imshow(plt.imread(p))
            ax.axis("off")
            ax.set_title(p.name)
            pdf.savefig(fig)
            plt.close(fig)

    return str(pdf_path)


def build_feature_table(config: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    df = load_metadata(config.preprocessed_dir)
    y = build_binary_target(df)
    subject_ids = get_subject_ids(df)

    demog_df, demog_names = build_demographic_features(df)

    ch_names = movement_channel_names()
    mv_feat_names = movement_feature_names(ch_names)
    q_feat_names = questionnaire_feature_names(config.questionnaire_items)
    feature_names = mv_feat_names + q_feat_names + demog_names

    rows: List[np.ndarray] = []
    for i, sid in enumerate(subject_ids):
        movement = load_movement_array(
            config.preprocessed_dir,
            sid,
            n_channels=config.movement_channels,
            n_timesteps=config.movement_timesteps,
        )
        mv_feats = extract_movement_features(movement, fs_hz=config.fs_hz)

        q_feats = load_questionnaire_array(
            config.preprocessed_dir,
            sid,
            n_items=config.questionnaire_items,
        ).astype(np.float32)

        d_feats = demog_df.iloc[i].values.astype(np.float32)

        row = np.concatenate([mv_feats, q_feats, d_feats]).astype(np.float32)
        rows.append(row)

    X = np.vstack(rows)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, feature_names, subject_ids


def _make_models(seed: int) -> Dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=600,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=6,
                        learning_rate=0.05,
                        max_iter=400,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def train_binary_pipeline(
    config: DatasetConfig,
    output_dir: Path,
    seed: int = 42,
    n_splits: int = 5,
    generate_analysis: bool = True,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_names, subject_ids = build_feature_table(config)

    np.save(output_dir / "X_features.npy", X)
    np.save(output_dir / "y_binary.npy", y)
    pd.DataFrame({"subject_id": subject_ids, "y": y}).to_csv(
        output_dir / "subjects_binary_labels.csv", index=False
    )
    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=True, indent=2)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = _make_models(seed)

    fold_rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []
    oof_by_model: Dict[str, np.ndarray] = {}

    best_name = None
    best_score = -1.0

    for model_name, model in models.items():
        model_fold_metrics = []
        oof_prob = np.zeros_like(y, dtype=np.float64)

        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_te)[:, 1]
            oof_prob[te_idx] = y_prob
            met = _compute_metrics(y_te, y_prob, threshold=0.5)

            met_row = {
                "model": model_name,
                "fold": fold_idx,
                **met,
            }
            fold_rows.append(met_row)
            model_fold_metrics.append(met)

        oof_by_model[model_name] = oof_prob

        dfm = pd.DataFrame(model_fold_metrics)
        agg = {
            "model": model_name,
            "balanced_accuracy_mean": float(dfm["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(dfm["balanced_accuracy"].std(ddof=0)),
            "roc_auc_mean": float(dfm["roc_auc"].mean()),
            "f1_mean": float(dfm["f1"].mean()),
            "sensitivity_mean": float(dfm["sensitivity"].mean()),
            "specificity_mean": float(dfm["specificity"].mean()),
        }
        summary_rows.append(agg)

        if agg["balanced_accuracy_mean"] > best_score:
            best_score = agg["balanced_accuracy_mean"]
            best_name = model_name

    folds_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="balanced_accuracy_mean", ascending=False
    )
    folds_df.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "cv_summary_metrics.csv", index=False)

    # Save out-of-fold predictions for each model.
    for model_name, oof in oof_by_model.items():
        pd.DataFrame(
            {
                "subject_id": subject_ids,
                "y_true": y,
                "y_prob": oof,
                "y_pred_t05": (oof >= 0.5).astype(int),
            }
        ).to_csv(output_dir / f"oof_predictions_{model_name}.csv", index=False)

    best_model = models[best_name]
    best_model.fit(X, y)

    artifact = {
        "model": best_model,
        "best_model_name": best_name,
        "feature_names": feature_names,
        "config": {
            "preprocessed_dir": str(config.preprocessed_dir),
            "fs_hz": config.fs_hz,
            "movement_channels": config.movement_channels,
            "movement_timesteps": config.movement_timesteps,
            "questionnaire_items": config.questionnaire_items,
            "seed": seed,
            "n_splits": n_splits,
        },
        "label_definition": {
            "positive": "Parkinson",
            "negative": "Healthy + Other diagnoses",
        },
        "class_mapping_source": {
            "0": "Healthy",
            "1": "Parkinson",
            "2": "Other diagnoses",
        },
    }

    model_path = output_dir / "model_best.joblib"
    joblib.dump(artifact, model_path)

    output_paths = {
        "model_path": str(model_path),
        "summary_path": str(output_dir / "cv_summary_metrics.csv"),
        "folds_path": str(output_dir / "cv_fold_metrics.csv"),
        "features_path": str(output_dir / "X_features.npy"),
    }

    if generate_analysis:
        best_oof = oof_by_model[best_name]
        best_metrics = _compute_metrics(y, best_oof, threshold=0.5)

        plot_paths = _save_classification_plots(
            y_true=y,
            y_prob=best_oof,
            output_dir=output_dir,
            threshold=0.5,
        )
        explain_paths = _save_explainability(
            fitted_pipeline=best_model,
            X=X,
            y=y,
            feature_names=feature_names,
            output_dir=output_dir,
            seed=seed,
        )
        analysis_paths = {**plot_paths, **explain_paths}

        md_report = _save_markdown_report(
            output_dir=output_dir,
            best_model_name=best_name,
            best_metrics=best_metrics,
            summary_df=summary_df,
            artifact_paths=analysis_paths,
            threshold=0.5,
        )
        pdf_report = _save_pdf_report(
            output_dir=output_dir,
            summary_df=summary_df,
            best_model_name=best_name,
            best_metrics=best_metrics,
            artifact_paths=analysis_paths,
        )

        output_paths.update(
            {
                "oof_best_path": str(output_dir / f"oof_predictions_{best_name}.csv"),
                "report_markdown": md_report,
                "report_pdf": pdf_report,
                **analysis_paths,
            }
        )

    return output_paths
