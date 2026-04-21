from pathlib import Path
from typing import Dict, Union

import joblib
import numpy as np

from .config import DatasetConfig
from .data import load_movement_array, load_questionnaire_array, load_metadata, build_demographic_features
from .features import extract_movement_features


def _subject_demog_features(config: DatasetConfig, subject_id: str) -> np.ndarray:
    df = load_metadata(config.preprocessed_dir)
    idx = df.index[df["id"] == subject_id]
    if len(idx) == 0:
        raise ValueError(f"Subject id {subject_id} not found in file_list.csv")
    demog_df, _ = build_demographic_features(df)
    return demog_df.iloc[int(idx[0])].values.astype(np.float32)


def build_subject_feature_vector(config: DatasetConfig, subject_id: str) -> np.ndarray:
    movement = load_movement_array(
        config.preprocessed_dir,
        subject_id,
        n_channels=config.movement_channels,
        n_timesteps=config.movement_timesteps,
    )
    mv_feats = extract_movement_features(movement, fs_hz=config.fs_hz)

    q_feats = load_questionnaire_array(
        config.preprocessed_dir,
        subject_id,
        n_items=config.questionnaire_items,
    ).astype(np.float32)

    d_feats = _subject_demog_features(config, subject_id)

    row = np.concatenate([mv_feats, q_feats, d_feats]).astype(np.float32)
    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    return row


def predict_one_subject(model_path: Path, subject_id: str) -> Dict[str, Union[str, int, float]]:
    artifact = joblib.load(model_path)
    cfg = artifact["config"]
    model = artifact["model"]

    config = DatasetConfig(
        preprocessed_dir=Path(cfg["preprocessed_dir"]),
        fs_hz=float(cfg["fs_hz"]),
        movement_channels=int(cfg["movement_channels"]),
        movement_timesteps=int(cfg["movement_timesteps"]),
        questionnaire_items=int(cfg["questionnaire_items"]),
    )

    sid = str(subject_id).zfill(3)
    x = build_subject_feature_vector(config, sid).reshape(1, -1)
    probs_raw = model.predict_proba(x)[0]
    classes = [int(c) for c in model.classes_]

    prob_by_class = {f"probability_class_{cls}": float(p) for cls, p in zip(classes, probs_raw)}
    pred_class = int(classes[int(np.argmax(probs_raw))])

    class_mapping = artifact.get("class_mapping_source", {})
    pred_name = class_mapping.get(str(pred_class), f"class_{pred_class}")

    out: Dict[str, Union[str, int, float]] = {
        "subject_id": sid,
        "predicted_label": pred_class,
        "predicted_label_name": pred_name,
    }
    out.update(prob_by_class)
    return out
