from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


MULTICLASS_LABELS: Dict[int, str] = {
    0: "Healthy",
    1: "Parkinson",
    2: "Other neurological diagnosis",
}


def load_metadata(preprocessed_dir: Path) -> pd.DataFrame:
    file_path = preprocessed_dir / "file_list.csv"
    df = pd.read_csv(file_path, dtype={"id": str})
    df["id"] = df["id"].str.zfill(3)
    return df


def build_multiclass_target(df: pd.DataFrame) -> np.ndarray:
    labels = df["label"].astype(int).values
    unknown = sorted(set(np.unique(labels)) - set(MULTICLASS_LABELS.keys()))
    if unknown:
        raise ValueError(f"Unexpected class labels found: {unknown}")
    return labels.astype(np.int32)


def get_subject_ids(df: pd.DataFrame) -> List[str]:
    return df["id"].tolist()


def load_movement_array(
    preprocessed_dir: Path,
    subject_id: str,
    n_channels: int,
    n_timesteps: int,
) -> np.ndarray:
    path = preprocessed_dir / "movement" / f"{subject_id}_ml.bin"
    arr = np.fromfile(path, dtype=np.float32)
    expected = n_channels * n_timesteps
    if arr.size != expected:
        raise ValueError(
            f"Unexpected size for {path}. Got {arr.size}, expected {expected}."
        )
    return arr.reshape(n_channels, n_timesteps)


def load_questionnaire_array(
    preprocessed_dir: Path,
    subject_id: str,
    n_items: int,
) -> np.ndarray:
    path = preprocessed_dir / "questionnaire" / f"{subject_id}_ml.bin"
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != n_items:
        raise ValueError(
            f"Unexpected questionnaire size for {path}. Got {arr.size}, expected {n_items}."
        )
    return arr


def build_demographic_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    base = pd.DataFrame(index=df.index)

    numeric_cols = ["age", "height", "weight", "age_at_diagnosis"]
    for col in numeric_cols:
        base[col] = pd.to_numeric(df[col], errors="coerce")

    bool_cols = ["appearance_in_kinship", "appearance_in_first_grade_kinship"]
    for col in bool_cols:
        vals = df[col].astype(str).str.strip().str.lower()
        base[col] = vals.map({"true": 1.0, "false": 0.0})

    cat_cols = ["gender", "handedness", "effect_of_alcohol_on_tremor"]
    cat = df[cat_cols].copy()
    for col in cat_cols:
        cat[col] = cat[col].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"})

    cat = pd.get_dummies(cat, columns=cat_cols, prefix=cat_cols, dtype=float)
    out = pd.concat([base, cat], axis=1)
    return out, out.columns.tolist()
