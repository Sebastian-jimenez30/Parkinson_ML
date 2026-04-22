from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    preprocessed_dir: Path
    fs_hz: float = 100.0
    movement_channels: int = 132
    movement_timesteps: int = 976
    questionnaire_items: int = 30


DEFAULT_PREPROCESSED_DIR = Path(
    "Data/pads-parkinsons-disease-smartwatch-dataset-1.0.0/preprocessed"
)
