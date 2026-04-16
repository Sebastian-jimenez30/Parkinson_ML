import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pd_binary_classifier.config import DatasetConfig, DEFAULT_PREPROCESSED_DIR
from pd_binary_classifier.training import train_binary_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a reproducible binary classifier: Parkinson vs Non-Parkinson"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / DEFAULT_PREPROCESSED_DIR,
        help="Path to preprocessed dataset folder",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
        help="Directory where model and reports are saved",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Disable post-training plots/explainability/report generation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = DatasetConfig(preprocessed_dir=args.data_dir)
    paths = train_binary_pipeline(
        config=config,
        output_dir=args.output_dir,
        seed=args.seed,
        n_splits=args.folds,
        generate_analysis=not args.no_analysis,
    )

    print("Training completed")
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
