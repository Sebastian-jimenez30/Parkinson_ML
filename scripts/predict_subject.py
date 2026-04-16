import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pd_binary_classifier.inference import predict_one_subject


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Parkinson probability for one subject id"
    )
    parser.add_argument(
        "--subject-id",
        type=str,
        required=True,
        help="Subject id, for example 001",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "outputs" / "model_best.joblib",
        help="Path to trained model artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = predict_one_subject(args.model_path, args.subject_id)

    print("Prediction result")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
