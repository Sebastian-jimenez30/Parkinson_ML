import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show CV results summary")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("outputs/cv_summary_metrics.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.summary_path)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
