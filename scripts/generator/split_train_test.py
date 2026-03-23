"""
Dataset Splitter
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(
        description="Split generated data into Train and Test sets."
    )
    parser.add_argument(
        "--input", required=True, type=str, help="Path to raw training.csv"
    )
    parser.add_argument(
        "--train-out", required=True, type=str, help="Output path for train.csv"
    )
    parser.add_argument(
        "--test-out", required=True, type=str, help="Output path for test.csv"
    )
    parser.add_argument(
        "--test-size", default=0.2, type=float, help="Proportion of data for testing"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run the generator first.")
        return

    # Load the raw data
    print(f"Loading raw dataset from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)

    # Ensure the target column exists
    if "aki" not in df.columns:
        print("Error: 'aki' column not found in dataset.")
        return

    # Stratified split guarantees the same percentage of AKI cases in both sets
    print(f"Splitting data (Test Size: {args.test_size * 100}%)...")
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df["aki"],
        random_state=42,  # Fixed seed for reproducibility
    )

    # Save the datasets
    train_df.to_csv(args.train_out, index=False)
    test_df.to_csv(args.test_out, index=False)

    print("Split Complete!")
    print(f"    Train Set: {len(train_df)} rows -> {args.train_out}")
    print(f"    Test Set:  {len(test_df)} rows -> {args.test_out}")


if __name__ == "__main__":
    main()
