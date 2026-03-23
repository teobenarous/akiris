"""
AKIRIS System Evaluation Script.
Parses application logs to evaluate real-time streaming performance against the ground truth.
Calculates F3 score, Latency, Specificity, and generates a Confusion Matrix.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def load_ground_truth(path: Path) -> tuple[set, set | None, set | None]:
    """
    Loads ground truth tuples.
    Returns: (Actual AKI events, NHS Predicted events, All evaluated events universe)
    """
    if not path.exists():
        return set(), None, None

    df = pd.read_csv(path)

    # Normalize date columns
    date_col = next((c for c in ["date", "creatinine_date"] if c in df.columns), None)
    if not date_col:
        raise ValueError("No valid date column found in ground truth.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["time_str"] = df[date_col].dt.strftime("%Y%m%d%H%M")

    # The universe of all blood tests evaluated
    universe = set(zip(df["mrn"].astype(str), df["time_str"]))

    # Detect new generator format (with NHS baseline data)
    if "aki" in df.columns and "nhs" in df.columns:
        actual_aki = set(
            zip(
                df[df["aki"] == "y"]["mrn"].astype(str),
                df[df["aki"] == "y"]["time_str"],
            )
        )
        nhs_predictions = set(
            zip(
                df[df["nhs"] == "y"]["mrn"].astype(str),
                df[df["nhs"] == "y"]["time_str"],
            )
        )
        return actual_aki, nhs_predictions, universe

    # Fallback legacy format
    actual_aki = set(zip(df["mrn"].astype(str), df["time_str"]))
    return actual_aki, None, None


def parse_logs(path: Path) -> tuple[set, list[float], float]:
    """Parses application JSON logs to extract Pager predictions and latency."""
    predictions = set()
    latencies = []
    timestamps = []

    if not path.exists():
        return predictions, latencies, 0.0

    with open(path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                msg = entry.get("msg", "")
                ts_str = entry.get("ts", "")

                if ts_str:
                    timestamps.append(datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f"))

                if msg.startswith("PAGED:"):
                    parts = msg.split()
                    mrn, event_time = parts[1], parts[3]
                    predictions.add((mrn, event_time[:12]))

                    try:
                        latencies.append(float(parts[5].strip("s)")))
                    except (IndexError, ValueError):
                        continue
            except (json.JSONDecodeError, ValueError):
                continue

    duration = (timestamps[-1] - timestamps[0]).total_seconds() if timestamps else 0.0
    return predictions, latencies, duration


def print_metrics(
    predictions: set, truth: set, universe: set | None, name: str
) -> None:
    """Calculates and beautifully formats all clinical metrics."""
    tp = len(predictions & truth)
    fp = len(predictions - truth)
    fn = len(truth - predictions)

    # Core Rates
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F3 Score (Beta=3 weights recall 9x higher than precision)
    beta = 3
    if (precision + recall) > 0:
        f3 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    else:
        f3 = 0.0

    print(f"--- {name} Performance ---")
    print(f"F3 Score (Target):  {f3:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"Precision:          {precision:.4f}")

    # Extended metrics if we have the universe of true negatives
    if universe is not None:
        tn = len(universe) - (tp + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        print(f"Specificity:        {specificity:.4f}\n")

        print("Confusion Matrix:")
        print(f"  TP: {tp:<5} | FP: {fp:<5}")
        print(f"  FN: {fn:<5} | TN: {tn:<5}")
    else:
        print("\n(Confusion matrix omitted: Universe of true negatives unknown)")

    print("=" * 55 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="AKIRIS Streaming Evaluation")
    parser.add_argument(
        "--logs",
        type=Path,
        default=Path("/Users/teo/Developer/akiris/logs/simulation.log"),
        help="Path to app logs",
    )
    # /Users/teo/Developer/akiris/data
    parser.add_argument(
        "--truth",
        type=Path,
        default=Path("/Users/teo/Developer/akiris/data/sample/ground_truth.csv"),
        help="Path to GT CSV",
    )
    args = parser.parse_args()

    if not args.truth.exists() or not args.logs.exists():
        sys.exit("Error: Missing logs or ground truth. Run the simulation first.")

    predictions, latencies, duration = parse_logs(args.logs)
    truth, nhs_predictions, universe = load_ground_truth(args.truth)

    print("\n" + "=" * 55)
    print("AKIRIS Real-Time Streaming Evaluation")
    print("=" * 55 + "\n")

    # Evaluate NHS Baseline if available
    if nhs_predictions is not None:
        print_metrics(nhs_predictions, truth, universe, name="Standard NHS Baseline")
    else:
        print("Running in Legacy Mode: NHS Baseline comparison unavailable.\n")

    # Evaluate ML Engine
    print_metrics(predictions, truth, universe, name="AKIRIS ML Engine")

    # System Telemetry
    p90 = np.percentile(latencies, 90) if latencies else 0.0
    m, s = divmod(int(duration), 60)

    print("--- System Telemetry ---")
    print(f"90th-Percentile Latency: {p90:.5f} seconds")
    print(f"Total Stream Duration:   {m}m {s}s")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()

