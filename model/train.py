"""
AKIRIS Model Training Pipeline.
Trains an XGBoost classifier with KDIGO-compliant features, calibrates probabilities
using Isotonic Regression via OOF cross-validation, evaluates on a holdout test set,
and exports a unified artifact (ONNX).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import convert_sklearn, update_registered_converter

# ONNX Export Libraries
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from xgboost import XGBClassifier

# Constants
RANDOM_SEED = 42
BETA_SCORE = 3


def load_and_preprocess_causal(
    filepath: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads CSV, applies strict causal preprocessing, and engineers KDIGO rolling windows.
    """
    df = pd.read_csv(filepath, low_memory=False)

    if "aki" not in df.columns:
        raise ValueError(f"Dataset {filepath} missing 'aki' column.")

    df = df.reset_index(names=["patient_id"])

    static_df = df[["patient_id", "age", "sex", "aki"]].copy()
    static_df["is_female"] = static_df["sex"].map({"f": 1, "m": 0}).fillna(0)
    static_df["patient_ever_aki"] = static_df["aki"].map({"y": 1, "n": 0})
    static_df = static_df.drop(columns=["sex", "aki"])

    long_df = pd.wide_to_long(
        df.drop(columns=["age", "sex", "aki"]),
        stubnames=["creatinine_date", "creatinine_result"],
        i="patient_id",
        j="seq_idx",
        sep="_",
        suffix=r"\w+",
    ).reset_index()

    long_df["creatinine_date"] = pd.to_datetime(
        long_df["creatinine_date"], errors="coerce"
    )
    long_df = long_df.dropna(subset=["creatinine_result"])

    # Sort chronologically and set index to datetime for pandas rolling windows
    long_df = long_df.sort_values(["patient_id", "creatinine_date"])
    long_df = long_df.set_index("creatinine_date")

    # --- KDIGO Time-Bounded Rolling Features ---
    grouped_rolling = long_df.groupby("patient_id")["creatinine_result"]
    rolling_48h = grouped_rolling.rolling("48h").min()
    rolling_7d = grouped_rolling.rolling("7D").min()

    long_df["rolling_min_48h"] = rolling_48h.reset_index(level=0, drop=True)
    long_df["rolling_min_7d"] = rolling_7d.reset_index(level=0, drop=True)

    # Reset index to restore creatinine_date as a normal column
    long_df = long_df.reset_index()
    long_df = long_df.merge(static_df, on="patient_id", how="left")

    # --- Standard Causal Features ---
    grouped = long_df.groupby("patient_id")
    long_df["baseline"] = grouped["creatinine_result"].cummin()
    long_df["current"] = long_df["creatinine_result"]
    long_df["peak"] = grouped["creatinine_result"].cummax()
    long_df["count"] = grouped.cumcount() + 1
    long_df["volatility"] = (
        grouped["creatinine_result"]
        .expanding()
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    value_delta = grouped["creatinine_result"].diff()
    time_delta = grouped["creatinine_date"].diff().dt.total_seconds() / 86400.0
    valid_mask = time_delta > 0

    long_df["velocity"] = 0.0
    long_df.loc[valid_mask, "velocity"] = (
        value_delta[valid_mask] / time_delta[valid_mask]
    ).clip(-10, 10)
    long_df["ratio"] = (long_df["current"] / long_df["baseline"]).clip(upper=10)
    long_df["abs_increase"] = (long_df["current"] - long_df["baseline"]).clip(
        lower=-5, upper=5
    )

    # KDIGO Feature: Ratio over 7-day rolling minimum
    long_df["kdigo_ratio_7d"] = (long_df["current"] / long_df["rolling_min_7d"]).clip(
        upper=10
    )

    is_last_test = grouped.cumcount(ascending=False) == 0
    long_df["event_aki"] = (long_df["patient_ever_aki"] & is_last_test).astype(int)

    feature_cols = [
        "age",
        "is_female",
        "baseline",
        "current",
        "peak",
        "volatility",
        "count",
        "velocity",
        "ratio",
        "abs_increase",
        "rolling_min_48h",
        "rolling_min_7d",
        "kdigo_ratio_7d",
    ]

    X = long_df[feature_cols].fillna(0)
    y = long_df["event_aki"]
    groups = long_df["patient_id"]

    return X.values, y.values, groups.values


def optimize_and_calibrate(
    model: XGBClassifier, X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[float, CalibratedClassifierCV]:
    """Trains base model, calibrates probabilities, and finds optimal F3 threshold."""
    print("[1/4] Generating OOF predictions and training Probability Calibrator...")

    # To get OOF probabilities for threshold tuning
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_raw_probs = cross_val_predict(
        model, X, y, cv=cv, groups=groups, method="predict_proba", n_jobs=-1
    )[:, 1]

    # Temporarily fit an isotonic regressor just to find the OOF threshold
    temp_calibrator = IsotonicRegression(out_of_bounds="clip")
    oof_calibrated_probs = temp_calibrator.fit_transform(oof_raw_probs, y)

    precisions, recalls, thresholds = precision_recall_curve(y, oof_calibrated_probs)

    f_scores = (
        (1 + BETA_SCORE**2)
        * (precisions * recalls)
        / ((BETA_SCORE**2 * precisions) + recalls + 1e-9)
    )
    best_idx = np.nanargmax(f_scores)
    best_threshold = float(
        thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    )

    print(
        f"    Optimal F{BETA_SCORE} Threshold (Calibrated) found: {best_threshold:.4f}"
    )

    print("[2/4] Training final pipeline on 100% of training data...")
    # Fit the base model on all data
    model.fit(X, y)

    # Handle Scikit-Learn 1.6+ FrozenEstimator requirement
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated_model = CalibratedClassifierCV(
            estimator=FrozenEstimator(model), method="isotonic"
        )
    except ImportError:
        calibrated_model = CalibratedClassifierCV(
            estimator=model, method="isotonic", cv="prefit"
        )

    # Fit the actual pipeline wrapper that we will export
    calibrated_model.fit(X, y)

    # UNWRAP Hack for ONNX compatibility
    if hasattr(calibrated_model, "calibrated_classifiers_"):
        for clf in calibrated_model.calibrated_classifiers_:
            if hasattr(clf, "estimator") and hasattr(clf.estimator, "estimator"):
                clf.estimator = clf.estimator.estimator

    return best_threshold, calibrated_model


def evaluate_performance(
    model: CalibratedClassifierCV,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
) -> None:
    """Calculates and prints comprehensive classification metrics using calibrated predictions."""
    print("\n[3/4] Evaluating Unified Model on Unseen Test Data...")

    calibrated_probs = model.predict_proba(X_test)[:, 1]
    predictions = (calibrated_probs >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, calibrated_probs)
    pr_auc = average_precision_score(y_test, calibrated_probs)

    f3 = fbeta_score(y_test, predictions, beta=BETA_SCORE, zero_division=0)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)

    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("=" * 50)
    print("AKIRIS Offline Test Metrics (Calibrated)")
    print("=" * 50)
    print(f"ROC-AUC Score:          {roc_auc:.4f}")
    print(f"PR-AUC Score:           {pr_auc:.4f}")
    print("-" * 50)
    print(f"F{BETA_SCORE} Score (Primary):     {f3:.4f}")
    print(f"Sensitivity (Recall):   {recall:.4f}  <- Prioritized (Minimizing FN)")
    print(f"Specificity:            {specificity:.4f}")
    print(f"Precision (PPV):        {precision:.4f}")
    print("-" * 50)
    print("Confusion Matrix:")
    print(f"  TP: {tp:<5} | FP: {fp:<5}")
    print(f"  FN: {fn:<5} | TN: {tn:<5}")
    print("=" * 50)


def export_artifacts(
    calibrated_model: CalibratedClassifierCV,
    threshold: float,
    model_out: Path,
    threshold_out: Path,
    num_features: int,
) -> None:
    print("\n[4/4] Exporting Unified Production Artifact (ONNX)...")
    threshold_out.parent.mkdir(parents=True, exist_ok=True)
    with open(threshold_out, "w") as f:
        f.write(str(threshold))

    update_registered_converter(
        XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )

    initial_type = [("float_input", FloatTensorType([None, num_features]))]

    onnx_model = convert_sklearn(
        calibrated_model,
        initial_types=initial_type,
        target_opset={"": 15, "ai.onnx.ml": 3},
        options={"zipmap": False},
    )

    with open(model_out, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"    Unified artifact saved to: {model_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AKIRIS Model Training & ONNX Export")
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("model/model.onnx"))
    parser.add_argument(
        "--threshold-out", type=Path, default=Path("model/threshold.txt")
    )
    args = parser.parse_args()

    if not args.train.exists():
        sys.exit(
            f"Error: Training file {args.train} does not exist. Run generation pipeline first."
        )

    # --- Pipeline Execution ---
    X_train, y_train, groups = load_and_preprocess_causal(args.train)

    num_negative = (y_train == 0).sum()
    num_positive = (y_train == 1).sum()
    dynamic_spw = num_negative / num_positive
    capped_spw = min(dynamic_spw, 50.0)

    model = XGBClassifier(
        tree_method="hist",
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=capped_spw,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        eval_metric="aucpr",
    )

    best_threshold, calibrated_model = optimize_and_calibrate(
        model, X_train, y_train, groups
    )

    if args.test.exists():
        X_test, y_test, _ = load_and_preprocess_causal(args.test)
        evaluate_performance(calibrated_model, X_test, y_test, best_threshold)
    else:
        print(f"\n[!] Warning: Test file {args.test} not found. Skipping evaluation.")

    export_artifacts(
        calibrated_model,
        best_threshold,
        args.model_out,
        args.threshold_out,
        num_features=X_train.shape[1],
    )


if __name__ == "__main__":
    main()

