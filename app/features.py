"""
Feature engineering module.
Transforms patient state into the feature vector required by the classifier.
"""

import numpy as np

from .config import FEATURE_ORDER
from .state import Patient

SECONDS_PER_DAY = 86400.0
MIN_TIME_DELTA = 0.0001  # Avoid div by zero


def compute_features(patient: Patient) -> np.ndarray:
    """
    Generates a 1x13 feature vector for inference.
    Applies strict clipping to match the offline training distribution.

    Features:
        - age (years)
        - is_female (1 for female, 0 for male)
        - baseline (minimum creatinine over entire history)
        - current (latest creatinine result)
        - peak (maximum creatinine over entire history)
        - volatility (standard deviation of all results)
        - count (total number of tests)
        - velocity (rate of change per day from previous test, clipped [-10, 10])
        - ratio (current / baseline, clipped max 10)
        - abs_increase (current - baseline, clipped [-5, 5])
        - rolling_min_48h (minimum creatinine in the last 48 hours)
        - rolling_min_7d (minimum creatinine in the last 7 days)
        - kdigo_ratio_7d (current / rolling_min_7d, clipped max 10)
    """
    if not patient.history:
        return np.zeros((1, len(FEATURE_ORDER)), dtype=np.float32)

    all_values = [r.value for r in patient.history]
    current_record = patient.history[-1]
    current_value = current_record.value
    current_date = current_record.date

    baseline = min(all_values)
    peak = max(all_values)
    count = len(all_values)

    volatility = float(np.std(all_values, ddof=1)) if count > 1 else 0.0

    velocity = 0.0
    if count > 1:
        prev_record = patient.history[-2]
        delta_days = (current_date - prev_record.date).total_seconds() / SECONDS_PER_DAY
        if delta_days > MIN_TIME_DELTA:
            velocity = max(
                -10.0, min((current_value - prev_record.value) / delta_days, 10.0)
            )

    ratio = min(current_value / baseline, 10.0) if baseline > 0 else 0.0
    abs_increase = max(-5.0, min(current_value - baseline, 5.0))

    # --- NEW KDIGO FEATURES ---
    # Filter history for tests within the last 48 hours and 7 days
    history_48h = [
        r.value
        for r in patient.history
        if (current_date - r.date).total_seconds() <= 48 * 3600
    ]
    history_7d = [
        r.value
        for r in patient.history
        if (current_date - r.date).total_seconds() <= 7 * 24 * 3600
    ]

    rolling_min_48h = min(history_48h) if history_48h else current_value
    rolling_min_7d = min(history_7d) if history_7d else current_value

    kdigo_ratio_7d = (
        min(current_value / rolling_min_7d, 10.0) if rolling_min_7d > 0 else 0.0
    )

    data = dict(
        age=patient.age,
        is_female=patient.is_female,
        baseline=baseline,
        current=current_value,
        peak=peak,
        volatility=volatility,
        count=count,
        velocity=velocity,
        ratio=ratio,
        abs_increase=abs_increase,
        rolling_min_48h=rolling_min_48h,
        rolling_min_7d=rolling_min_7d,
        kdigo_ratio_7d=kdigo_ratio_7d,
    )

    vector = [data[feature_name] for feature_name in FEATURE_ORDER]
    return np.array([vector], dtype=np.float32)

