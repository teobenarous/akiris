import unittest
from datetime import datetime

from app.config import FEATURE_ORDER
from app.features import compute_features
from app.state import Patient


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.p = Patient(mrn="123", dob=datetime(1990, 1, 1), sex="F")

    def test_full_feature_vector_logic(self):
        """Verifies the features are correctly computed for the ML model."""
        # (a) add history
        self.p.add_result(datetime(2024, 1, 1), 100.0)
        self.p.add_result(datetime(2024, 1, 2), 120.0)

        # (b) add current result (48h after Jan 2, 72h after Jan 1)
        self.p.add_result(datetime(2024, 1, 4), 110.0)

        # (c) compute features
        feats = compute_features(self.p)[0]
        f_map = dict(zip(FEATURE_ORDER, feats))

        self.assertEqual(f_map["age"], 36.0)
        self.assertEqual(f_map["is_female"], 1.0)
        self.assertEqual(f_map["baseline"], 100.0)
        self.assertEqual(f_map["current"], 110.0)
        self.assertEqual(f_map["peak"], 120.0)
        self.assertAlmostEqual(f_map["volatility"], 10.0)
        self.assertEqual(f_map["count"], 3.0)

        # Velocity (change per day from last result: (110 - 120) / 2 days = -5.0)
        self.assertAlmostEqual(f_map["velocity"], -5.0)

        # Ratio (110 / 100 = 1.1)
        self.assertAlmostEqual(f_map["ratio"], 1.1)

        # Absolute increase (110 - 100 = 10 -> Clipped to max 5.0)
        self.assertEqual(f_map["abs_increase"], 5.0)

        # KDIGO Rolling Features
        # 48h window covers Jan 2 (120) and Jan 4 (110). Min is 110.
        self.assertEqual(f_map["rolling_min_48h"], 110.0)
        # 7d window covers all three. Min is 100.
        self.assertEqual(f_map["rolling_min_7d"], 100.0)
        # KDIGO ratio 7d (110 / 100 = 1.1)
        self.assertAlmostEqual(f_map["kdigo_ratio_7d"], 1.1)

    def test_single_value_stats(self):
        """Edge case: first result (no history)."""
        p = Patient(mrn="999", sex="M")
        p.add_result(datetime(2024, 1, 1), 50.0)

        feats = compute_features(p)[0]
        f_map = dict(zip(FEATURE_ORDER, feats))

        self.assertEqual(f_map["volatility"], 0.0)
        self.assertEqual(f_map["velocity"], 0.0)
        self.assertEqual(f_map["count"], 1.0)

        # New KDIGO features should default to the current value / 1.0 ratio
        self.assertEqual(f_map["rolling_min_48h"], 50.0)
        self.assertEqual(f_map["rolling_min_7d"], 50.0)
        self.assertEqual(f_map["kdigo_ratio_7d"], 1.0)

    def test_clipping_bounds(self):
        """Verifies that extreme values are strictly clipped to match offline training."""
        p = Patient(mrn="888", sex="M")
        p.add_result(datetime(2024, 1, 1), 10.0)

        # Add a massive spike 1 day later
        p.add_result(datetime(2024, 1, 2), 200.0)

        feats = compute_features(p)[0]
        f_map = dict(zip(FEATURE_ORDER, feats))

        # Velocity: (200 - 10) / 1 day = 190 -> Clipped to 10.0
        self.assertEqual(f_map["velocity"], 10.0)

        # Ratio: 200 / 10 = 20 -> Clipped to 10.0
        self.assertEqual(f_map["ratio"], 10.0)
        self.assertEqual(f_map["kdigo_ratio_7d"], 10.0)

        # Absolute increase: 200 - 10 = 190 -> Clipped to 5.0
        self.assertEqual(f_map["abs_increase"], 5.0)

    def test_empty_history_returns_zeros(self):
        """Verify a patient with no blood tests safely returns an empty vector."""
        p_empty = Patient(mrn="404")
        feats = compute_features(p_empty)

        self.assertEqual(feats.shape, (1, 13))
        self.assertEqual(feats.sum(), 0.0)  # All zeros

