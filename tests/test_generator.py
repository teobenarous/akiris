# Adapted from the tests written by Andrew Eland:
# https://github.com/andreweland/swemls/blob/main/generator/generator_test.py

import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.metrics import fbeta_score

# Constants
NHS_F3_THRESHOLD = 0.7
CREATININE_UPPER_LIMIT = 1500.0


class TestGeneratorIntegration(unittest.TestCase):
    """
    Integration tests for the synthetic data generator and NHS baseline algorithm.
    Ensures data generated is physiologically valid and the baseline performs as expected.
    """

    def setUp(self):
        self.work_dir = Path(tempfile.mkdtemp())
        # Pointing to the new refactored locations
        self.generator_script = "scripts/generator/generator.py"
        self.nhs_script = "scripts/generator/nhs.py"

    def tearDown(self):
        shutil.rmtree(self.work_dir)

    def test_nhs_self_test(self):
        """Verify the NHS algorithm's internal unit tests pass."""
        result = subprocess.run(
            [sys.executable, self.nhs_script, "--test"],
            capture_output=True,  # Keeps the "Ran 6 tests" out of your main terminal
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"NHS self-test failed: {result.stderr}")

    def test_generator_pipeline_and_f3_score(self):
        """
        Runs the generator, validates creatinine distributions,
        and verifies the NHS algorithm achieves the minimum F3 threshold.
        """
        # 1. Run Generator
        result = subprocess.run(
            [
                sys.executable,
                self.generator_script,
                "--days=25",
                "--data=data/assets",  # Point to the raw census files
                f"--output={self.work_dir}",
            ],
            capture_output=True,
        )
        self.assertEqual(
            result.returncode, 0, f"Generator failed: {result.stderr.decode()}"
        )

        dataset_csv = self.work_dir / "dataset.csv"
        self.assertTrue(dataset_csv.exists(), "Generator did not output dataset.csv")

        # 2. Validate Creatinine Values (Physiological limits)
        with open(dataset_csv, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            creatinine_cols = [
                i for i, h in enumerate(headers) if h.startswith("creatinine_result_")
            ]

            for row in reader:
                for col in creatinine_cols:
                    if row[col]:  # If not empty
                        val = float(row[col])
                        self.assertLessEqual(
                            val,
                            CREATININE_UPPER_LIMIT,
                            f"Physiological anomaly: Creatinine value {val} exceeds limit.",
                        )

        # 3. Generate NHS Predictions
        aki_predictions_csv = self.work_dir / "aki_predictions.csv"
        result = subprocess.run(
            [
                sys.executable,
                self.nhs_script,
                f"--input={dataset_csv}",
                f"--output={aki_predictions_csv}",
            ],
            capture_output=True,
        )
        self.assertEqual(
            result.returncode, 0, f"NHS prediction failed: {result.stderr.decode()}"
        )

        # 4. Calculate F3 Score
        df_expected = pd.read_csv(dataset_csv)
        df_predicted = pd.read_csv(aki_predictions_csv)

        # Convert 'y'/'n' to binary
        y_true = (
            df_expected["aki"]
            .map({"y": 1, "n": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )
        y_pred = (
            df_predicted["aki"]
            .map({"y": 1, "n": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )

        f3_score = fbeta_score(y_true, y_pred, beta=3, zero_division=0)

        # 5. Assert Threshold
        self.assertGreaterEqual(
            f3_score,
            NHS_F3_THRESHOLD,
            f"NHS Baseline F3 score ({f3_score:.4f}) fell below threshold ({NHS_F3_THRESHOLD})",
        )
