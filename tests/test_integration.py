import shutil
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import requests

from app.config import SETTINGS
from app.main import process_message
from app.pager import PagerService
from app.predictor import AKIPredictor
from app.state import PatientStore


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.path = Path(self.test_dir)
        SETTINGS.PERSISTENCE_PATH = self.path / "state.pkl"
        SETTINGS.HISTORY_PATH = self.path / "history.csv"
        SETTINGS.JOURNAL_PATH = self.path / "journal.jsonl"

        self.store = PatientStore()
        self.mock_predictor = MagicMock(spec=AKIPredictor)
        self.mock_pager = MagicMock(spec=PagerService)
        self.mock_predictor.predict.return_value = False

    def tearDown(self):
        # Clean up temporary WAL and state files
        shutil.rmtree(self.test_dir)

    def test_end_to_end_lifecycle(self):
        """
        Integration: Admit -> Alert (AKI) -> Discharge -> Ignore Result
        """
        # (1) Admit
        msg_admit = (
            "MSH|^~\\&|SIM|SRH|||20240101||ADT^A01\rPID|1||123||Doe^John||19800101|M"
        )
        process_message(msg_admit, self.store, self.mock_predictor, self.mock_pager)
        # Verify patient created
        self.assertIn(
            "123",
            self.store.patients,
            "Patient '123' was not created. Check HL7 parsing.",
        )
        p = self.store.patients["123"]
        self.assertEqual(p.sex, "M")
        self.assertFalse(p.is_discharged)

        # (2) High result (AKI)
        self.mock_predictor.predict.return_value = True
        msg_aki = (
            "MSH|^~\\&|SIM|SRH|||20240102||ORU^R01\r"
            "PID|1||123\r"
            "OBX|1|SN|CREATININE||200"
        )
        process_message(msg_aki, self.store, self.mock_predictor, self.mock_pager)
        # Verify Alert Sent
        self.mock_pager.send_page.assert_called_with("123", ANY, ANY)

        # (3) Discharge
        msg_discharge = "MSH|^~\\&|SIM|SRH|||20240103||ADT^A03\rPID|1||123"
        process_message(msg_discharge, self.store, self.mock_predictor, self.mock_pager)
        self.assertTrue(p.is_discharged)

        # (4) Result
        self.mock_pager.reset_mock()
        process_message(msg_aki, self.store, self.mock_predictor, self.mock_pager)
        # Verify NO alert
        self.mock_pager.send_page.assert_not_called()

    def test_latency_slo(self):
        """
        Performance Test: Ensure processing takes < 30ms (SLO).
        """
        # Valid HL7 message
        msg = (
            "MSH|^~\\&|SIM|SRH|||20240101||ORU^R01\r"
            "PID|1||123\r"
            "OBX|1|SN|CREATININE||100"
        )

        start = time.time()
        for _ in range(100):
            process_message(msg, self.store, self.mock_predictor, self.mock_pager)
        duration = time.time() - start

        avg_latency = duration / 100
        self.assertLess(
            avg_latency,
            0.03,
            f"Average latency {avg_latency:.4f}s exceeded 30ms target",
        )

    def test_pager_api_failure(self):
        """
        Failure Test: Verify application continues if Pager API throws exception.
        """
        # Use a real PagerService to test the internal error handling
        real_pager = PagerService(workers=1)

        error = requests.exceptions.ConnectionError("Connection Timeout")

        # Suppress exponential backoff sleep to make the test run fast
        with patch("app.pager.time.sleep"), patch("requests.post", side_effect=error):
            with self.assertLogs(level="WARNING") as log_capture:
                # Trigger an AKI alert
                real_pager.send_page("123", datetime.now(), time.time())

                # Wait for thread to finish
                real_pager.shutdown()

                # Verify error was logged but app didn't crash
                self.assertTrue(
                    any("Pager network error" in r for r in log_capture.output)
                )

    def test_malformed_message_handling(self):
        """Failure Test: Malformed HL7 should return ACK and not crash."""
        bad_msg = "MSH|^~\\&|...|INVALID_HL7_NO_SEGMENTS"
        try:
            ack = process_message(
                bad_msg, self.store, self.mock_predictor, self.mock_pager
            )
            self.assertIn(b"MSA|AA", ack)  # Should still acknowledge receipt
        except Exception as e:
            self.fail(f"App crashed on malformed message: {e}")
