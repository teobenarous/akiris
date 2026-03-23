import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from app.config import SETTINGS
from app.state import Patient, PatientStore


class TestStatePersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.path = Path(self.test_dir)
        SETTINGS.PERSISTENCE_PATH = self.path / "state.pkl"
        SETTINGS.HISTORY_PATH = self.path / "history.csv"
        SETTINGS.JOURNAL_PATH = self.path / "journal.jsonl"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_patient_logic(self):
        """Unit test for in-memory patient sorting logic."""
        p = Patient(mrn="1")
        # Out of order insertion
        p.add_result(datetime(2024, 1, 5), 10.0)
        p.add_result(datetime(2024, 1, 1), 5.0)

        # Should sort automatically
        self.assertEqual(p.history[0].value, 5.0)
        self.assertEqual(p.history[-1].value, 10.0)

    def test_snapshot_roundtrip(self):
        """Verifies full state serialization and deserialization."""
        store = PatientStore()
        # Use log methods to populate (simulating app usage)
        store.log_demographics("123", sex="F", dob=None)

        store.save()

        # Reload
        new_store = PatientStore()
        new_store.hydrate()
        self.assertEqual(new_store.patients["123"].sex, "F")

    def test_wal_write_and_replay(self):
        """
        Verify that operations are logged to WAL and replayed on hydration without a snapshot.
        """
        store = PatientStore()
        # (1) Log events
        store.log_demographics("WAL_PATIENT", sex="F", dob=datetime(1990, 1, 1))
        store.log_result("WAL_PATIENT", datetime(2024, 1, 1), 100.0)

        # (2) Verify journal file content
        self.assertTrue(SETTINGS.JOURNAL_PATH.exists(), "Journal file was not created.")
        with open(SETTINGS.JOURNAL_PATH) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2, "Expected 2 journal entries.")

        # Verify JSON structure
        rec1 = json.loads(lines[0])
        self.assertEqual(rec1["op"], "demo")
        self.assertEqual(rec1["mrn"], "WAL_PATIENT")

        # (3) Simulate Crash: Create new store without calling save()
        new_store = PatientStore()
        new_store.hydrate()

        # (4) Verify data recovered from WAL
        p = new_store.patients.get("WAL_PATIENT")
        self.assertIsNotNone(p, "Patient lost after crash simulation.")
        self.assertEqual(p.sex, "F")
        self.assertEqual(len(p.history), 1)
        self.assertEqual(p.history[0].value, 100.0)

    def test_wal_truncation_on_snapshot(self):
        """Verify journal is cleared after a successful snapshot to prevent unbounded growth."""
        store = PatientStore()
        store.log_result("123", datetime(2024, 1, 1), 50.0)

        # Pre-condition: Journal has data
        self.assertGreater(os.path.getsize(SETTINGS.JOURNAL_PATH), 0)

        # Action: Save Snapshot
        store.save()

        # Post-condition: Journal empty, Snapshot exists
        self.assertEqual(
            os.path.getsize(SETTINGS.JOURNAL_PATH), 0, "Journal was not truncated."
        )
        self.assertTrue(SETTINGS.PERSISTENCE_PATH.exists())

    def test_corrupt_wal_recovery(self):
        """
        Failure Test: Verify system falls back gracefully if state file is corrupt
        and handles garbage in the journal log.
        """
        # (1) Create a corrupt pickle file
        with open(SETTINGS.PERSISTENCE_PATH, "wb") as f:
            f.write(b"NOT_A_PICKLE_FILE")

        # (2) Create a mixed-quality journal
        SETTINGS.JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS.JOURNAL_PATH, "w") as f:
            f.write(json.dumps({"op": "demo", "mrn": "VALID_1"}) + "\n")
            f.write(
                "THIS_IS_GARBAGE_JSON_DATA_FROM_DISK_CORRUPTION\n"
            )  # Should be skipped
            f.write(json.dumps({"op": "demo", "mrn": "VALID_2"}) + "\n")

        # (3) Attempt hydration inside the log capturer
        with self.assertLogs("app.state", level="WARNING") as cm:
            store = PatientStore()
            try:
                store.hydrate()
            except Exception as e:
                self.fail(f"Store crashed on corrupt recovery: {e}")
            self.assertTrue(any("corrupt" in msg for msg in cm.output))

        # (4) Verify valid data recovered despite the corruption
        self.assertIn("VALID_1", store.patients)
        self.assertIn("VALID_2", store.patients)

    def test_legacy_csv_ingestion(self):
        """Verify fallback loading from history.csv works and ignores bad data."""
        with open(SETTINGS.HISTORY_PATH, "w") as f:
            f.write("mrn,sex,dob,creatinine_date,creatinine_result\n")
            f.write("1,M,1980-01-01,2024-01-01 10:00:00,150.0\n")  # Valid
            f.write("2,F,BAD_DATE,2024-01-01,100.0\n")  # Bad DOB, valid test
            f.write(",,,,, \n")  # Empty row

        store = PatientStore()
        store._load_csv_history()

        self.assertIn("1", store.patients)
        self.assertEqual(store.patients["1"].history[0].value, 150.0)
        self.assertIn("2", store.patients)
        self.assertEqual(store.patients["2"].history[0].value, 100.0)  # Handled bad DOB

    @patch("app.state.os.fsync", side_effect=OSError("Disk Full"))
    def test_wal_fsync_failure_handled(self, _):
        """Verify WAL failures log critically but do not crash the app."""
        store = PatientStore()
        with self.assertLogs("app.state", level="CRITICAL"):
            store.log_discharge("123")

