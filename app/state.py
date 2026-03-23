"""
State management module.
Handles patient demographics and test result history.
Implements Write-Ahead Logging (WAL) for persistence.
"""

import csv
import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .config import SETTINGS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TestResult:
    """Immutable record of a creatinine test result."""

    date: datetime
    value: float


@dataclass
class Patient:
    """Represents a single patient's state."""

    mrn: str
    dob: Optional[datetime] = None
    sex: str = "u"  # Default 'u' (unknown) avoids null checks
    history: list[TestResult] = field(default_factory=list)
    is_discharged: bool = False

    @property
    def age(self) -> int:
        """Calculates age in years."""
        if not self.dob:
            return 0
        today = datetime.now()
        # tuple comparison automatically handles month/day boundaries
        return (
            today.year
            - self.dob.year
            - ((today.month, today.day) < (self.dob.month, self.dob.day))
        )

    @property
    def is_female(self) -> int:
        """Returns 1 for female, 0 for male/unknown."""
        return 1 if self.sex.lower() in ("f", "female") else 0

    def add_result(self, date: datetime, value: float) -> None:
        """Adds a test result. Only triggers O(N log N) sort if the new data arrives out of order."""
        res = TestResult(date, value)
        self.history.append(res)

        # Check if the new last item is older than the second-to-last item
        if len(self.history) > 1 and self.history[-2].date > self.history[-1].date:
            self.history.sort(key=lambda x: x.date)

    def update_demographics(
        self, sex: Optional[str] = None, dob: Optional[datetime] = None
    ) -> None:
        """Update demographics."""
        if sex:
            self.sex = sex
        if dob:
            self.dob = dob

    def discharge(self):
        self.is_discharged = True


class PatientStore:
    """In-memory database for patient state with WAL (Write-Ahead Log) persistence."""

    def __init__(self):
        self.patients: dict[str, Patient] = dict()
        self._ensure_journal_exists()

    def get_or_create(self, mrn: str) -> Patient:
        if mrn not in self.patients:
            self.patients[mrn] = Patient(mrn=mrn)
        return self.patients[mrn]

    # Journaling (WAL) Methods

    @staticmethod
    def _ensure_journal_exists():
        """Creates the journal directory if needed."""
        if not SETTINGS.JOURNAL_PATH.parent.exists():
            SETTINGS.JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _append_journal(entry: dict) -> None:
        """
        Appends a structured log entry to the journal file immediately.
        Uses os.fsync to ensure data survives a hard crash.
        """
        try:
            with open(SETTINGS.JOURNAL_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.critical(f"WAL Write Failed: {e}")

    def log_result(self, mrn: str, date: datetime, value: float) -> None:
        """Updates state and logs to WAL."""
        p = self.get_or_create(mrn)
        p.add_result(date, value)
        self._append_journal(
            dict(op="result", mrn=mrn, date=date.isoformat(), val=value)
        )

    def log_demographics(
        self, mrn: str, sex: Optional[str], dob: Optional[datetime]
    ) -> None:
        """Updates state and logs to WAL."""
        p = self.get_or_create(mrn)
        p.update_demographics(sex, dob)

        entry = dict(op="demo", mrn=mrn)
        if sex:
            entry["sex"] = sex
        if dob:
            entry["dob"] = dob.isoformat()
        self._append_journal(entry)

    def log_discharge(self, mrn: str) -> None:
        """Updates state and logs to WAL."""
        p = self.get_or_create(mrn)
        p.discharge()
        self._append_journal(dict(op="discharge", mrn=mrn))

    # Persistence / Hydration

    def _replay_journal(self) -> None:
        """Replays events from the journal on top of the loaded snapshot."""
        if not SETTINGS.JOURNAL_PATH.exists():
            return

        count = 0
        try:
            with open(SETTINGS.JOURNAL_PATH, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        mrn = rec["mrn"]
                        p = self.get_or_create(mrn)

                        op = rec.get("op")
                        if op == "result":
                            dt = datetime.fromisoformat(rec["date"])
                            p.add_result(dt, rec["val"])
                        elif op == "demo":
                            dob = (
                                datetime.fromisoformat(rec["dob"])
                                if rec.get("dob")
                                else None
                            )
                            p.update_demographics(rec.get("sex"), dob)
                        elif op == "discharge":
                            p.discharge()
                        count += 1
                    except (json.JSONDecodeError, KeyError, ValueError):
                        logger.warning(f"Skipping corrupt journal line: {line[:50]}...")

            if count > 0:
                logger.info(f"WAL Replay: Recovered {count} events from journal.")
        except Exception as e:
            logger.error(f"WAL Replay Failed: {e}")

    def hydrate(self) -> None:
        """Loads state from pickle (fast) -> Replays WAL (durability) -> or Fallback CSV."""
        loaded_snapshot = False

        # (1) Try loading snapshot
        if SETTINGS.PERSISTENCE_PATH.exists():
            try:
                with open(SETTINGS.PERSISTENCE_PATH, "rb") as f:
                    self.patients = pickle.load(f)
                logger.info(f"Hydrated from snapshot: {len(self.patients)} patients.")
                loaded_snapshot = True
            except (pickle.PickleError, EOFError) as e:
                logger.warning(f"Snapshot corrupt ({e}). Starting fresh/CSV.")

        # (2) If no snapshot, load CSV history
        if not loaded_snapshot:
            self._load_csv_history()

        # (3) Always replay journal (it contains everything since the last successful snapshot)
        self._replay_journal()

    def save(self) -> None:
        """
        Atomically saves state to disk (Pickle).
        On success, truncates the journal because the snapshot now includes those events.
        """
        tmp_path = SETTINGS.PERSISTENCE_PATH.with_suffix(".tmp")
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(self.patients, f)
            shutil.move(str(tmp_path), str(SETTINGS.PERSISTENCE_PATH))

            # Truncate journal now that data is safe in snapshot
            with open(SETTINGS.JOURNAL_PATH, "w") as f:
                f.truncate(0)

            logger.info(
                f"Snapshot saved & Journal truncated. Total patients: {len(self.patients)}"
            )
        except Exception as e:
            logger.error(f"Persistence failure: {e}")

    # Legacy CSV Support
    def _ingest_csv_row(self, row: dict) -> None:
        """Helper to parse a single CSV row safely."""
        mrn = row.get("mrn")
        if not mrn:
            return

        patient = self.get_or_create(mrn)
        if row.get("sex"):
            patient.sex = row["sex"]
        if row.get("dob"):
            try:
                patient.dob = datetime.strptime(row["dob"], "%Y-%m-%d")
            except ValueError:
                pass

        try:
            dt_str = row.get("creatinine_date", row.get("date"))
            val_str = row.get("creatinine_result", row.get("result", row.get("value")))
            if dt_str and val_str:
                fmt = "%Y-%m-%d %H:%M:%S" if " " in dt_str else "%Y-%m-%d"
                patient.add_result(datetime.strptime(dt_str, fmt), float(val_str))
        except (ValueError, TypeError):
            pass

    def _load_csv_history(self) -> None:
        if not SETTINGS.HISTORY_PATH.exists():
            logger.warning("No history CSV found. Starting empty.")
            return
        try:
            with open(SETTINGS.HISTORY_PATH, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self._ingest_csv_row(row)
            logger.info(f"Rebuilt state from CSV: {len(self.patients)} patients.")
        except Exception as e:
            logger.error(f"CSV Ingestion failed: {e}")
