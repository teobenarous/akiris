"""
Configuration management.
Loads settings from environment variables.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    # Network
    MLLP_ADDRESS: str = os.environ.get("MLLP_ADDRESS", "localhost:8440")
    PAGER_ADDRESS: str = os.environ.get("PAGER_ADDRESS", "localhost:8441")
    PROMETHEUS_PORT: int = int(os.environ.get("PROMETHEUS_PORT", "8000"))

    # Artifacts
    MODEL_PATH: Path = Path(os.environ.get("MODEL_PATH", "/model/model.onnx"))
    THRESHOLD_PATH: Path = Path(
        os.environ.get("THRESHOLD_PATH", "/model/threshold.txt")
    )

    # Persistence
    HISTORY_PATH: Path = Path(os.environ.get("HISTORY_PATH", "/data/history.csv"))
    PERSISTENCE_PATH: Path = Path(
        os.environ.get("PERSISTENCE_PATH", "/state/patients.pkl")
    )
    JOURNAL_PATH: Path = Path(os.environ.get("JOURNAL_PATH", "/state/journal.jsonl"))
    SAVE_INTERVAL: int = int(os.environ.get("SAVE_INTERVAL", "100"))

    # Runtime
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()
    RECONNECT: bool = os.environ.get("RECONNECT", "true").lower() == "true"

    @property
    def mllp_host_port(self) -> tuple[str, int]:
        """Splits MLLP_ADDRESS into (host, port) for socket binding."""
        host, port = self.MLLP_ADDRESS.split(":")
        return host, int(port)


# Instance to be imported by other modules
SETTINGS = Settings()

# Matches model training
FEATURE_ORDER = [
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
