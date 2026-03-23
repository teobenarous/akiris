"""
HL7 Message Parsing.
Provides utilities for extracting data from raw HL7 strings.
"""

from datetime import datetime
from typing import Optional


def parse_hl7_date(date_str: str) -> Optional[datetime]:
    """
    Parses HL7 timestamp formats (YYYYMMDD[HH[MM[SS]]]).
    Returns None if format is invalid.
    """
    if not date_str:
        return None
    formats = [("%Y%m%d%H%M%S", 14), ("%Y%m%d%H%M", 12), ("%Y%m%d", 8)]
    for fmt, length in formats:
        if len(date_str) >= length:
            try:
                return datetime.strptime(date_str[:length], fmt)
            except ValueError:
                continue
    return None


class HL7Message:
    """
    Wrapper for raw HL7 message strings.
    Does not validate strict HL7 compliance, only extracts fields needed for AKI.
    """

    def __init__(self, raw_msg: str):
        self.raw = raw_msg
        # store as list to preserve order and allow duplicate segments (e.g., multiple OBX fields)
        self.segments: list[str] = [s for s in raw_msg.split("\r") if s.strip()]

    def get_segment(self, segment_code: str) -> Optional[list[str]]:
        """Returns the fields of the first segment matching the code (e.g., 'PID')."""
        for seg in self.segments:
            if seg.startswith(f"{segment_code}|"):
                return seg.split("|")
        return None

    @property
    def message_type(self) -> str:
        """Extracted from MSH-9 (e.g., 'ADT^A01')."""
        msh = self.get_segment("MSH")
        return msh[8] if msh and len(msh) > 8 else ""

    @property
    def message_control_id(self) -> str:
        """Extracted from MSH-10. Used for ACKs."""
        msh = self.get_segment("MSH")
        return msh[9] if msh and len(msh) > 9 else ""

    @property
    def mrn(self) -> Optional[str]:
        """Extracted from PID-3 (Patient ID)."""
        pid = self.get_segment("PID")
        if pid and len(pid) > 3:
            # handle eventual '12345^NHS^...' format by taking first component
            return pid[3].split("^")[0]
        return None

    def _resolve_timestamp(self) -> datetime:
        """Helper to find the best available timestamp."""
        # (a) Try OBR-7 (Observation datetime)
        obr = self.get_segment("OBR")
        if obr and len(obr) > 7:
            ts = parse_hl7_date(obr[7])
            if ts:
                return ts

        # (b) Try MSH-7 (Message sent datetime)
        msh = self.get_segment("MSH")
        if msh and len(msh) > 6:
            ts = parse_hl7_date(msh[6])
            if ts:
                return ts

        # (c) Fallback (system time)
        return datetime.now()

    def get_obx_value(self, test_identifier: str) -> Optional[tuple[float, datetime]]:
        """
        Scans OBX segments for a specific test code.

        Returns:
            (Value, Timestamp)

        Timestamp Priority:
            1. OBR-7 (Observation Time)
            2. MSH-7 (Message Time)
            3. System Time (Fallback)
        """
        # Determine timestamp once for the message context
        obs_time = self._resolve_timestamp()

        for seg in self.segments:
            if seg.startswith("OBX|"):
                fields = seg.split("|")
                # OBX-3 contains the identifier (e.g. 'CREATININE')
                if len(fields) > 5 and test_identifier in fields[3]:
                    try:
                        return float(fields[5]), obs_time
                    except ValueError:
                        continue
        return None
