import unittest
from datetime import datetime

from app.hl7 import HL7Message, parse_hl7_date


class TestHL7Parsing(unittest.TestCase):
    """Validates HL7 parsing logic and robustness."""

    def test_message_type_extraction(self):
        # MSH-9 is at index 8 (0-based split)
        # Structure: MSH | ^~\& | App | Fac | RApp | RFac | Dt | Sec | TYPE
        # Indices:    0  |   1  |  2  |  3  |   4  |   5  | 6  |  7  |  8
        adt = "MSH|^~\\&|SIM|SRH|||||ADT^A01|||2.5"
        oru = "MSH|^~\\&|SIM|SRH|||||ORU^R01|||2.5"

        self.assertEqual(HL7Message(adt).message_type, "ADT^A01")
        self.assertEqual(HL7Message(oru).message_type, "ORU^R01")

    def test_mrn_extraction(self):
        # PID-3 is at index 3
        # Structure: PID | ID | ID_Type | MRN | ...
        # Indices:    0  | 1  |    2    |  3

        # Standard
        raw = "MSH|^~\\&|...|||||ADT^A01\rPID|1||12345||Doe"
        self.assertEqual(HL7Message(raw).mrn, "12345")

        # Composite (123^NHS^...)
        raw_comp = "MSH|^~\\&|...|||||ADT^A01\rPID|1||999^NHS^M11||Doe"
        self.assertEqual(HL7Message(raw_comp).mrn, "999")

        # Missing
        self.assertIsNone(HL7Message("MSH|^~\\&|...").mrn)

    def test_date_parsing(self):
        # Seconds precision
        self.assertEqual(parse_hl7_date("20240101120000"), datetime(2024, 1, 1, 12, 0))
        # Day precision
        self.assertEqual(parse_hl7_date("20240101"), datetime(2024, 1, 1))
        # Invalid
        self.assertIsNone(parse_hl7_date("not-a-date"))
        self.assertIsNone(parse_hl7_date(None))

    def test_obx_value_extraction(self):
        # OBR-7 is at index 7
        raw = (
            "MSH|^~\\&|...|||||ORU^R01|||2.5\r"
            "PID|1||123\r"
            "OBR|1||||||20240101120000\r"  # Index 7 is the timestamp here
            "OBX|1|SN|GLUCOSE||5.5\r"
            "OBX|2|SN|CREATININE||150.0\r"  # Target
            "OBX|3|SN|UREA||10.0"
        )
        msg = HL7Message(raw)
        val, ts = msg.get_obx_value("CREATININE")

        self.assertEqual(val, 150.0)
        self.assertEqual(ts, datetime(2024, 1, 1, 12, 0, 0))

    def test_timestamp_fallback_logic(self):
        """
        Verify precedence: OBR-7 > MSH-7 > System Time
        """
        # 1. OBR is missing, use MSH-7 (index 6)
        # MSH | ^~\& | App | Fac | RApp | RFac | DATE
        #  0  |   1  |  2  |  3  |   4  |   5  |  6
        raw_msh_only = (
            "MSH|^~\\&|SIM|SRH|||20230101090000||ORU^R01\rOBX|1|SN|CREATININE||100"
        )
        msg = HL7Message(raw_msh_only)
        # Ensure we actually extracted the right date and didn't fall back to NOW
        # If extraction fails, we get current time (2025/2026), causing assertion error
        _, ts = msg.get_obx_value("CREATININE")
        self.assertEqual(ts, datetime(2023, 1, 1, 9, 0, 0))

        # 2. Both missing (uses datetime.now())
        raw_none = "MSH|^~\\&|...|||||||ORU^R01\rOBX|1|SN|CREATININE||100"
        _, ts_now = HL7Message(raw_none).get_obx_value("CREATININE")

        # Allow small delta for execution time
        delta = abs((datetime.now() - ts_now).total_seconds())
        self.assertLess(delta, 1.0)

    def test_obx_value_malformed(self):
        """Verify invalid float conversions are skipped cleanly."""
        raw_bad = "MSH|^~\\&||||||||ORU^R01\rOBX|1|SN|CREATININE||NOT_A_FLOAT"
        result = HL7Message(raw_bad).get_obx_value("CREATININE")
        self.assertIsNone(result)

