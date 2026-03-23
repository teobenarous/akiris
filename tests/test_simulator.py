# Adapted from the tests written by Andrew Eland:
# https://github.com/andreweland/swemls/blob/main/simulator/simulator_test.go

import http.client
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from simulator import simulator

# --- Test Data ---
ADT_A01 = [
    r"MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||202401201630||ADT^A01|||2.5",
    r"PID|1||478237423||ELIZABETH HOLMES||19840203|F",
]

ORU_R01 = [
    r"MSH|^~\&|SIMULATION|SOUTH RIVERSIDE|||202401201800||ORU^R01|||2.5",
    r"PID|1||478237423",
    r"OBR|1||||||202401202243",
    r"OBX|1|SN|CREATININE||103.4",
]

ACK = [
    r"MSH|^~\&|||||20240129093837||ACK|||2.5",
    r"MSA|AA",
]

# --- Constants ---
TEST_MLLP_PORT = 18440
TEST_PAGER_PORT = 18441


class TestSimulator(unittest.TestCase):
    """
    Integration tests for the HL7/MLLP simulator.
    Verifies networking, framing, and Pager HTTP logic.
    """

    def setUp(self):
        # Create an isolated environment for the test run
        self.test_dir = Path(tempfile.mkdtemp())
        self.messages_file = self.test_dir / "test_messages.mllp"

        # (1) Prepare a dummy MLLP file for the simulator to read
        with open(self.messages_file, "wb") as f:
            for msg in (ADT_A01, ORU_R01):
                f.write(self._to_mllp(msg))

        # (2) Start the simulator as a background process
        # We use 'python -m simulator.simulator' to respect the package structure
        self.process = subprocess.Popen(
            [
                "python3",
                "-m",
                "simulator.simulator",
                f"--mllp={TEST_MLLP_PORT}",
                f"--pager={TEST_PAGER_PORT}",
                f"--messages={self.messages_file}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # (3) Wait for the simulator to become healthy
        if not self._wait_until_healthy():
            self.process.kill()
            raise RuntimeError("Simulator failed to start in time.")

    def tearDown(self):
        # Graceful shutdown via the /shutdown endpoint
        try:
            with urllib.request.urlopen(
                f"http://localhost:{TEST_PAGER_PORT}/shutdown"
            ) as _:
                pass
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()
        finally:
            if self.process.stdout:
                self.process.stdout.close()
            if self.process.stderr:
                self.process.stderr.close()
            if self.test_dir.exists():
                shutil.rmtree(self.test_dir)

    # --- Helper Methods ---
    @staticmethod
    def _to_mllp(segments: list[str]) -> bytes:
        """Wraps HL7 segments in MLLP framing."""
        m = bytes([simulator.MLLP_START_OF_BLOCK])
        m += bytes("\r".join(segments) + "\r", "ascii")
        m += bytes([simulator.MLLP_END_OF_BLOCK, simulator.MLLP_CARRIAGE_RETURN])
        return m

    @staticmethod
    def _from_mllp(buffer: bytes) -> list[str]:
        """Strips MLLP framing and returns segments."""
        return str(buffer[1:-3], "ascii").split("\r")

    def _wait_until_healthy(self, timeout=10) -> bool:
        """Polls the /healthy endpoint until 200 OK or timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(
                    f"http://localhost:{TEST_PAGER_PORT}/healthy"
                ) as r:
                    if r.status == 200:
                        return True
            except (urllib.error.URLError, http.client.RemoteDisconnected):
                pass
            time.sleep(0.5)
        return False

    # --- Test Cases ---

    def test_mllp_handshake_and_flow(self):
        """Verify that the simulator sends messages and waits for ACKs."""
        received = []
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(("localhost", TEST_MLLP_PORT))

            # Read ADT_A01
            data = s.recv(1024)
            received.append(self._from_mllp(data))
            s.sendall(self._to_mllp(ACK))

            # Read ORU_R01
            data = s.recv(1024)
            received.append(self._from_mllp(data))
            s.sendall(self._to_mllp(ACK))

        self.assertEqual(received[0], ADT_A01)
        self.assertEqual(received[1], ORU_R01)

    def test_pager_endpoint_valid_request(self):
        """Test the /page endpoint with valid MRN and timestamp."""
        data = b"478237423,202401202243"
        req = urllib.request.Request(
            f"http://localhost:{TEST_PAGER_PORT}/page", data=data, method="POST"
        )
        with urllib.request.urlopen(req) as r:
            self.assertEqual(r.status, 200)

    def test_pager_endpoint_invalid_mrn(self):
        """Verify that alphanumeric MRNs are rejected (BAD REQUEST)."""
        data = b"INVALID_MRN_123"
        req = urllib.request.Request(
            f"http://localhost:{TEST_PAGER_PORT}/page", data=data, method="POST"
        )
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)

        self.assertEqual(cm.exception.code, 400)
        cm.exception.close()

    def test_pager_endpoint_bad_timestamp(self):
        """Verify that malformed timestamps are rejected."""
        data = b"12345,bad-date-format"
        req = urllib.request.Request(
            f"http://localhost:{TEST_PAGER_PORT}/page", data=data, method="POST"
        )
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)

        self.assertEqual(cm.exception.code, 400)
        cm.exception.close()
