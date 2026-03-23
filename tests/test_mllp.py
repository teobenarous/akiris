import socket
import unittest
from unittest.mock import MagicMock

from app.mllp import CARRIAGE_RETURN, END_BLOCK, START_BLOCK, read_messages


class MockSocket:
    """
    A simple mock for a TCP socket to simulate byte-stream reading.
    Allows testing of TCP fragmentation via the chunk_size parameter.
    """

    def __init__(self, data: bytes, chunk_size: int = 4096):
        self.data = data
        self.cursor = 0
        self.chunk_size = chunk_size

    def recv(self, size: int) -> bytes:
        if self.cursor >= len(self.data):
            return b""

        # Simulate fragmentation by taking the minimum of the requested size or the allowed chunk
        actual_size = min(size, self.chunk_size)
        chunk = self.data[self.cursor : self.cursor + actual_size]
        self.cursor += actual_size
        return chunk


class TestMLLP(unittest.TestCase):
    """Test suite for MLLP framing, network resilience, and socket reading logic."""

    def setUp(self):
        """Set up standard valid MLLP frames for reuse."""
        self.sample_content = "MSH|^~\\&|TEST"
        self.valid_frame = (
            START_BLOCK
            + self.sample_content.encode("ascii")
            + END_BLOCK
            + CARRIAGE_RETURN
        )

    def test_valid_frame(self):
        """Verify that a standard, well-formed MLLP frame is parsed correctly."""
        sock = MockSocket(self.valid_frame)
        msgs = list(read_messages(sock))

        self.assertEqual(msgs, [self.sample_content])

    def test_fragmented_frame(self):
        """Verify TCP fragmentation (byte-by-byte delivery) is handled seamlessly."""
        sock = MockSocket(self.valid_frame, chunk_size=1)
        msgs = list(read_messages(sock))

        self.assertEqual(msgs, [self.sample_content])

    def test_junk_data_resilience(self):
        """Verify bytes outside the standard Start/End blocks are ignored."""
        junk_frame = (
            b"noise_before"
            + START_BLOCK
            + b"REAL_PAYLOAD"
            + END_BLOCK
            + CARRIAGE_RETURN
            + b"noise_after"
        )
        sock = MockSocket(junk_frame)
        msgs = list(read_messages(sock))

        self.assertEqual(msgs, ["REAL_PAYLOAD"])

    def test_socket_timeout_yields_none(self):
        """Verify a socket timeout yields None to allow main loop heartbeat checks."""
        sock = MagicMock(spec=socket.socket)
        sock.recv.side_effect = socket.timeout("timeout")

        generator = read_messages(sock)

        self.assertIsNone(next(generator))

    def test_error_breaks_generator(self):
        """Verify network errors (e.g., ConnectionReset) cleanly exit the generator."""
        sock = MagicMock(spec=socket.socket)
        sock.recv.side_effect = OSError("Connection reset by peer")

        generator = read_messages(sock)

        # If the generator catches the error and exits gracefully, it yields nothing.
        self.assertEqual(list(generator), [])

    def test_unicode_decode_error_skipped(self):
        """Verify non-ASCII garbage bytes are skipped without crashing the pipeline."""
        bad_frame = START_BLOCK + b"\xff\xfe\xfa" + END_BLOCK + CARRIAGE_RETURN
        sock = MockSocket(bad_frame)

        generator = read_messages(sock)

        self.assertEqual(list(generator), [])

