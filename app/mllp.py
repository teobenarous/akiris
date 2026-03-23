"""
MLLP Networking.
Handles the framing of HL7 messages over TCP.
Frame: <VT> payload <FS><CR>
"""

import logging
import socket
from datetime import datetime
from typing import Generator

from .monitoring import MLLP_ERRORS_TOTAL

START_BLOCK = b"\x0b"  # vertical tab
END_BLOCK = b"\x1c"  # file separator
CARRIAGE_RETURN = b"\x0d"

logger = logging.getLogger(__name__)


def create_ack() -> bytes:
    """
    Constructs an HL7 ACK (application accept) response.
    """
    # timestamp for the ACK header
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # ACK Body
    ack_msh = f"MSH|^~\\&|||||{timestamp}||ACK|||2.5"
    ack_msa = "MSA|AA"
    hl7_response = f"{ack_msh}\r{ack_msa}\r"

    # wrap in MLLP
    return START_BLOCK + hl7_response.encode("ascii") + END_BLOCK + CARRIAGE_RETURN


def read_messages(sock: socket.socket) -> Generator[str | None, None, None]:
    """
    Yields parsed HL7 messages from a TCP socket.
    Includes buffer protection against garbage data.
    """
    buffer = b""

    while True:
        try:
            chunk = sock.recv(4096)
            if not chunk:
                break  # connection closed
        except socket.timeout:
            # Wake up the main loop to check for shutdown signals
            yield None
            continue
        except OSError as e:
            # catches timeouts, connection resets, broken pipes
            logger.error(f"Network error: {e}")
            MLLP_ERRORS_TOTAL.labels(type="socket_error").inc()
            break

        buffer += chunk

        # extract all available messages
        while True:
            start_idx = buffer.find(START_BLOCK)
            end_idx = buffer.find(END_BLOCK)

            # We found an end, but no start before it. Discard everything up to the end.
            if start_idx == -1 and end_idx != -1:
                buffer = buffer[end_idx + 2 :]
                continue

            # complete frame
            if start_idx != -1 and end_idx != -1:
                # sanity check: start must precede end
                if start_idx < end_idx:
                    msg_bytes = buffer[start_idx + 1 : end_idx]
                    try:
                        msg_str = msg_bytes.decode("ascii")
                        yield msg_str
                    except UnicodeDecodeError:
                        logger.warning("Skipping non-ASCII message.")
                        MLLP_ERRORS_TOTAL.labels(type="decode_error").inc()

                # advance buffer past this message (or garbage if start > end)
                buffer = buffer[end_idx + 2 :]
            else:
                # incomplete message, wait for more data
                break
