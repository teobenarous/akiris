import socket
import unittest
from unittest.mock import MagicMock, patch

from app.main import run_loop


class TestSystemFailures(unittest.TestCase):
    def setUp(self):
        """Ensure global state is reset before each test."""
        import app.main

        app.main.SHUTDOWN_FLAG = False

    @patch("app.main.socket.create_connection")
    @patch("app.main.time.sleep")  # Prevent actual waiting
    def test_tcp_disconnect_reconnect(self, mock_sleep, mock_connect):
        """
        Failure Test: Verify loop attempts to reconnect on socket failure.

        Scenario:
        1. Connects successfully -> Receives data -> Remote close
        2. Reconnects -> Fails (Network Error)
        3. Reconnects -> Success -> Receives data
        """
        # Setup dependencies
        store = MagicMock()
        predictor = MagicMock()
        pager = MagicMock()

        # --- Socket 1: Initial Success ---
        sock_1 = MagicMock()
        sock_1.__enter__.return_value = sock_1
        # Returns one message, then empty bytes (indicating remote close)
        sock_1.recv.side_effect = [
            b"\x0bMSH|^~\\&|...|||||ADT^A01\rPID|1||123\x1c\r",
            b"",
        ]

        # --- Socket 2: Reconnection Success ---
        sock_2 = MagicMock()
        sock_2.__enter__.return_value = sock_2
        # Returns one message, then empty bytes
        sock_2.recv.side_effect = [
            b"\x0bMSH|^~\\&|...|||||ADT^A01\rPID|1||123\x1c\r",
            b"",
        ]

        # --- Connection Lifecycle ---
        # 1. First call: Returns sock_1
        # 2. Second call: Raises ConnectionError (Simulates network down)
        # 3. Third call: Returns sock_2 (Simulates network restored)
        mock_connect.side_effect = [sock_1, socket.error("Network Down"), sock_2]

        # --- Loop Control ---
        # Stop the loop only after we've hit the error and tried to reconnect
        def stop_loop(*_):
            import app.main

            # Check call count to ensure we went through the reconnect cycle
            if mock_connect.call_count >= 3:
                app.main.SHUTDOWN_FLAG = True

        mock_sleep.side_effect = stop_loop

        # Run the loop
        try:
            run_loop(store, predictor, pager)
        except Exception as e:
            self.fail(f"Loop crashed: {e}")

        # Assertions
        # Ensure we actually tried to connect 3 times
        self.assertEqual(mock_connect.call_count, 3)

        # Ensure we slept (backed off) when the error occurred
        self.assertTrue(mock_sleep.called)

    @patch("app.main.socket.create_connection")
    @patch("app.main.time.sleep")
    @patch("app.main.sys.exit")
    @patch("app.main.read_messages")
    def test_circuit_breaker_trips(self, mock_read, mock_exit, *_):
        """Verify 5 consecutive unhandled exceptions trip the circuit breaker."""
        # Force a generic exception inside the connection loop
        mock_read.side_effect = Exception("Unknown severe memory fault")
        # Make the mock actually stop the execution flow
        mock_exit.side_effect = SystemExit

        store = MagicMock()
        predictor = MagicMock()
        pager = MagicMock()

        # Run loop. It will hit the exception 5 times, then call sys.exit(1).
        # We must catch the SystemExit so it doesn't fail the test runner.
        with self.assertRaises(SystemExit):
            run_loop(store, predictor, pager)

        # Verify sys.exit was called exactly once with status code 1
        mock_exit.assert_called_once_with(1)

    def test_ignore_discharged_patient(self):
        """Verify ORU results for discharged patients are ignored."""
        from app.main import process_message

        store = MagicMock()
        predictor = MagicMock()
        pager = MagicMock()

        # Setup patient as discharged
        patient = MagicMock()
        patient.is_discharged = True
        store.get_or_create.return_value = patient

        msg = "MSH|^~\\&|SIM|SRH|||2024||ORU^R01\rPID|1||123\rOBX|1|SN|CREATININE||200"
        process_message(msg, store, predictor, pager)

        predictor.predict.assert_not_called()
        pager.send_page.assert_not_called()

    @patch("app.main.socket.create_connection")
    @patch("app.main.sys.platform", "linux")
    @patch("app.main.socket.TCP_KEEPCNT", 6, create=True)
    @patch("app.main.socket.TCP_KEEPINTVL", 5, create=True)
    @patch("app.main.socket.TCP_KEEPIDLE", 4, create=True)
    @patch("app.main.read_messages")
    def test_os_specific_keep_alive_linux(self, mock_read, mock_connect):
        """Verify Linux-specific TCP Keep-Alive flags are set."""
        mock_sock = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_sock

        import socket

        import app.main

        app.main.SHUTDOWN_FLAG = False  # Ensure the loop starts

        # Flip the shutdown flag as soon as it tries to read messages
        def stop_loop(*_):
            app.main.SHUTDOWN_FLAG = True
            yield None

        mock_read.side_effect = stop_loop

        app.main.run_loop(MagicMock(), MagicMock(), MagicMock())

        # Verify the Linux socket option was applied during the connection phase
        mock_sock.setsockopt.assert_any_call(
            socket.IPPROTO_TCP, app.main.socket.TCP_KEEPIDLE, 60
        )

    @patch("app.main.SETTINGS.SAVE_INTERVAL", 2)
    @patch("app.main.socket.create_connection")
    @patch("app.main.read_messages")
    def test_periodic_save_triggered(self, mock_read, _):
        """Verify state is saved to disk every SAVE_INTERVAL messages."""
        store = MagicMock()

        # Yield exactly 2 messages, then simulate shutdown
        def fake_stream(*_):
            yield "MSH|^~\\&|...|||||ORU^R01\rPID|1||123\rOBX|1|SN|CREATININE||100"
            yield "MSH|^~\\&|...|||||ORU^R01\rPID|1||124\rOBX|1|SN|CREATININE||110"
            import app.main

            app.main.SHUTDOWN_FLAG = True

        mock_read.side_effect = fake_stream

        import app.main

        app.main.SHUTDOWN_FLAG = False
        app.main.run_loop(store, MagicMock(), MagicMock())

        # 2 messages processed % SAVE_INTERVAL (2) == 0. Should save once.
        store.save.assert_called_once()

    @patch("app.main.SETTINGS.RECONNECT", False)
    @patch("app.main.socket.create_connection")
    @patch("app.main.read_messages", return_value=())
    def test_no_reconnect_exits_cleanly(self, *_):
        """Verify the loop breaks gracefully if the socket closes and RECONNECT is False."""
        import app.main

        app.main.SHUTDOWN_FLAG = False

        # The loop should process the empty list, hit the reconnect check, and break.
        # We don't need to force a shutdown flag.
        app.main.run_loop(MagicMock(), MagicMock(), MagicMock())

        self.assertFalse(app.main.SHUTDOWN_FLAG)

    def test_message_processing_branches(self):
        """Exercises the negative inference, discharge, and ignore branches."""
        from app.main import process_message
        from app.state import PatientStore

        store = PatientStore()
        predictor = MagicMock()
        predictor.predict.return_value = False  # Force a negative prediction
        pager = MagicMock()

        # 1. Admit Patient
        process_message(
            "MSH|^~\\&|SIM|SRH|||2024||ADT^A01\rPID|1||999", store, predictor, pager
        )

        # 2. Blood Test -> Negative AKI
        process_message(
            "MSH|^~\\&|SIM|SRH|||2024||ORU^R01\rPID|1||999\rOBX|1|SN|CREATININE||100",
            store,
            predictor,
            pager,
        )
        pager.send_page.assert_not_called()

        # 3. Discharge Patient
        process_message(
            "MSH|^~\\&|SIM|SRH|||2024||ADT^A03\rPID|1||999", store, predictor, pager
        )
        self.assertTrue(store.patients["999"].is_discharged)

        # Clear the mock's history
        predictor.reset_mock()

        # 4. Blood Test for Discharged Patient
        process_message(
            "MSH|^~\\&|SIM|SRH|||2024||ORU^R01\rPID|1||999\rOBX|1|SN|CREATININE||100",
            store,
            predictor,
            pager,
        )

        # Predictor should not be called because the patient is discharged
        predictor.predict.assert_not_called()

