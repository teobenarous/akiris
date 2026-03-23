import time
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import requests

from app.pager import PagerService


class TestPagerService(unittest.TestCase):
    def setUp(self):
        # We test the static retry method directly to avoid thread race conditions
        self.pager = PagerService(workers=1)
        self.test_time = datetime(2024, 1, 1, 12, 0)
        self.arrival_time = time.time()

    @patch("app.pager.requests.post")
    def test_page_success_200(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        self.pager._do_request_with_retry("123", self.test_time, self.arrival_time)
        mock_post.assert_called_once()  # Should not retry

    @patch("app.pager.requests.post")
    def test_page_client_error_400(self, mock_post):
        """Verify 4xx errors are dropped immediately without retrying."""
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_post.return_value = mock_resp

        with self.assertLogs("app.pager", level="ERROR"):
            self.pager._do_request_with_retry("123", self.test_time, self.arrival_time)
            mock_post.assert_called_once()  # Should NOT retry

    @patch("app.pager.time.sleep")  # Don't actually sleep in tests
    @patch("app.pager.requests.post")
    def test_page_server_error_500_retry(self, mock_post, mock_sleep):
        """Verify 5xx errors trigger exactly 3 retries with exponential backoff."""
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_post.return_value = mock_resp

        with self.assertLogs("app.pager", level="WARNING"):
            self.pager._do_request_with_retry("123", self.test_time, self.arrival_time)
            self.assertEqual(mock_post.call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)  # Sleeps after attempt 1 and 2

    @patch("app.pager.time.sleep")
    @patch("app.pager.requests.post", side_effect=requests.RequestException("Timeout"))
    def test_page_network_timeout(self, mock_post, _):
        """Verify network exceptions trigger the retry loop."""
        with self.assertLogs("app.pager", level="WARNING"):
            self.pager._do_request_with_retry("123", self.test_time, self.arrival_time)
            self.assertEqual(mock_post.call_count, 3)

