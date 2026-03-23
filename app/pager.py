"""
Handles HTTP notifications to the hospital pager system with resilience and monitoring.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests

from .config import SETTINGS
from .monitoring import PAGER_LATENCY_SECONDS, PAGER_REQUESTS_TOTAL

logger = logging.getLogger(__name__)


class PagerService:
    """
    Service for sending alerts.
    Features:
    - Non-blocking execution (ThreadPool)
    - Latency tracking (SLA monitoring)
    - Automatic retries for transient failures
    """

    def __init__(self, workers: int = 10):
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def send_page(self, mrn: str, event_time: datetime, arrival_time: float) -> None:
        """Submits a paging task to the worker pool."""
        self.pool.submit(self._do_request_with_retry, mrn, event_time, arrival_time)

    @staticmethod
    def _do_request_with_retry(
        mrn: str, event_time: datetime, arrival_time: float
    ) -> None:
        """
        Executes HTTP POST with exponential backoff.
        Retries on 5xx errors or timeouts.
        Stops on 4xx errors (client error) or success.
        """
        url = f"http://{SETTINGS.PAGER_ADDRESS}/page"
        ts_str = event_time.strftime("%Y%m%d%H%M")
        payload = f"{mrn},{ts_str}"
        headers = {"Content-Type": "text/plain"}

        attempt = 0
        max_retries = 3
        backoff = 1  # seconds

        while attempt < max_retries:
            try:
                resp = requests.post(url, data=payload, headers=headers, timeout=2)

                # Success (200 OK)
                if resp.status_code == 200:
                    PAGER_REQUESTS_TOTAL.labels(status="success").inc()

                    # Calculate latency based on processing time
                    processing_latency = time.time() - arrival_time
                    PAGER_LATENCY_SECONDS.observe(processing_latency)

                    logger.info(
                        f"PAGED: {mrn} at {ts_str} (Latency: {processing_latency:.4f}s)"
                    )
                    return

                # Client Error (4xx) - Do not retry
                if 400 <= resp.status_code < 500:
                    logger.error(f"Pager rejected {mrn}: {resp.status_code}")
                    PAGER_REQUESTS_TOTAL.labels(status="error").inc()
                    return

                # Server Error (5xx) - Retry
                logger.warning(f"Pager server error {resp.status_code}. Retrying...")
                PAGER_REQUESTS_TOTAL.labels(status="error").inc()

            except requests.RequestException as e:
                logger.warning(f"Pager network error: {e}")
                PAGER_REQUESTS_TOTAL.labels(status="timeout").inc()

            # Backoff and retry
            attempt += 1
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= 2

        logger.error(f"Failed to page {mrn} after {max_retries} attempts.")

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)
