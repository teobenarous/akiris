"""
Prometheus metrics registry.
"""

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from .config import SETTINGS

# Counters
MESSAGES_TOTAL = Counter(
    "aki_messages_received_total",
    "Total number of HL7 messages received via MLLP.",
    ["message_type"],
)

BLOOD_TESTS_TOTAL = Counter(
    "aki_blood_tests_total", "Total number of creatinine test results processed."
)

AKI_PREDICTIONS_TOTAL = Counter(
    "aki_predictions_total",
    "Model inference outcomes.",
    ["result"],  # labels: positive, negative
)

PAGER_REQUESTS_TOTAL = Counter(
    "aki_pager_requests_total",
    "Outcome of HTTP requests to the pager system.",
    ["status"],  # labels: success, error, timeout
)

MLLP_RECONNECTS_TOTAL = Counter(
    "aki_mllp_reconnects_total",
    "Number of times the MLLP socket connection was re-established.",
)

MLLP_ERRORS_TOTAL = Counter(
    "aki_mllp_errors_total",
    "Network errors encountered during MLLP processing.",
    ["type"],  # labels: timeout, connection_reset, broken_pipe
)

# Histograms
LATENCY_BUCKETS = (
    0.001,
    0.002,
    0.004,
    0.006,
    0.008,
    0.01,
    0.02,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    float("inf"),
)

PAGER_LATENCY_SECONDS = Histogram(
    "aki_e2e_latency_seconds",
    "Time from blood test timestamp to successful page alert.",
    buckets=LATENCY_BUCKETS,
)

BLOOD_TEST_VALUES = Histogram(
    "aki_blood_test_values",
    "Distribution of creatinine test results.",
    buckets=(50, 75, 100, 125, 150, 200, 300, 400, float("inf")),
)

# Gauges
ACTIVE_CONNECTIONS = Gauge(
    "aki_active_connections",
    "Current number of active MLLP connections (should be 0 or 1).",
)


def start_metrics_server():
    """Starts the Prometheus HTTP server on the configured port."""
    start_http_server(SETTINGS.PROMETHEUS_PORT)
