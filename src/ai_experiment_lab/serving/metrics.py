"""Prometheus metrics for the prediction API.

Each app gets its own registry (not the process-global one), so several apps can coexist in
tests without "duplicated timeseries" errors. Label values are bounded: the path label is the
route template (``/predict``), never the raw URL, so unknown URLs cannot create unbounded series.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
PROBABILITY_BUCKETS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
CONTENT_TYPE = CONTENT_TYPE_LATEST


class Metrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "ael_http_requests_total", "HTTP requests by route and status code",
            ["method", "path", "status"], registry=self.registry)
        self.latency = Histogram(
            "ael_http_request_duration_seconds", "HTTP request latency in seconds",
            ["method", "path"], buckets=LATENCY_BUCKETS, registry=self.registry)
        self.predictions = Counter(
            "ael_predictions_total", "Predictions made, by predicted class",
            ["prediction"], registry=self.registry)
        self.probability = Histogram(
            "ael_churn_probability", "Distribution of predicted churn probabilities",
            buckets=PROBABILITY_BUCKETS, registry=self.registry)
        self.model_loaded = Gauge(
            "ael_model_loaded", "1 if a model is loaded, else 0", registry=self.registry)
        self.model_info = Gauge(
            "ael_model_info", "Served model identity (value is always 1)",
            ["registered_name", "version"], registry=self.registry)

    def observe_request(self, method: str, path: str, status: int, seconds: float) -> None:
        self.requests.labels(method, path, str(status)).inc()
        self.latency.labels(method, path).observe(seconds)

    def observe_predictions(self, preds) -> None:
        """``preds`` is an iterable of ``(predicted_label, probability_or_None)``.

        The label set is bounded (two classes, at most 50 classes, or the constant "value" for
        regression). The probability histogram only gets binary probabilities.
        """
        for label, probability in preds:
            self.predictions.labels(str(label)).inc()
            if probability is not None:
                self.probability.observe(probability)

    def set_model(self, meta: dict | None) -> None:
        self.model_info.clear()
        if meta is None:
            self.model_loaded.set(0)
            return
        self.model_loaded.set(1)
        self.model_info.labels(
            str(meta.get("registered_name", "unknown")), str(meta.get("registry_version", "unknown"))
        ).set(1)

    def render(self) -> bytes:
        return generate_latest(self.registry)
