"""FastAPI service that serves the exported champion model (ADR-005).

Environment variables:
    MODEL_DIR        folder with model.skops + metadata.json   (default: models/champion)
    PREDICTION_LOG   optional path of a JSONL file; each prediction (features, probability,
                     model version, UTC time) is appended for later drift analysis (Phase 7)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response

from ai_experiment_lab.registry.export import load_exported
from ai_experiment_lab.serving.metrics import CONTENT_TYPE, Metrics
from ai_experiment_lab.serving.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    Health,
    Prediction,
)

log = logging.getLogger("ai_experiment_lab.serving")
DEFAULT_MODEL_DIR = "models/champion"


class ModelService:
    """Holds the loaded pipeline and turns validated requests into predictions."""

    def __init__(self, model_dir: str | Path, prediction_log: str | Path | None = None,
                 metrics: Metrics | None = None):
        self.metrics = metrics
        self.pipeline, self.meta = load_exported(model_dir)
        self.columns: list[str] = list(self.meta["input_columns"])
        self.threshold = float(self.meta.get("threshold", 0.5))
        self.positive_label = self.meta.get("positive_label", "Yes")
        self.version = self.meta.get("registry_version")
        self._log_path = Path(prediction_log) if prediction_log else None
        self._lock = threading.Lock()

        expected = set(CustomerFeatures.model_fields)
        if set(self.columns) != expected:
            raise RuntimeError(
                "Model input columns do not match the API schema: "
                f"only in model {sorted(set(self.columns) - expected)}, "
                f"only in API {sorted(expected - set(self.columns))}"
            )
        trained_with = (self.meta.get("libraries") or {}).get("scikit-learn")
        running = importlib_metadata.version("scikit-learn")
        if trained_with and trained_with != running:
            log.warning("scikit-learn %s serving a model trained with %s", running, trained_with)

    def predict(self, customers: list[CustomerFeatures]) -> list[Prediction]:
        frame = pd.DataFrame([c.model_dump() for c in customers])[self.columns]
        proba = self.pipeline.predict_proba(frame)[:, 1]
        out = [
            Prediction(
                churn_probability=round(float(p), 6),
                prediction=self.positive_label if p >= self.threshold else "No",
                threshold=self.threshold,
                model_version=self.version,
            )
            for p in proba
        ]
        if self.metrics is not None:
            self.metrics.observe_predictions(out)
        self._log(customers, out)
        return out

    def _log(self, customers: list[CustomerFeatures], preds: list[Prediction]) -> None:
        if self._log_path is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        lines = [
            json.dumps(
                {"ts": now, "model_version": p.model_version,
                 "churn_probability": p.churn_probability, "features": c.model_dump()}
            )
            for c, p in zip(customers, preds, strict=True)
        ]
        try:
            with self._lock:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
        except OSError:
            # A full disk or a read-only mount must never turn a good prediction into a 500.
            log.exception("Could not write the prediction log %s", self._log_path)


def create_app(model_dir: str | Path | None = None,
               prediction_log: str | Path | None = None) -> FastAPI:
    """Build the app. Arguments override the environment variables (used by tests)."""

    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mdir = model_dir or os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
        plog = prediction_log or os.environ.get("PREDICTION_LOG")
        try:
            app.state.service = ModelService(mdir, plog, metrics)
            metrics.set_model(app.state.service.meta)
            log.info("Loaded model from %s (version %s)", mdir, app.state.service.version)
        except Exception:
            # Stay up so /health can report 'degraded' instead of crash-looping silently.
            log.exception("Could not load model from %s", mdir)
            app.state.service = None
            metrics.set_model(None)
        yield

    app = FastAPI(
        title="AI Experiment Lab: churn prediction",
        version="0.5.0",
        description="Serves the registry champion (see docs/decisions/ADR-005).",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            if path != "/metrics":
                metrics.observe_request(request.method, path, status, time.perf_counter() - start)

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(metrics.render(), media_type=CONTENT_TYPE)

    def service(request: Request) -> ModelService:
        svc = request.app.state.service
        if svc is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return svc

    @app.get("/health", response_model=Health)
    def health(request: Request) -> Health:
        loaded = request.app.state.service is not None
        return Health(status="ok" if loaded else "degraded", model_loaded=loaded)

    @app.get("/model")
    def model_info(request: Request) -> dict[str, Any]:
        meta = service(request).meta
        keep = ("registered_name", "registry_version", "alias", "model_name", "stage",
                "experiment_id", "metrics", "dataset_name", "dataset_sha256",
                "code_version", "libraries", "threshold", "input_columns")
        return {k: meta[k] for k in keep if k in meta}

    @app.post("/predict", response_model=Prediction)
    def predict(customer: CustomerFeatures, request: Request) -> Prediction:
        return service(request).predict([customer])[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        return BatchResponse(predictions=service(request).predict(batch.customers))

    return app


app = create_app()
