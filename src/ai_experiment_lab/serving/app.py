"""FastAPI service that serves the exported champion model (ADR-005).

Environment variables:
    MODEL_DIR        folder with model.skops + metadata.json   (default: models/champion)
    PREDICTION_LOG   optional path of a JSONL file; each prediction (features, probability,
                     model version, UTC time) is appended for later drift analysis (Phase 7)
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ai_experiment_lab.registry.export import load_exported
from ai_experiment_lab.serving.contract import Contract, GenericPrediction, load_contract
from ai_experiment_lab.serving.lab import register_lab
from ai_experiment_lab.serving.metrics import CONTENT_TYPE, Metrics
from ai_experiment_lab.serving.schemas import (
    BatchResponse,
    Health,
    Prediction,
)
from ai_experiment_lab.serving.ui import render_index, ui_spec

log = logging.getLogger("ai_experiment_lab.serving")
DEFAULT_MODEL_DIR = "models/champion"


class ModelService:
    """Holds the loaded pipeline and turns validated requests into predictions."""

    def __init__(self, model_dir: str | Path, prediction_log: str | Path | None = None,
                 metrics: Metrics | None = None, contract: Contract | None = None):
        self.metrics = metrics
        self.contract = contract or load_contract(model_dir)
        self.pipeline, self.meta = load_exported(model_dir)
        self.columns: list[str] = list(self.meta["input_columns"])
        self.threshold = float(self.meta.get("threshold", 0.5))
        self.positive_label = self.meta.get("positive_label") or "Yes"
        self.negative_label = self.meta.get("negative_label") or "No"
        self.version = self.meta.get("registry_version")
        self._log_path = Path(prediction_log) if prediction_log else None
        self._lock = threading.Lock()

        expected = set(self.contract.columns)
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

    def predict(self, rows: list[Any]) -> list[Any]:
        frame = pd.DataFrame([r.model_dump(by_alias=True) for r in rows])[self.columns]
        out, scores, labels = self._score(frame)
        if self.metrics is not None:
            self.metrics.observe_predictions(
                [(lab, sc if self.contract.task == "binary" else None)
                 for lab, sc in zip(labels, scores, strict=True)])
        self._log(frame, out, labels, scores)
        return out

    def _score(self, frame: pd.DataFrame) -> tuple[list[Any], list[float], list[str]]:
        task, c = self.contract.task, self.contract
        if task == "regression":
            vals = [float(v) for v in self.pipeline.predict(frame)]
            out = [GenericPrediction(prediction=round(v, 6), model_version=self.version) for v in vals]
            return out, vals, ["value"] * len(vals)
        proba = self.pipeline.predict_proba(frame)
        if task == "multiclass":
            classes = c.classes
            out, scores, labels = [], [], []
            for row in proba:
                top = int(np.argmax(row))
                out.append(GenericPrediction(
                    prediction=classes[top],
                    probabilities={k: round(float(p), 6) for k, p in zip(classes, row, strict=True)},
                    model_version=self.version))
                scores.append(float(row[top]))
                labels.append(classes[top])
            return out, scores, labels
        p1 = [float(p) for p in proba[:, 1]]
        labels = [self.positive_label if p >= self.threshold else self.negative_label for p in p1]
        if c.legacy:
            out = [Prediction(churn_probability=round(p, 6), prediction=lab,
                              threshold=self.threshold, model_version=self.version)
                   for p, lab in zip(p1, labels, strict=True)]
        else:
            out = [GenericPrediction(prediction=lab, probability=round(p, 6),
                                     threshold=self.threshold, model_version=self.version)
                   for p, lab in zip(p1, labels, strict=True)]
        return out, p1, labels

    def _log(self, frame: pd.DataFrame, preds: list[Any], labels: list[str],
             scores: list[float]) -> None:
        if self._log_path is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        lines = []
        for feats, p, lab, sc in zip(frame.to_dict("records"), preds, labels, scores, strict=True):
            rec = {"ts": now, "model_version": p.model_version, "score": round(sc, 6),
                   "prediction": lab, "features": _jsonable(feats)}
            if self.contract.legacy:
                rec["churn_probability"] = p.churn_probability
            lines.append(json.dumps(rec))
        try:
            with self._lock:
                self._log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
        except OSError:
            # A full disk or a read-only mount must never turn a good prediction into a 500.
            log.exception("Could not write the prediction log %s", self._log_path)


def _jsonable(rec: dict[str, Any]) -> dict[str, Any]:
    """Numpy scalars and NaN to plain JSON values (missing becomes null)."""
    out = {}
    for k, v in rec.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out[k] = None
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def create_app(model_dir: str | Path | None = None,
               prediction_log: str | Path | None = None) -> FastAPI:
    """Build the app. Arguments override the environment variables (used by tests).

    The request and response shapes come from the model's own metadata (``serving/contract.py``),
    so the app is built after reading ``metadata.json`` but before the model itself is loaded.
    """

    metrics = Metrics()
    mdir = model_dir or os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
    contract = load_contract(mdir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        plog = prediction_log or os.environ.get("PREDICTION_LOG")
        try:
            app.state.service = ModelService(mdir, plog, metrics, contract)
            metrics.set_model(app.state.service.meta)
            log.info("Loaded model from %s (version %s)", mdir, app.state.service.version)
        except Exception:
            # Stay up so /health can report 'degraded' instead of crash-looping silently.
            log.exception("Could not load model from %s", mdir)
            app.state.service = None
            metrics.set_model(None)
        yield

    app = FastAPI(
        title=("AI Experiment Lab: churn prediction" if contract.legacy
               else f"AI Experiment Lab: {contract.title}"),
        version="0.6.0",
        description="Serves the registry champion (see docs/decisions/ADR-005 and ADR-010).",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        """A 422 that lists where and why, without echoing the submitted values.

        FastAPI's default handler puts the rejected value in the response, which cannot be turned
        into JSON when it is NaN or Infinity (a request body may legally contain those tokens):
        the client would get a 500 instead of a 422.
        """
        detail = [{"type": e.get("type"), "loc": list(e.get("loc", ())), "msg": e.get("msg")}
                  for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

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

    spec = ui_spec(contract)
    page = render_index(spec)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        """The bare URL opens the web page for people; the API documentation is at /docs."""
        return RedirectResponse(url="app")

    @app.get("/app", include_in_schema=False)
    def web_app() -> HTMLResponse:
        """Form for one row and CSV upload for many (plain HTML, see serving/ui.py)."""
        return HTMLResponse(page)

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
                "code_version", "libraries", "threshold", "input_columns",
                "task", "target", "title", "classes", "positive_label", "negative_label")
        return {k: meta[k] for k in keep if k in meta}

    @app.get("/schema")
    def schema() -> dict[str, Any]:
        """The inputs this model expects (names, types, allowed values) and what it predicts."""
        return spec

    row_model, batch_model = contract.row_model, contract.batch_model
    response_model = Prediction if contract.legacy else GenericPrediction

    def predict(row, request: Request):
        return service(request).predict([row])[0]

    def predict_batch(batch, request: Request):
        preds = service(request).predict(batch.customers)
        return {"predictions": preds}

    # FastAPI reads these annotations to validate the body; they are set here because the
    # request model is built at runtime, from the model's schema.
    predict.__annotations__["row"] = row_model
    predict_batch.__annotations__["batch"] = batch_model
    app.add_api_route("/predict", predict, methods=["POST"], response_model=response_model,
                      response_model_exclude_none=not contract.legacy,
                      summary="Predict for one row")
    batch_response = BatchResponse if contract.legacy else _generic_batch_response()
    app.add_api_route("/predict/batch", predict_batch, methods=["POST"],
                      response_model=batch_response,
                      response_model_exclude_none=not contract.legacy,
                      summary="Predict for up to 1000 rows")

    if register_lab(app) is not None:
        app.state.lab_enabled = True

    return app


def _generic_batch_response():
    from pydantic import BaseModel

    class GenericBatchResponse(BaseModel):
        predictions: list[GenericPrediction]

    return GenericBatchResponse


app = create_app()
