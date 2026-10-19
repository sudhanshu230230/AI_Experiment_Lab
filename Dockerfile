# Serving image for the champion model (Phase 5, ADR-006).
# Build from the repository root:  docker build -t ael-api .
# Run:                             docker run --rm -p 7860:7860 ael-api
FROM python:3.12-slim

# Use the scikit-learn version the model was trained with (see models/champion/metadata.json,
# "libraries"). Override with: --build-arg SKLEARN_VERSION=x.y.z
ARG SKLEARN_VERSION=1.6.1
# Only needed when the model contains an XGBoost model (metadata.json lists "xgboost" under
# "libraries" and scripts/check_model_export.py says so). Empty = not installed.
ARG XGBOOST_VERSION=

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models/champion \
    PORT=7860

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Dependencies first (cached unless pyproject.toml or the source changes).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]" "scikit-learn==${SKLEARN_VERSION}" \
    && if [ -n "${XGBOOST_VERSION}" ]; then pip install "xgboost==${XGBOOST_VERSION}"; fi

# The model is copied last: a new model does not reinstall the dependencies.
COPY models/champion ./models/champion

USER app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '7860'), timeout=3)"

CMD ["sh", "-c", "uvicorn ai_experiment_lab.serving.app:app --host 0.0.0.0 --port ${PORT}"]
