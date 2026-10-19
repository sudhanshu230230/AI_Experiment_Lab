"""Smoke test for a running prediction API (local, Docker, or deployed).

Usage:  python scripts/smoke_test_api.py http://127.0.0.1:7860
        python scripts/smoke_test_api.py URL --expect-version 4   # also check the served model version
        python scripts/smoke_test_api.py https://churn-api.onrender.com --wait 120

--wait SECONDS keeps polling /health until the service answers (free hosts sleep when idle and
need about a minute to wake up). Uses only the standard library. Exit code 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

EXAMPLE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 5,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5,
    "TotalCharges": "420.35",
}


def call(base: str, path: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            try:
                return r.status, json.loads(raw)
            except ValueError:          # an HTML page, for example
                return r.status, None
    except urllib.error.HTTPError as e:
        return e.code, None


def wait_until_up(base: str, seconds: float) -> bool:
    """Poll /health until it returns 200 or the time runs out."""
    deadline = time.time() + seconds
    while True:
        try:
            status, _ = call(base, "/health")
            if status == 200:
                return True
        except (urllib.error.URLError, OSError):
            pass
        if time.time() >= deadline:
            return False
        print("waiting for the service to wake up ...")
        time.sleep(5)


def main(base: str, wait: float = 0, expect_version: int | None = None) -> int:
    base = base.rstrip("/")
    if wait and not wait_until_up(base, wait):
        print(f"FAIL  service did not answer /health within {wait:.0f} s")
        return 1
    results: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        results.append((name, ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}")

    status, body = call(base, "/health")
    check("GET /health is ok and model loaded", status == 200 and body == {"status": "ok", "model_loaded": True})
    status, body = call(base, "/model")
    check("GET /model returns registry metadata", status == 200 and "registry_version" in body)
    if expect_version is not None:
        served = body.get("registry_version") if status == 200 and body else None
        check(f"served model is registry version {expect_version} (got {served})", served == expect_version)
    status, spec = call(base, "/schema")
    generic = status == 200 and isinstance(spec, dict) and not spec.get("legacy", True)
    payload = spec["presets"][0]["values"] if generic else EXAMPLE
    task = spec["task"] if generic else "binary"

    status, body = call(base, "/predict", payload)
    if generic and task == "regression":
        check("POST /predict returns a number", status == 200 and isinstance(body["prediction"], (int, float)))
    elif generic and task == "multiclass":
        check("POST /predict returns a class and its probabilities",
              status == 200 and abs(sum(body["probabilities"].values()) - 1) < 1e-3)
    else:
        key = "probability" if generic else "churn_probability"
        check("POST /predict returns a probability in [0, 1]", status == 200 and 0 <= body[key] <= 1)
    if status == 200:
        print("      " + " ".join(f"{k}={v}" for k, v in body.items() if k != "probabilities"))

    if generic:
        field = next((f for f in spec["fields"] if f["input"] == "select" and f["send"] == "string"), None)
        bad = {**payload, field["name"]: "__not_a_category__"} if field else \
            {**payload, spec["fields"][0]["name"]: "not a number"}
        status, _ = call(base, "/predict", bad)
        check("POST /predict rejects an invalid value (422)", status == 422)
    else:
        status, body = call(base, "/predict", {**EXAMPLE, "Contract": "Weekly"})
        check("POST /predict rejects an unknown category (422)", status == 422)
        status, body = call(base, "/predict", {**EXAMPLE, "TotalCharges": " "})
        check("POST /predict accepts blank TotalCharges", status == 200)
    status, body = call(base, "/predict/batch", {"customers": [payload, payload]})
    check("POST /predict/batch returns 2 predictions", status == 200 and len(body["predictions"]) == 2)
    status, _ = call(base, "/app")
    check("GET /app serves the web page", status == 200)

    failed = [n for n, ok in results if not ok]
    print("\nALL CHECKS PASSED" if not failed else f"\n{len(failed)} CHECK(S) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smoke test for the prediction API")
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:7860")
    ap.add_argument("--wait", type=float, default=0, help="seconds to wait for the service to come up")
    ap.add_argument("--expect-version", type=int, default=None,
                    help="fail unless /model reports this registry_version")
    args = ap.parse_args()
    sys.exit(main(args.url, args.wait, args.expect_version))
