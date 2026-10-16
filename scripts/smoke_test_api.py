"""Smoke test for a running prediction API (local, Docker, or deployed).

Usage:  python scripts/smoke_test_api.py http://127.0.0.1:7860
Uses only the standard library. Exit code 0 = all checks passed.
"""

from __future__ import annotations

import json
import sys
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
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


def main(base: str) -> int:
    base = base.rstrip("/")
    results: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        results.append((name, ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}")

    status, body = call(base, "/health")
    check("GET /health is ok and model loaded", status == 200 and body == {"status": "ok", "model_loaded": True})
    status, body = call(base, "/model")
    check("GET /model returns registry metadata", status == 200 and "registry_version" in body)
    status, body = call(base, "/predict", EXAMPLE)
    check("POST /predict returns a probability in [0, 1]",
          status == 200 and 0 <= body["churn_probability"] <= 1)
    if status == 200:
        print(f"      churn_probability={body['churn_probability']} prediction={body['prediction']} "
              f"model_version={body['model_version']}")
    status, body = call(base, "/predict", {**EXAMPLE, "Contract": "Weekly"})
    check("POST /predict rejects an unknown category (422)", status == 422)
    status, body = call(base, "/predict", {**EXAMPLE, "TotalCharges": " "})
    check("POST /predict accepts blank TotalCharges", status == 200)
    status, body = call(base, "/predict/batch", {"customers": [EXAMPLE, EXAMPLE]})
    check("POST /predict/batch returns 2 predictions", status == 200 and len(body["predictions"]) == 2)

    failed = [n for n, ok in results if not ok]
    print("\nALL CHECKS PASSED" if not failed else f"\n{len(failed)} CHECK(S) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"))
