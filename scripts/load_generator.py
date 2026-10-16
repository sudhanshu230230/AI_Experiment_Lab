"""Send synthetic traffic to the prediction API so the dashboard and drift reports have data.

Usage:
    python scripts/load_generator.py --url http://127.0.0.1:7860 --rate 5 --duration 120
    python scripts/load_generator.py --drift ...        # shifted customer mix (more risky customers)
    python scripts/load_generator.py --bad-fraction 0.1 # 10% invalid requests (shows 422s)

Standard library only. The generated customers are random, not real people.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request

YN = ["Yes", "No"]
YNI = ["Yes", "No", "No internet service"]
PAYMENT = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]


def customer(rng: random.Random, drift: bool) -> dict:
    """Random customer. With ``drift`` the mix shifts to short-tenure, month-to-month, fiber, high bills."""
    if drift:
        tenure = int(rng.betavariate(1.2, 6) * 72)
        contract = rng.choices(["Month-to-month", "One year", "Two year"], [0.9, 0.07, 0.03])[0]
        internet = rng.choices(["DSL", "Fiber optic", "No"], [0.1, 0.85, 0.05])[0]
        monthly = round(rng.uniform(75, 118), 2)
        payment = rng.choices(PAYMENT, [0.75, 0.15, 0.05, 0.05])[0]
    else:
        tenure = int(rng.betavariate(1.5, 1.8) * 72)
        contract = rng.choices(["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24])[0]
        internet = rng.choices(["DSL", "Fiber optic", "No"], [0.34, 0.44, 0.22])[0]
        monthly = round(rng.uniform(18, 115), 2)
        payment = rng.choices(PAYMENT, [0.34, 0.23, 0.22, 0.21])[0]
    addon = (lambda: "No internet service") if internet == "No" else (lambda: rng.choice(YN))
    return {
        "gender": rng.choice(["Male", "Female"]), "SeniorCitizen": int(rng.random() < 0.16),
        "Partner": rng.choice(YN), "Dependents": rng.choice(YN), "tenure": tenure,
        "PhoneService": "Yes", "MultipleLines": rng.choice(["Yes", "No"]),
        "InternetService": internet, "OnlineSecurity": addon(), "OnlineBackup": addon(),
        "DeviceProtection": addon(), "TechSupport": addon(), "StreamingTV": addon(),
        "StreamingMovies": addon(), "Contract": contract, "PaperlessBilling": rng.choice(YN),
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": "" if tenure == 0 else str(round(tenure * monthly, 2)),
    }


def post(url: str, payload: dict) -> int:
    req = urllib.request.Request(url + "/predict", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--rate", type=float, default=5.0, help="requests per second")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds")
    ap.add_argument("--drift", action="store_true", help="send a shifted customer mix")
    ap.add_argument("--bad-fraction", type=float, default=0.0, help="share of invalid requests")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    counts: dict[int, int] = {}
    end = time.time() + a.duration
    while time.time() < end:
        payload = customer(rng, a.drift)
        if rng.random() < a.bad_fraction:
            payload["Contract"] = "Weekly"
        code = post(a.url.rstrip("/"), payload)
        counts[code] = counts.get(code, 0) + 1
        time.sleep(1.0 / a.rate)
    print("status counts:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
