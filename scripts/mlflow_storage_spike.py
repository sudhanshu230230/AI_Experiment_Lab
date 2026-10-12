"""Spike for ADR-001: is an MLflow store (SQLite database + file artifacts) reliable on a given disk?

Run it against Google Drive in Colab and against local disk for comparison:

    python scripts/mlflow_storage_spike.py write      --root /content/drive/MyDrive/ai-experiment-lab/spike
    python scripts/mlflow_storage_spike.py concurrent --root /content/drive/MyDrive/ai-experiment-lab/spike
    # restart the Colab runtime (Runtime > Disconnect and delete runtime), remount Drive, then:
    python scripts/mlflow_storage_spike.py verify     --root /content/drive/MyDrive/ai-experiment-lab/spike

Every command ends with a line starting with RESULT: PASS or RESULT: FAIL.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import mlflow
from mlflow import MlflowClient

EXPERIMENT = "storage-spike"
TAG = "spike_batch"


def open_store(root: Path) -> tuple[MlflowClient, str]:
    """Open (creating if needed) a SQLite-backed MLflow store with artifacts under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{(root / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        art = (root / "artifacts").resolve()
        art.mkdir(parents=True, exist_ok=True)
        exp_id = client.create_experiment(EXPERIMENT, artifact_location=art.as_uri())
    else:
        exp_id = exp.experiment_id
    return client, exp_id


def write_runs(root: Path, n: int, label: str) -> tuple[int, int, float, list[str]]:
    """Log ``n`` runs (3 params, 3 metrics, 1 artifact each). Returns (ok, failed, seconds, errors)."""
    client, exp_id = open_store(root)
    ok, errors = 0, []
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "note.txt"
        for i in range(n):
            try:
                artifact.write_text(f"{label} run {i}\n")
                run = client.create_run(exp_id, tags={TAG: label})
                rid = run.info.run_id
                for k, v in (("model", "demo"), ("seed", str(i)), ("label", label)):
                    client.log_param(rid, k, v)
                for step, name in enumerate(("f1", "roc_auc", "pr_auc")):
                    client.log_metric(rid, name, 0.5 + 0.01 * i + 0.001 * step)
                client.log_artifact(rid, str(artifact))
                client.set_terminated(rid)
                ok += 1
            except Exception as exc:  # noqa: BLE001 - spike: record every failure kind
                errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
    return ok, n - ok, time.perf_counter() - start, errors


def count_runs(root: Path, label: str | None = None) -> tuple[int, int, int]:
    """Return (runs, runs_with_full_metadata, runs_with_artifact) from a fresh connection."""
    client, exp_id = open_store(root)
    flt = f"tags.{TAG} = '{label}'" if label else ""
    runs = client.search_runs([exp_id], filter_string=flt, max_results=5000)
    full = sum(1 for r in runs if len(r.data.params) == 3 and len(r.data.metrics) == 3)
    with_art = sum(1 for r in runs if any(a.path == "note.txt" for a in client.list_artifacts(r.info.run_id)))
    return len(runs), full, with_art


def new_label(prefix: str = "batch") -> str:
    """Unique per invocation, so repeated runs on a non-empty store are counted exactly."""
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def cmd_write(args) -> int:
    root = Path(args.root)
    args.label = args.label or new_label()
    print(f"batch label: {args.label}")
    ok, failed, secs, errors = write_runs(root, args.n, args.label)
    print(f"wrote {ok}/{args.n} runs in {secs:.1f}s ({secs / max(args.n, 1):.2f}s per run); failures: {failed}")
    for e in sorted(set(errors)):
        print("  error:", e)
    runs, full, with_art = count_runs(root, args.label)
    print(f"read back: {runs} runs, {full} with full params/metrics, {with_art} with artifact")
    good = failed == 0 and runs == args.n and full == args.n and with_art == args.n
    print("RESULT:", "PASS" if good else "FAIL", "write")
    return 0 if good else 1


def cmd_concurrent(args) -> int:
    """Two processes write at the same time, as would happen if two Colab sessions shared a store."""
    root = Path(args.root)
    open_store(root)  # create the store first so both writers start from the same state
    script = str(Path(__file__).resolve())
    stamp = new_label("concurrent")
    labels = [f"{stamp}-proc{i}" for i in (1, 2)]
    procs = [
        subprocess.Popen(
            [sys.executable, script, "write", "--root", str(root), "--n", str(args.n), "--label", lab],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for lab in labels
    ]
    outputs = [p.communicate()[0] for p in procs]
    codes = [p.returncode for p in procs]
    locked = sum(o.lower().count("locked") for o in outputs)
    for i, o in enumerate(outputs, 1):
        print(f"--- writer {i} (exit {codes[i - 1]}) ---\n{o.strip()[-600:]}")
    r1, _, _ = count_runs(root, labels[0])
    r2, _, _ = count_runs(root, labels[1])
    print(f"concurrent: proc1 stored {r1}/{args.n}, proc2 stored {r2}/{args.n}, 'locked' mentions: {locked}")
    good = r1 == args.n and r2 == args.n
    print("RESULT:", "PASS" if good else "FAIL", "concurrent")
    return 0 if good else 1


def cmd_verify(args) -> int:
    """Run after a runtime restart: is everything written earlier still there and readable?"""
    runs, full, with_art = count_runs(Path(args.root))
    print(f"found {runs} runs, {full} with full params/metrics, {with_art} with artifact")
    good = runs > 0 and full == runs and with_art == runs
    print("RESULT:", "PASS" if good else "FAIL", "verify (persistence)")
    return 0 if good else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("write", cmd_write), ("concurrent", cmd_concurrent), ("verify", cmd_verify)):
        sp = sub.add_parser(name)
        sp.add_argument("--root", required=True, help="Directory for mlflow.db and artifacts")
        sp.add_argument("--n", type=int, default=30, help="Runs to write (per writer for 'concurrent')")
        sp.add_argument("--label", default=None, help="Batch label (default: unique per run)")
        sp.set_defaults(fn=fn)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
