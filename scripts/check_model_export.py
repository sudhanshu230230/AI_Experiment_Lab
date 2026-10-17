"""Check the committed model export before it is deployed.

Usage:  python scripts/check_model_export.py [models/champion] [--dockerfile Dockerfile]
        python scripts/check_model_export.py --print-version      # just print registry_version

Fails (exit 1) if the export cannot be loaded, lacks the metadata the API needs, or was trained
with a scikit-learn version different from the one the Dockerfile installs (a mismatch loads
silently in some cases and gives wrong predictions in others).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def dockerfile_sklearn(dockerfile: Path) -> str | None:
    m = re.search(r"^ARG SKLEARN_VERSION=(\S+)", dockerfile.read_text(), re.MULTILINE)
    return m.group(1) if m else None


def check(model_dir: Path, dockerfile: Path) -> list[str]:
    """Return a list of problems (empty = fine)."""
    problems: list[str] = []
    meta_path = model_dir / "metadata.json"
    if not (model_dir / "model.skops").exists() or not meta_path.exists():
        return [f"{model_dir} must contain model.skops and metadata.json"]
    meta = json.loads(meta_path.read_text())
    for key in ("registered_name", "registry_version", "input_columns", "libraries"):
        if key not in meta:
            problems.append(f"metadata.json is missing '{key}'")
    trained = meta.get("libraries", {}).get("scikit-learn")
    image = dockerfile_sklearn(dockerfile) if dockerfile.exists() else None
    if trained and image and trained != image:
        problems.append(f"model trained with scikit-learn {trained} but the Dockerfile installs {image}; "
                        f"change ARG SKLEARN_VERSION in the Dockerfile to {trained}")
    try:
        from ai_experiment_lab.registry import load_exported
        load_exported(model_dir)
    except Exception as exc:  # noqa: BLE001 - any load failure is reported as a problem, not a crash
        problems.append(f"model failed to load: {type(exc).__name__}: {exc}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", nargs="?", default="models/champion")
    ap.add_argument("--dockerfile", default="Dockerfile")
    ap.add_argument("--print-version", action="store_true")
    args = ap.parse_args(argv)
    model_dir = Path(args.model_dir)
    if args.print_version:
        print(json.loads((model_dir / "metadata.json").read_text())["registry_version"])
        return 0
    problems = check(model_dir, Path(args.dockerfile))
    meta = json.loads((model_dir / "metadata.json").read_text()) if (model_dir / "metadata.json").exists() else {}
    for p in problems:
        print("FAIL ", p)
    if not problems:
        print(f"OK  {meta.get('registered_name')} version {meta.get('registry_version')} "
              f"(source run {meta.get('source_run_id')}, scikit-learn {meta.get('libraries', {}).get('scikit-learn')})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
