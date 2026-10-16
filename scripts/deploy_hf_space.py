"""Deploy the serving image to a Hugging Face Space (Docker SDK).

A Space is its own git repo whose root README.md needs YAML front matter, so instead of pushing
this repository as is, we stage exactly what the image needs into a temporary folder and upload
that: Dockerfile, pyproject.toml, src/, models/champion/ and a Space README.

Usage:
    python scripts/deploy_hf_space.py --space USER/NAME --dry-run         # stage only, no network
    python scripts/deploy_hf_space.py --space USER/NAME                   # upload (needs HF_TOKEN)

The token is read from the HF_TOKEN environment variable and never written anywhere.
Requires:  pip install huggingface_hub
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: Churn Prediction API
emoji: 📉
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

FastAPI service serving the champion churn model from the AI Experiment Lab project.

- `GET /docs`: interactive API documentation
- `GET /health`, `GET /model`, `POST /predict`, `POST /predict/batch`

Source and documentation: see the project repository.
"""

# README.md is included because pyproject.toml declares it as the package readme.
FILES = ["Dockerfile", "pyproject.toml"]
DIRS = ["src", "models/champion"]


def stage(dest: Path) -> list[str]:
    """Copy what the image needs into ``dest``; return the staged relative paths."""
    if not (ROOT / "models/champion/model.skops").exists():
        raise SystemExit("models/champion/model.skops not found: export the champion first (notebook 05).")
    for name in FILES:
        shutil.copy2(ROOT / name, dest / name)
    for name in DIRS:
        shutil.copytree(
            ROOT / name, dest / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
        )
    # The Dockerfile copies README.md for the package metadata; the Space's README is the front matter file.
    (dest / "README.md").write_text(SPACE_README, encoding="utf-8")
    return sorted(str(p.relative_to(dest)).replace("\\", "/") for p in dest.rglob("*") if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="Space id, e.g. username/churn-api")
    ap.add_argument("--dry-run", action="store_true", help="stage the files and list them; do not upload")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        files = stage(dest)
        print(f"Staged {len(files)} files for {args.space}:")
        for f in files:
            print("  ", f)
        if args.dry_run:
            return 0

        token = os.environ.get("HF_TOKEN")
        if not token:
            print("HF_TOKEN is not set. Create a write token at huggingface.co/settings/tokens.", file=sys.stderr)
            return 1
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(
            folder_path=str(dest), repo_id=args.space, repo_type="space",
            commit_message="Deploy champion model",
        )
        host = args.space.replace("/", "-").lower()
        print(f"Uploaded. Build logs: https://huggingface.co/spaces/{args.space}")
        print(f"When running, the API is at https://{host}.hf.space (try /docs and /health)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
