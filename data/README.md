# Data directory

Datasets are **not committed to Git** (see `.gitignore`).

- `raw/`: original, immutable source files.
- `processed/`: derived datasets produced by the pipeline.

Where the data lives:

| Environment | Location |
|---|---|
| Colab | `/content/drive/MyDrive/ai-experiment-lab/data/` |
| Local (VS Code) | `data/raw/` (small samples only) |

Dataset provenance, licence and version are recorded in
`docs/decisions/ADR-003-initial-dataset.md`.
