# models/

`models/champion/` holds the model the API serves (ADR-005):

- `model.skops`: fitted preprocessing + model Pipeline (skops format).
- `metadata.json`: registry name and version, source MLflow run, metrics, dataset hash, code and library versions, input columns.

It is produced by `notebooks/05_model_registry.ipynb` in Colab and committed because it is small
(logistic regression). A larger model would be stored as a release asset or in object storage instead.
Everything else under `models/` stays out of git (see `.gitignore`).
