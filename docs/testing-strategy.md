# Testing Strategy

**Status:** Draft. Expanded as modules are implemented.

| Level | Scope | Tooling | When added |
|---|---|---|---|
| Smoke | Package imports, version present | pytest | Phase 0 (exists) |
| Unit: data | Ingestion, profiling stats, each validation rule (pass and fail cases) | pytest, small synthetic frames | Phase 1 |
| Unit: preprocessing | No leakage (fit on train only), identical output train vs inference, unseen categories | pytest | Phase 1 |
| Unit: models and evaluation | Metric correctness on known inputs, seed reproducibility | pytest | Phase 1 |
| Integration | Dataset to trained pipeline on a tiny fixture dataset | pytest | Phase 1-2 |
| API | `/health`, `/predict` valid and invalid payloads, `/model-info` | pytest, FastAPI TestClient | Phase 5 |
| Model quality gate | Candidate vs production metric thresholds | pytest or script in CI | Phase 8 |
| Static | Lint | ruff | Phase 0 (exists) |

**Rules:** tests use small synthetic fixtures, never the full dataset; ML-heavy tests that
need Colab compute are marked and excluded from CI.
