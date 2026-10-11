# ADR-001: MLflow tracking and artifact storage

- **Status:** Proposed (spike in progress; decision is finalised when the results below are filled in)
- **Date:** 2026-10-10
- **Phase:** 2

## Context
Training runs in Google Colab, whose local disk is wiped when the runtime ends. MLflow
runs, parameters, metrics and model artifacts must survive between sessions and be
reachable later from the API and dashboard. No paid infrastructure is available.

## Findings from the MLflow documentation (checked 2026-10-10)
Source: [MLflow backend store documentation](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/).
- The **local file system backend is legacy**: marked "to be deprecated soon", in maintenance mode,
  with no further updates. This rules out the `file:` tracking URI that the Colab setup cell
  currently sets (it is unused until Phase 2 code lands, and will be changed after this spike).
- A **relational database is the recommended backend and the default**. SQLite is the default
  (`sqlite:///mlflow.db`); PostgreSQL, MySQL and MSSQL are also supported.
- SQLite allows **one concurrent writer** (stated in the docs for trace processing).
- The documentation says **nothing about running SQLite on a network filesystem** such as a
  Drive mount. SQLite file locking is known to be unreliable on some network and FUSE
  filesystems, so this must be tested, not assumed.

## Options considered
1. **SQLite database and artifacts both on the Drive mount.** Simplest, persistent, free.
   Risk: locking or corruption on the Drive filesystem; slower writes.
2. **SQLite on Colab local disk, artifacts on Drive, database copied to Drive at the end of each session.**
   Avoids locking on Drive. Risk: runs since the last copy are lost if the runtime dies; copying
   needs a discipline (or a helper function).
3. **Hosted MLflow (for example DagsHub).** Persistent, shareable, always-on UI.
   Costs: third-party account, free-tier limits, an extra dependency.
4. **Self-hosted server on a free cloud tier with PostgreSQL.** Closest to production. Costs: setup
   effort; free tiers may sleep or reset storage.
5. **Legacy `file:` store.** Rejected: deprecated (see findings).

## Spike (how option 1 is tested)
`scripts/mlflow_storage_spike.py`, run via `notebooks/spike_mlflow_storage.ipynb`.
It writes 30 runs (3 params, 3 metrics, 1 artifact each), reads them back from a fresh connection,
runs two writers concurrently, and verifies persistence after deleting the Colab runtime.
It was tested on local disk before the Colab run (30 runs in under a second, no errors, concurrent
writers fine), so any failure on Drive is attributable to Drive.

**Option 1 is accepted if all of these hold on Drive:**
1. `write`: all 30 runs written and read back with full metadata and artifacts, zero errors.
2. `verify` after a runtime restart: everything still present and readable.
3. Time per run is under 2 seconds (otherwise logging 50 Optuna trials becomes tedious).
4. `concurrent`: either passes, or fails in a way we can live with (we run one training session
   at a time, and will document "one writer" as a rule).

If 1 or 2 fails, use option 2 (and re-run the spike on that layout) or option 3.

## Spike results
*To be filled in after the Colab run.*

| Check | Local disk | Google Drive |
|---|---|---|
| write: runs OK / failed | | |
| time per run | | |
| concurrent writers | | |
| persistence after restart | | |

## Decision
*To be written after the results.*

## Consequences (apply whichever option is chosen)
- The tracking URI is configuration (`configs/default.yaml`), never hardcoded in code.
- The Colab setup cell will set the URI from config; the `file:` URI will be removed.
- One training session writes to the store at a time.
- Moving to a hosted or self-hosted server later is a config change plus a data migration.

## Interview note
MLflow's file store is being deprecated, so I used a SQLite database. Because Colab is
ephemeral and SQLite behaves differently on network drives, I tested the storage layout
(including concurrent writers and a runtime restart) before building on it.
